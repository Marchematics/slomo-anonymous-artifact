"""Optional local Qwen2.5-VL answerer for the final selected evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .answering import ContextItem, build_vlm_messages, clean_open_answer, extract_choice


def _install_pcre_compat() -> None:
    """Expose the installed pcre2 binding under GPTQModel's legacy name."""

    try:
        import pcre  # noqa: F401
        return
    except ModuleNotFoundError as exc:
        if exc.name != "pcre":
            raise
    import pcre2
    import sys

    if not hasattr(pcre2, "Flag"):
        class _PcreFlag:
            MULTILINE = pcre2.M
            DOTALL = pcre2.S
            CASELESS = pcre2.I

        pcre2.Flag = _PcreFlag
    sys.modules["pcre"] = pcre2


@dataclass(frozen=True)
class QwenAnswererConfig:
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    device: str = "cuda"
    load_in_4bit: bool = False
    max_new_tokens: int = 64
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 768 * 28 * 28
    enable_thinking: bool | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1 or self.min_pixels < 1 or self.max_pixels < self.min_pixels:
            raise ValueError("invalid Qwen answerer generation/image budget")


@dataclass(frozen=True)
class ChoiceScores:
    """Forced-choice scores and uncertainty for one multiple-choice prompt."""

    scores: tuple[float, ...]
    top1: int
    top2: int
    margin: float

    def __post_init__(self) -> None:
        if len(self.scores) < 2:
            raise ValueError("choice scoring requires at least two options")
        if not 0 <= self.top1 < len(self.scores) or not 0 <= self.top2 < len(self.scores):
            raise ValueError("choice ranking is outside the score vector")
        if self.top1 == self.top2 or self.margin < 0.0:
            raise ValueError("choice ranking must contain distinct top options and nonnegative margin")


class QwenVLAnswerer:
    """Lazy local answerer; model weights load only when this class is built."""

    def __init__(self, config: QwenAnswererConfig | None = None) -> None:
        cfg = config or QwenAnswererConfig()
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Qwen answerer requested CUDA, but CUDA is unavailable")
        try:
            from transformers import AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("install transformers, accelerate, bitsandbytes, and qwen-vl-utils") from exc
        kwargs = {"torch_dtype": torch.float16 if cfg.device.startswith("cuda") else torch.float32}
        if cfg.device.startswith("cuda"):
            kwargs["device_map"] = "auto"
        if cfg.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self.config = cfg
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_name,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.max_pixels,
        )
        # Qwen3-VL and Qwen2.5-VL share the processor/runtime contract but use
        # different model classes. Keeping the dispatch here lets the
        # competition runner benchmark a stronger local answerer without
        # changing routing artifacts or prediction schemas.
        if "Qwen3.5" in cfg.model_name or "Qwen3_5" in cfg.model_name:
            from transformers import Qwen3_5ForConditionalGeneration

            model_class = Qwen3_5ForConditionalGeneration
            kwargs["torch_dtype"] = torch.bfloat16 if cfg.device.startswith("cuda") else torch.float32
        elif "Qwen3-VL" in cfg.model_name:
            from transformers import Qwen3VLForConditionalGeneration

            model_class = Qwen3VLForConditionalGeneration
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_class = Qwen2_5_VLForConditionalGeneration
        # AutoAWQ 0.2.x imports a class renamed by newer Transformers. The
        # AWQ model carries its own quantization config, so install the exact
        # compatibility alias only for that optional loading path.
        if "AWQ" in cfg.model_name:
            import transformers.activations as activations

            if not hasattr(activations, "PytorchGELUTanh"):
                activations.PytorchGELUTanh = activations.GELUTanh
        # GPTQModel is imported by both the AWQ and GPTQ Transformers
        # integration paths. Install the compatibility alias before either
        # path reaches ``from_pretrained``.
        if "GPTQ" in cfg.model_name or "AWQ" in cfg.model_name:
            _install_pcre_compat()
        if "AWQ" in cfg.model_name:
            # GPTQModel's automatic AWQ path currently prefers Marlin, whose
            # CUDA kernel rejects some Qwen-VL projection widths (for example
            # the vision MLP width 4304).  The released checkpoint is GEMM
            # AWQ, so request the compatible GEMM backend explicitly.
            from transformers import AutoConfig

            loaded_config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
            if isinstance(getattr(loaded_config, "quantization_config", None), dict):
                quant_config = loaded_config.quantization_config
                quant_config["backend"] = "gemm"
                # Qwen-VL stores this skip as ``visual``, while the current
                # Transformers matcher sees module paths rooted at
                # ``model.visual``. Keep both spellings so the vision tower
                # is not accidentally sent through the AWQ kernel.
                skips = list(quant_config.get("modules_to_not_convert", []))
                if "model.visual" not in skips:
                    skips.append("model.visual")
                quant_config["modules_to_not_convert"] = skips
            kwargs["config"] = loaded_config
        self.model = model_class.from_pretrained(cfg.model_name, **kwargs)
        if not cfg.device.startswith("cuda"):
            self.model.to(cfg.device)
        self.model.eval()

    def _prepare_inputs(
        self,
        question: str,
        contexts: Sequence[ContextItem],
        *,
        options: Sequence[str] = (),
        transcript: str = "",
        story_memory: str = "",
        qa_memory: str = "",
        question_type: str = "general",
        answer_style: str = "concise",
        extra_instruction: str = "",
        task_mode: str = "answer",
    ) -> dict[str, object]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("install qwen-vl-utils for Qwen2.5-VL image processing") from exc
        messages = build_vlm_messages(
            question,
            contexts,
            options=options,
            transcript=transcript,
            story_memory=story_memory,
            qa_memory=qa_memory,
            question_type=question_type,
            answer_style=answer_style,
            extra_instruction=extra_instruction,
            task_mode=task_mode,
        )
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if "Qwen3.5" in self.config.model_name or "Qwen3_5" in self.config.model_name:
            # Qwen3.5 defaults to a long reasoning block.  MCQA runners need
            # the answer token directly so generation is bounded and parsable.
            template_kwargs["enable_thinking"] = (
                False if self.config.enable_thinking is None else self.config.enable_thinking
            )
        text = self.processor.apply_chat_template(messages, **template_kwargs)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    @torch.inference_mode()
    def answer(
        self,
        question: str,
        contexts: Sequence[ContextItem],
        *,
        options: Sequence[str] = (),
        transcript: str = "",
        story_memory: str = "",
        qa_memory: str = "",
        question_type: str = "general",
        answer_style: str = "concise",
        extra_instruction: str = "",
        task_mode: str = "answer",
    ) -> str:
        """Generate an open-ended answer, evidence JSON, audit JSON, or MCQA letter."""
        inputs = self._prepare_inputs(
            question,
            contexts,
            options=options,
            transcript=transcript,
            story_memory=story_memory,
            qa_memory=qa_memory,
            question_type=question_type,
            answer_style=answer_style,
            extra_instruction=extra_instruction,
            task_mode=task_mode,
        )
        # Greedy decoding avoids multinomial probability checks that can fail
        # on a long sequence of mixed-resolution image prompts in fp16.
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
        )
        prompt_length = inputs["input_ids"].shape[1]
        generated = generated[:, prompt_length:]
        answer = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        if options:
            choice = extract_choice(answer, len(options))
            return chr(65 + choice) if choice is not None else clean_open_answer(answer)
        return clean_open_answer(answer)

    @torch.inference_mode()
    def score_options(
        self,
        question: str,
        contexts: Sequence[ContextItem],
        *,
        options: Sequence[str],
        transcript: str = "",
        story_memory: str = "",
        qa_memory: str = "",
        question_type: str = "general",
        extra_instruction: str = "",
    ) -> ChoiceScores:
        """Score option letters from the next-token distribution.

        The prompt ends at the assistant generation boundary, so the returned
        values are directly comparable within one question. They are not
        calibrated probabilities across different prompts.
        """

        if len(options) < 2:
            raise ValueError("forced-choice scoring requires at least two options")
        if len(options) > 26:
            raise ValueError("forced-choice scoring supports at most 26 options")
        inputs = self._prepare_inputs(
            question,
            contexts,
            options=options,
            transcript=transcript,
            story_memory=story_memory,
            qa_memory=qa_memory,
            question_type=question_type,
            answer_style="concise",
            extra_instruction=extra_instruction,
        )
        outputs = self.model(**inputs, use_cache=False)
        next_token_logits = outputs.logits[0, -1, :].float()
        token_ids: list[int] = []
        tokenizer = self.processor.tokenizer
        for index in range(len(options)):
            letter = chr(65 + index)
            ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f"option letter {letter!r} is not a single tokenizer token; "
                    "forced-choice scoring cannot be used with this tokenizer"
                )
            token_ids.append(int(ids[0]))
        scores = tuple(float(next_token_logits[token_id].cpu()) for token_id in token_ids)
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        return ChoiceScores(
            scores=scores,
            top1=ranked[0],
            top2=ranked[1],
            margin=float(scores[ranked[0]] - scores[ranked[1]]),
        )

    @torch.inference_mode()
    def score_option_texts(
        self,
        question: str,
        contexts: Sequence[ContextItem],
        *,
        options: Sequence[str],
        transcript: str = "",
        question_type: str = "general",
        extra_instruction: str = "",
    ) -> ChoiceScores:
        """Rank options by length-normalized conditional text likelihood.

        This is an alternative to single-token letter scoring.  The prompt is
        explicitly changed to request verbatim option text, and each option is
        scored under the same multimodal prefix.  Scores are not comparable to
        ``score_options`` across methods; they are only a within-question
        selector.
        """

        if len(options) < 2:
            raise ValueError("text scoring requires at least two options")
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("install qwen-vl-utils for Qwen2.5-VL image processing") from exc

        messages = build_vlm_messages(
            question,
            contexts,
            options=options,
            transcript=transcript,
            question_type=question_type,
            answer_style="concise",
            extra_instruction=extra_instruction,
        )
        if isinstance(messages[-1].get("content"), list):
            final_block = messages[-1]["content"][-1]
            if isinstance(final_block, dict) and final_block.get("type") == "text":
                final_block["text"] = "Now give the final answer. Copy exactly one option text verbatim."

        rendered_prefix = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(
                {
                    "enable_thinking": (
                        False if self.config.enable_thinking is None else self.config.enable_thinking
                    )
                }
                if "Qwen3.5" in self.config.model_name or "Qwen3_5" in self.config.model_name
                else {}
            ),
        )
        image_inputs, video_inputs = process_vision_info(messages)
        prefix_kwargs: dict[str, object] = {
            "text": [rendered_prefix],
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        prefix_inputs = self.processor(**prefix_kwargs)
        prefix_ids = prefix_inputs["input_ids"][0]
        device = next(self.model.parameters()).device

        scores: list[float] = []
        tokenizer = self.processor.tokenizer
        for option in options:
            candidate_messages = list(messages) + [
                {"role": "assistant", "content": [{"type": "text", "text": str(option)}]}
            ]
            rendered = self.processor.apply_chat_template(
                candidate_messages,
                tokenize=False,
                add_generation_prompt=False,
                **(
                    {
                        "enable_thinking": (
                            False if self.config.enable_thinking is None else self.config.enable_thinking
                        )
                    }
                    if "Qwen3.5" in self.config.model_name or "Qwen3_5" in self.config.model_name
                    else {}
                ),
            )
            encoded_option = tokenizer.encode(str(option), add_special_tokens=False)
            candidate_kwargs: dict[str, object] = {
                "text": [rendered],
                "images": image_inputs,
                "videos": video_inputs,
                "padding": True,
                "return_tensors": "pt",
            }
            candidate_inputs = self.processor(**candidate_kwargs)
            input_ids = candidate_inputs["input_ids"][0]
            prefix_len = int(prefix_ids.shape[0])
            if not torch.equal(input_ids[:prefix_len].cpu(), prefix_ids.cpu()):
                raise RuntimeError("option-text prompt prefix changed between candidates")
            option_ids = torch.tensor(encoded_option, dtype=input_ids.dtype)
            start = prefix_len
            if encoded_option and not torch.equal(input_ids[start : start + len(encoded_option)].cpu(), option_ids):
                raise RuntimeError("option text is not aligned to the assistant token span")
            if not encoded_option:
                scores.append(float("-inf"))
                continue
            model_inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in candidate_inputs.items()
            }
            outputs = self.model(**model_inputs, use_cache=False)
            # Avoid materializing a float copy of the full sequence x vocab
            # logits tensor (which can exceed 2 GB for Qwen-VL).  Only the
            # answer-token positions are needed for this option's score.
            answer_logits = outputs.logits[0, start - 1 : start - 1 + len(encoded_option)]
            target = input_ids[start : start + len(encoded_option)].to(device)
            log_probs = answer_logits.log_softmax(dim=-1)
            scores.append(float(log_probs.gather(-1, target.unsqueeze(-1)).mean().cpu()))
            del outputs, answer_logits, log_probs, candidate_inputs, model_inputs

        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        return ChoiceScores(
            scores=tuple(scores),
            top1=ranked[0],
            top2=ranked[1],
            margin=float(scores[ranked[0]] - scores[ranked[1]]),
        )
