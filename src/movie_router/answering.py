"""Model-agnostic prompt packing and answer normalization for selected windows."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ContextItem:
    """One selected temporal window passed to the final answerer."""

    index: int
    start_seconds: float
    subtitle: str = ""
    image: Any = None


def infer_question_type(question: str) -> str:
    """Assign a conservative answer-plan label from the question wording.

    This is deliberately a prompt-routing heuristic, not a claim that the
    question has a single semantic type.  The fallback keeps the original
    generic prompt for ambiguous questions.
    """

    normalized = " ".join(str(question).lower().split())
    if not normalized:
        return "general"
    dialogue_cues = (
        "what is said",
        "what was said",
        "what does he say",
        "what does she say",
        "what did he say",
        "what did she say",
        "tell",
        "told",
        "ask",
        "said",
        "say to",
        "conversation",
        "dialogue",
        "respond",
        "reply",
    )
    visual_cues = (
        "look like",
        "wear",
        "wearing",
        "color",
        "colour",
        "appearance",
        "holding",
        "carry",
        "carrying",
        "object",
        "in the trunk",
        "what is in",
        "where does",
        "where is",
        "where are",
        "location",
        "room",
        "enter",
        "door",
        "vehicle",
        "car",
    )
    temporal_cues = (
        "before",
        "after",
        "earlier",
        "later",
        "first",
        "next",
        "then",
        "eventually",
        "at the beginning",
        "in the final",
        "at the end",
        "following",
    )
    causal_cues = (
        "why",
        "how did",
        "how does",
        "how was",
        "how is",
        "what caused",
        "reason",
        "motivat",
        "purpose",
        "meaning",
        "imply",
        "signif",
    )
    spatial_cues = (
        "direction",
        "compass",
        "north",
        "south",
        "east",
        "west",
        "map",
        "path",
        "distance",
        "line-of-sight",
        "straight-line",
        "landmark",
        "walking tour",
        "real world",
    )
    if any(cue in normalized for cue in dialogue_cues):
        return "dialogue"
    if any(cue in normalized for cue in visual_cues):
        return "visual"
    if any(cue in normalized for cue in spatial_cues):
        return "spatial"
    if any(cue in normalized for cue in temporal_cues):
        return "temporal"
    if any(cue in normalized for cue in causal_cues):
        return "causal"
    return "general"


_QUESTION_TYPE_INSTRUCTIONS = {
    "visual": (
        "This is primarily a visual-fact question. Inspect the supplied frames first. "
        "Report the visible person, object, color, place, or action exactly; do not invent "
        "details from general plot knowledge."
    ),
    "dialogue": (
        "This is primarily a dialogue question. Use the timestamped transcript to identify "
        "the exact speaker and statement. Do not replace what was said with a nearby visual fact."
    ),
    "temporal": (
        "This is a chronology question. Establish the order of the relevant events from the "
        "timestamps and answer the requested before/after/first/final relation explicitly."
    ),
    "causal": (
        "This is a causal or narrative question. Find the event that explains the action or "
        "outcome, check for later revelations, and answer with the direct reason rather than "
        "merely repeating what happened."
    ),
    "spatial": (
        "This is a spatial-navigation question. Track the camera direction, path, landmarks, "
        "or distance across the supplied timestamps. For compass questions, infer the net "
        "orientation change from the sequence rather than guessing from one frame. For map "
        "questions, compare the full path geometry with every candidate map."
    ),
    "general": (
        "Answer the exact question directly. Use the most specific evidence supplied and do "
        "not add unsupported plot details."
    ),
}


def build_vlm_messages(
    question: str,
    contexts: Sequence[ContextItem],
    *,
    options: Sequence[str] = (),
    transcript: str = "",
    story_memory: str = "",
    qa_memory: str = "",
    question_type: str = "general",
    answer_style: str = "concise",
    max_subtitle_chars: int = 600,
    extra_instruction: str = "",
    task_mode: str = "answer",
) -> list[dict[str, Any]]:
    """Build chronological multimodal messages without binding to a VLM API."""

    if not question.strip() or max_subtitle_chars < 1:
        raise ValueError("question must be nonempty and max_subtitle_chars positive")
    if question_type not in _QUESTION_TYPE_INSTRUCTIONS:
        raise ValueError(f"unknown question_type: {question_type}")
    if task_mode not in {"answer", "facts", "audit"}:
        raise ValueError(f"unknown task_mode: {task_mode}")
    if answer_style not in {"concise", "supported", "reasoned"}:
        raise ValueError(f"unknown answer_style: {answer_style}")
    if answer_style == "concise":
        answer_instruction = "Return a concise textual answer, never an option letter or multiple-choice label."
    elif answer_style == "supported":
        answer_instruction = (
            "Return a direct, semantically complete textual answer. Include the key entity, "
            "reason, action, or temporal qualifier required by the question. Use at most two "
            "short sentences and do not speculate or hedge."
        )
    else:
        answer_instruction = (
            "First resolve the answer internally from the evidence, checking the relevant "
            "entity, temporal order, and causal relation. Then return only the concise final "
            "textual answer; never reveal the reasoning, uncertainty, or an option letter."
        )
    if task_mode == "facts":
        if options:
            raise ValueError("facts task cannot be combined with multiple-choice options")
        system = (
            "Extract directly observable evidence from the supplied movie material. "
            "Do not answer the question, propose a candidate answer, summarize the story, "
            "or infer motivation, causality, identity, or temporal conclusions. "
            "Preserve contradictory observations and use timestamps. "
            "Return only the requested JSON object."
        )
    elif task_mode == "audit":
        if options:
            raise ValueError("audit task cannot be combined with multiple-choice options")
        system = (
            "Audit a frozen draft answer against the supplied directly extracted movie facts. "
            "The draft is the current champion and must be preserved by default. "
            "Do not invent facts, use outside knowledge, or answer from question priors. "
            "Only mark REPAIR when a core semantic slot is directly contradicted or a required "
            "relation is directly supported by fact IDs. Only mark CANONICALIZE for a safe deletion "
            "of redundant wording. If facts are insufficient, mark INSUFFICIENT and preserve the draft. "
            "Return only the requested JSON object."
        )
    else:
        system = (
            "Answer the question using only the supplied movie evidence. "
            "Respect the temporal order. "
            + _QUESTION_TYPE_INSTRUCTIONS[question_type]
            + (
                "For multiple choice, return only the option letter."
                if options
                else answer_instruction
            )
        )
    prompt = (
        f"Question type: {question_type}\n"
        f"Question: {question.strip()}\n"
    )
    if options:
        prompt += "Options:\n" + "\n".join(
            f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
        ) + "\n"
    if extra_instruction.strip():
        prompt += extra_instruction.strip() + "\n"
    if story_memory.strip():
        prompt += (
            "A compact global story memory is supplied below. Use it to resolve "
            "long-range identities, motivations, chronology, and later revelations. "
            "Treat it as a fallible summary and prefer direct transcript or visual "
            "evidence when the two conflict:\n"
            + story_memory.strip()
            + "\n"
        )
    if qa_memory.strip():
        prompt += (
            "Weak cross-question notes from other questions about this same movie are supplied below. "
            "They are independently generated and may be wrong; use them only to connect entities "
            "or chronology, and prefer direct evidence when they conflict:\n"
            + qa_memory.strip()
            + "\n"
        )
    if transcript.strip():
        prompt += (
            "A timestamped transcript of the movie is also supplied. Use it to recover spoken facts, "
            "but prefer visually grounded evidence when the transcript is noisy:\n"
            + transcript.strip()
            + "\n"
        )
    prompt += "Evidence follows in chronological order:\n"
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for context in sorted(contexts, key=lambda item: (item.start_seconds, item.index)):
        evidence_text = f"[t={context.start_seconds:.1f}s]"
        if context.subtitle:
            evidence_text += f" {context.subtitle[:max_subtitle_chars]}"
        content.append({"type": "text", "text": evidence_text})
        if context.image is not None:
            content.append({"type": "image", "image": context.image})
    if task_mode == "facts":
        final_text = (
            "Do not answer the question. Extract only facts directly stated in the transcript "
            "or directly visible in the frames. Return exactly one valid JSON object with these "
            "keys: required_roles and facts. Use compact facts with only keys id, time, modality, "
            "s (subject), p (directly observed predicate), o (object), and q (short direct quote). "
            "Return at most 3 high-value facts, keep every field short, and omit q when unavailable. "
            "Do not include answer, candidate_answer, conclusion, explanation, therefore, probably, "
            "motivation, or any field that states what the answer is. If evidence is insufficient, "
            "return an empty facts list. Return JSON only, with no markdown fences."
        )
    elif task_mode == "audit":
        final_text = (
            "Audit the frozen draft answer. Return exactly one valid JSON object with keys: status, "
            "answer_type, slots, revised_answer, support_ids, reason. status must be one of KEEP, "
            "CANONICALIZE, REPAIR, INSUFFICIENT. slots must list each important semantic slot with "
            "name, value, supported, and fact_ids. Use KEEP when the draft is adequately supported. "
            "Use CANONICALIZE only when deleting redundant wording without changing facts. Use REPAIR "
            "only when the facts directly support a minimal correction. Use INSUFFICIENT when the facts "
            "cannot justify a change. revised_answer must be empty for KEEP and INSUFFICIENT. For a "
            "change, revised_answer may contain only content supported by listed fact_ids. Do not add "
            "new entities, actions, causes, temporal relations, negation, or numbers. JSON only, no markdown."
        )
    else:
        final_text = (
            "Now give the final answer. "
            + (
                "Return only one option letter."
                if options
                else (
                    "Return only the shortest correct textual answer, preferably one sentence. "
                    "Do not add analysis, evidence labels, confidence, or a hedge."
                    if answer_style == "concise"
                    else "Return the direct answer with only the minimal supporting detail "
                    "needed for semantic completeness. Do not add analysis, confidence, or a hedge."
                )
            )
        )
    content.append({"type": "text", "text": final_text})
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def extract_choice(raw_answer: str, option_count: int) -> int | None:
    """Extract a zero-based option index from a free-form VLM response."""

    if option_count < 1 or option_count > 26:
        raise ValueError("option_count must lie in [1, 26]")
    answer = str(raw_answer).strip()
    match = re.search(r"(?:answer|choice|option)\s*[:\-]?\s*([A-Z])\b", answer, re.I)
    if match is None:
        match = re.search(r"\b([A-Z])\b", answer, re.I)
    if match is None:
        numeric = re.search(r"\b(?:option\s*)?([0-9]+)\b", answer, re.I)
        if numeric is None:
            return None
        index = int(numeric.group(1))
        if 1 <= index <= option_count:
            return index - 1
        if 0 <= index < option_count:
            return index
        return None
    index = ord(match.group(1).upper()) - ord("A")
    return index if index < option_count else None


def clean_open_answer(raw_answer: str) -> str:
    """Remove common answer prefixes while retaining the generated content."""

    answer = re.sub(r"^\s*(?:answer|final answer)\s*[:\-]\s*", "", str(raw_answer), flags=re.I)
    return " ".join(answer.strip().split())


def write_predictions_csv(
    path: str | Path,
    question_ids: Iterable[str],
    answers: Iterable[str],
    *,
    answer_column: str = "answer",
) -> None:
    """Write a minimal prediction file and refuse accidental overwrites."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite prediction file: {destination}")
    ids, values = list(question_ids), list(answers)
    if len(ids) != len(values) or len(set(ids)) != len(ids):
        raise ValueError("question_ids must be unique and align with answers")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", answer_column])
        writer.writeheader()
        writer.writerows(
            {"question_id": question_id, answer_column: answer}
            for question_id, answer in zip(ids, values)
        )


def validate_predictions_csv(
    path: str | Path,
    expected_question_ids: Iterable[str],
    *,
    answer_column: str = "answer",
) -> None:
    """Validate row count, ordering, uniqueness, and non-empty answers."""

    expected = list(expected_question_ids)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"question_id", answer_column}:
        raise ValueError("prediction CSV has an unexpected header")
    actual = [row.get("question_id", "") for row in rows]
    answers = [row.get(answer_column, "").strip() for row in rows]
    if actual != expected:
        raise ValueError("prediction question IDs do not exactly match the expected order")
    if len(actual) != len(set(actual)) or any(not answer for answer in answers):
        raise ValueError("prediction IDs must be unique and answers non-empty")
