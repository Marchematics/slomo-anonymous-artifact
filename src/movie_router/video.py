"""Video window extraction and frozen embedding adapters for SF20K caches."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class VideoWindows:
    frames: np.ndarray
    starts: np.ndarray
    duration_seconds: float


def read_frame_at(
    video_path: str | Path,
    timestamp_seconds: float,
    *,
    frame_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Read one RGB frame at a cached window start timestamp."""

    if timestamp_seconds < 0.0:
        raise ValueError("timestamp_seconds must be nonnegative")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("install opencv-python to read video frames") from exc
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0.0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"video has invalid FPS/frame count: {video_path}")
    frame_index = min(int(round(timestamp_seconds * fps)), frame_count - 1)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"could not read frame at {timestamp_seconds}s from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame_size
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def parse_vtt(path: str | Path) -> tuple[tuple[float, float, str], ...]:
    """Parse simple WebVTT cues into ``(start, end, text)`` tuples."""

    content = Path(path).read_text(encoding="utf-8", errors="replace")
    cues: list[tuple[float, float, str]] = []

    def timestamp(value: str) -> float:
        parts = value.strip().replace(",", ".").split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours, minutes, seconds = "0", parts[0], parts[1]
        else:
            raise ValueError(f"invalid VTT timestamp: {value}")
        return 3600.0 * float(hours) + 60.0 * float(minutes) + float(seconds)

    lines = content.splitlines()
    index = 0
    while index < len(lines):
        match = re.match(r"^\s*(\S+)\s+-->\s+(\S+)", lines[index])
        if not match:
            index += 1
            continue
        start, end = timestamp(match.group(1)), timestamp(match.group(2))
        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            text_lines.append(re.sub(r"<[^>]+>", "", lines[index]).strip())
            index += 1
        text = " ".join(value for value in text_lines if value)
        if text:
            cues.append((start, end, text))
    return tuple(cues)


def subtitles_for_windows(
    cues: tuple[tuple[float, float, str], ...],
    starts: np.ndarray,
    *,
    window_seconds: float,
) -> tuple[str, ...]:
    """Aggregate subtitle cues that overlap each fixed video window."""

    if window_seconds <= 0.0:
        raise ValueError("window_seconds must be positive")
    return tuple(
        " ".join(
            text
            for cue_start, cue_end, text in cues
            if cue_end > float(start) and cue_start < float(start) + window_seconds
        )
        for start in np.asarray(starts, dtype=float)
    )


_TRANSCRIPT_STOPWORDS = frozenset(
    "a an and are as at be been but by did do does for from had has have he her him his how i if in is it me my of on or our she that the their them they this to was we were what when where which who why will with you your"
    .split()
)


def rank_question_relevant_cues(
    cues: tuple[tuple[float, float, str], ...], question: str
) -> tuple[int, ...]:
    """Rank transcript cues by lightweight question-term TF-IDF relevance."""

    if not cues:
        return ()

    def tokens(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", value.lower())
            if token not in _TRANSCRIPT_STOPWORDS and len(token) > 1
        }

    query_tokens = tokens(question)
    if not query_tokens:
        return tuple(range(len(cues)))
    cue_tokens = [tokens(text) for _start, _end, text in cues]
    document_frequency: dict[str, int] = {}
    for cue_terms in cue_tokens:
        for term in cue_terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    scored: list[tuple[float, int]] = []
    for index, (cue_terms, (_start, _end, text)) in enumerate(zip(cue_tokens, cues)):
        overlap = query_tokens.intersection(cue_terms)
        score = sum(
            math.log((len(cues) + 1) / (document_frequency[term] + 1)) + 1.0
            for term in overlap
        ) / math.sqrt(max(len(text.split()), 1))
        if score > 0.0:
            scored.append((score, index))
    return tuple(index for _score, index in sorted(scored, reverse=True))


def question_relevant_transcript(
    cues: tuple[tuple[float, float, str], ...],
    question: str,
    *,
    max_chars: int,
    neighbor_cues: int = 1,
) -> str:
    """Pack question-relevant VTT cues while retaining local dialogue context.

    A prefix cap can discard the answer in a late scene.  This lightweight
    lexical selector ranks cues with question-term TF-IDF, expands selected
    cues by nearby dialogue, and restores chronological order.  It is a
    retrieval heuristic for transcript packing, not a semantic answerer.
    """

    if max_chars < 1 or neighbor_cues < 0:
        raise ValueError("max_chars must be positive and neighbor_cues nonnegative")
    if not cues:
        return ""
    lines = [f"[t={start:.1f}s] {text}" for start, _end, text in cues]
    joined = "\n".join(lines)
    if len(joined) <= max_chars:
        return joined

    ranked = rank_question_relevant_cues(cues, question)
    if not ranked:
        return joined[:max_chars]

    selected: set[int] = set()
    for index in ranked:
        expansion = range(
            max(0, index - neighbor_cues),
            min(len(cues), index + neighbor_cues + 1),
        )
        proposal = selected.union(expansion)
        packed = "\n".join(lines[item] for item in sorted(proposal))
        if len(packed) <= max_chars:
            selected = proposal
        elif index not in selected:
            single = "\n".join(lines[item] for item in sorted(selected.union({index})))
            if len(single) <= max_chars:
                selected.add(index)
    if not selected:
        selected.add(ranked[0])
    packed = "\n".join(lines[item] for item in sorted(selected))
    return packed[:max_chars]


def extract_center_windows(
    video_path: str | Path,
    *,
    window_seconds: float = 8.0,
    stride_seconds: float = 4.0,
    frame_size: tuple[int, int] = (224, 224),
    max_windows: int = 256,
) -> VideoWindows:
    """Extract one RGB center frame per fixed temporal window using OpenCV."""

    if window_seconds <= 0.0 or stride_seconds <= 0.0 or max_windows < 1:
        raise ValueError("window_seconds, stride_seconds, and max_windows must be positive")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("install opencv-python to extract video windows") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0.0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"video has invalid FPS/frame count: {video_path}")
    duration = frame_count / fps
    last_start = max(0.0, duration - window_seconds)
    starts = np.arange(0.0, last_start + 1e-6, stride_seconds, dtype=float)
    if starts.size > max_windows:
        positions = np.linspace(0, starts.size - 1, max_windows).round().astype(int)
        starts = starts[positions]
    height, width = frame_size
    frames: list[np.ndarray] = []
    valid_starts: list[float] = []
    for start in starts:
        target_frame = int(round((start + min(window_seconds / 2.0, duration / 2.0)) * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, min(target_frame, frame_count - 1))
        ok, frame = capture.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
        valid_starts.append(float(start))
    capture.release()
    if not frames:
        raise ValueError(f"no readable frames in video: {video_path}")
    return VideoWindows(np.stack(frames).astype(np.uint8), np.asarray(valid_starts), duration)


class ColorHashEmbedder:
    """Dependency-free smoke encoder with a stable text/image dimension.

    This is a cache-pipeline diagnostic, not a competitive representation.
    Use ``OpenCLIPEmbedder`` for the actual visual/text retrieval run.
    """

    dimension = 96

    def encode_images(self, frames: np.ndarray) -> np.ndarray:
        images = np.asarray(frames, dtype=np.uint8)
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("frames must have shape (batch, height, width, 3)")
        features: list[np.ndarray] = []
        for image in images:
            channels = [image[..., index] for index in range(3)]
            histograms = [np.histogram(channel, bins=32, range=(0, 256))[0] for channel in channels]
            vector = np.concatenate(histograms).astype(np.float32)
            vector /= np.maximum(np.linalg.norm(vector), 1e-8)
            features.append(vector)
        return np.stack(features)

    def encode_text(self, texts: Iterable[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for text in texts:
            vector = np.empty(self.dimension, dtype=np.float32)
            encoded = str(text).lower().encode("utf-8")
            for index in range(self.dimension):
                digest = hashlib.sha256(encoded + index.to_bytes(4, "little")).digest()
                integer = int.from_bytes(digest[:8], "little")
                vector[index] = (integer / 2**63) - 1.0
            vector /= np.maximum(np.linalg.norm(vector), 1e-8)
            vectors.append(vector)
        return np.stack(vectors)


class OpenCLIPEmbedder:
    """Frozen OpenCLIP image/text encoder with lazy optional dependencies."""

    def __init__(
        self,
        *,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
        batch_size: int = 32,
        weights_only: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("install open_clip_torch and torch for OpenCLIP encoding") from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for OpenCLIP but is unavailable")
        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
            weights_only=weights_only,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.dimension = int(self.model.visual.output_dim)

    def encode_images(self, frames: np.ndarray) -> np.ndarray:
        images = np.asarray(frames, dtype=np.uint8)
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("install Pillow for OpenCLIP image preprocessing") from exc
        tensors = [self.preprocess(Image.fromarray(image)) for image in images]
        outputs: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(tensors), self.batch_size):
                batch = self._torch.stack(tensors[start : start + self.batch_size]).to(self.device)
                value = self.model.encode_image(batch, normalize=True).detach().cpu().numpy()
                outputs.append(value.astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def encode_text(self, texts: Iterable[str]) -> np.ndarray:
        values = list(texts)
        outputs: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                tokens = self.tokenizer(values[start : start + self.batch_size]).to(self.device)
                value = self.model.encode_text(tokens, normalize=True).detach().cpu().numpy()
                outputs.append(value.astype(np.float32))
        return np.concatenate(outputs, axis=0)


def build_embedder(name: str, *, device: str = "cuda", batch_size: int = 32):
    """Construct one of the explicit cache encoders."""

    if name == "color_hash":
        return ColorHashEmbedder()
    if name == "open_clip":
        return OpenCLIPEmbedder(device=device, batch_size=batch_size)
    raise ValueError(f"unknown encoder: {name!r}")
