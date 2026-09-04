"""Assemble fixed-size question/candidate caches from extracted movie windows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .features import make_pair_features
from .sf20k import SF20KQuestion
from .video import VideoWindows


def build_candidate_arrays(
    questions: Sequence[SF20KQuestion],
    windows_by_movie: Mapping[str, VideoWindows],
    embeddings_by_movie: Mapping[str, np.ndarray],
    encoder: Any,
    *,
    max_candidates: int = 32,
    subtitles_by_movie: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, np.ndarray]:
    """Create router inputs and an answer-intervention proxy.

    Candidate windows are ranked by frozen query-to-frame similarity. The
    intervention proxy is the change from ``question`` to ``question + answer``
    similarity. It is intentionally recorded as a proxy target, not as a
    timestamp ground truth.
    """

    if not questions or max_candidates < 1:
        raise ValueError("questions must be nonempty and max_candidates positive")
    grouped: dict[str, list[tuple[int, SF20KQuestion]]] = defaultdict(list)
    for index, question in enumerate(questions):
        grouped[question.video_id].append((index, question))
    missing = set(grouped).difference(windows_by_movie).union(set(grouped).difference(embeddings_by_movie))
    if missing:
        raise KeyError(f"missing extracted windows or embeddings for movies: {sorted(missing)}")

    question_embeddings = encoder.encode_text([question.question for question in questions])
    answer_embeddings = encoder.encode_text(
        [f"{question.question} Answer: {question.answer}" for question in questions]
    )
    if question_embeddings.ndim != 2 or answer_embeddings.shape != question_embeddings.shape:
        raise ValueError("encoder text outputs must have shape (questions, dimension)")
    dimension = question_embeddings.shape[1]
    if subtitles_by_movie is not None:
        for video_id, movie_windows in windows_by_movie.items():
            if video_id in grouped and len(subtitles_by_movie.get(video_id, ())) != movie_windows.starts.size:
                raise ValueError(f"subtitle window count mismatch for movie {video_id}")
    features: list[np.ndarray] = []
    coarse_rows: list[np.ndarray] = []
    utility_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    timestamp_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    durations = np.zeros(len(questions), dtype=np.float32)
    subtitle_embeddings_by_movie: dict[str, np.ndarray] = {}
    if subtitles_by_movie is not None:
        for video_id, texts in subtitles_by_movie.items():
            if video_id in grouped:
                subtitle_embeddings_by_movie[video_id] = encoder.encode_text(texts)

    for question_index, question in enumerate(questions):
        movie_windows = windows_by_movie[question.video_id]
        movie_embeddings = np.asarray(embeddings_by_movie[question.video_id], dtype=np.float32)
        if movie_embeddings.shape != (movie_windows.starts.size, dimension):
            raise ValueError(f"embedding shape mismatch for movie {question.video_id}")
        query = question_embeddings[question_index]
        answer = answer_embeddings[question_index]
        extra_features = None
        if subtitles_by_movie is not None and question.video_id in subtitles_by_movie:
            subtitle_embeddings = subtitle_embeddings_by_movie[question.video_id]
            if subtitle_embeddings.shape != movie_embeddings.shape:
                raise ValueError(f"subtitle embedding shape mismatch for movie {question.video_id}")
            fused_embeddings = movie_embeddings + 0.25 * subtitle_embeddings
            fused_embeddings /= np.maximum(np.linalg.norm(fused_embeddings, axis=1, keepdims=True), 1e-8)
            subtitle_similarity = subtitle_embeddings @ query
            extra_features = subtitle_similarity[None, :, None]
        else:
            fused_embeddings = movie_embeddings
        similarities = fused_embeddings @ query
        order = np.argsort(-similarities, kind="stable")[:max_candidates]
        count = order.size
        candidate = np.zeros((max_candidates, dimension), dtype=np.float32)
        coarse = np.full(max_candidates, -1e6, dtype=np.float32)
        utility = np.zeros(max_candidates, dtype=np.float32)
        timestamps = np.zeros(max_candidates, dtype=np.float32)
        mask = np.zeros(max_candidates, dtype=bool)
        candidate[:count] = fused_embeddings[order]
        coarse[:count] = similarities[order]
        utility[:count] = (fused_embeddings[order] @ answer) - similarities[order]
        timestamps[:count] = movie_windows.starts[order]
        mask[:count] = True
        movie_duration = float(movie_windows.duration_seconds)
        durations[question_index] = movie_duration
        pair_features = make_pair_features(
            query[None, :],
            candidate[None, :, :],
            coarse_scores=coarse[None, :],
            timestamps=timestamps[None, :],
            movie_durations=np.asarray([max(movie_duration, 1.0)]),
            extra_features=(
                np.pad(
                    extra_features[:, order, :],
                    ((0, 0), (0, max_candidates - count), (0, 0)),
                )
                if extra_features is not None
                else None
            ),
        )[0]
        features.append(pair_features)
        coarse_rows.append(coarse)
        utility_rows.append(np.clip(utility, -1.0, 1.0))
        candidate_rows.append(candidate)
        timestamp_rows.append(timestamps)
        mask_rows.append(mask)

    return {
        "features": np.stack(features).astype(np.float32),
        "question_ids": np.asarray([question.question_id for question in questions]),
        "video_ids": np.asarray([question.video_id for question in questions]),
        "coarse_scores": np.stack(coarse_rows).astype(np.float32),
        "utility_gain": np.stack(utility_rows).astype(np.float32),
        "candidate_embeddings": np.stack(candidate_rows).astype(np.float32),
        "timestamps": np.stack(timestamp_rows).astype(np.float32),
        "candidate_mask": np.stack(mask_rows),
        "movie_durations": durations,
    }
