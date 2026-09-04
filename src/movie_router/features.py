"""Feature assembly from frozen query/candidate embeddings."""

from __future__ import annotations

import numpy as np


def make_pair_features(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    *,
    coarse_scores: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    movie_durations: np.ndarray | None = None,
    extra_features: np.ndarray | None = None,
) -> np.ndarray:
    """Build symmetric query-candidate features for the tiny router.

    The output contains query, candidate, absolute difference, product, and
    optional scalar/context features.  Equal embedding dimensions are
    required so the same function works for visual, text, or fused embeddings.
    """

    query = np.asarray(query_embeddings, dtype=np.float32)
    candidate = np.asarray(candidate_embeddings, dtype=np.float32)
    if query.ndim != 2 or candidate.ndim != 3 or candidate.shape[0] != query.shape[0]:
        raise ValueError("query_embeddings and candidate_embeddings have incompatible shapes")
    questions, candidates, dimension = candidate.shape
    if query.shape[1] != dimension or dimension < 1:
        raise ValueError("query and candidate embedding dimensions must match")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(candidate)):
        raise ValueError("embeddings must be finite")

    query_expanded = np.broadcast_to(query[:, None, :], candidate.shape)
    pieces = [query_expanded, candidate, np.abs(query_expanded - candidate), query_expanded * candidate]
    scalar_pieces: list[np.ndarray] = []
    if coarse_scores is not None:
        coarse = np.asarray(coarse_scores, dtype=np.float32)
        if coarse.shape != (questions, candidates):
            raise ValueError("coarse_scores has the wrong shape")
        scalar_pieces.append(coarse[..., None])
    if timestamps is not None:
        time = np.asarray(timestamps, dtype=np.float32)
        if time.shape != (questions, candidates):
            raise ValueError("timestamps has the wrong shape")
        if movie_durations is None:
            scale = np.maximum(np.max(time, axis=1, keepdims=True), 1.0)
        else:
            duration = np.asarray(movie_durations, dtype=np.float32)
            if duration.shape != (questions,):
                raise ValueError("movie_durations must have shape (questions,)")
            if np.any(duration <= 0.0) or not np.all(np.isfinite(duration)):
                raise ValueError("movie_durations must be positive and finite")
            scale = duration[:, None]
        scalar_pieces.append((time / scale)[..., None])
    if extra_features is not None:
        extra = np.asarray(extra_features, dtype=np.float32)
        if extra.ndim != 3 or extra.shape[:2] != (questions, candidates):
            raise ValueError("extra_features must have shape (questions, candidates, dimensions)")
        scalar_pieces.append(extra)
    if scalar_pieces:
        pieces.extend(scalar_pieces)
    return np.concatenate(pieces, axis=-1).astype(np.float32, copy=False)
