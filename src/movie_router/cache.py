"""Validation and serialization of frozen candidate caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CandidateCacheShape:
    questions: int
    candidates: int
    feature_dim: int
    embedding_dim: int
    has_utility_gain: bool


def validate_candidate_cache(
    arrays: Mapping[str, np.ndarray], *, require_utility_gain: bool = False
) -> CandidateCacheShape:
    """Validate the NPZ contract before a GPU run or leaderboard inference."""

    required = {"features", "question_ids", "coarse_scores", "candidate_embeddings", "timestamps"}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"candidate cache is missing fields: {sorted(missing)}")
    features = np.asarray(arrays["features"])
    if features.ndim != 3:
        raise ValueError("features must have shape (questions, candidates, feature_dim)")
    questions, candidates, feature_dim = features.shape
    if min(questions, candidates, feature_dim) < 1:
        raise ValueError("candidate cache dimensions must be positive")
    expected_matrix = (questions, candidates)
    for name in ("coarse_scores", "timestamps"):
        value = np.asarray(arrays[name])
        if value.shape != expected_matrix:
            raise ValueError(f"{name} must have shape {expected_matrix}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
    embeddings = np.asarray(arrays["candidate_embeddings"])
    if embeddings.ndim != 3 or embeddings.shape[:2] != expected_matrix or embeddings.shape[2] < 1:
        raise ValueError("candidate_embeddings must have shape (questions, candidates, embedding_dim)")
    if not np.all(np.isfinite(embeddings)) or not np.all(np.isfinite(features)):
        raise ValueError("features and candidate_embeddings must be finite")
    question_ids = np.asarray(arrays["question_ids"])
    if question_ids.shape != (questions,):
        raise ValueError("question_ids must have shape (questions,)")
    if len(set(question_ids.tolist())) != questions:
        raise ValueError("question_ids must be unique")
    has_utility = "utility_gain" in arrays
    if has_utility:
        utility = np.asarray(arrays["utility_gain"])
        if utility.shape != expected_matrix or not np.all(np.isfinite(utility)):
            raise ValueError("utility_gain must be a finite (questions, candidates) matrix")
    if "candidate_mask" in arrays:
        mask = np.asarray(arrays["candidate_mask"])
        if mask.shape != expected_matrix or mask.dtype.kind not in "bui":
            raise ValueError("candidate_mask must be a boolean/integer candidate matrix")
        if not np.all(mask.astype(bool).any(axis=1)):
            raise ValueError("each question must have at least one valid candidate")
    if require_utility_gain and not has_utility:
        raise ValueError("training cache requires utility_gain")
    return CandidateCacheShape(questions, candidates, feature_dim, embeddings.shape[2], has_utility)


def load_candidate_cache(path: str | Path, *, require_utility_gain: bool = False) -> dict[str, np.ndarray]:
    """Load and validate a cache with pickle disabled."""

    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    validate_candidate_cache(arrays, require_utility_gain=require_utility_gain)
    return arrays


def save_candidate_cache(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a validated cache and refuse accidental overwrites."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite candidate cache: {destination}")
    validate_candidate_cache(arrays)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
