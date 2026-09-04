"""Construction of balanced answer-intervention supervision.

The router does not require timestamp annotations.  A caller supplies a
question/candidate matrix and an answer-support utility such as the change in
answer log likelihood after adding one candidate.  This module turns that
utility into a balanced relation stream while retaining the continuous payoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class InterventionLabels:
    """Balanced candidate-level labels selected independently per question."""

    row_indices: Array
    relation: Array
    payoff: Array
    question_indices: Array

    def __post_init__(self) -> None:
        lengths = {
            np.asarray(self.row_indices).size,
            np.asarray(self.relation).size,
            np.asarray(self.payoff).size,
            np.asarray(self.question_indices).size,
        }
        if len(lengths) != 1:
            raise ValueError("intervention fields must have equal length")
        relation = np.asarray(self.relation)
        if not np.all(np.isin(relation, [0, 1])):
            raise ValueError("relation labels must be binary")

    @property
    def size(self) -> int:
        return int(np.asarray(self.row_indices).size)

    @property
    def positive_fraction(self) -> float:
        if self.size == 0:
            return float("nan")
        return float(np.mean(np.asarray(self.relation) == 1))


def _as_matrix(name: str, value: Any, shape: tuple[int, int]) -> Array:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def build_intervention_labels(
    question_ids: Array,
    coarse_scores: Array,
    utility_gain: Array,
    *,
    positive_threshold: float = 0.0,
    negative_threshold: float = 0.0,
    hard_negative_fraction: float = 0.5,
    max_pairs_per_question: int = 4,
    payoff_clip: float = 1.0,
    seed: int = 0,
    valid_mask: Array | None = None,
) -> InterventionLabels:
    """Build a 50/50 positive/hard-negative intervention stream.

    ``utility_gain[b, j]`` is a proxy for the answer improvement caused by
    candidate ``j`` for question ``b``.  Positives require a positive margin.
    Negatives are drawn from the highest-coarse-score candidates whose utility
    is non-positive, which makes them retrieval-confusable negatives.  A
    question is included only when it has both classes.  Sampling is balanced
    inside each question, so questions with many candidate windows cannot
    dominate the relation learner.
    """

    ids = np.asarray(question_ids)
    if ids.ndim != 1:
        raise ValueError("question_ids must be one-dimensional")
    n = ids.size
    coarse = np.asarray(coarse_scores, dtype=float)
    utility = np.asarray(utility_gain, dtype=float)
    if coarse.ndim != 2 or utility.shape != coarse.shape or coarse.shape[0] != n:
        raise ValueError("question_ids, coarse_scores, and utility_gain have incompatible shapes")
    _as_matrix("coarse_scores", coarse, coarse.shape)
    _as_matrix("utility_gain", utility, utility.shape)
    if valid_mask is None:
        valid = np.ones_like(coarse, dtype=bool)
    else:
        valid = np.asarray(valid_mask).astype(bool)
        if valid.shape != coarse.shape or not np.all(valid.any(axis=1)):
            raise ValueError("valid_mask must match candidates and keep one candidate per question")
    if positive_threshold < negative_threshold:
        raise ValueError("positive_threshold must not be below negative_threshold")
    if not 0.0 < hard_negative_fraction <= 1.0:
        raise ValueError("hard_negative_fraction must lie in (0, 1]")
    if max_pairs_per_question < 1 or payoff_clip <= 0.0:
        raise ValueError("max_pairs_per_question and payoff_clip must be positive")

    rng = np.random.default_rng(seed)
    row_indices: list[int] = []
    relation: list[int] = []
    payoff: list[float] = []
    selected_questions: list[Any] = []
    for row in range(n):
        valid_columns = np.flatnonzero(valid[row])
        if positive_threshold == negative_threshold:
            positive = valid_columns[utility[row, valid_columns] > positive_threshold]
        else:
            positive = valid_columns[utility[row, valid_columns] >= positive_threshold]
        rank = valid_columns[np.argsort(-coarse[row, valid_columns], kind="stable")]
        pool_size = max(1, int(np.ceil(valid_columns.size * hard_negative_fraction)))
        hard_pool = rank[:pool_size]
        negative = hard_pool[utility[row, hard_pool] <= negative_threshold]

        # If the hard pool contains no negative, use the lowest-utility
        # non-positive candidate among the retrieved set.  It remains a
        # retrieval-confusable negative because the pool is still top-ranked.
        if negative.size == 0:
            non_positive = valid_columns[utility[row, valid_columns] <= negative_threshold]
            if non_positive.size:
                order = np.argsort(-coarse[row, non_positive], kind="stable")
                negative = non_positive[order[:1]]
        if positive.size == 0 or negative.size == 0:
            continue

        count = min(max_pairs_per_question, positive.size, negative.size)
        pos = rng.choice(positive, size=count, replace=False)
        neg = rng.choice(negative, size=count, replace=False)
        for column in pos:
            row_indices.append(row * coarse.shape[1] + int(column))
            relation.append(1)
            payoff.append(float(np.clip(utility[row, column], -payoff_clip, payoff_clip)))
            selected_questions.append(ids[row])
        for column in neg:
            row_indices.append(row * coarse.shape[1] + int(column))
            relation.append(0)
            payoff.append(float(np.clip(utility[row, column], -payoff_clip, payoff_clip)))
            selected_questions.append(ids[row])

    return InterventionLabels(
        row_indices=np.asarray(row_indices, dtype=np.int64),
        relation=np.asarray(relation, dtype=np.int64),
        payoff=np.asarray(payoff, dtype=np.float32),
        question_indices=np.asarray(selected_questions),
    )
