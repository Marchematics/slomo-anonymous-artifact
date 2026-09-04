"""Numerically stable prior correction and evidence-budget selection."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SelectionConfig:
    """Inference-time controls for one fixed answerer/frame budget."""

    max_k: int = 8
    action_threshold: float = 0.0
    beta: float = 1.0
    prior_lambda: float = 1.0
    prior_bias: float = 0.0
    mmr_gamma: float = 0.0
    allow_empty: bool = True

    def __post_init__(self) -> None:
        if self.max_k < 0:
            raise ValueError("max_k must be nonnegative")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if self.prior_lambda < 0.0 or self.mmr_gamma < 0.0:
            raise ValueError("prior_lambda and mmr_gamma must be nonnegative")


@dataclass(frozen=True)
class SelectionResult:
    """Selected windows and all intermediate quantities for auditing."""

    selected_indices: tuple[Array, ...]
    evidence_probability: Array
    payoff: Array
    action: Array
    effective_library_size: Array

    @property
    def selected_counts(self) -> Array:
        return np.asarray([indices.size for indices in self.selected_indices], dtype=np.int64)


def _validate_matrix(name: str, value: Array) -> Array:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _stable_softmax(values: Array, beta: float) -> Array:
    scaled = beta * values
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(np.clip(scaled, -80.0, 80.0))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def effective_library_size(coarse_scores: Array, *, beta: float = 1.0) -> Array:
    """Return ``1/sum softmax(beta*score)^2`` for each candidate pool.

    The value is a concentration diagnostic, not a literal shot-level
    library size.  A one-candidate pool returns one; larger pools lie in
    ``[1, K]`` up to floating-point precision.
    """

    scores = np.asarray(coarse_scores, dtype=float)
    if scores.ndim not in (1, 2) or not np.all(np.isfinite(scores)):
        raise ValueError("coarse_scores must be a finite vector or matrix")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    one_dimensional = scores.ndim == 1
    matrix = scores[None, :] if one_dimensional else scores
    if matrix.shape[1] == 0:
        raise ValueError("candidate pools cannot be empty")
    weights = _stable_softmax(matrix, beta)
    result = 1.0 / np.sum(weights**2, axis=1)
    return result[0] if one_dimensional else result


def prior_corrected_probability(
    evidence_logit: Array,
    coarse_scores: Array,
    *,
    prior_lambda: float = 1.0,
    prior_bias: float = 0.0,
    beta: float = 1.0,
) -> Array:
    """Convert balanced evidence into a calibrated candidate probability."""

    evidence = np.asarray(evidence_logit, dtype=float)
    scores = np.asarray(coarse_scores, dtype=float)
    if evidence.shape != scores.shape:
        raise ValueError("evidence_logit and coarse_scores must have equal shapes")
    if evidence.ndim not in (1, 2) or not np.all(np.isfinite(evidence)):
        raise ValueError("evidence_logit must be a finite vector or matrix")
    if prior_lambda < 0.0:
        raise ValueError("prior_lambda must be nonnegative")
    k_eff = np.asarray(effective_library_size(scores, beta=beta), dtype=float)
    denominator = np.maximum(k_eff - 1.0, 1e-8)
    shift = prior_bias - prior_lambda * np.log(denominator)
    if evidence.ndim == 2:
        shift = shift[:, None]
    logits = np.clip(evidence + shift, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _cosine_matrix(embeddings: Array) -> Array:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def mmr_select(
    action: Array,
    embeddings: Array,
    timestamps: Array,
    *,
    max_k: int,
    threshold: float = 0.0,
    gamma: float = 0.0,
    allow_empty: bool = True,
) -> Array:
    """Select high-payoff windows with optional redundancy suppression.

    The returned indices are chronological.  Selection is performed in action
    order, so chronology never changes which evidence was selected.
    """

    scores = np.asarray(action, dtype=float)
    vectors = np.asarray(embeddings, dtype=float)
    times = np.asarray(timestamps, dtype=float)
    if scores.ndim != 1 or vectors.ndim != 2 or times.shape != scores.shape:
        raise ValueError("action, embeddings, and timestamps have incompatible shapes")
    if vectors.shape[0] != scores.size or not np.all(np.isfinite(scores) | np.isneginf(scores)):
        raise ValueError("candidate arrays have incompatible or non-finite values")
    if max_k < 0 or gamma < 0.0:
        raise ValueError("max_k and gamma must be nonnegative")
    if max_k == 0:
        return np.empty(0, dtype=np.int64)
    eligible = np.flatnonzero(scores > threshold)
    if eligible.size == 0:
        if allow_empty:
            return np.empty(0, dtype=np.int64)
        eligible = np.asarray([int(np.argmax(scores))])

    # Stable ordering gives deterministic tie-breaking by original candidate
    # order, which is important when cached retrieval scores are tied.
    ranked = eligible[np.argsort(-scores[eligible], kind="stable")]
    selected: list[int] = [int(ranked[0])]
    similarity = _cosine_matrix(vectors) if gamma > 0.0 else None
    while len(selected) < min(max_k, ranked.size):
        remaining = [int(index) for index in ranked if int(index) not in selected]
        if not remaining:
            break
        if similarity is None:
            best = remaining[0]
        else:
            adjusted = np.asarray(
                [scores[index] - gamma * np.max(similarity[index, selected]) for index in remaining]
            )
            best = remaining[int(np.argmax(adjusted))]
        selected.append(best)
    return np.asarray(sorted(selected, key=lambda index: (times[index], index)), dtype=np.int64)


def select_candidates(
    evidence_logit: Array,
    payoff: Array,
    coarse_scores: Array,
    embeddings: Array,
    timestamps: Array,
    *,
    config: SelectionConfig | None = None,
    candidate_mask: Array | None = None,
) -> SelectionResult:
    """Apply BF evidence, payoff, prior correction, and temporal selection."""

    cfg = config or SelectionConfig()
    evidence = _validate_matrix("evidence_logit", evidence_logit)
    utility = _validate_matrix("payoff", payoff)
    coarse = _validate_matrix("coarse_scores", coarse_scores)
    if evidence.shape != utility.shape or evidence.shape != coarse.shape:
        raise ValueError("evidence, payoff, and coarse scores must have equal shapes")
    batch, candidates = evidence.shape
    vector_array = np.asarray(embeddings, dtype=float)
    time_array = np.asarray(timestamps, dtype=float)
    if vector_array.ndim != 3 or vector_array.shape[:2] != (batch, candidates):
        raise ValueError("embeddings must have shape (batch, candidates, dimension)")
    if time_array.shape != (batch, candidates):
        raise ValueError("timestamps must have shape (batch, candidates)")
    if candidate_mask is None:
        mask = np.ones((batch, candidates), dtype=bool)
    else:
        mask = np.asarray(candidate_mask).astype(bool)
        if mask.shape != (batch, candidates):
            raise ValueError("candidate_mask must have shape (batch, candidates)")
        if not np.all(mask.any(axis=1)):
            raise ValueError("each question must have at least one valid candidate")

    probability = prior_corrected_probability(
        evidence,
        coarse,
        prior_lambda=cfg.prior_lambda,
        prior_bias=cfg.prior_bias,
        beta=cfg.beta,
    )
    action = probability * np.maximum(utility, 0.0)
    action = np.where(mask, action, -np.inf)
    k_eff = np.asarray(effective_library_size(coarse, beta=cfg.beta))
    selected = tuple(
        mmr_select(
            action[row],
            vector_array[row],
            time_array[row],
            max_k=cfg.max_k,
            threshold=cfg.action_threshold,
            gamma=cfg.mmr_gamma,
            allow_empty=cfg.allow_empty,
        )
        for row in range(batch)
    )
    return SelectionResult(selected, probability, utility, action, k_eff)
