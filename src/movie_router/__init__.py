"""Lightweight evidence/payoff routing for long-form movie QA."""

from .data import InterventionLabels, build_intervention_labels
from .cache import CandidateCacheShape, load_candidate_cache, save_candidate_cache, validate_candidate_cache
from .builder import build_candidate_arrays
from .answering import (
    ContextItem,
    build_vlm_messages,
    clean_open_answer,
    extract_choice,
    validate_predictions_csv,
    write_predictions_csv,
)
from .features import make_pair_features
from .model import BFMovieRouter, RouterTrainConfig, predict_router, train_router
from .routing import (
    SelectionConfig,
    SelectionResult,
    effective_library_size,
    prior_corrected_probability,
    select_candidates,
)
from .sf20k import (
    SF20KQuestion,
    iter_hf_annotations,
    load_annotations,
    load_hf_annotations,
    split_movie_ids,
)
from .qwen_answerer import ChoiceScores, QwenAnswererConfig, QwenVLAnswerer

__all__ = [
    "BFMovieRouter",
    "ContextItem",
    "CandidateCacheShape",
    "InterventionLabels",
    "RouterTrainConfig",
    "SelectionConfig",
    "SelectionResult",
    "QwenAnswererConfig",
    "QwenVLAnswerer",
    "ChoiceScores",
    "SF20KQuestion",
    "build_intervention_labels",
    "build_candidate_arrays",
    "build_vlm_messages",
    "clean_open_answer",
    "load_candidate_cache",
    "make_pair_features",
    "iter_hf_annotations",
    "load_annotations",
    "load_hf_annotations",
    "effective_library_size",
    "extract_choice",
    "predict_router",
    "prior_corrected_probability",
    "select_candidates",
    "save_candidate_cache",
    "train_router",
    "validate_candidate_cache",
    "validate_predictions_csv",
    "split_movie_ids",
    "write_predictions_csv",
]
