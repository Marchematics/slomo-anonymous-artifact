"""Small, dependency-light adapters for SF20K annotations and movie splits."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class SF20KQuestion:
    """Normalized annotation record used by both OEQA and MCQA adapters."""

    question_id: str
    video_id: str
    question: str
    answer: str
    video_url: str | None = None
    movie_title: str | None = None
    options: tuple[str, ...] = ()
    correct_index: int | None = None
    correct_letter: str | None = None


def _value(record: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = record.get(key, default)
    if value is None:
        return default
    return value


def normalize_question(record: Mapping[str, Any]) -> SF20KQuestion:
    """Normalize a raw HF/JSON/CSV row without discarding MCQA fields."""

    required = ("question_id", "video_id", "question")
    missing = [key for key in required if not str(_value(record, key, "")).strip()]
    if missing:
        raise ValueError(f"annotation is missing required fields: {missing}")
    raw_options = _value(record, "options")
    if isinstance(raw_options, (list, tuple)):
        options = tuple(str(value).strip() for value in raw_options if str(value).strip())
    else:
        options = tuple(
            str(_value(record, f"option_{index}", "")).strip()
            for index in range(5)
            if str(_value(record, f"option_{index}", "")).strip()
        )
    correct_raw = _value(record, "correct_answer", _value(record, "correct_index"))
    correct_index: int | None
    if correct_raw in (None, ""):
        correct_index = None
    else:
        try:
            numeric = float(correct_raw)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError
            correct_index = int(numeric)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"correct_answer is not an integer: {correct_raw!r}") from exc
        if options and not 0 <= correct_index < len(options):
            raise ValueError("correct_answer is outside the available options")
    answer = str(_value(record, "answer", "")).strip()
    if not answer and correct_index is not None and options:
        answer = options[correct_index]
    raw_letter = _value(record, "correct_letter")
    return SF20KQuestion(
        question_id=str(record["question_id"]),
        video_id=str(record["video_id"]),
        question=str(record["question"]).strip(),
        answer=answer,
        video_url=(str(_value(record, "video_url")).strip() or None),
        movie_title=(str(_value(record, "movie_title")).strip() or None),
        options=options,
        correct_index=correct_index,
        correct_letter=(str(raw_letter).strip() if raw_letter not in (None, "") else None),
    )


def _read_path(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("data", value.get("rows", value))
        if not isinstance(value, list):
            raise ValueError("JSON annotations must contain a list or a data/rows list")
        return value
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [
                record
                for record in csv.DictReader(handle)
                if record
                and str(record.get("question_id") or "").strip()
                and str(record.get("question") or "").strip()
            ]
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"unsupported annotation file extension: {path.suffix}")


def load_annotations(source: str | Path | Iterable[Mapping[str, Any]]) -> tuple[SF20KQuestion, ...]:
    """Load local annotations or normalize an already-loaded HF iterable."""

    if isinstance(source, (str, Path)):
        records = _read_path(Path(source))
    else:
        records = list(source)
    questions = tuple(normalize_question(record) for record in records)
    ids = [question.question_id for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("question_id values must be unique")
    return questions


def iter_hf_annotations(
    repo_id: str = "rghermi/sf20k",
    *,
    split: str = "train",
    streaming: bool = True,
    skip_invalid: bool = False,
) -> Iterable[SF20KQuestion]:
    """Yield normalized HF annotations without downloading movie videos.

    SF20K publishes split CSVs. Reading the CSV directly avoids the slow
    Arrow streaming path and also handles the blank lines present in the
    published files.
    """

    del streaming  # retained for API compatibility with the earlier HF adapter
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("install huggingface_hub to load SF20K annotations") from exc
    competition_files = {
        "public_test": "sf20k_public_test_questions.csv",
        "test_public": "sf20k_public_test_questions.csv",
        "private_test": "sf20k_private_test_questions.csv",
        "test_private": "sf20k_private_test_questions.csv",
    }
    if repo_id == "rghermi/sf20k-qa":
        filename = competition_files.get(split, split)
    else:
        filename = {
            "train": "train.csv",
            "test": "test.csv",
            "test_expert": "test_expert.csv",
            "test_silent": "test_silent.csv",
        }.get(split, split)
    if not filename.endswith(".csv"):
        filename += ".csv"
    path = hf_hub_download(repo_id, filename, repo_type="dataset")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        records = csv.DictReader(handle)
        for record in records:
            if not record or all(value in (None, "") for value in record.values()):
                continue
            try:
                question = normalize_question(record)
            except ValueError:
                if not skip_invalid:
                    raise
                continue
            yield question


def load_hf_annotations(repo_id: str = "rghermi/sf20k", *, split: str = "train") -> tuple[SF20KQuestion, ...]:
    """Load one HF split explicitly; importing datasets remains optional."""

    return tuple(iter_hf_annotations(repo_id, split=split, streaming=False))


def split_movie_ids(
    video_ids: Iterable[str],
    *,
    train_fraction: float = 0.8,
    calibration_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, tuple[str, ...]]:
    """Make a deterministic movie-disjoint train/calibration/validation split."""

    if train_fraction <= 0.0 or calibration_fraction <= 0.0:
        raise ValueError("train_fraction and calibration_fraction must be positive")
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError("fractions must leave a positive validation fraction")
    unique = sorted({str(value) for value in video_ids})
    if len(unique) < 3:
        raise ValueError("at least three distinct movies are required")
    ranked = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    n = len(ranked)
    train_count = min(max(1, int(n * train_fraction)), n - 2)
    calibration_count = min(max(1, int(n * calibration_fraction)), n - train_count - 1)
    train_end = train_count
    calibration_end = train_end + calibration_count
    return {
        "train": tuple(ranked[:train_end]),
        "calibration": tuple(ranked[train_end:calibration_end]),
        "validation": tuple(ranked[calibration_end:]),
    }
