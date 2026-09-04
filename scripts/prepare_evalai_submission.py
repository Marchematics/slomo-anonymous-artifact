"""Convert validated local answers to the official SLoMO EvalAI schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.movie_router.answering import validate_predictions_csv, write_predictions_csv  # noqa: E402
from src.movie_router.sf20k import load_annotations  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-column", default="answer")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing submission: {args.out}")
    expected = [question.question_id for question in load_annotations(args.annotations)]
    if args.answers.suffix.lower() == ".jsonl":
        rows = []
        for line in args.answers.read_text(encoding="utf-8").splitlines():
            if line.strip():
                import json

                row = json.loads(line)
                rows.append(
                    {
                        "question_id": str(row["question_id"]),
                        args.source_column: str(row.get(args.source_column, row.get("answer", ""))),
                    }
                )
    else:
        with args.answers.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or set(rows[0]) != {"question_id", args.source_column}:
            raise ValueError("source answers have an unexpected header")
    by_id = {row.get("question_id", ""): row.get(args.source_column, "") for row in rows}
    if set(by_id) != set(expected):
        raise ValueError("source answers do not exactly cover the annotation question IDs")
    predictions = [by_id[question_id] for question_id in expected]
    write_predictions_csv(args.out, expected, predictions, answer_column="prediction")
    validate_predictions_csv(args.out, expected, answer_column="prediction")
    print(f"wrote {len(predictions)} predictions to {args.out}")


if __name__ == "__main__":
    main()
