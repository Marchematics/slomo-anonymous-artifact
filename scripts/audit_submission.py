#!/usr/bin/env python3
"""Check exact question coverage and schema for an EvalAI submission CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def ids_from_annotations(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.append(str(json.loads(line)["question_id"]))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args()
    expected = ids_from_annotations(args.annotations)
    with args.submission.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["question_id", "prediction"]:
            raise ValueError(f"expected question_id,prediction; got {reader.fieldnames}")
        rows = list(reader)
    actual = [str(row.get("question_id", "")) for row in rows]
    if len(actual) != len(set(actual)):
        raise ValueError("submission contains duplicate question IDs")
    if set(actual) != set(expected):
        raise ValueError(f"ID mismatch: missing={len(set(expected)-set(actual))}, extra={len(set(actual)-set(expected))}")
    empty = [row["question_id"] for row in rows if not str(row.get("prediction", "")).strip()]
    if empty:
        raise ValueError(f"empty predictions: {len(empty)}")
    print(json.dumps({"rows": len(rows), "unique_ids": len(set(actual)), "empty": 0}, indent=2))


if __name__ == "__main__":
    main()
