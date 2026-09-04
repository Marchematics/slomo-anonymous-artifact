from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.audit_submission import main


def test_submission_audit_accepts_exact_rows(tmp_path: Path, monkeypatch) -> None:
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(
        "\n".join(json.dumps({"question_id": qid}) for qid in ("q2", "q1")) + "\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    with submission.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "prediction"])
        writer.writeheader()
        writer.writerows([
            {"question_id": "q1", "prediction": "answer one"},
            {"question_id": "q2", "prediction": "answer two"},
        ])
    monkeypatch.setattr("sys.argv", ["audit_submission.py", "--annotations", str(annotations), "--submission", str(submission)])
    main()

