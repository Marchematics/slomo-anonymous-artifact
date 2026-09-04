"""Deterministic contracts for evidence-certified answer switching."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


RELATION_TYPES = {
    "entity",
    "object",
    "action",
    "location",
    "temporal",
    "count",
    "cause",
    "state",
    "spatial",
    "global",
}
SLOT_TYPES = {
    "PERSON",
    "OBJECT",
    "LOCATION",
    "TEMPORAL",
    "COUNT",
    "ACTION",
    "CAUSE",
    "MEANING",
    "STATE",
    "GLOBAL",
}
SLOT_SWITCH_TYPES = {"PERSON", "OBJECT", "LOCATION", "TEMPORAL", "COUNT", "ACTION"}
EVIDENCE_STATUSES = {"direct", "missing", "contradicted"}
EVIDENCE_MODALITIES = {"asr", "frame", "multimodal"}
FAILURE_REASONS = {
    "none",
    "missing_evidence",
    "subject_mismatch",
    "time_mismatch",
    "relation_mismatch",
    "global_undercoverage",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def read_predictions(path: Path) -> tuple[list[str], dict[str, str]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            raw = list(csv.DictReader(handle))
    else:
        raw = read_jsonl(path)
    order: list[str] = []
    values: dict[str, str] = {}
    for row in raw:
        question_id = str(row.get("question_id", "")).strip()
        answer = " ".join(
            str(row.get("prediction", row.get("answer", ""))).split()
        )
        if not question_id or not answer or question_id in values:
            raise ValueError(f"invalid prediction row in {path}: {row}")
        order.append(question_id)
        values[question_id] = answer
    if not order:
        raise ValueError(f"prediction file is empty: {path}")
    return order, values


def read_annotations(path: Path) -> list[dict[str, object]]:
    rows = read_jsonl(path)
    seen: set[str] = set()
    for row in rows:
        question_id = str(row.get("question_id", "")).strip()
        if (
            not question_id
            or question_id in seen
            or not str(row.get("video_id", "")).strip()
            or not str(row.get("question", "")).strip()
        ):
            raise ValueError(f"invalid annotation row in {path}: {row}")
        seen.add(question_id)
    return rows


def classify_slot(question: str) -> str:
    """Assign a conservative answer-slot contract from question wording."""

    text = " ".join(str(question).lower().split())
    if re.match(r"^(who|whose|whom)\b", text) or "name of" in text:
        return "PERSON"
    if re.match(r"^(how many|how much|what number|number of)\b", text):
        return "COUNT"
    if re.match(r"^(where|in what location|what place)\b", text) or "location" in text:
        return "LOCATION"
    if any(cue in text for cue in ("before", "after", "earlier", "later", "first", "last", "eventually")):
        return "TEMPORAL"
    if any(cue in text for cue in ("object", "holding", "carry", "carrying", "look like", "wear", "wearing", "what is in")):
        return "OBJECT"
    if any(cue in text for cue in ("why", "reason", "purpose", "motivated", "motivation", "caused")):
        return "CAUSE"
    if any(cue in text for cue in ("meaning", "imply", "theme", "symbol", "significance")):
        return "MEANING"
    if any(cue in text for cue in ("feel", "emotion", "relationship", "state", "condition")):
        return "STATE"
    if any(cue in text for cue in ("what happens", "what does", "what did", "action")):
        return "ACTION"
    return "GLOBAL"


def answer_sha256(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def _parent_first(seed: int, question_id: str) -> bool:
    digest = hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).digest()
    return digest[0] % 2 == 0


def transcript_available(path: Path) -> bool:
    if not path.is_file():
        return False
    content = path.read_text(encoding="utf-8", errors="ignore")
    payload = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and line.strip() != "WEBVTT" and "-->" not in line
    ]
    return any(not line.isdigit() for line in payload)


def build_task_bundle(
    annotations: Sequence[Mapping[str, object]],
    parent: Mapping[str, str],
    alternate: Mapping[str, str],
    *,
    seed: int,
    protected_ids: set[str] | None = None,
    subtitle_dir: Path | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Build normal/swapped source-blind tasks and private role maps."""

    protected = protected_ids or set()
    annotation_ids = [str(row["question_id"]) for row in annotations]
    expected = set(annotation_ids)
    if set(parent) != expected or set(alternate) != expected:
        raise ValueError("annotations, parent, and alternate IDs must match exactly")
    unknown = protected - expected
    if unknown:
        raise ValueError(f"protected IDs are unknown: {sorted(unknown)[:5]}")

    normal_tasks: list[dict[str, object]] = []
    swapped_tasks: list[dict[str, object]] = []
    normal_maps: list[dict[str, object]] = []
    swapped_maps: list[dict[str, object]] = []
    for annotation in annotations:
        question_id = str(annotation["question_id"])
        if question_id in protected or parent[question_id] == alternate[question_id]:
            continue
        parent_is_a = _parent_first(seed, question_id)
        labels = (
            {"A": "parent", "B": "alternate"}
            if parent_is_a
            else {"A": "alternate", "B": "parent"}
        )
        answers = {"parent": parent[question_id], "alternate": alternate[question_id]}
        asr_path = (
            subtitle_dir / f"{annotation['video_id']}.vtt"
            if subtitle_dir is not None
            else None
        )
        common = {
            "question_id": question_id,
            "video_id": str(annotation["video_id"]),
            "question": str(annotation["question"]),
            "asr_available": bool(asr_path and transcript_available(asr_path)),
            "slot_type": classify_slot(str(annotation["question"])),
        }
        normal_tasks.append(
            {
                **common,
                "hypothesis_a": answers[labels["A"]],
                "hypothesis_b": answers[labels["B"]],
            }
        )
        swapped_tasks.append(
            {
                **common,
                "hypothesis_a": answers[labels["B"]],
                "hypothesis_b": answers[labels["A"]],
            }
        )

        def role_map(a_role: str, b_role: str) -> dict[str, object]:
            return {
                "question_id": question_id,
                "label_roles": {"A": a_role, "B": b_role},
                "answer_sha256": {
                    "A": answer_sha256(answers[a_role]),
                    "B": answer_sha256(answers[b_role]),
                },
            }

        normal_maps.append(role_map(labels["A"], labels["B"]))
        swapped_maps.append(role_map(labels["B"], labels["A"]))
    return normal_tasks, swapped_tasks, normal_maps, swapped_maps


def normalize_evidence_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def normalize_transcript_text(value: str) -> str:
    """Remove VTT structure so quotes may span adjacent subtitle cues."""

    lines = []
    for raw in str(value).splitlines():
        line = raw.strip()
        if not line or line == "WEBVTT" or line.isdigit() or "-->" in line:
            continue
        lines.append(line)
    return normalize_evidence_text(" ".join(lines))


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _max_nonoverlapping(intervals: Sequence[tuple[float, float]]) -> int:
    selected = 0
    end = float("-inf")
    for start, stop in sorted(intervals, key=lambda value: (value[1], value[0])):
        if start >= end:
            selected += 1
            end = stop
    return selected


def validate_certificate(
    task: Mapping[str, object],
    certificate: Mapping[str, object],
    *,
    transcript: str,
    frame_root: Path,
) -> dict[str, object]:
    """Validate evidence bytes and return the uniquely certified A/B label."""

    errors: list[str] = []
    if str(certificate.get("question_id", "")) != str(task["question_id"]):
        errors.append("question_id_mismatch")
    relation_type = str(certificate.get("relation_type", ""))
    if relation_type not in RELATION_TYPES:
        errors.append("invalid_relation_type")
    failure_reason = str(certificate.get("failure_reason", ""))
    if failure_reason not in FAILURE_REASONS:
        errors.append("invalid_failure_reason")

    raw_evidence = certificate.get("evidence", [])
    if not isinstance(raw_evidence, list):
        raw_evidence = []
        errors.append("invalid_evidence_list")
    evidence: dict[str, dict[str, object]] = {}
    valid_evidence: set[str] = set()
    transcript_norm = normalize_transcript_text(transcript)
    for raw in raw_evidence:
        if not isinstance(raw, dict):
            errors.append("invalid_evidence_item")
            continue
        evidence_id = str(raw.get("evidence_id", "")).strip()
        if not evidence_id or evidence_id in evidence:
            errors.append("duplicate_or_empty_evidence_id")
            continue
        evidence[evidence_id] = raw
        modality = str(raw.get("modality", ""))
        if modality not in EVIDENCE_MODALITIES:
            errors.append(f"{evidence_id}:invalid_modality")
            continue
        try:
            start = float(raw.get("start"))
            end = float(raw.get("end"))
        except (TypeError, ValueError):
            errors.append(f"{evidence_id}:invalid_time")
            continue
        if start < 0 or end < start:
            errors.append(f"{evidence_id}:invalid_time")
            continue
        quote = str(raw.get("quote", "")).strip()
        frame_paths = raw.get("frame_paths", [])
        if not isinstance(frame_paths, list):
            errors.append(f"{evidence_id}:invalid_frame_paths")
            continue
        asr_ok = True
        frame_ok = True
        if modality in {"asr", "multimodal"}:
            quote_norm = normalize_evidence_text(quote)
            asr_ok = bool(quote_norm and quote_norm in transcript_norm)
            if not asr_ok:
                errors.append(f"{evidence_id}:unbound_asr_quote")
        if modality in {"frame", "multimodal"}:
            paths = [Path(str(value)) for value in frame_paths]
            frame_ok = bool(paths) and all(
                _inside(path, frame_root) and path.is_file() for path in paths
            )
            if not frame_ok:
                errors.append(f"{evidence_id}:unbound_frame")
        if asr_ok and frame_ok:
            valid_evidence.add(evidence_id)

    slot_payload = certificate.get("slot_evidence")
    if isinstance(slot_payload, dict):
        task_slot = str(task.get("slot_type", ""))
        certificate_slot = str(certificate.get("slot_type", ""))
        if task_slot not in SLOT_TYPES or certificate_slot != task_slot:
            errors.append("slot_type_mismatch")
        if task_slot not in SLOT_SWITCH_TYPES:
            errors.append("slot_type_disabled")
        slot_supported = {"A": False, "B": False}
        slot_contradicted = {"A": False, "B": False}
        slot_missing = {"A": False, "B": False}
        for label, key in (("A", "hypothesis_a"), ("B", "hypothesis_b")):
            value = slot_payload.get(key)
            if not isinstance(value, dict):
                errors.append(f"slot_{label}:invalid")
                continue
            status = str(value.get("status", ""))
            fill = " ".join(str(value.get("slot_fill", "")).split())
            ids = value.get("evidence_ids", [])
            if status not in EVIDENCE_STATUSES or not isinstance(ids, list):
                errors.append(f"slot_{label}:invalid_status_or_ids")
                continue
            id_set = {str(item) for item in ids}
            if status in {"direct", "contradicted"} and (
                not id_set or not id_set.issubset(valid_evidence)
            ):
                errors.append(f"slot_{label}:unbound_status")
                continue
            if status == "direct":
                if not fill:
                    errors.append(f"slot_{label}:empty_fill")
                    continue
                slot_supported[label] = True
                if task_slot in {"PERSON", "COUNT"}:
                    evidence_words = set()
                    for evidence_id in id_set:
                        evidence_words.update(
                            normalize_evidence_text(str(evidence[evidence_id].get("quote", ""))).split()
                        )
                    fill_words = set(normalize_evidence_text(fill).split())
                    if fill_words and not fill_words.intersection(evidence_words):
                        errors.append(f"slot_{label}:fill_not_grounded")
                        slot_supported[label] = False
            elif status == "contradicted":
                slot_contradicted[label] = True
            else:
                slot_missing[label] = True
        if not errors and failure_reason == "none":
            candidates = [
                label
                for label, other in (("A", "B"), ("B", "A"))
                if slot_supported[label] and (slot_contradicted[other] or slot_missing[other])
            ]
            if len(candidates) == 1:
                return {
                    "question_id": str(task["question_id"]),
                    "valid": True,
                    "certified_label": candidates[0],
                    "errors": [],
                }
        return {
            "question_id": str(task["question_id"]),
            "valid": not errors,
            "certified_label": None,
            "errors": errors,
        }

    raw_observables = certificate.get("mandatory_observables", [])
    if not isinstance(raw_observables, list) or not 1 <= len(raw_observables) <= 4:
        raw_observables = []
        errors.append("invalid_observable_count")
    supported = {"A": True, "B": True}
    contradicted = {"A": False, "B": False}
    direct_evidence = {"A": set(), "B": set()}
    for index, raw in enumerate(raw_observables):
        if not isinstance(raw, dict):
            errors.append(f"observable_{index}:invalid")
            supported = {"A": False, "B": False}
            continue
        for label, key in (("A", "hypothesis_a"), ("B", "hypothesis_b")):
            status = str(raw.get(f"{key}_status", ""))
            ids = raw.get(f"{key}_evidence_ids", [])
            if status not in EVIDENCE_STATUSES or not isinstance(ids, list):
                errors.append(f"observable_{index}:{label}:invalid_status_or_ids")
                supported[label] = False
                continue
            id_set = {str(value) for value in ids}
            if status in {"direct", "contradicted"} and (
                not id_set or not id_set.issubset(valid_evidence)
            ):
                errors.append(f"observable_{index}:{label}:unbound_status")
                supported[label] = False
            if status != "direct":
                supported[label] = False
            else:
                direct_evidence[label].update(id_set)
            if status == "contradicted":
                contradicted[label] = True

    if relation_type == "global":
        intervals = [
            (float(item["start"]), float(item["end"]))
            for evidence_id, item in evidence.items()
            if evidence_id in valid_evidence
        ]
        if _max_nonoverlapping(intervals) < 3:
            errors.append("global_undercoverage")
            supported = {"A": False, "B": False}

    if not bool(task.get("asr_available", False)):
        for label in ("A", "B"):
            visual_ids = {
                evidence_id
                for evidence_id in direct_evidence[label]
                if str(evidence[evidence_id].get("modality")) in {"frame", "multimodal"}
            }
            if supported[label] and len(visual_ids) < 2:
                errors.append(f"{label}:empty_asr_requires_two_visual_anchors")
                supported[label] = False

    certified_label: str | None = None
    if failure_reason == "none" and not errors:
        candidates = [
            label
            for label, other in (("A", "B"), ("B", "A"))
            if supported[label] and contradicted[other]
        ]
        if len(candidates) == 1:
            certified_label = candidates[0]
    return {
        "question_id": str(task["question_id"]),
        "valid": not errors,
        "certified_label": certified_label,
        "errors": errors,
    }


def resolve_order_swapped_decision(
    normal_validation: Mapping[str, object],
    swapped_validation: Mapping[str, object],
    normal_role_map: Mapping[str, object],
    swapped_role_map: Mapping[str, object],
) -> dict[str, object]:
    question_id = str(normal_role_map["question_id"])
    if str(swapped_role_map.get("question_id")) != question_id:
        raise ValueError("normal and swapped maps have different question IDs")

    def selection(
        validation: Mapping[str, object], role_map: Mapping[str, object]
    ) -> tuple[str, str] | None:
        label = validation.get("certified_label")
        if label not in {"A", "B"}:
            return None
        roles = role_map.get("label_roles", {})
        hashes = role_map.get("answer_sha256", {})
        if not isinstance(roles, dict) or not isinstance(hashes, dict):
            return None
        role = str(roles.get(label, ""))
        answer_hash = str(hashes.get(label, ""))
        if role not in {"parent", "alternate"} or not answer_hash:
            return None
        return role, answer_hash

    normal = selection(normal_validation, normal_role_map)
    swapped = selection(swapped_validation, swapped_role_map)
    if normal is None or swapped is None or normal != swapped:
        return {
            "question_id": question_id,
            "decision": "ABSTAIN",
            "selected_role": "parent",
            "answer_sha256": "",
            "order_consistent": False,
        }
    role, answer_hash = normal
    return {
        "question_id": question_id,
        "decision": "SWITCH" if role == "alternate" else "KEEP",
        "selected_role": role,
        "answer_sha256": answer_hash,
        "order_consistent": True,
    }


def evaluate_decisions(
    records: Sequence[Mapping[str, object]],
    decisions: Mapping[str, Mapping[str, object]],
    *,
    parent_system: str,
    alternate_system: str,
    bootstrap_draws: int = 2000,
    seed: int = 20260823,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for record in records:
        question_id = str(record["question_id"])
        parent = record.get(parent_system)
        alternate = record.get(alternate_system)
        if not isinstance(parent, dict) or not isinstance(alternate, dict):
            raise ValueError(f"missing systems for {question_id}")
        parent_correct = bool(parent.get("correct"))
        alternate_correct = bool(alternate.get("correct"))
        switched = decisions.get(question_id, {}).get("decision") == "SWITCH"
        final_correct = alternate_correct if switched else parent_correct
        rows.append(
            {
                "question_id": question_id,
                "video_id": str(record["video_id"]),
                "parent_correct": parent_correct,
                "alternate_correct": alternate_correct,
                "switched": switched,
                "final_correct": final_correct,
                "delta": int(final_correct) - int(parent_correct),
            }
        )
    parent_hits = sum(bool(row["parent_correct"]) for row in rows)
    alternate_hits = sum(bool(row["alternate_correct"]) for row in rows)
    final_hits = sum(bool(row["final_correct"]) for row in rows)
    switched_rows = [row for row in rows if row["switched"]]
    wins = sum(row["delta"] == 1 for row in switched_rows)
    losses = sum(row["delta"] == -1 for row in switched_rows)
    decisive = wins + losses

    by_movie: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_movie.setdefault(str(row["video_id"]), []).append(row)
    movies = sorted(by_movie)
    rng = np.random.default_rng(seed)
    samples = np.zeros(bootstrap_draws, dtype=float)
    for draw in range(bootstrap_draws):
        selected = rng.choice(movies, size=len(movies), replace=True)
        sampled = [row for movie in selected for row in by_movie[str(movie)]]
        samples[draw] = sum(int(row["delta"]) for row in sampled) / max(1, len(sampled))
    ci_low, ci_high = np.quantile(samples, [0.025, 0.975]).tolist()
    movie_net = {
        movie: sum(int(row["delta"]) for row in movie_rows)
        for movie, movie_rows in by_movie.items()
    }
    positive_total = sum(max(0, value) for value in movie_net.values())
    concentration = (
        max((max(0, value) for value in movie_net.values()), default=0)
        / positive_total
        if positive_total
        else 0.0
    )
    return {
        "questions": len(rows),
        "parent_hits": parent_hits,
        "alternate_hits": alternate_hits,
        "final_hits": final_hits,
        "net_gain": final_hits - parent_hits,
        "switches": len(switched_rows),
        "wins": wins,
        "losses": losses,
        "decisive_switch_precision": wins / decisive if decisive else 0.0,
        "switch_answer_accuracy": (
            sum(bool(row["alternate_correct"]) for row in switched_rows)
            / len(switched_rows)
            if switched_rows
            else 0.0
        ),
        "movie_bootstrap_accuracy_delta_95ci": [ci_low, ci_high],
        "max_positive_movie_share": concentration,
        "per_movie_net_gain": movie_net,
    }
