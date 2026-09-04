#!/usr/bin/env python3
"""Answer all SF20K questions for each movie with one GPT-5.6 call.

The runner is useful for a first full-corpus pass: the model sees one shared
transcript and chronological frame pack per movie, then returns one JSON value
per question. It supports the labelled overlap set for direct validation and
the unlabelled public set for candidate generation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import threading
import time
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_gpt56_sf20k_overlap import (  # noqa: E402
    movie_frames,
    response_text,
    transcript,
)
from src.movie_router.answering import infer_question_type  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--subtitle-dir", type=Path, default=None)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--base-url", default="https://code.conpera.ai")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--transcript-chars", type=int, default=80000)
    parser.add_argument("--max-output-tokens", type=int, default=5000)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--movie-id", action="append", default=[])
    parser.add_argument("--exclude-movie-id", action="append", default=[])
    parser.add_argument("--limit-movies", type=int, default=None)
    parser.add_argument("--multiple-choice", action="store_true")
    parser.add_argument("--open-answer", action="store_true")
    parser.add_argument("--type-aware", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("batch response is not a JSON object")
    return value


def normalize_letter(value: Any, option_count: int) -> str:
    text = str(value).strip().upper()
    match = re.search(r"\b([A-Z])\b", text)
    if not match:
        raise ValueError(f"could not parse option from batch value {value!r}")
    letter = match.group(1)
    if not ord("A") <= ord(letter) < ord("A") + option_count:
        raise ValueError(f"option outside range from batch value {value!r}")
    return letter


def question_block(questions: list[dict[str, Any]], multiple_choice: bool, type_aware: bool) -> str:
    lines: list[str] = []
    for row in questions:
        type_note = f" [{infer_question_type(str(row['question']))}]" if type_aware else ""
        lines.append(f"{row['question_id']}{type_note}: {row['question']}")
        if multiple_choice and row.get("options"):
            lines.extend(
                f"  {chr(ord('A') + index)}. {option}"
                for index, option in enumerate(row["options"])
            )
    return "\n".join(lines)


def make_prompt(
    video_id: str,
    questions: list[dict[str, Any]],
    film_transcript: str,
    frame_count: int,
    multiple_choice: bool,
    type_aware: bool,
) -> str:
    if multiple_choice:
        output_rule = (
            "Return one JSON object with the exact question IDs as keys and one uppercase "
            "option letter (A, B, C, D, or E) as each value."
        )
    else:
        output_rule = (
            "Return one JSON object with the exact question IDs as keys and concise factual "
            "English answer phrases as values. Answer each question independently."
        )
    type_instruction = (
        "Respect the inferred question type: who asks for a person/name, where for a location, "
        "how-many for a number, when for a time, visual for a visible object/action, and causal "
        "for a reason. Do not answer a different aspect of the movie. "
        if type_aware
        else ""
    )
    return (
        "Answer every question about the same short film. First reconstruct a compact fact "
        "ledger and chronological timeline silently. Resolve names, relationships, event "
        "order, causes, repeated objects, and the final state consistently across answers. "
        f"The film evidence contains {frame_count} chronological frames. {type_instruction}{output_rule} "
        "Do not include markdown, explanations, extra keys, or the question text. "
        "Use only facts supported by the transcript or visible frames.\n\n"
        f"MOVIE ID: {video_id}\n\n"
        f"TRANSCRIPT:\n{film_transcript}\n\n"
        f"QUESTIONS:\n{question_block(questions, multiple_choice, type_aware)}"
    )


def main() -> None:
    args = parse_args()
    if args.frames < 1 or args.frame_size < 64 or args.workers < 1:
        raise ValueError("invalid frame or worker configuration")
    token = os.environ.get(args.api_key_env, "").strip()
    if not token:
        raise RuntimeError(f"{args.api_key_env} is empty or unset")

    all_rows = load_rows(args.annotations)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[str(row["video_id"])].append(row)
    selected_movies = sorted(grouped)
    if args.movie_id:
        wanted = set(args.movie_id)
        selected_movies = [movie_id for movie_id in selected_movies if movie_id in wanted]
    if args.exclude_movie_id:
        excluded = set(args.exclude_movie_id)
        selected_movies = [movie_id for movie_id in selected_movies if movie_id not in excluded]
    if args.limit_movies is not None:
        selected_movies = selected_movies[: args.limit_movies]
    grouped = {movie_id: grouped[movie_id] for movie_id in selected_movies}
    rows = [row for movie_id in selected_movies for row in grouped[movie_id]]
    if not rows:
        raise ValueError("no annotation rows selected")

    completed: dict[str, dict[str, Any]] = {}
    if args.resume and args.out.exists():
        for row in load_rows(args.out):
            qid = str(row.get("question_id", ""))
            answer = str(row.get("answer", "")).strip()
            if qid and answer:
                completed[qid] = row
    pending_movies = [
        movie_id
        for movie_id in selected_movies
        if any(str(row["question_id"]) not in completed for row in grouped[movie_id])
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    endpoint = args.base_url.rstrip("/") + "/responses"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    cache_lock = threading.Lock()
    write_lock = threading.Lock()
    evidence_cache: dict[str, tuple[str, list[tuple[float, str]]]] = {}

    def evidence(movie_id: str) -> tuple[str, list[tuple[float, str]]]:
        with cache_lock:
            if movie_id not in evidence_cache:
                first = grouped[movie_id][0]
                video_path = Path(str(first.get("video_path", "")))
                subtitle_path = Path(str(first.get("asr_path", "")))
                if args.video_dir is not None:
                    video_path = args.video_dir / f"{movie_id}.mp4"
                if args.subtitle_dir is not None:
                    subtitle_path = args.subtitle_dir / f"{movie_id}.vtt"
                evidence_cache[movie_id] = (
                    transcript(subtitle_path, args.transcript_chars),
                    movie_frames(video_path, args.frames, args.frame_size),
                )
            return evidence_cache[movie_id]

    def answer_movie(movie_id: str) -> list[dict[str, Any]]:
        questions = grouped[movie_id]
        film_transcript, frames = evidence(movie_id)
        multiple_choice = (args.multiple_choice and not args.open_answer) or (
            not args.open_answer and all(row.get("options") for row in questions)
        )
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": make_prompt(
                    movie_id,
                    questions,
                    film_transcript,
                    len(frames),
                    multiple_choice,
                    args.type_aware,
                ),
            }
        ]
        for timestamp, image in frames:
            content.append({"type": "input_text", "text": f"[FRAME t={timestamp:.2f}s]"})
            content.append({"type": "input_image", "image_url": image, "detail": "high"})
        payload = {
            "model": args.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "You are a precise batch movie-QA solver. Return JSON only."}],
                },
                {"role": "user", "content": content},
            ],
            "reasoning": {"effort": args.reasoning_effort},
            "max_output_tokens": args.max_output_tokens,
            "stream": True,
            "store": False,
        }
        last: Exception | None = None
        for attempt in range(args.max_retries + 1):
            try:
                with httpx.Client(timeout=args.timeout, trust_env=True) as client:
                    with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                        if response.status_code >= 400:
                            detail = response.read().decode("utf-8", errors="replace")[:1200]
                            raise RuntimeError(f"HTTP {response.status_code}: {detail}")
                        raw = response_text(response)
                parsed = parse_object(raw)
                output: list[dict[str, Any]] = []
                for row in questions:
                    qid = str(row["question_id"])
                    value = parsed.get(qid, "")
                    if multiple_choice:
                        answer = normalize_letter(value, len(row["options"]))
                        predicted = row["options"][ord(answer) - ord("A")]
                        correct = (
                            answer == str(row.get("correct_letter"))
                            if row.get("correct_letter")
                            else None
                        )
                    else:
                        answer = " ".join(str(value).strip().split())
                        if not answer:
                            raise ValueError(f"missing answer for {qid}")
                        predicted = answer
                        correct = None
                    output.append(
                        {
                            "question_id": qid,
                            "video_id": movie_id,
                            "answer": answer,
                            "prediction": predicted,
                            "correct": correct,
                            "model": args.model,
                            "frames": len(frames),
                            "timestamps": [round(ts, 3) for ts, _image in frames],
                            "raw_batch_answer": raw,
                        }
                    )
                return output
            except Exception as exc:  # pragma: no cover - provider dependent
                last = exc
                if attempt < args.max_retries:
                    time.sleep(min(2**attempt, 16))
        raise RuntimeError(f"failed for {movie_id}: {last}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(answer_movie, movie_id): movie_id for movie_id in pending_movies}
        for index, future in enumerate(as_completed(futures), start=1):
            batch = future.result()
            with write_lock, args.out.open("a", encoding="utf-8") as handle:
                for row in batch:
                    completed[str(row["question_id"])] = row
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            print(f"answered movie {index}/{len(pending_movies)} {futures[future]} ({len(batch)} questions)", flush=True)

    selected_qids = {str(row["question_id"]) for row in rows}
    completed_selected = selected_qids.intersection(completed)
    if len(completed_selected) != len(selected_qids):
        raise RuntimeError(f"incomplete run: {len(completed_selected)}/{len(selected_qids)}")
    labelled = [row for row in completed.values() if row.get("correct") is not None]
    hits = sum(bool(row["correct"]) for row in labelled)
    args.out.with_suffix(".config.json").write_text(
        json.dumps(
            {
                "annotations": str(args.annotations.resolve()),
                "out": str(args.out.resolve()),
                "model": args.model,
                "base_url": args.base_url,
                "frames": args.frames,
                "frame_size": args.frame_size,
                "transcript_chars": args.transcript_chars,
                "reasoning_effort": args.reasoning_effort,
                "workers": args.workers,
                "movies": len(selected_movies),
                "rows": len(rows),
                "labelled_rows": len(labelled),
                "correct": hits,
                "accuracy": hits / len(labelled) if labelled else None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "movies": len(selected_movies),
                "rows": len(rows),
                "labelled_rows": len(labelled),
                "correct": hits,
                "accuracy": hits / len(labelled) if labelled else None,
                "out": str(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
