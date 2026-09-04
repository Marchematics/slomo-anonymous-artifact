#!/usr/bin/env python3
"""Run a GPT-5.6 visual multiple-choice probe on the SF20K overlap set.

The overlap annotations contain released correct options, so this runner can
measure the larger model directly without an EvalAI submission. Each record is
resumable and retains the raw model response for auditing.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any

import cv2
import httpx
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.movie_router.video import parse_vtt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default="https://code.conpera.ai")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=448)
    parser.add_argument("--transcript-chars", type=int, default=50000)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def transcript(path: Path, max_chars: int) -> str:
    if not path.is_file():
        return "(no transcript available)"
    cues = parse_vtt(path)
    text = "\n".join(f"[{start:.1f}s] {body}" for start, _end, body in cues)
    if len(text) <= max_chars:
        return text or "(empty transcript)"
    # Keep both the setup and the ending because questions often refer to
    # either a causal premise or the final state of the film.
    head = max_chars // 2
    return text[:head] + "\n[...middle omitted...]\n" + text[-(max_chars - head) :]


def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def movie_frames(video_path: Path, count: int, size: int) -> list[tuple[float, str]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"could not open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    if total <= 0 or fps <= 0:
        capture.release()
        raise ValueError(f"invalid video metadata: {video_path}")
    result: list[tuple[float, str]] = []
    try:
        for index in range(count):
            frame_number = min(total - 1, int((index + 0.5) * total / count))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, size / max(height, width))
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result.append((frame_number / fps, encode_image(image)))
    finally:
        capture.release()
    return result


def response_text(response: httpx.Response) -> str:
    chunks: list[str] = []
    for line in response.iter_lines():
        if not line:
            continue
        value = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
        if value.startswith("data:"):
            value = value[5:].strip()
        if not value or value in {"[DONE]", "event: ping"} or value.startswith(":"):
            continue
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "response.output_text.delta":
            chunks.append(str(payload.get("delta", "")))
        elif payload.get("type") == "response.completed":
            body = payload.get("response") or {}
            for item in body.get("output", []) or []:
                for part in item.get("content", []) or []:
                    if part.get("type") == "output_text":
                        text = str(part.get("text", ""))
                        if text and text not in "".join(chunks):
                            chunks.append(text)
    output = "".join(chunks).strip()
    if not output:
        raise ValueError("Responses stream contained no output text")
    return output


def parse_letter(raw: str, count: int) -> str:
    text = raw.strip().upper()
    matches = re.findall(
        r"(?:FINAL\s+ANSWER|ANSWER|CHOICE|OPTION)\s*[:=\-]?\s*([A-Z])\b",
        text,
    )
    if not matches:
        matches = [
            value
            for value in re.findall(r"\b([A-Z])\b", text[-120:])
            if ord(value) - ord("A") < count
        ]
    if not matches:
        raise ValueError(f"could not parse option letter from {raw!r}")
    letter = matches[-1]
    if not ord("A") <= ord(letter) < ord("A") + count:
        raise ValueError(f"option outside range from {raw!r}")
    return letter


def prompt(row: dict[str, Any], film_transcript: str, frame_count: int) -> str:
    options = "\n".join(
        f"{chr(ord('A') + index)}. {value}"
        for index, value in enumerate(row["options"])
    )
    return (
        "Answer one SF20K short-film multiple-choice question. Use the chronological "
        f"video frames and timestamped transcript as evidence. There are {frame_count} "
        "frames in chronological order. Reconstruct the relevant event, relationship, "
        "motivation, or object before choosing. Ignore option position priors. Return "
        "exactly one uppercase option letter and nothing else.\n\n"
        f"TRANSCRIPT:\n{film_transcript}\n\n"
        f"QUESTION:\n{row['question']}\n\n"
        f"OPTIONS:\n{options}\n\n"
        "FINAL ANSWER (one letter only):"
    )


def main() -> None:
    args = parse_args()
    if args.frames < 1 or args.frame_size < 64 or args.workers < 1:
        raise ValueError("invalid frame or worker configuration")
    token = os.environ.get(args.api_key_env, "").strip()
    if not token:
        raise RuntimeError(f"{args.api_key_env} is empty or unset")

    rows = load_rows(args.annotations)
    if args.question_id:
        wanted = set(args.question_id)
        rows = [row for row in rows if str(row["question_id"]) in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and args.out.exists():
        for row in load_rows(args.out):
            completed[str(row["question_id"])] = row
    pending = [row for row in rows if str(row["question_id"]) not in completed]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    endpoint = args.base_url.rstrip("/") + "/responses"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    write_lock = threading.Lock()
    frame_cache: dict[str, list[tuple[float, str]]] = {}
    transcript_cache: dict[str, str] = {}
    cache_lock = threading.Lock()

    def cached_evidence(row: dict[str, Any]) -> tuple[str, list[tuple[float, str]]]:
        video_id = str(row["video_id"])
        with cache_lock:
            if video_id not in frame_cache:
                frame_cache[video_id] = movie_frames(
                    Path(str(row["video_path"])), args.frames, args.frame_size
                )
            if video_id not in transcript_cache:
                transcript_cache[video_id] = transcript(
                    Path(str(row["asr_path"])), args.transcript_chars
                )
            return transcript_cache[video_id], frame_cache[video_id]

    def answer(row: dict[str, Any]) -> dict[str, Any]:
        film_transcript, frames = cached_evidence(row)
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": prompt(row, film_transcript, len(frames))}
        ]
        for timestamp, image in frames:
            content.append({"type": "input_text", "text": f"[FRAME t={timestamp:.2f}s]"})
            content.append({"type": "input_image", "image_url": image, "detail": "high"})
        payload = {
            "model": args.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": "You are a precise visual QA solver."}]},
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
                letter = parse_letter(raw, len(row["options"]))
                predicted_index = ord(letter) - ord("A")
                return {
                    "question_id": str(row["question_id"]),
                    "video_id": str(row["video_id"]),
                    "question": str(row["question"]),
                    "reference_letter": str(row["correct_letter"]),
                    "predicted_letter": letter,
                    "correct": letter == str(row["correct_letter"]),
                    "prediction": str(row["options"][predicted_index]),
                    "options": row["options"],
                    "raw_answer": raw,
                    "model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "frames": len(frames),
                    "timestamps": [round(timestamp, 3) for timestamp, _image in frames],
                }
            except Exception as exc:  # pragma: no cover - provider dependent
                last = exc
                if attempt < args.max_retries:
                    time.sleep(min(2**attempt, 16))
        raise RuntimeError(f"failed for {row['question_id']}: {last}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(answer, row): row for row in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            completed[str(result["question_id"])] = result
            with write_lock, args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            print(
                f"answered {index}/{len(pending)} {result['question_id']} "
                f"pred={result['predicted_letter']} correct={result['correct']}",
                flush=True,
            )

    if len(completed) != len(rows):
        raise RuntimeError(f"incomplete run: {len(completed)}/{len(rows)}")
    hits = sum(bool(row["correct"]) for row in completed.values())
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
                "rows": len(rows),
                "correct": hits,
                "accuracy": hits / len(rows) if rows else None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "correct": hits, "accuracy": hits / len(rows), "out": str(args.out)}))


if __name__ == "__main__":
    main()
