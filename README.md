# Anonymous Long-Video QA Artifact

This repository contains the anonymous reproducibility artifact for a
movie-level long-video question answering system. It is intentionally limited
to code, configuration guidance, and technical documentation. Datasets,
videos, model weights, generated answers, credentials, account identifiers,
and submission files are excluded.

## Method

The released runner preserves the central inference contract:

```text
one movie
  -> full timestamped transcript
  -> chronological visual observations
  -> all questions for that movie in one request
  -> one JSON answer bank
```

The shared context lets the answerer maintain names, relationships, event
order, and ending state across questions. The runner is compatible with an
OpenAI-style Responses endpoint. API credentials are read only from an
environment variable.

## Reproduce

Install the dependencies in `requirements.txt`, then provide an annotation
JSONL file with `question_id`, `video_id`, `question`, `video_url` and local
video/subtitle files named `<video_id>.mp4` and `<video_id>.vtt`.

```bash
export OPENAI_API_KEY='...'

python scripts/run_gpt56_sf20k_movie_batch.py \
  --annotations path/to/annotations.jsonl \
  --video-dir path/to/videos \
  --subtitle-dir path/to/subtitles \
  --out outputs/answers.jsonl \
  --model gpt-5.6-luna \
  --base-url https://example.invalid \
  --api-key-env OPENAI_API_KEY \
  --frames 16 \
  --frame-size 512 \
  --transcript-chars 80000 \
  --max-output-tokens 5000 \
  --reasoning-effort high \
  --open-answer

python scripts/prepare_evalai_submission.py \
  --answers outputs/answers.jsonl \
  --annotations path/to/annotations.jsonl \
  --out outputs/submission.csv

python scripts/audit_submission.py \
  --annotations path/to/annotations.jsonl \
  --submission outputs/submission.csv
```

The endpoint, model identifier, and quotas are deployment-specific. They are
not hard-coded as credentials or as a claim of current service availability.

## Submission checklist

The artifact follows the SLoMO/EvalAI submission conventions recorded in the
competition materials:

- use the correct challenge phase shown by the live competition portal;
- submit a CSV with exactly `question_id,prediction` columns;
- include every required question ID exactly once;
- provide a non-empty prediction for every row;
- keep public and private splits separate;
- do not use private labels, options, or evaluation outputs when generating
  predictions;
- record the final file hash and software configuration outside the anonymous
  source tree when submitting an artifact.

The live requirements remain authoritative:

- <https://slomo-workshop.github.io/eccv2026/>
- <https://eval.ai/>

## Anonymous release boundary

The Git history for this artifact starts from a fresh anonymous commit. No
author names, team names, account handles, API keys, absolute local paths,
private annotations, model caches, answer banks, or leaderboard submission IDs
are part of the release tree.

## Files

- `scripts/run_gpt56_sf20k_movie_batch.py`: movie-batched multimodal runner;
- `scripts/run_gpt56_sf20k_overlap.py`: frame, transcript, and response helpers;
- `scripts/prepare_evalai_submission.py`: official CSV conversion and schema
  validation;
- `scripts/audit_submission.py`: local coverage and non-empty-row audit;
- `docs/technical_document.md`: concise technical description;
- `docs/technical_document.tex`: two-page printable version.

