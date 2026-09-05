# Movie-Level Long-Video QA

This repository releases a movie-level long-video question answering system
for SLoMO. It contains the runner, submission utilities, configuration
guidance, chart data, tests, and a two-page technical report with a third-page
reference list.

## Method

The runner uses one movie as the inference unit:

```text
one movie
  -> full timestamped transcript
  -> chronological visual observations
  -> all questions for that movie in one request
  -> one JSON answer bank
```

The shared context maintains names, relationships, event order, and ending
state across questions. The runner accepts an OpenAI-compatible Responses
endpoint.

The final SLoMO submission by `Math` ranked first with `66.36%` accuracy and a
`3.02` score (`357/538` correct answers).

The released protocol uses `gpt-5.6-luna` without fine-tuning, timestamped VTT
transcripts from faster-whisper large-v3-turbo, and 16 centered uniformly
spaced chronological frames. Each request includes every question from one
movie and returns a question-ID keyed JSON answer bank.

## Reproduce

Install the dependencies in `requirements.txt`, then provide an annotation
JSONL file with `question_id`, `video_id`, `question`, and `video_url`, plus
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

## Submission checklist

The artifact follows the SLoMO/EvalAI submission conventions recorded in the
competition materials:

- use the correct challenge phase shown by the live competition portal;
- submit a CSV with exactly `question_id,prediction` columns;
- include every required question ID exactly once;
- provide a non-empty prediction for every row;
- keep public and private splits separate;
- record the submission configuration with the generated CSV.

The live requirements are documented at:

- <https://slomo-workshop.github.io/eccv2026/>
- <https://eval.ai/>

## Files

- `scripts/run_gpt56_sf20k_movie_batch.py`: movie-batched multimodal runner;
- `scripts/run_gpt56_sf20k_overlap.py`: frame, transcript, and response helpers;
- `scripts/prepare_evalai_submission.py`: official CSV conversion and schema
  validation;
- `scripts/audit_submission.py`: local coverage and non-empty-row audit;
- `technical_report.tex`: clean technical report using the ECCV style files;
- `technical_report.pdf`: two-page main report with third-page references;
- `docs/technical_report.md`: Markdown version of the report;
- `figures/figure_data.json`: traceable chart data;
- `figures/evidence_dashboard.pdf`: compact five-panel vector evidence figure;
- `figures/system_architecture.pdf`: movie-level joint-inference architecture;
- `references.bib`: verified bibliography for the technical report;
- `scripts/make_report_figures.py`: deterministic PDF chart generator.

The report uses ECCV typography and page geometry while presenting a compact
technical description of the final system.

The compact evidence dashboard uses direct labels, small multiples, and a
color-safe scientific palette. Its visual conventions are informed by
[figures4papers](https://github.com/ChenLiu-1996/figures4papers); the plotting
code and all inputs in this repository are original and reproducible.
