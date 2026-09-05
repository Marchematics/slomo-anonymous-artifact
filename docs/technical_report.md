# SLoMO Technical Report

## System Overview

This artifact documents an open-ended question answering system for short
films. The movie is the unit of context: all questions for one film are
answered in a shared multimodal request. This preserves aliases, changing
relationships, event order, causes, and ending state across a question set.

The release contains inference and submission utilities, configuration
guidance, tests, chart data, and this report. Videos, subtitles, model weights,
generated answers, credentials, account identifiers, and evaluation records
remain external inputs.

## Method

For movie `m`, shared evidence `E_m` and the complete question set `Q_m` are
passed together to the answerer:

```text
(E_m, Q_m) -> {a_m,1, a_m,2, ..., a_m,n}
```

The evidence packet contains a timestamped transcript and chronological frames.
The prompt requests concise English answers with exact question IDs, consistent
names and event structure, and no unsupported visual details. Frames are
resized, JPEG encoded, and marked with timestamps. Long transcripts use a
deterministic head-and-tail bound.

## Inference Pipeline

1. Group annotation rows by `video_id`.
2. Parse subtitle cues and retain start timestamps.
3. Sample and encode chronological frames.
4. Send one request containing shared evidence and all movie questions.
5. Parse and validate the complete JSON answer bank.
6. Convert the validated bank to `question_id,prediction` CSV.

Answers are persisted only after a complete movie response has been validated.
Transient provider failures may be retried. Missing IDs, empty answers,
malformed JSON, and incomplete movie banks are errors.

## Reproducibility

The runner uses an OpenAI-compatible Responses endpoint and reads its key from
an environment variable. The repository does not store credentials or data.
The local audit checks the CSV header, exact ID coverage, uniqueness, and
non-empty predictions. Public and private splits remain separate.

Core settings are 16 chronological frames, a 512-pixel maximum frame side,
JPEG quality 82, a timestamped transcript, temperature 0, high reasoning
effort, and one JSON object per movie.

## Results and Ablations

The historical official score chain is included as a reference chart. It shows
the large discontinuity associated with introducing a movie-batched multimodal
answer source. The factor chart is a frozen six-movie, 60-question local
diagnostic: joint answering reaches 27/60, independent question calls 18/60,
no frames 24/60, Target16 29/60, and Dense48 30/60. The latter values are
local semantic diagnostics, not official leaderboard scores.

## Limitations

Performance depends on transcript quality, visual coverage, endpoint behavior,
and context budget. The artifact does not package competition data or weights,
and it does not claim universal endpoint availability. Reproduction requires
authorized data and model access followed by an independent schema audit.

## Public Sources

- SLoMO Workshop and Challenge: <https://slomo-workshop.github.io/eccv2026/>
- EvalAI: <https://eval.ai/>
- SF20K dataset: <https://huggingface.co/datasets/rghermi/sf20k>
