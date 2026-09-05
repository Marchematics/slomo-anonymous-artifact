# Movie-Level Joint Inference for Open-Ended Long-Video QA

The PDF uses the ECCV style files for typography and page geometry. It is a
technical/reproducibility document, not an ECCV submission template: there is
no paper ID, review line numbering, or submission metadata.

## System Overview

The movie is the unit of context: all questions for one film are answered in a
shared multimodal request. This preserves aliases, changing relationships,
event order, causes, and ending state across a question set and produces one
coherent answer bank. The final SLoMO submission ranked first with 66.36%
accuracy and a 3.02 score.

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

The complete answer bank is converted to the required
`question_id,prediction` CSV.

## Reproducibility

The runner uses an OpenAI-compatible Responses endpoint. The conversion and
audit utilities produce a complete `question_id,prediction` CSV with one
answer for every requested ID.

Core settings are 16 chronological frames, a 512-pixel maximum frame side,
JPEG quality 82, a timestamped transcript, temperature 0, high reasoning
effort, and one JSON object per movie.

## Results and Ablations

The report uses a compact five-panel evidence dashboard. It combines the
development trajectory, joint-context comparison, visual evidence, visual
allocation, and a source comparison. Joint answering reaches 27/60, compared
with 18/60 for independent question calls, a gain of nine correct answers. The
source study compares GPT-5.6-Luna, Qwen3-VL Flash, Qwen-VL Plus, Qwen3.8-Max,
and Kimi-K3 on a common 141-question movie-level overlap set. Every source
receives the same evidence and questions. GPT-5.6-Luna reaches 74 correct
answers.
The no-frame, Target16, and Dense48 conditions quantify the visual
configurations around the shared movie context.

## Public Sources

- SLoMO Workshop and Challenge: <https://slomo-workshop.github.io/eccv2026/>
- EvalAI: <https://eval.ai/>
- SF20K dataset: <https://huggingface.co/datasets/rghermi/sf20k>
