# Anonymous Technical Document

## 1. Problem and artifact scope

The task is open-ended question answering over short films. Each example
contains a movie identifier, a natural-language question, and an answer that
must be submitted without exposing evaluation labels. The core engineering
problem is long-range evidence integration: a question may depend on a name
introduced early in the film, a relation that changes later, or the final
state of an event chain.

This artifact releases the inference and submission code only. Raw videos,
subtitles, model weights, generated predictions, credentials, and evaluation
records are intentionally external inputs. This boundary supports anonymous
review and prevents accidental disclosure of test information.

## 2. Movie-level answer-bank construction

The main runner groups questions by `video_id`. For each movie it builds one
request containing a timestamped transcript and a chronological set of visual
observations. All questions for that movie are placed in the same request. The
model is asked to return exactly one JSON object whose keys are the original
question IDs and whose values are concise English answers.

The computation can be viewed as:

\[
  (E_m,Q_m) \longrightarrow
  \{a_{m,1},a_{m,2},\ldots,a_{m,n_m}\},
\]

where \(E_m\) is shared evidence for movie \(m\), \(Q_m\) is its complete
question set, and the answer bank is emitted in one autoregressive context.
The prompt explicitly asks for consistent names, aliases, relationships,
chronology, causes, repeated objects, and ending state. It also forbids facts
that are unsupported by the supplied transcript or frames.

The implementation keeps the evidence contract explicit. Transcript cues are
serialized with their start timestamps. Frames are sampled in chronological
order, resized to a fixed maximum side, encoded as JPEG, and placed after the
text prompt with timestamp markers. The transcript is bounded with a
head-and-tail policy so both setup and ending evidence remain visible when a
source transcript is longer than the configured limit.

## 3. Reproducibility and failure handling

The runner accepts an OpenAI-compatible Responses endpoint and reads its API
key from a process environment variable. The key is never written to output
files or logs. Movie requests can be retried for transient provider failures;
answers are persisted only after the complete JSON object has been parsed and
validated against every question ID in that movie. A missing key, malformed
JSON object, missing answer, or incomplete movie bank is treated as an error.

This all-or-nothing movie boundary is deliberate. It prevents a partial
request from silently mixing answers produced under different evidence or
decoding conditions. A separate conversion script maps a validated answer
bank into the competition schema. The audit script checks exact ID coverage,
duplicate rows, empty predictions, and the required header.

## 4. Competition-facing compliance

The release follows the operational requirements used by the SLoMO/EvalAI
workflow. Public and private annotations are supplied separately by the user;
the repository contains neither. A submission artifact must use the live
challenge phase, contain exactly `question_id,prediction`, cover every required
ID once, and contain no empty prediction. The code does not read answer keys,
options, or leaderboard results during inference.

The repository is a fresh anonymous Git history. Identity-bearing material is
outside the tree, including account names, team names, local absolute paths,
API tokens, private data, generated answer banks, and submission IDs. The live
competition and EvalAI portals remain the authority for deadlines, phase
availability, metadata fields, and upload policy.

## 5. Known limitations

The method is inference-oriented and depends on the quality of the supplied
transcript, frame pack, model endpoint, and context budget. The artifact does
not claim that every endpoint or model identifier remains available. It also
does not package competition data or an official score. Reproduction requires
authorized access to the relevant dataset and model service, followed by an
independent local audit of the generated CSV.

