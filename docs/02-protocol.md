# Experimental Protocol and Data Schema (v1)

This protocol puts the methodology in [01-methodology.md](01-methodology.md) into practice. The code enforces it; this document explains what the code enforces and why.

## 1. Authoring a case

- **One file per case.** Each case is a YAML file in `cases/`, named after its `case_id`. The schema is `judgelab/schema.py` → `Case`, and the worked example is [IN-THEFT-001](../cases/IN-THEFT-001.yaml).
- **Evidence pool.** `evidence` holds every item the experiment might use. `baseline` lists the items shown in the baseline condition, in their canonical order.
- **Self-contained item text.** Never refer to another item's label (such as "PW-2"), because labels are assigned at render time by presentation position. Refer to people only by role. Use no names, pronouns, or caste, religion or gender markers.
- **Variants** are alternative texts for one item. Each variant declares exactly one of the following, and the validator rejects anything else:
  - `changes: {attribute: level}` — a content manipulation of one attribute;
  - `presentation: <label>` — a wording manipulation where the content is equivalent.
- **Relations.** `relations` (corroborates / contradicts) and `independence: derived:<id>` describe how items relate to each other. They are metadata for the analysis and are never shown to the model.
- **Researcher notes.** `researcher_notes` is never rendered to the model.

## 2. Defining an experiment

- **One file per experiment** in `experiments/`; see [pilot-01](../experiments/pilot-01.yaml).
- **Conditions.** Each condition names a `ref`, which defaults to `baseline`, and a list of `ops` applied to that ref:

  | Op | Meaning |
  |---|---|
  | `add: X` | add item X |
  | `remove: X` | remove item X |
  | `variant: {X: name}` | use one of X's variants |
  | `replace: {X: Y}` | replace X with Y; used for category swaps |
  | `position: {X: first\|last}` | pin X first or last within its side |

- **Single-change check.** `validate` and `plan` render each condition and its ref in canonical order and require two things:
  1. The non-evidence text (system prompt, charge, law, facts, the accused's statement) is byte-identical.
  2. The evidence differs by exactly one inserted, deleted or substituted item, or by one moved item. A change that only pins an item's position is allowed, because its effect is only visible per trial.
- **Shared item IDs.** Item IDs referenced by conditions must exist in every case listed in the experiment.
- **Labels for analysis.** `factor`/`level` label each condition in the exported data, and `exploratory: true` marks probes that are outside the confirmatory family.

## 3. Rendering and order

- **Templates** are `prompts/system_<version>.md` and `prompts/user_<version>.md`. Each trial stores hashes of both.
- **Order.** Prosecution evidence always comes before defence evidence. With `order: randomize_within_side`, unpinned items within each side are shuffled per trial, using a seed derived from (experiment seed, case, condition, run). Pinned items go first or last.
- **Labels** (PW-n, DW-n, Ex.P-n, Ex.D-n, MO-n) follow the presentation order. Each trial stores its `label_map` so that the items the model cites can be mapped back to item IDs.

## 4. Running

```bash
python -m judgelab validate cases experiments
```
```bash
python -m judgelab plan experiments/pilot-01.yaml
```
```bash
python -m judgelab run pilot-01 --max-usd 5
```

- **Plan.** `plan` writes one pending trial per case × condition × model × run, shuffled once by the experiment seed. Re-planning an unchanged spec adds nothing. A changed spec under the same `experiment_id` is refused, so a frozen design cannot be altered after the fact.
- **Run.** `run` executes only pending trials.
  - **Statelessness.** Every call is a new stateless request.
  - **Prompt integrity.** Each prompt is re-rendered from the stored case and checked against its stored hash before it is sent.
  - **Retries.** Only connection errors, 408/409/429 and 5xx responses are retried, up to 3 attempts, and every attempt is logged.
  - **No re-asking.** Refusals and invalid outputs are recorded as outcomes and never re-asked.
  - **Budget.** The run stops at the smaller of `--max-usd` and the spec's `budget.max_usd`, and at `budget.max_calls`. Because the check happens before each call is submitted, the cap can be overshot by at most `--concurrency` calls.
- **Rate and quota limits.** Each model's `rpm` paces its calls. If a model hits a rate limit, a quota limit or a billing error, its trial goes back to `pending` (the attempts stay logged), and the run stops sending to that model. Running again later resumes from there. Billing errors are not retried.
- **Dry run.** `--dry-run` (placed before the subcommand) uses a fake model and the separate `data/dryrun.sqlite`, so a rehearsal can never mix with real data.
- **Live-run guard.** A live run is refused while any model ID in the spec still contains `TODO`.

## 4a. Local models (Ollama)

- **Request.** Each analysis is one `POST /api/chat` to Ollama on this computer. It contains exactly two messages (instructions and case record), the same response schema as the cloud models (Ollama's `format` field), and `stream: false`. No history is carried between analyses.
- **Address.** The address comes from `OLLAMA_HOST` (default `http://localhost:11434`). Addresses that aren't on this computer are refused unless `JUDGELAB_ALLOW_REMOTE_OLLAMA=1` is set. The web interface can't change the address, and the dashboard itself listens on localhost only.
- **Pre-flight check.** Before any analysis, the runner checks that Ollama is running and that every model in the design is installed. It records each model's digest and the Ollama version, stored as `name@digest (ollama x.y.z)`. Models are never downloaded automatically.
- **Seeds.** Each analysis gets its own seed, derived from (experiment seed, case, scenario, model, repeat). Repeats differ from one another, but each analysis can be reproduced exactly with the same model digest.
- **Settings recorded.** `num_ctx` (context size) and `num_predict` (maximum response length) are recorded. If the case plus the answer reaches `num_ctx`, the result is recorded as `context_overflow`, because Ollama would otherwise silently drop the start of the case.
- **Failures.** If Ollama stops or a model disappears mid-run, the affected trials go back to `pending` and that model is paused for the run; nothing is ever sent to a cloud provider instead. Timeouts (`timeout_s`, default 300 s) are retried up to 3 times.
- **One at a time.** Local runs process one analysis at a time by default, and a lock file stops the same experiment running twice.

## 5. What is recorded

See `judgelab/db.py`. One row per trial holds:

- the exact prompt and its hash;
- the evidence order and label map;
- the model configuration, including the prices used.

One row per attempt holds:

- the raw response text and the model ID the provider returned;
- request ID, tokens, cost, stop reason and HTTP status/error;
- the parse status and the parsed fields.

Parse status is one of `valid`, `invalid_json`, `schema_violation`, `refusal`, `truncated`, `context_overflow` or `api_error`.

Each attempt also stores `request_json` (the settings actually sent; never the prompt text) and `response_meta_json` (provider metrics, such as Ollama's timings). Both columns were added by an additive migration, so attempts made before it have NULL there.

## 6. Generation parameters

| Setting | Rule |
|---|---|
| Temperature | Sent only when it is set in the spec. Current Claude models reject custom temperature, so the protocol uses each provider's default and records exactly what was sent (`params_json`). |
| Thinking / reasoning | Kept to a minimum to save tokens, and recorded as part of the model configuration: `thinking: false` for Claude Haiku 5.5, `effort: none` for GPT and `effort: minimal` (the thinking level) for Gemini. Any other setting is a different model configuration and gets its own model `key`. |
| Prompt version | `v2` ("lean"): the same task and the same 7 response fields, with reasoning capped at about 60 words. This keeps output to about 270 tokens per call. `v1` is kept for comparison. |
| Structured output | Each provider's native JSON-schema mode, using the schema from `response_schema()`. Value ranges (0–100, 1–7) are enforced by the parser. |

## 7. Exclusions

A trial is excluded only by setting `status='excluded'` with an `exclusion_reason`. Nothing is deleted, and every exclusion appears in the export.
