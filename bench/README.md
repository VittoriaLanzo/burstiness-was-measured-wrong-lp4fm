# `bench/` — frontier-model burstiness bench

Generates a news article for each of **200 headlines** (the frozen human reference set) under
**5 non-overlapping prompting modes**, across several frontier LLM APIs at **maximum reasoning
effort**, and checkpoints every output to disk so a long, paid run is resumable and loss-proof.
This is the *prompt-level control* track (paper Q1): how close can prompt engineering alone push
an LLM's rhythm toward the human `B_n` distribution.

> ✅ **Prompt wording is FINALIZED** (`prompts.PROMPTS_FINALIZED = True`, `bench/prompts.py:42`):
> the 5 modes are locked and real runs no longer need `--force`. Changing the wording now would
> break comparability with completed legs.

> On Windows, use the **Anaconda** interpreter (the global Python is broken
> there): `python`. On macOS/Linux, plain `python3` from the
> repo root works (this is how production runs execute).

## The five modes (bench/prompts.py)

Each mode is a function `(gen, brief) -> final article text`, where `gen(user)` is one model
completion. They cover the 5 orthogonal prompting mechanisms (The Prompt Report, Schulhoff et al.
2024); wording is metric-free by design (never names the measured axis).

| mode | mechanism | calls |
|---|---|---|
| `control` | void baseline (brief sent as-is) | 1 |
| `instruction` | M1 tell: the `VOICE` directive | 1 |
| `fewshot` | M2 show: 3 frozen human exemplars (off-topic, leakage-checked) | 1 |
| `plan` | M3 plan-before: plan the shape, then write (plan not emitted) | 1 |
| `self_refine` | M4 fix-after: draft, critique, rewrite | 3 |

The 200 briefs come straight from the frozen HF dataset (`prompts.load_briefs`,
`ANONYMIZED/paper-dataset`); each brief is a neutral headline + a head-to-head length
(the paired human article's word count). `best_of_N` (M5 select) is **not** a mode: it optimizes
the metric by sampling, so it is reported only as a labeled, compute-matched upper bound.

## Models under test

| key | model | provider | reasoning | API key env |
|---|---|---|---|---|
| `opus-4.8` | `claude-opus-4-8` | Anthropic | adaptive thinking + `effort=high`, `max_tokens=32000` (NEVER max: at effort=max the self_refine rewrite spends the whole budget thinking and emits zero prose; 15/15 cells empty at 16k, 9/15 still truncated at 32k. high also matches Gemini's thinking_level=high) | `ANTHROPIC_API_KEY` |
| `fable-5` | `claude-fable-5` | Anthropic | adaptive thinking + `effort=max` | `ANTHROPIC_API_KEY` |
| `gpt-5.5` | `gpt-5.5` | OpenAI (Responses API) | `reasoning.effort=xhigh` | `OPENAI_API_KEY` |
| `gemini-3.1-pro` | `gemini-3.1-pro-preview` | Google (`google-genai`) | `thinking_level=high` | `GEMINI_API_KEY` |
| `kimi-k2.6` | `kimi-k2.6` | Moonshot (OpenAI-compat) | thinking on | `MOONSHOT_API_KEY` |
| `deepseek-v4` | `deepseek-v4-pro` | DeepSeek (OpenAI-compat) | `reasoning_effort=max`; 32K cap (runaway reasoning at 16K) | `DEEPSEEK_API_KEY` |
| `grok-4.3` | `grok-4.3` | xAI (OpenAI-compat) | built-in, no effort dial | `XAI_API_KEY` |
| `grok-4.5` | `grok-4.5` | xAI (OpenAI-compat) | built-in, no effort dial | `XAI_API_KEY` |
| `fugu-ultra` | `fugu-ultra` | Sakana (OpenAI-compat) | `reasoning.effort=xhigh` | `SAKANA_API_KEY` |
| `glm-5.2` | `glm-5.2` | Z.ai/Zhipu (OpenAI-compat) | `reasoning_effort=max` | `ZAI_API_KEY` |

**Nothing is currently excluded** (`config.EXCLUDED_MODELS` is empty). The earlier Qwen
exclusion (circularity: it was the synthetic-data-generation family) was lifted 2026-07-02 when
the T5 training was retired; Qwen3.6 became the first benchmarked model, run locally via
`bench/gen_vllm.py` rather than as an API leg. See `bench/config.py:105-108`.

## Keys (read from env; never hardcoded)

Export the vars above, **or** drop a gitignored `bench/.env`:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
MOONSHOT_API_KEY=...
DEEPSEEK_API_KEY=...
XAI_API_KEY=...
SAKANA_API_KEY=...
ZAI_API_KEY=...
```

`config.py` loads `bench/.env` (or `./.env`) at import without overriding real env vars. A leg
whose key (or SDK) is missing is cleanly **skipped**, not failed.

## Run

macOS/Linux (production path):

```bash
python3 -m pip install -r bench/requirements.txt   # only the SDKs for legs you run
python3 -m bench.prompts                           # offline self-check of the 5 modes
python3 -m bench.run --check                       # preflight every leg (SDK+key), NO API calls
python3 -m bench.run --dry-run --limit 3           # plan + first rendered cell, NO API calls
python3 -m bench.run --smoke                       # ONE real cell per ready leg
python3 -m bench.run                               # full run (resumable checkpoints)
python3 -m bench.run --models opus-4.8 --limit 5
python3 -m bench.run --models deepseek-v4 --workers 24   # concurrent cells per leg (default 1 = sequential; throughput is often provider-capped)
python3 -m bench.run --iterations 5                # override cycle count (also: BENCH_ITERATIONS)
python3 -m bench.run --all-legs                    # explicitly run EVERY enabled leg (spends on all providers; refused without this flag)
```

Windows:

```powershell
$py = "python"
& $py -m bench.run --check                         # same flags as above
```

Order of operations: `--check` → `--dry-run` → `--smoke` → full. Prompts are finalized, so
`--force` is no longer needed; it remains only as an emergency override.

## Failure handling (not happy-path only)

- `--check` reports each leg READY / NOT READY (missing SDK → `pip install …`; missing key →
  which env var), with **no** API calls.
- A cell error prints `type: message` + full traceback + the leg context, and is **not** recorded
  (so it retries next run).
- A leg that errors `FAIL_FAST` (3) times before any success is declared **broken** and stopped —
  no hammering a dead key/model-id hundreds of times.
- Empty output (refusal/truncation) is **WARNed** and recorded for inspection.
- The end summary distinguishes `ok` / `ok-with-errors` / `FAILED` / `skipped`, and the process
  **exits non-zero** if any leg failed.

## The array & checkpointing

Output space: **mode × model × brief × iteration**, id `prompt_id = <mode>__s<brief_id>`.

- Results are written **per model** to `bench/results/<model_key>.jsonl`, one JSON line per cell,
  flushed + `fsync`'d immediately.
- On start, each model's existing cells are loaded and **skipped** — a crash, timeout, or
  non-answer never loses earlier outputs; re-run to resume.
- Reconstruct with `storage.load_records(model_key)` (fields: `eval_prompt_id` (the mode),
  `scenario_id` (the brief id), `prompt_id`, `iteration`, `text`, `reasoning`, `usage`, `ts`).

## Notes & caveats

- Model IDs reflect current (2026) flagships; the Gemini/GPT IDs are still "preview" and may roll
  to GA — re-check before a long run. `glm-5.2` stream-usage is off until verified.
- `max_tokens=16000` is shared, but on Anthropic it caps thinking AND text together, so opus-4.8 carries a per-model override to 32000 (parity of TEXT budget, not raw cap; see bench/config.py). Measured 2026-07-13: effort=max is a runaway there, not a budget problem
- Built with the **ponytail** minimalism plugin — least code that does the job.


## Batch-mode runner (Anthropic legs)

`python3 -m bench.run_batch --model opus-4.8 --effort high` runs a full leg
through the Message Batches API at half price: wave A (single-call modes +
self_refine drafts) -> wave B (critiques) -> wave C (rewrites). Resumable at
every stage via `<key>.batchstate.json`; output is storage-compatible so
`make leg` works unchanged. `--dry-run` prints a deliberately conservative
cost preview. A preflight archives any pre-existing results file so one leg
never mixes generation configs.

## Live-run procedure (learned 2026-07-11)

- Long live runs must be launched detached (`nohup ... &`) or wrapped in
  `caffeinate -is`: machine sleep killed two runs mid-leg. Batch-mode runs are
  immune (Anthropic does the computing).
- Provider content-filter refusals are recorded as empty cells
  (usage.refusal), never retried into fail-fast. The refusal pattern itself is
  data: see knowledge/PROVIDER-REFUSAL-AUDIT.md.
- Cost estimation: price sheet x reasoning-mode output multiplier x mode-mix
  multiplier. Thinking modes produced 5-8x terse-model output volume
  (kimi 15.15M out tokens); self_refine is ~40-50% of a thinking leg's cost.
