# `bench/` — frontier-model burstiness bench

Generates a news article for each of 200 headlines (the frozen human reference set) under
5 non-overlapping prompting modes, across several LLM APIs at high or maximum reasoning
effort, and checkpoints every output to disk so a long run is resumable. This is the
prompt-level control track: how close prompt engineering alone pushes an LLM's rhythm toward
the human `B_n` distribution.

Prompt wording is finalized (`prompts.PROMPTS_FINALIZED = True` in `bench/prompts.py`): the
5 modes are locked and real runs do not need `--force`. Changing the wording would break
comparability with completed legs.

## The five modes (bench/prompts.py)

Each mode is a function `(gen, brief) -> final article text`, where `gen(user)` is one model
completion. They cover 5 orthogonal prompting mechanisms (The Prompt Report, Schulhoff et al.
2024); the wording is metric-free (it never names the measured axis).

| mode | mechanism | calls |
|---|---|---|
| `control` | baseline: the brief sent as-is | 1 |
| `instruction` | tell: the `VOICE` directive | 1 |
| `fewshot` | show: 3 frozen human exemplars (off-topic, leakage-checked) | 1 |
| `plan` | plan before writing: plan the shape, then write (plan not emitted) | 1 |
| `self_refine` | fix after writing: draft, critique, rewrite | 3 |

The 200 briefs come from the frozen HF dataset (`prompts.load_briefs`,
`burstiness-engine/sentence-length-dispersion`, `paper-dataset/prompts_200_human.jsonl`); each brief is a neutral headline plus a head-to-head length
(the paired human article's word count). `best_of_N` (select) is not a mode: it optimizes
the metric by sampling, so it could only be reported as a labeled, compute-matched upper bound.

## Models

The registry is `bench/config.py` (`MODELS`).

| key | model | provider | reasoning | API key env |
|---|---|---|---|---|
| `opus-4.8` | `claude-opus-4-8` | Anthropic | adaptive thinking; registry `effort=max`, the leg ran at `effort=high` via `run_batch`; `max_tokens=32000` (Anthropic bills thinking tokens against `max_tokens`: at `effort=max` the self_refine rewrite spends the whole budget thinking and emits no prose, all cells empty at 16k and 9 of 15 still truncated at 32k; `high` also matches Gemini's `thinking_level=high`) | `ANTHROPIC_API_KEY` |
| `fable-5` | `claude-fable-5` | Anthropic | adaptive thinking + `effort=max` | `ANTHROPIC_API_KEY` |
| `gpt-5.5` | `gpt-5.5` | OpenAI (Responses API) | `reasoning.effort=xhigh` | `OPENAI_API_KEY` |
| `gemini-3.1-pro` | `gemini-3.1-pro-preview` | Google (`google-genai`) | `thinking_level=high` | `GEMINI_API_KEY` |
| `kimi-k2.6` | `kimi-k2.6` | Moonshot (OpenAI-compat) | thinking on | `MOONSHOT_API_KEY` |
| `deepseek-v4` | `deepseek-v4-pro` | DeepSeek (OpenAI-compat) | `reasoning_effort=max`; 32K cap (at 16K the model can spend the whole budget on reasoning) | `DEEPSEEK_API_KEY` |
| `grok-4.3` | `grok-4.3` | xAI (OpenAI-compat) | built-in, no effort dial | `XAI_API_KEY` |
| `grok-4.5` | `grok-4.5` | xAI (OpenAI-compat) | built-in, no effort dial | `XAI_API_KEY` |
| `fugu-ultra` | `fugu-ultra` | Sakana (OpenAI-compat) | `reasoning.effort=xhigh` | `SAKANA_API_KEY` |
| `glm-5.2` | `glm-5.2` | Z.ai/Zhipu (OpenAI-compat) | `reasoning_effort=max` | `ZAI_API_KEY` |

Legs run: `opus-4.8` (through the batch runner), `gemini-3.1-pro`, `kimi-k2.6`, `deepseek-v4`,
`grok-4.3`, `grok-4.5`, `glm-5.2`. Registered only, never run: `fable-5`, `gpt-5.5`,
`fugu-ultra`; `config.EXCLUDED_MODELS` records the reason for each (fable-5 has no published
list price, so its leg cannot be costed; gpt-5.5 was dropped on cost, the harness having no
OpenAI batch lane; fugu-ultra has no recorded reason). The two open-weights legs, Qwen3.6-27B
and Gemma 3 27B, are not API legs: they run locally through `bench/gen_vllm.py`.

## Keys (read from env)

Export the vars above, or put them in a gitignored `bench/.env`:

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
whose key (or SDK) is missing is skipped, not failed.

## Run

From the repo root:

```bash
python3 -m pip install -r bench/requirements.txt   # only the SDKs for legs you run
python3 -m bench.prompts                           # offline self-check of the 5 modes
python3 -m bench.run --check                       # preflight every leg (SDK+key), no API calls
python3 -m bench.run --dry-run --limit 3           # plan + first rendered cell, no API calls
python3 -m bench.run --smoke                       # one real cell per ready leg
python3 -m bench.run                               # full run (resumable checkpoints)
python3 -m bench.run --models opus-4.8 --limit 5
python3 -m bench.run --models deepseek-v4 --workers 24   # concurrent cells per leg (default 1 = sequential; throughput is often provider-capped)
python3 -m bench.run --iterations 5                # override cycle count (also: BENCH_ITERATIONS)
python3 -m bench.run --all-legs                    # run every enabled leg (spends on all providers; refused without this flag)
```

Order of operations: `--check`, `--dry-run`, `--smoke`, full. `--force` overrides the
`PROMPTS_FINALIZED` guard and is not needed.

## Failure handling

- `--check` reports each leg READY / NOT READY (missing SDK: which `pip install`; missing key:
  which env var), with no API calls.
- A cell error prints `type: message`, the full traceback and the leg context, and is not
  recorded, so it retries on the next run.
- A leg that errors `FAIL_FAST` (3) times before any success is declared broken and stopped.
- A provider content-filter refusal is recorded as an empty cell with `usage.refusal` set and
  does not count toward fail-fast. The per-leg refusal counts are in each leg's `meta.json`.
- Other empty output (truncation) is warned about and recorded for inspection.
- The end summary distinguishes `ok` / `ok-with-errors` / `FAILED` / `skipped`, and the process
  exits non-zero if any leg failed.

## The array and checkpointing

Output space: mode × model × brief × iteration, id `prompt_id = <mode>__s<brief_id>`.

- Results are written per model to `bench/results/<model_key>.jsonl`, one JSON line per cell,
  flushed and `fsync`'d immediately.
- On start, each model's existing cells are loaded and skipped, so a crash, timeout, or
  non-answer never loses earlier outputs; re-run to resume.
- Reconstruct with `storage.load_records(model_key)` (fields: `eval_prompt_id` (the mode),
  `scenario_id` (the brief id), `prompt_id`, `iteration`, `text`, `reasoning`, `usage`, `ts`).

## Notes

- The Gemini model id is a preview id (`gemini-3.1-pro-preview`). `glm-5.2` and `fugu-ultra`
  have `include_usage` off (stream usage support unverified).
- `max_tokens=16000` is shared, but on Anthropic it caps thinking and text together, so
  `opus-4.8` carries a per-model override to 32000: parity of text budget, not of raw cap
  (see `bench/config.py`). `deepseek-v4` carries the same override.

## Batch-mode runner (Anthropic legs)

`python3 -m bench.run_batch --model opus-4.8 --effort high` runs a full leg
through the Message Batches API at half the list price: wave A (single-call modes +
self_refine drafts) -> wave B (critiques) -> wave C (rewrites). Resumable at
every stage via `<key>.batchstate.json`; output is storage-compatible, so
downstream scoring is unchanged. `--dry-run` prints a cost preview. A preflight
archives any pre-existing results file so one leg never mixes generation configs.
A leg with cells that hit `max_tokens` before emitting any text exits non-zero and
is not scoreable; the fix is a larger `max_tokens` for that model in `bench/config.py`.
