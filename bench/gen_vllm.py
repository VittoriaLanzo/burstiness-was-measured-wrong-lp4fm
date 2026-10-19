#!/usr/bin/env python3
"""Offline batched generation of the prompt-ceiling benchmark on a local vLLM.

Runs every (mode, brief, seed) cell for the 5 prompting modes through a locally served
open-weights model (default Qwen3.6-27B) with vLLM offline batching, and checkpoints to an
append-only JSONL that mirrors bench/storage.py's fields, so scoring downstream is unchanged.
All wording comes from bench/prompts.py: the modes, the brief builder, and the self_refine
templates. This file is only the batch orchestration.

Design:
  - Qwen3.6-27B (dense, bf16) has about 54 GB of weights, more than one 48 GB GPU holds, so
    tensor_parallel_size=2 is the default. The dense model is used rather than the 35B-A3B
    MoE: MoE routing adds seed-to-seed metric variance, and the MoE hangs CUDA-graph capture on
    Ampere GPUs (vLLM issue #40124), which forces eager mode and removes its speed advantage.
  - Offline llm.chat(list) rather than the OpenAI-compatible server: continuous batching
    saturates the GPUs with no client-side concurrency to manage, and it applies the chat
    template, which a raw generate() would skip and which an instruct model requires.
  - enable_prefix_caching, with the shared prefix (VOICE, and for fewshot the exemplars) first
    and byte-identical, so the long shared prefix is prefilled once and reused across cells.
  - self_refine is batched across items in staged passes (all drafts, then all critiques, then
    all rewrites), so every pass keeps the batch full.

Thinking is on by default, for parity with the API models, which run at maximum reasoning; the
<think> block is stripped and only the article is scored. Set BENCH_THINKING=0 to disable.

Context overflow (check_prompts): exemplars are corpus-capped (120-500 words) and each stage's
output is capped by max_tokens, so the longest prompt stays around 3-4K tokens, well under
MAX_MODEL_LEN=16384. check_prompts asserts prompt + output <= the limit before any GPU time is
spent; vLLM itself errors rather than silently truncating an over-length prompt.

Checkpoint and resume:
  - Each cell is appended and fsync'd to the JSONL as its batch completes, so a crash loses at
    most CHUNK cells and a restart skips everything already written. CHUNK (default 50) is also
    the batch size: below the GPU's saturation concurrency the batch underfills and throughput
    drops.
  - With BENCH_HF_REPO set, a CommitScheduler pushes the output folder to that HF dataset every
    5 minutes on a background thread, as backup and final artifact.

Errors: each batch is wrapped; if the first batch fails with 0 successes the run aborts (likely
OOM or a config problem); a later batch error logs the traceback and context and is skipped
(its cells retry on the next run); empty outputs are warned about and counted; a final summary
reports made/errors/empty. vLLM's own log level is VLLM_LOGGING_LEVEL (default INFO here).

    python -m bench.gen_vllm --check-prompts   # tokenizer-only overflow preflight; no model load, no GPU
    python -m bench.gen_vllm --smoke           # 2 briefs end-to-end (+ HF push), then exit
    python -m bench.gen_vllm                    # full run (resumable)
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import re
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

from . import prompts

# Set before the (lazy, inside main) vllm imports: log level, and the spawn start method for the
# tensor-parallel worker processes.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# On SIGTERM (e.g. from an outer `timeout` wrapper killing a stuck run), dump every thread's
# stack first so the log names the hung line. faulthandler.register is Unix-only; the module
# still imports where it is absent.
if hasattr(faulthandler, "register"):
    faulthandler.register(signal.SIGTERM, all_threads=True)

MODEL = os.environ.get("BENCH_VLLM_MODEL", "Qwen/Qwen3.6-27B")
TP = int(os.environ.get("BENCH_TP", "2"))
MAX_MODEL_LEN = int(os.environ.get("BENCH_MAX_MODEL_LEN", "16384"))
GPU_MEM_UTIL = float(os.environ.get("BENCH_GPU_MEM_UTIL", "0.90"))  # capture needs headroom; alloc fails inside capture can deadlock
MAX_NUM_SEQS = int(os.environ.get("BENCH_MAX_NUM_SEQS", "256"))
N_SEEDS = int(os.environ.get("BENCH_ITERATIONS", "3"))
CHUNK = int(os.environ.get("BENCH_CHUNK", "50"))      # batch size, checkpoint and crash-loss window
TEMPERATURE = 0.7                                      # decoding held constant across modes (see prompts.py)
TOP_P = float(os.environ.get("BENCH_TOP_P", "0.95"))
THINKING = os.environ.get("BENCH_THINKING", "1") == "1"
# Per-stage output caps. With THINKING on, the cap is shared with the reasoning: Qwen3.6 spends
# 1-4k tokens reasoning before the article, so an undersized cap truncates mid-reasoning and no
# article appears (compare the _MAX_TOKENS note in config.py). Overridable per stage.
MAX_TOK = {"article": int(os.environ.get("BENCH_MAX_TOK_ARTICLE") or ("6144" if THINKING else "1024")),
           "critique": int(os.environ.get("BENCH_MAX_TOK_CRITIQUE") or ("2048" if THINKING else "512"))}
# enforce_eager disables CUDA graphs, which costs throughput. The dense 27B graph path works on
# Ampere (A40), so the default is off. The 35B-A3B MoE hangs CUDA-graph capture on Ampere (vLLM
# issue #40124: 100% utilization spin, init never returns; the BENCH_INIT_TIMEOUT_S watchdog
# catches it), so a hybrid-MoE model on Ampere needs BENCH_ENFORCE_EAGER=1.
ENFORCE_EAGER = os.environ.get("BENCH_ENFORCE_EAGER", "0") == "1"
# vLLM's custom all-reduce uses direct CUDA IPC peer-to-peer and ignores NCCL_P2P_DISABLE. On a
# virtualized PCIe GPU pair with broken P2P, the small decode all-reduces replayed inside CUDA
# graphs take its one-shot IPC path and spin at 100% GPU indefinitely, while the profile run
# passes because its large all-reduces exceed the custom kernel's size cap and go to NCCL over
# shared memory. Default: custom all-reduce disabled, so tensor-parallel collectives go through
# NCCL/SHM, at a small latency cost.
DISABLE_CUSTOM_AR = os.environ.get("BENCH_DISABLE_CUSTOM_AR", "1") == "1"
# Engine-init watchdog: LLM() can hang indefinitely before KV-cache allocation when NCCL's first
# collective hangs on broken GPU P2P (weights loaded, 0% GPU). The window covers weight load,
# profiling and graph capture with the model snapshot already downloaded; capture over PCIe/SHM
# can be slow, hence 1800 s. 0 disables. When it fires, all thread stacks are dumped
# (faulthandler) before exit.
INIT_TIMEOUT_S = int(os.environ.get("BENCH_INIT_TIMEOUT_S", "1800"))
OUT_DIR = Path(os.environ.get("BENCH_OUT_DIR", "/workspace/out"))
HF_REPO = os.environ.get("BENCH_HF_REPO")             # e.g. "burstiness-engine/qwen36-news-gen"; unset -> no HF mirror
MODEL_KEY = os.environ.get("BENCH_MODEL_KEY", "qwen3.6-27b")
SINGLE_CALL = ["control", "instruction", "fewshot", "plan"]
_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


def strip_think(text):
    """Drop the model's reasoning so only the article is scored. Handles both shapes: a tagged
    <think>...</think> block, and the Qwen3 chat-template shape where the opening tag is part
    of the prompt, so the output starts mid-reasoning and only the closing tag appears."""
    text = text or ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return _THINK.sub("", text).strip()


def visible(text):
    """The scoreable article from a raw completion, or None when thinking overran max_tokens
    without closing. A None cell is dropped (never written), so done_ids never contains it and
    the append-only resume regenerates it on the next run."""
    if THINKING and "</think>" not in (text or ""):
        return None
    return strip_think(text)


def cid(mode, brief_id, it):
    return f"{mode}__s{brief_id}__it{it}"


def single_prompt(mode, brief):
    """The one user prompt a single-call mode sends (and self_refine's draft). Reuses prompts.py wording."""
    return prompts.preview(mode, brief)


def check_prompts(tok, briefs):
    """Assert no prompt + its capped output exceeds MAX_MODEL_LEN. Tokenizer only; no GPU."""
    worst, where = 0, ""
    # The stubs bound the visible text a later stage embeds in its prompt (drafts and critiques
    # are stripped articles, instruction-bounded at about 300 words), not the generation cap:
    # with THINKING the cap is mostly reasoning that never reaches the next prompt, so using the
    # raw cap here would overflow spuriously. The +MAX_TOK output term below keeps the full cap,
    # which is what the engine can emit.
    stub_a, stub_c = min(MAX_TOK["article"], 1024), min(MAX_TOK["critique"], 512)
    draft_stub, crit_stub = "word " * stub_a, "word " * stub_c
    for b in briefs:
        for mode in SINGLE_CALL + ["self_refine"]:
            n = len(tok.encode(single_prompt(mode, b["brief"]))) + MAX_TOK["article"]
            if n > worst:
                worst, where = n, f"{mode} s{b['id']}"
        rewrite = prompts.REWRITE.replace("{draft}", draft_stub).replace("{feedback}", crit_stub)
        n = len(tok.encode(rewrite)) + MAX_TOK["article"]
        if n > worst:
            worst, where = n, f"self_refine-rewrite s{b['id']}"
    ok = worst <= MAX_MODEL_LEN
    print(f"worst-case prompt+out ~= {worst} tokens at [{where}]  (limit {MAX_MODEL_LEN})  ->  {'OK' if ok else 'OVERFLOW'}")
    if not ok:
        raise SystemExit("would overflow MAX_MODEL_LEN: raise it, shorten exemplars, or lower max_tokens")


def done_ids(path):
    if not path.exists():
        return set()
    return {json.loads(l)["cell_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def chat(llm, prompts_list, max_tokens, seed):
    """One batched chat completion. Applies the instruct chat template (a raw generate() would not)."""
    from vllm import SamplingParams  # lazy: keep this module importable (for the smoke) without vllm
    sp = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=max_tokens, seed=seed)
    msgs = [[{"role": "user", "content": p}] for p in prompts_list]
    return llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": THINKING}, use_tqdm=True)


def run_chunked(label, items, max_tokens, seed, sink, chat_fn, stats, chunk=CHUNK):
    """items: list of (payload, prompt). Generate in `chunk` batches and sink each output.

    Module-level (not nested in main) so the error paths (mid-run skip-and-retry and the
    first-batch fail-fast) can be exercised with a mock chat_fn. `stats` (made/errors/empty) is
    mutated in place.
    """
    nchunks = -(-len(items) // chunk) if items else 0
    for i in range(0, len(items), chunk):
        c = items[i:i + chunk]
        t0 = time.time()
        try:
            outs = chat_fn([p for _, p in c], max_tokens, seed)
        except Exception as e:
            stats["errors"] += 1
            print(f"\n!! ERROR {label} seed{seed} batch {i // chunk + 1}/{nchunks} "
                  f"(size {len(c)}): {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            if stats["made"] == 0:
                print("\n!! FAIL-FAST: first batch failed with 0 successes -- likely OOM/config. "
                      "Stopping.\n", flush=True)
                sys.exit(1)
            print("   skipping this batch; its cells retry on the next run.\n", flush=True)
            continue
        for (payload, _p), o in zip(c, outs):
            sink(payload, o.outputs[0].text)
        dt = time.time() - t0
        print(f"  {label} seed{seed} batch {i // chunk + 1}/{nchunks}: {len(c)} cells in "
              f"{dt:.1f}s ({len(c) / max(dt, 1e-6):.1f}/s) | made={stats['made']} "
              f"empty={stats['empty']} err={stats['errors']}", flush=True)


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-prompts", action="store_true", help="tokenizer-only overflow preflight; no GPU")
    ap.add_argument("--smoke", action="store_true", help="2 briefs end-to-end then exit")
    args = ap.parse_args()

    briefs = prompts.load_briefs()

    if args.check_prompts:  # no model load, no GPU
        from transformers import AutoTokenizer
        check_prompts(AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True), briefs)
        return 0

    n_seeds = N_SEEDS
    if args.smoke:
        briefs, n_seeds = briefs[:2], 1

    print(f"[config] model={MODEL} tp={TP} max_len={MAX_MODEL_LEN} gpu_util={GPU_MEM_UTIL} "
          f"max_num_seqs={MAX_NUM_SEQS} chunk={CHUNK} seeds={n_seeds} thinking={THINKING} "
          f"enforce_eager={ENFORCE_EAGER} init_timeout={INIT_TIMEOUT_S}s temp={TEMPERATURE} "
          f"top_p={TOP_P} briefs={len(briefs)} hf_repo={HF_REPO} out={OUT_DIR}", flush=True)

    from vllm import LLM, SamplingParams

    def _init_wedged():  # daemon Timer thread: only fires if LLM() overruns the watchdog
        import faulthandler
        print(f"\n!! FAIL-FAST: engine init exceeded BENCH_INIT_TIMEOUT_S={INIT_TIMEOUT_S}s -- wedged, "
              "not slow. If the log ends in NCCL init, GPU P2P is the usual culprit (set "
              "NCCL_P2P_DISABLE=1 and check `nvidia-smi topo -m`). If it ends in "
              "CUDA graph capture, retry with BENCH_ENFORCE_EAGER=1 (vLLM #40124). All thread "
              "stacks follow. os._exit because the wedge sits in non-Python threads that ignore "
              "SystemExit.", flush=True)
        faulthandler.dump_traceback(all_threads=True)
        os._exit(70)

    watchdog = threading.Timer(INIT_TIMEOUT_S, _init_wedged)
    watchdog.daemon = True
    if INIT_TIMEOUT_S > 0:
        watchdog.start()
    llm = LLM(model=MODEL, tensor_parallel_size=TP, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=GPU_MEM_UTIL, max_num_seqs=MAX_NUM_SEQS,
              enable_prefix_caching=True, enforce_eager=ENFORCE_EAGER,
              disable_custom_all_reduce=DISABLE_CUSTOM_AR, trust_remote_code=True)
    watchdog.cancel()
    check_prompts(llm.get_tokenizer(), briefs)  # hard gate before spending GPU time

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{MODEL_KEY}.jsonl"
    done = done_ids(out)
    print(f"[resume] {len(done)} cells already on disk -> skipping them", flush=True)

    sched = None
    if HF_REPO:
        from huggingface_hub import CommitScheduler
        sched = CommitScheduler(repo_id=HF_REPO, repo_type="dataset", folder_path=str(OUT_DIR),
                                path_in_repo="data", every=5)
    lock = sched.lock if sched else _NullLock()
    stats = {"made": 0, "errors": 0, "empty": 0, "truncated": 0}
    t_start = time.time()

    def chat_fn(prompts_list, max_tokens, seed):  # the module-level chat, bound to this run's llm
        return chat(llm, prompts_list, max_tokens, seed)

    def keep(d, key, text):  # refine intermediates: drop truncated thinking, never propagate it
        v = visible(text)
        if v is None:
            stats["truncated"] += 1
            print(f"   WARN truncated thinking (refine intermediate s{key}) -- dropped; resume retries the cell", flush=True)
        else:
            d[key] = v

    def record(cell, mode, bid, it, text):
        text = visible(text)
        if text is None:
            stats["truncated"] += 1
            print(f"   WARN truncated thinking {cell} -- cell NOT recorded; a resume rerun regenerates it", flush=True)
            return
        if not text.strip():
            stats["empty"] += 1
            print(f"   WARN empty output {cell} (refusal/truncation?) -- recorded for inspection", flush=True)
        rec = {"cell_id": cell, "model_key": MODEL_KEY, "eval_prompt_id": mode, "scenario_id": bid,
               "prompt_id": f"{mode}__s{bid}", "iteration": it, "text": text, "reasoning": None, "usage": None}
        with lock, out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        stats["made"] += 1

    for it in range(n_seeds):
        # single-call modes: one batched pass per seed
        jobs = [((m, b["id"]), single_prompt(m, b["brief"]))
                for b in briefs for m in SINGLE_CALL if cid(m, b["id"], it) not in done]
        run_chunked("single", jobs, MAX_TOK["article"], it,
                    lambda pl, text: record(cid(pl[0], pl[1], it), pl[0], pl[1], it, text),
                    chat_fn=chat_fn, stats=stats)

        # self_refine: staged (drafts -> critiques -> rewrites); intermediate stages held in memory
        todo = [b for b in briefs if cid("self_refine", b["id"], it) not in done]
        drafts, crits = {}, {}
        run_chunked("refine-draft", [((b["id"],), single_prompt("self_refine", b["brief"])) for b in todo],
                    MAX_TOK["article"], it, lambda pl, text: keep(drafts, pl[0], text),
                    chat_fn=chat_fn, stats=stats)
        run_chunked("refine-critique", [((bid,), prompts.FEEDBACK.replace("{draft}", d)) for bid, d in drafts.items()],
                    MAX_TOK["critique"], it, lambda pl, text: keep(crits, pl[0], text),
                    chat_fn=chat_fn, stats=stats)
        # only rewrite items whose draft AND critique both survived (partial-failure safe)
        rw = [((bid,), prompts.REWRITE.replace("{draft}", drafts[bid]).replace("{feedback}", crits[bid]))
              for bid in drafts if bid in crits]
        run_chunked("refine-rewrite", rw, MAX_TOK["article"], it,
                    lambda pl, text: record(cid("self_refine", pl[0], it), "self_refine", pl[0], it, text),
                    chat_fn=chat_fn, stats=stats)

    if sched:
        sched.trigger().result()  # final flush to HF
    total = len(done_ids(out))
    print(f"\n== summary == wrote {stats['made']} cells this run ({stats['empty']} empty, "
          f"{stats['truncated']} truncated-thinking dropped, {stats['errors']} batch errors) "
          f"in {time.time() - t_start:.0f}s | {total} total on disk -> {out}", flush=True)
    if stats["truncated"]:
        print("NOTE: truncated-thinking cells were dropped, not recorded -- re-run to regenerate them "
              "(resume re-queues exactly those), or raise BENCH_MAX_TOK_ARTICLE.", flush=True)
    if stats["errors"]:
        print("NOTE: some batches errored and were skipped -- re-run to fill them (resume is idempotent).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
