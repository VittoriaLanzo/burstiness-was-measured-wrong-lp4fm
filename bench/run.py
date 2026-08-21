"""Driver. Generates every cell of the (mode x model x brief x iteration) array and
checkpoints each output, so a long paid run is resumable and loss-proof.

A "mode" is a prompting condition (bench/prompts.py MODES): a function (gen, brief) -> final
article text, where gen(user) is one model completion. Most modes make one call; self_refine
makes three (draft, critique, rewrite). The 200 briefs are the news headlines from the frozen
HF dataset (prompts.load_briefs).

Failure handling is loud, not happy-path-only:
  - --check runs a no-API preflight (SDK importable? key set?) per leg.
  - A leg with a missing SDK/key is clearly SKIPPED (distinct from failure).
  - A cell that errors prints the full traceback + leg context; the cell is not recorded
    (so it retries next run).
  - If a leg errors FAIL_FAST times before any success, it is declared broken and stopped.
  - An empty output is WARNed (refusal/truncation) but still recorded for inspection.
  - The end-of-run summary distinguishes ok / ok-with-errors / FAILED / skipped, and the
    process exits non-zero if any leg failed.

    python -m bench.run --check                # preflight, no API calls, no cost
    python -m bench.run --dry-run              # plan + first rendered cell, no API calls
    python -m bench.run --smoke                # one real cell per ready leg (needs --force while draft)
    python -m bench.run --force                # real run (prompts not finalized -> needs --force)
"""
from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config, prompts, providers, storage

FAIL_FAST = 3  # errors before any success => the leg is broken; stop it


def cells(briefs):
    """Yield (mode, brief_id, brief_text, iteration) for the whole array."""
    for mode in prompts.MODES:
        for b in briefs:
            for it in range(config.N_ITERATIONS):
                yield mode, b["id"], b["brief"], it


def _make_gen(m):
    """A single-call generator bound to model m, plus the usage list it accumulates into.

    Modes call gen(user) once (or, for self_refine, three times); usages collects each call's
    token counts so the driver can store the true multi-call total.
    """
    usages = []

    def gen(user):
        text, _reasoning, usage = providers.generate(m, "", user)  # system is empty: the voice lives in the user turn
        usages.append(usage)
        return text

    return gen, usages


def _sum_usage(usages):
    us = [u for u in usages if u]
    if not us:
        return None
    return {"input_tokens": sum(u.get("input_tokens") or 0 for u in us),
            "output_tokens": sum(u.get("output_tokens") or 0 for u in us),
            "calls": len(us)}


def check(models):
    print("== preflight (no API calls) ==")
    ready = 0
    for m in models:
        if not m.get("enabled", True):
            print(f"  {m['key']:<16} DISABLED")
            continue
        problems = providers.preflight(m)
        if problems:
            print(f"  {m['key']:<16} NOT READY")
            for p in problems:
                print(f"        - {p}")
        else:
            ready += 1
            print(f"  {m['key']:<16} READY      ({m['provider']}, {m['model_id']})")
    print(f"\nready legs: {ready}/{sum(1 for m in models if m.get('enabled', True))}"
          f"   |   prompts finalized: {prompts.PROMPTS_FINALIZED}")
    if not prompts.PROMPTS_FINALIZED:
        print("note: real runs are blocked until prompts are finalized (or you pass --force).")


def _guard_prompts(force):
    if prompts.PROMPTS_FINALIZED or force:
        return True
    print("=" * 72)
    print("REFUSING TO RUN: prompts are not finalized (prompts.PROMPTS_FINALIZED is False).")
    print("This is shared work-in-progress. Finalize the wording and set")
    print("PROMPTS_FINALIZED = True in bench/prompts.py, or pass --force to override.")
    print("(--dry-run and --check never require this.)")
    print("=" * 72)
    return False


def _run_cell(m, mode, brief):
    """Worker body: generate one cell. Returns (text, usages); exceptions propagate."""
    gen, usages = _make_gen(m)
    return prompts.MODES[mode](gen, brief), usages


def run(models, limit=None, smoke=False, force=False, workers=1):
    if not _guard_prompts(force):
        sys.exit(2)
    briefs = prompts.load_briefs()
    if limit:
        briefs = briefs[:limit]
    summary = []  # (key, status, made, errors, empties)

    for m in models:
        if not m.get("enabled", True):
            summary.append((m["key"], "disabled", 0, 0, 0))
            continue
        problems = providers.preflight(m)
        if problems:
            print(f"SKIP {m['key']}: " + "; ".join(problems))
            summary.append((m["key"], "skipped", 0, 0, 0))
            continue

        done = storage.done_cells(m["key"])
        pending = [(mode, bid, brief, it) for mode, bid, brief, it in cells(briefs)
                   if (f"{mode}__s{bid}", it) not in done]
        if smoke:
            pending = pending[:1]
        made = errors = empties = 0
        broken = False

        # Workers only generate; this (main) thread stays the sole storage writer,
        # so the append-only checkpoint file keeps its crash-safety guarantees.
        pool = ThreadPoolExecutor(max_workers=max(1, workers))
        try:
            futs = {pool.submit(_run_cell, m, mode, brief): (mode, bid, it)
                    for mode, bid, brief, it in pending}
            for fut in as_completed(futs):
                mode, bid, it = futs[fut]
                prompt_id = f"{mode}__s{bid}"
                try:
                    text, usages = fut.result()
                except Exception as e:
                    # Provider-side content moderation (e.g. Moonshot 400 content_filter on
                    # violent-news headlines) is a REFUSAL, not a harness fault: record the
                    # cell empty (the scorer's protocol already excludes empties, and the
                    # refusal rate is itself reportable) instead of retry-looping into
                    # fail-fast on a block of filtered headlines.
                    if "content_filter" in str(e) or "high risk" in str(e):
                        empties += 1
                        print(f"   WARN {m['key']} {mode} s{bid} it{it}: provider content filter "
                              f"refused the prompt -- recorded EMPTY (protocol exclusion).")
                        storage.append(
                            m["key"], eval_prompt_id=mode, scenario_id=bid, prompt_id=prompt_id,
                            iteration=it, text="", reasoning=None,
                            usage={"refusal": "provider_content_filter"},
                        )
                        continue
                    errors += 1
                    print(f"\n!! ERROR {m['key']} [{m['provider']}/{m['model_id']}] {mode} s{bid} it{it}")
                    print(f"   {type(e).__name__}: {e}")
                    traceback.print_exc()
                    if made == 0 and errors >= FAIL_FAST:
                        print(f"\n!! FAIL-FAST {m['key']}: {errors} errors and 0 successes -- the "
                              f"leg looks broken (key/model-id/SDK/endpoint). Stopping it.\n")
                        broken = True
                        break
                    continue  # not recorded -> retried next run
                if not (text and text.strip()):
                    empties += 1
                    print(f"   WARN {m['key']} {mode} s{bid} it{it}: EMPTY output "
                          f"(refusal/truncation?) -- recorded, inspect it.")
                storage.append(
                    m["key"], eval_prompt_id=mode, scenario_id=bid, prompt_id=prompt_id,
                    iteration=it, text=text, reasoning=None, usage=_sum_usage(usages),
                )
                made += 1
                print(f"  ok {m['key']} {mode} s{bid} it{it} ({len(text)} chars, {len(usages)} calls)")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        status = "FAILED" if broken else ("ok" if errors == 0 else "ok-with-errors")
        summary.append((m["key"], status, made, errors, empties))

    print("\n== run summary ==")
    failed = False
    for key, status, made, errs, emp in summary:
        extra = (f", {errs} errors" if errs else "") + (f", {emp} empty" if emp else "")
        print(f"  {key:<16} {status:<16} {made} cells{extra}")
        if status in ("FAILED", "ok-with-errors"):
            failed = True
    if failed:
        print("\nOne or more legs failed or errored -- exit 1 (this is NOT a clean run).")
        sys.exit(1)


def dry_run(models, limit=None):
    briefs = prompts.load_briefs()
    if limit:
        briefs = briefs[:limit]
    per_model = len(prompts.MODES) * len(briefs) * config.N_ITERATIONS
    enabled = [m["key"] for m in models if m.get("enabled", True)]
    print(f"briefs={len(briefs)}  modes={list(prompts.MODES)}  iterations={config.N_ITERATIONS}")
    print(f"cells per model = {per_model}  |  enabled = {enabled}  |  total = {per_model * len(enabled)}")
    print(f"prompts finalized: {prompts.PROMPTS_FINALIZED}  (real runs blocked until True or --force)")
    if briefs:
        print("\n--- example cell (mode=instruction, brief 0) ---")
        print(prompts.preview("instruction", briefs[0]["brief"])[:1200])


def main():
    ap = argparse.ArgumentParser(description="Burstiness frontier-model bench")
    ap.add_argument("--models", help="comma-separated model keys (default: all enabled)")
    ap.add_argument("--limit", type=int, help="limit number of briefs (for a quick run)")
    ap.add_argument("--iterations", type=int, help="override N_ITERATIONS")
    ap.add_argument("--smoke", action="store_true", help="one real cell per ready leg")
    ap.add_argument("--dry-run", action="store_true", help="plan only; no API calls")
    ap.add_argument("--check", action="store_true", help="preflight each leg (SDK+key); no API calls")
    ap.add_argument("--force", action="store_true", help="run even though prompts aren't finalized")
    ap.add_argument("--all-legs", action="store_true",
                    help="explicitly allow running every enabled leg (spends on all providers)")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent cells per leg (default 1 = sequential; API legs tolerate 8-16)")
    a = ap.parse_args()

    if a.iterations:
        config.N_ITERATIONS = a.iterations
    models = config.MODELS
    if a.models:
        keys = set(a.models.split(","))
        models = [m for m in models if m["key"] in keys]

    if a.check:
        check(models)
    elif a.dry_run:
        dry_run(models, a.limit)
    else:
        if not a.models and not a.all_legs:
            enabled = [m["key"] for m in models if m.get("enabled", True)]
            print("=" * 72)
            print(f"REFUSING TO RUN ALL {len(enabled)} LEGS: no --models filter was given.")
            print(f"This would spend on EVERY ready provider: {', '.join(enabled)}.")
            print("A pasted command that line-breaks before --models silently becomes this")
            print("(2026-07-10 near-miss). Name legs with --models key1,key2 or pass")
            print("--all-legs if the full-registry run is genuinely intended.")
            print("=" * 72)
            sys.exit(2)
        run(models, limit=a.limit, smoke=a.smoke, force=a.force, workers=a.workers)


if __name__ == "__main__":
    main()
