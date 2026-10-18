"""Batch-mode leg runner for Anthropic models (Message Batches API, half the list price).

The live runner (bench/run.py) pays the full per-token price and holds a worker
pool open for the whole leg. The Message Batches API bills every token at half
price, so this runner submits the leg as batches. self_refine (draft -> critique
-> rewrite) needs three sequential calls per cell, so the leg goes out as three
waves, each a single batch, with intermediate drafts and critiques carried in a
state file between waves.

  Wave A: control / instruction / fewshot / plan cells + self_refine drafts
  Wave B: self_refine critiques (built from wave A drafts)
  Wave C: self_refine rewrites  (recorded as the final self_refine cells)

Output schema is bench/storage.append, byte-compatible with the live runner, so
hygiene, normalization and scoring work unchanged. Resumable at every stage:
completed cells are skipped, submitted batch ids are persisted in
<results>/<key>.batchstate.json and re-polled instead of resubmitted.

Refusals (stop_reason == "refusal") are recorded as empty cells with
usage.refusal = "provider_content_filter".

  python3 -m bench.run_batch --model opus-4.8 [--effort high] [--limit N]
                             [--poll 120] [--dry-run]

Cost preview with --dry-run (counts requests, estimates spend, submits nothing).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bench import prompts, storage
from bench.config import MODELS

MODES_SINGLE = ["control", "instruction", "fewshot", "plan"]
SEEDS = 3

# Published list prices, USD per Mtok, as (input, output), taken from the provider's pricing
# page. A model with no published list price gets None, and batch_price refuses to cost it.
#
#   claude-opus-4-8  anthropic.com/pricing
#   claude-fable-5   no published list price
LIST_PRICE_USD: dict[str, tuple[float, float] | None] = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-fable-5": None,
}
# The Message Batches API bills at half the list rate; batch prices are derived from the list
# prices rather than stated separately.
BATCH_DISCOUNT = 0.5


def batch_price(model_id: str) -> tuple[float, float]:
    """Batched (input, output) USD per Mtok. Refuses rather than guessing."""
    if model_id not in LIST_PRICE_USD:
        sys.exit(f"{model_id} has no entry in LIST_PRICE_USD. Add it with a verified source "
                 f"before running or costing this leg.")
    listed = LIST_PRICE_USD[model_id]
    if listed is None:
        sys.exit(f"{model_id} has no published list price, so this leg cannot be costed "
                 f"honestly. Verify the price, add it to LIST_PRICE_USD, then re-run.")
    return listed[0] * BATCH_DISCOUNT, listed[1] * BATCH_DISCOUNT


# Mean tokens per recorded cell, measured on opus-4.8 5-brief smoke runs. Keyed by effort and
# split by mode class, because a self_refine cell sums three chained calls. At max, a
# single-shot cell emits about 1,700 output tokens and a self_refine cell about 38,500; at
# high, 582 and 2,416. Effort, not model choice, is the cost driver, and the self_refine chain
# is where it compounds. An effort that has not been measured is bracketed between the two
# that have (see estimate_leg_cost).
#
#                    single-shot cell      self_refine cell (draft+critique+rewrite, summed)
PER_CELL_TOKENS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "high": ((806, 582), (2559, 2416)),
    "max":  ((806, 1701), (2840, 38533)),   # the rewrite runs away; 60% of cells truncate even at 32k
}
SELF_REFINE_SHARE = 0.2   # 1 of the 5 modes


def estimate_leg_cost(n_cells: int, effort: str, pin: float, pout: float) -> tuple[float, float]:
    """(low, high) USD for a batched leg. Equal bounds when the effort has been measured."""
    if effort not in PER_CELL_TOKENS:
        # Bracket between the two measured efforts.
        lo, _ = estimate_leg_cost(n_cells, "high", pin, pout)
        _, hi = estimate_leg_cost(n_cells, "max", pin, pout)
        return lo, hi
    (s_in, s_out), (r_in, r_out) = PER_CELL_TOKENS[effort]
    n_refine = n_cells * SELF_REFINE_SHARE
    n_single = n_cells - n_refine
    tin = (n_single * s_in + n_refine * r_in) / 1e6
    tout = (n_single * s_out + n_refine * r_out) / 1e6
    cost = tin * pin + tout * pout
    return cost, cost


def _prompt_for(mode: str, brief: str) -> str:
    if mode == "control":
        return brief
    if mode == "instruction":
        return prompts._wrap("", brief)
    if mode == "fewshot":
        return prompts._wrap(prompts.FEWSHOT_LEAD + prompts.EXEMPLARS + prompts.FEWSHOT_TAIL, brief)
    if mode == "plan":
        return prompts._wrap(prompts.PLAN, brief)
    if mode == "draft":  # self_refine stage 1: identical to instruction, on purpose
        return prompts._wrap("", brief)
    raise ValueError(mode)


def _params(model: dict, user: str, effort: str) -> dict:
    return {
        "model": model["model_id"],
        "max_tokens": model["max_tokens"],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort},
        "messages": [{"role": "user", "content": user}],
    }


def _extract(message) -> tuple[str, dict]:
    text = "".join(b.text for b in message.content if b.type == "text")
    usage = {"input_tokens": message.usage.input_tokens,
             "output_tokens": message.usage.output_tokens}
    return text, usage


def _classify(msg, text: str, usage: dict) -> tuple[str, dict]:
    """Mark why a cell is empty.

    Two ways a cell comes back with no prose, with opposite meanings:

      refusal    the provider declined the content. A datum: the leg records it and the
                 scorer's seed-averaging excludes it.
      max_tokens the model spent its whole budget thinking and never emitted a text block.
                 Not a datum: for Anthropic models thinking tokens are billed against
                 max_tokens, so this is a budget problem, and an unmarked empty cell would
                 read downstream as a refusal that never happened.
    """
    if msg.stop_reason == "refusal":
        return "", {**usage, "refusal": "provider_content_filter"}
    if msg.stop_reason == "max_tokens" and not text.strip():
        return "", {**usage, "truncated": "max_tokens_before_text"}
    return text, usage


def _submit(client, requests, label):
    print(f"[{label}] submitting {len(requests)} requests")
    batch = client.messages.batches.create(requests=requests)
    print(f"[{label}] batch {batch.id} ({batch.processing_status})")
    return batch.id


def _await_batch(client, batch_id, poll_s):
    while True:
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            print(f"[poll] {batch_id} ended: {b.request_counts}")
            return
        print(f"[poll] {batch_id}: {b.processing_status} "
              f"(done {b.request_counts.succeeded + b.request_counts.errored}"
              f"/{b.request_counts.processing + b.request_counts.succeeded + b.request_counts.errored})")
        time.sleep(poll_s)


def _results(client, batch_id):
    out = {}
    for r in client.messages.batches.results(batch_id):
        out[r.custom_id] = r.result
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="opus-4.8")
    ap.add_argument("--effort", default=None, help="override the registry effort (recommend: high)")
    ap.add_argument("--limit", type=int, default=0, help="brief count cap (smoke)")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    model = next((m for m in MODELS if m["key"] == a.model), None)
    if not model or model.get("provider") != "anthropic":
        sys.exit(f"--model must be a registered anthropic leg, got {a.model}")
    effort = a.effort or model.get("effort", "high")

    briefs = prompts.load_briefs()
    if a.limit:
        briefs = briefs[: a.limit]
    brief_by_id = {b["id"]: b["brief"] for b in briefs}

    results_f = storage.RESULTS_DIR / f"{a.model}.jsonl"
    # Config purity preflight: cells written before this batch run (e.g. by the live runner at
    # the registry's effort) would mix generation configs inside one leg, so any pre-existing
    # results file is archived unless a batchstate already exists (then the prior cells are
    # this runner's own earlier waves).
    state_probe = storage.RESULTS_DIR / f"{a.model}.batchstate.json"
    # A --dry-run is a cost preview and touches nothing on disk, so it only reports what a
    # real run would archive.
    stale = results_f.exists() and not state_probe.exists()
    if stale:
        import datetime
        bak = results_f.with_suffix(f".pre-batch-{datetime.date.today()}.jsonl")
        if a.dry_run:
            print(f"preflight (dry-run, nothing written): a real run would archive "
                  f"{results_f.name} -> {bak.name} (config purity)")
        else:
            results_f.rename(bak)
            print(f"preflight: archived {results_f.name} -> {bak.name} (config purity; "
                  f"cells there were generated outside this batch config)")

    done = set()
    # Once the stale file is archived (really, or hypothetically under --dry-run) the leg
    # starts empty. Only count existing cells when they are our own earlier waves, so the
    # dry-run estimate matches what a real run would actually cost.
    if results_f.exists() and not stale:
        for line in results_f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["eval_prompt_id"], r["scenario_id"], r["iteration"]))

    state_f = storage.RESULTS_DIR / f"{a.model}.batchstate.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {
        "wave_a": None, "wave_b": None, "wave_c": None, "drafts": {}, "critiques": {}}

    def save_state():
        storage.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        state_f.write_text(json.dumps(state))

    # ---- build wave A ----
    wave_a_reqs = []
    for b in briefs:
        for it in range(SEEDS):
            for mode in MODES_SINGLE:
                if (mode, b["id"], it) not in done:
                    wave_a_reqs.append({
                        "custom_id": f"{mode}__{b['id']}__{it}",
                        "params": _params(model, _prompt_for(mode, b["brief"]), effort)})
            key = f"self_refine__{b['id']}__{it}"
            if ("self_refine", b["id"], it) not in done and key not in state["drafts"]:
                wave_a_reqs.append({
                    "custom_id": f"draft__{b['id']}__{it}",
                    "params": _params(model, _prompt_for("draft", b["brief"]), effort)})

    n_cells = len(briefs) * SEEDS * 5
    print(f"{a.model} @ effort={effort}: {n_cells} cells total, {len(done)} done, "
          f"wave A = {len(wave_a_reqs)} requests")
    if a.dry_run:
        pin, pout = batch_price(model["model_id"])
        lo, hi = estimate_leg_cost(n_cells, effort, pin, pout)
        band = f"~${lo:.0f}" if lo == hi else f"~${lo:.0f}-{hi:.0f}"
        print(f"dry-run: batched estimate {band} "
              f"({'from measured per-cell token counts' if effort in PER_CELL_TOKENS else 'unmeasured effort, wide band'})")
        return

    import anthropic
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env

    # ---- wave A ----
    # A wave sits in one of three states: None (never submitted), a batch id (in flight),
    # or "consumed" (its results are recorded). A wave is submitted when there is pending
    # work and no batch is in flight, which includes the "consumed" case: that is how a
    # smoke run (--limit N) expands into the full leg without regenerating the cells it
    # already has.
    if wave_a_reqs and state["wave_a"] in (None, "consumed"):
        state["wave_a"] = _submit(client, wave_a_reqs, "wave A")
        save_state()
    if state["wave_a"] and state["wave_a"] != "consumed":
        _await_batch(client, state["wave_a"], a.poll)
        for cid, res in _results(client, state["wave_a"]).items():
            stage, sid, it = cid.rsplit("__", 2)
            sid, it = int(sid), int(it)
            if res.type != "succeeded":
                print(f"  !! {cid}: {res.type} (re-queued next run)")
                continue
            msg = res.message
            text, usage = _extract(msg)
            text, usage = _classify(msg, text, usage)
            if stage == "draft":
                state["drafts"][f"self_refine__{sid}__{it}"] = {"text": text, "usage": usage}
            else:
                if (stage, sid, it) not in done:
                    storage.append(a.model, eval_prompt_id=stage, scenario_id=sid,
                                   prompt_id=f"{stage}__s{sid}", iteration=it,
                                   text=text, reasoning=None, usage=usage)
                    done.add((stage, sid, it))
        state["wave_a"] = "consumed"
        save_state()

    # ---- wave B: critiques ----
    wave_b_reqs = []
    for key, d in state["drafts"].items():
        if key not in state["critiques"] and d["text"].strip():
            _, sid, it = key.rsplit("__", 2)
            wave_b_reqs.append({
                "custom_id": f"critique__{sid}__{it}",
                "params": _params(model, prompts.FEEDBACK.replace("{draft}", d["text"]), effort)})
    if wave_b_reqs and state["wave_b"] in (None, "consumed"):
        state["wave_b"] = _submit(client, wave_b_reqs, "wave B")
        save_state()
    if state["wave_b"] and state["wave_b"] != "consumed":
        _await_batch(client, state["wave_b"], a.poll)
        for cid, res in _results(client, state["wave_b"]).items():
            _, sid, it = cid.rsplit("__", 2)
            if res.type == "succeeded":
                text, usage = _extract(res.message)
                state["critiques"][f"self_refine__{sid}__{it}"] = {"text": text, "usage": usage}
        state["wave_b"] = "consumed"
        save_state()

    # ---- wave C: rewrites (the recorded self_refine cells) ----
    wave_c_reqs = []
    for key, c in state["critiques"].items():
        _, sid, it = key.rsplit("__", 2)
        if ("self_refine", int(sid), int(it)) in done:
            continue
        d = state["drafts"][key]
        user = prompts.REWRITE.replace("{draft}", d["text"]).replace("{feedback}", c["text"])
        wave_c_reqs.append({"custom_id": f"self_refine__{sid}__{it}",
                            "params": _params(model, user, effort)})
    if wave_c_reqs and state["wave_c"] in (None, "consumed"):
        state["wave_c"] = _submit(client, wave_c_reqs, "wave C")
        save_state()
    if state["wave_c"] and state["wave_c"] != "consumed":
        _await_batch(client, state["wave_c"], a.poll)
        for cid, res in _results(client, state["wave_c"]).items():
            stage, sid, it = cid.rsplit("__", 2)
            sid, it = int(sid), int(it)
            if res.type != "succeeded":
                print(f"  !! {cid}: {res.type} (re-queued next run)")
                continue
            msg = res.message
            text, usage = _extract(msg)
            key = f"self_refine__{sid}__{it}"
            # sum the chain's usage into the recorded cell, like the live runner
            for stage_u in (state["drafts"].get(key), state["critiques"].get(key)):
                if stage_u:
                    usage["input_tokens"] += stage_u["usage"].get("input_tokens", 0)
                    usage["output_tokens"] += stage_u["usage"].get("output_tokens", 0)
            text, usage = _classify(msg, text, usage)
            if ("self_refine", sid, it) not in done:
                storage.append(a.model, eval_prompt_id="self_refine", scenario_id=sid,
                               prompt_id=f"self_refine__s{sid}", iteration=it,
                               text=text, reasoning=None, usage=usage)
                done.add(("self_refine", sid, it))
        state["wave_c"] = "consumed"
        save_state()

    # also record self_refine cells whose draft was refused outright
    for key, d in state["drafts"].items():
        _, sid, it = key.rsplit("__", 2)
        sid, it = int(sid), int(it)
        if ("self_refine", sid, it) not in done and not d["text"].strip():
            storage.append(a.model, eval_prompt_id="self_refine", scenario_id=sid,
                           prompt_id=f"self_refine__s{sid}", iteration=it,
                           text="", reasoning=None, usage=d["usage"])
            done.add(("self_refine", sid, it))

    total = sum(1 for line in results_f.read_text().splitlines() if line.strip()) if results_f.exists() else 0
    pin = pout = 0
    truncated_by_mode: dict[str, int] = {}
    cells_by_mode: dict[str, int] = {}
    if results_f.exists():
        for line in results_f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            u = r.get("usage") or {}
            pin += u.get("input_tokens", 0) or 0
            pout += u.get("output_tokens", 0) or 0
            mode = r.get("eval_prompt_id", "?")
            cells_by_mode[mode] = cells_by_mode.get(mode, 0) + 1
            if u.get("truncated"):
                truncated_by_mode[mode] = truncated_by_mode.get(mode, 0) + 1
    pi, po = batch_price(model["model_id"])
    print(f"== {a.model}: {total}/{n_cells} cells recorded; "
          f"{pin/1e6:.2f}M in, {pout/1e6:.2f}M out ~= ${pin/1e6*pi + pout/1e6*po:.2f} batched")

    # A leg is not complete just because every cell has a row. Truncated cells carry no prose
    # and are not data; left unflagged they would enter the scorer as empties, be seed-averaged
    # away, and the leg would report a mode it never measured. The run therefore exits non-zero.
    if truncated_by_mode:
        print()
        for mode, n_trunc in sorted(truncated_by_mode.items()):
            pct = 100.0 * n_trunc / max(cells_by_mode.get(mode, 1), 1)
            print(f"!! {mode}: {n_trunc}/{cells_by_mode.get(mode)} cells ({pct:.0f}%) hit max_tokens "
                  f"before emitting any text")
        sys.exit(
            f"\nLEG NOT USABLE: {sum(truncated_by_mode.values())} cells were truncated, not refused.\n"
            f"The model spent its whole budget thinking. Raise max_tokens for {a.model} in\n"
            f"bench/config.py (thinking is billed against it on Anthropic), purge the truncated\n"
            f"cells, and re-run. Do NOT score this leg: the affected mode was never measured.")


if __name__ == "__main__":
    main()
