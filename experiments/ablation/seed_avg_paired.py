#!/usr/bin/env python3
"""Multi-seed paired (matched-headline) test: bench generations vs the human corpus.

Why this exists: the bench (bench/gen_vllm.py) generates N_SEEDS completions per
(mode, headline). Pooling every seed as its own pair would reuse each human document
N times (pseudo-replication: Wilcoxon assumes independent pairs, p-values come out
overconfident). Per the G1 lesson (var(L) is noisy; always multi-seed), the sound
design is: average each headline's metric vector ACROSS seeds first, then run one
paired test on the n independent headline pairs. A per-seed rank-biserial consistency
matrix for B_n is printed alongside, so seed-to-seed variation stays visible.

Reuses metrics() and wilcoxon() from paired_test.py unchanged. Model-free, stdlib
plus the repo. Bootstrap is seeded, so reruns reproduce the committed JSON exactly.

    python -m experiments.ablation.seed_avg_paired HUMAN.jsonl GEN.jsonl [label]

HUMAN.jsonl rows: {id, human_text, ...}   (the HF paper-dataset prompts_200_human.jsonl)
GEN.jsonl   rows: {eval_prompt_id, scenario_id, iteration, text, ...}  (bench output)
Writes seed_avg_paired_<label>.json next to GEN.jsonl.
"""
from __future__ import annotations

import collections
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path for `experiments`
from experiments.ablation.paired_test import metrics, wilcoxon

random.seed(0)
DIMS = ["B_n", "M", "gini", "seg_entropy"]


def main():
    human_path, gen_path = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else "run"

    human = {str(r["id"]): r["human_text"]
             for r in (json.loads(l) for l in open(human_path, encoding="utf-8") if l.strip())}
    rows = [json.loads(l) for l in open(gen_path, encoding="utf-8") if l.strip()]

    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        if r["text"].strip():
            by[r["eval_prompt_id"]][str(r["scenario_id"])][r["iteration"]] = r["text"]
    modes = sorted(by)

    hm = {sid: metrics(t) for sid, t in human.items()}
    out = {"label": label, "dims": {}, "consistency_B_n_rank_biserial": {},
           "note": "Per headline: metric vector averaged across available seeds, then ONE paired "
                   "Wilcoxon on independent headline pairs (pooling seeds as pairs would "
                   "pseudo-replicate the human docs). delta = human - model."}

    for dim in DIMS:
        out["dims"][dim] = {}
        for mode in modes:
            diffs = []
            for sid, its in sorted(by[mode].items()):
                h = hm.get(sid)
                if not h:
                    continue
                ams = [m for m in (metrics(t) for t in its.values()) if m]
                if not ams:
                    continue
                diffs.append(h[dim] - st.fmean(a[dim] for a in ams))
            z, p, rb = wilcoxon(diffs)
            dz = st.fmean(diffs) / (st.pstdev(diffs) or 1e-9)
            meds = sorted(st.median(random.choice(diffs) for _ in diffs) for _ in range(2000))
            out["dims"][dim][mode] = {
                "n_pairs": len(diffs), "median_delta_h_minus_ai": round(st.median(diffs), 3),
                "median_delta_95ci": [round(meds[50], 3), round(meds[1949], 3)],
                "wilcoxon_z": z, "wilcoxon_p": p, "matched_rank_biserial": rb,
                "paired_dz": round(dz, 3)}
            print(f"{dim:<12}{mode:<13}n={len(diffs):<5}median_d={st.median(diffs):>7.3f}  "
                  f"rb={rb:>7}  p={p:<9}dz={round(dz, 3)}", flush=True)

    for mode in modes:
        rbs = {}
        its_present = sorted({it for sids in by[mode].values() for it in sids})
        for it in its_present:
            diffs = []
            for sid, its in sorted(by[mode].items()):
                h, t = hm.get(sid), its.get(it)
                a = metrics(t) if (h and t) else None
                if h and a:
                    diffs.append(h["B_n"] - a["B_n"])
            _, _, rb = wilcoxon(diffs)
            rbs[f"it{it}"] = rb
        out["consistency_B_n_rank_biserial"][mode] = rbs
        print(f"consistency B_n {mode:<13}{rbs}", flush=True)

    dest = Path(gen_path).parent / f"seed_avg_paired_{label}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
