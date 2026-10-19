#!/usr/bin/env python3
"""Paired (matched-headline) human-vs-AI test on burstiness_pairs.jsonl.

Per dimension: paired delta = metric(human) - metric(ai); Wilcoxon signed-rank
(normal approximation, tie-corrected), sign test, paired dz, median delta with a
seeded 2,000-resample bootstrap CI. The matched design (same headline/era/register,
only authorship varies) is the most powerful version of the ablation. Model-free;
M is bias-corrected by adding 1/(n-1).

metrics() and wilcoxon() are also the scorer used by seed_avg_paired.py and
review_analyses.py. The corpus is the private HF dataset burstiness-engine/burstiness-engine-ablation;
hf-datasets.zip carries it as ablation-corpus/burstiness_pairs.jsonl. Point at it with
BURST_PAIRS=/path/to/burstiness_pairs.jsonl or place it in ./data/.
Writes paired_test_results.json next to the corpus.
"""
from __future__ import annotations
import json, math, os, random, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path for `experiments`
from experiments.burstiness_metrics import sentence_lengths
from experiments.burstiness_pp import burstiness_B_corrected, gini, memory_M_corrected, segmentation_entropy

random.seed(0)


def _corpus_path():
    """Resolve the pairs corpus lazily so metrics()/wilcoxon() stay importable by other
    harnesses (seed_avg_paired.py) without requiring the corpus to exist."""
    p = os.environ.get("BURST_PAIRS") or str(Path(__file__).resolve().parent / "data" / "burstiness_pairs.jsonl")
    if not Path(p).exists():
        raise SystemExit(
            f"paired corpus not found at {p}. Download burstiness_pairs.jsonl from the HF dataset "
            "burstiness-engine/burstiness-engine-ablation into experiments/ablation/data/ (or set BURST_PAIRS)."
        )
    return p


def m_corr(L):
    return memory_M_corrected(L)


def metrics(text):
    L = [float(x) for x in sentence_lengths(text)]
    if len(L) < 5:
        return None
    bn, m = burstiness_B_corrected(L), m_corr(L)
    if bn is None or m is None:
        return None
    return {"B_n": bn, "M": m, "gini": gini(L), "seg_entropy": segmentation_entropy(text)}


def wilcoxon(diffs):
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 6:
        return float("nan"), float("nan"), float("nan")
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    Wp = sum(ranks[i] for i in range(n) if d[i] > 0)
    Wm = sum(ranks[i] for i in range(n) if d[i] < 0)
    mu = n * (n + 1) / 4.0
    sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (Wp - mu) / sig if sig > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))  # two-sided
    rb = (Wp - Wm) / (Wp + Wm) if (Wp + Wm) > 0 else 0.0  # matched-pairs rank-biserial
    return round(z, 2), round(p, 5), round(rb, 3)


def main():
    path = _corpus_path()
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    dims = ["B_n", "M", "gini", "seg_entropy"]
    out = []
    pair_metrics = []
    for r in rows:
        h, a = metrics(r["human_text"]), metrics(r["ai_text"])
        if h and a:
            pair_metrics.append((h, a))
    n_used = len(pair_metrics)
    for dim in dims:
        diffs = [h[dim] - a[dim] for h, a in pair_metrics]
        pos = sum(1 for x in diffs if x > 0)
        neg = sum(1 for x in diffs if x < 0)
        z, p, rb = wilcoxon(diffs)
        # sign test normal approx
        ns = pos + neg
        sz = (pos - ns / 2) / math.sqrt(ns / 4) if ns > 0 else 0.0
        sp = math.erfc(abs(sz) / math.sqrt(2))
        dz = st.fmean(diffs) / (st.pstdev(diffs) or 1e-9)
        # bootstrap CI on median delta
        meds = sorted(st.median(random.choice(diffs) for _ in diffs) for _ in range(2000))
        out.append({"dim": dim, "median_delta_h_minus_ai": round(st.median(diffs), 3),
                    "median_delta_95ci": [round(meds[50], 3), round(meds[1949], 3)],
                    "pos": pos, "neg": neg, "sign_p": round(sp, 5),
                    "wilcoxon_z": z, "wilcoxon_p": p, "matched_rank_biserial": rb,
                    "paired_dz": round(dz, 3)})
    res = {"n_pairs": n_used, "dims": out,
           "note": "Paired by shared headline/event (matched design). delta=human-ai. "
                   "Positive = human higher. M bias-corrected."}
    Path(path).parent.joinpath("paired_test_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
