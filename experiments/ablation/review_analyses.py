#!/usr/bin/env python3
"""Supplementary analyses of the prompt-ceiling benchmark, computed from the released artifacts.

Every quantity is computed with the canonical scorer's own pieces (paired_test.metrics /
wilcoxon, seeded bootstrap of 2,000 resamples), on the same uniformly normalized
generations, so nothing here is a parallel implementation of the protocol.

  human_spread        B_n distribution of the 200 human articles (margin calibration)
  equivalence         instruction-mode CI-bounded equivalence for every family, every margin
  mode_contrasts      direct paired mode-vs-mode tests within family (seed-averaged per headline)
  variance_baseline   raw var(L) paired deltas beside B_n, and each statistic's correlation
                      with mean sentence length on the human articles
  guard_rates         Goodhart-guard flag rates on human articles and on every leg/mode
  guard_filtered_B_n  the B_n benchmark re-scored without guard-flagged generations and
                      without the human pairs the guard flags (side by side with the main scoring)
  floor10             the B_n benchmark re-scored with a ten-sentence floor (default floor: five)
  lengths             achieved word-count ratio and sentence counts per leg/mode

    python -m experiments.ablation.review_analyses HUMAN.jsonl NORM_DIR [--out PATH]

NORM_DIR holds one normalized generation file per leg, named <model_id>.jsonl
(normalize_generation.py output). Writes experiments/results/review_analyses.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.ablation.paired_test import m_corr, metrics, wilcoxon  # noqa: E402
from experiments.burstiness_metrics import sentence_lengths  # noqa: E402
from experiments.burstiness_pp import burstiness_B_corrected, gaming_flags  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "experiments" / "results" / "benchmark"
MODES = ["control", "instruction", "fewshot", "plan", "self_refine"]
MARGINS = [0.03, 0.05, 0.08]
CONTRASTS = [("instruction", "control"), ("fewshot", "control"), ("plan", "control"),
             ("self_refine", "control"), ("fewshot", "instruction"), ("self_refine", "instruction")]


def _boot_ci(d, rng):
    meds = sorted(st.median(rng.choice(d) for _ in d) for _ in range(2000))
    return [meds[50], meds[1949]]


def _q(a, p):
    a = sorted(a)
    return a[int(p * (len(a) - 1))]


def _corr(a, b):
    ma, mb = st.fmean(a), st.fmean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (len(a) * st.pstdev(a) * st.pstdev(b))


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _seed_avg(rows, fn):
    """mode -> headline -> mean of fn(text) over the seeds where fn is defined."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["text"].strip():
            v = fn(r["text"])
            if v is not None:
                by[r["eval_prompt_id"]][str(r["scenario_id"])].append(v)
    return {m: {k: st.fmean(v) for k, v in d.items()} for m, d in by.items()}


def _paired(diffs, rng):
    z, p, rb = wilcoxon(diffs)
    return {"n_pairs": len(diffs), "median_delta": round(st.median(diffs), 3),
            "median_delta_95ci": _boot_ci(diffs, rng), "wilcoxon_p": p, "matched_rank_biserial": rb}


def _var_L(text):
    L = sentence_lengths(text)
    return st.pvariance(L) if len(L) >= 5 else None


def _bn_floor10(text):
    L = [float(x) for x in sentence_lengths(text)]
    if len(L) < 10 or m_corr(L) is None:
        return None
    return burstiness_B_corrected(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("human", type=Path)
    ap.add_argument("norm_dir", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "review_analyses.json")
    a = ap.parse_args()
    rng = random.Random(0)

    hrows = {str(r["id"]): r for r in _load(a.human)}
    htext = {k: r["human_text"] for k, r in hrows.items()}
    hm = {k: metrics(t) for k, t in htext.items()}
    hb = [m["B_n"] for m in hm.values() if m]
    out = {"note": __doc__.split("\n\n")[0]}

    # human spread
    out["human_spread"] = {"n": len(hb), "median": round(st.median(hb), 3), "sd": round(st.pstdev(hb), 3),
                           "iqr": [round(_q(hb, .25), 3), round(_q(hb, .75), 3)],
                           "p10_p90": [round(_q(hb, .10), 3), round(_q(hb, .90), 3)]}
    sd = st.pstdev(hb)
    out["human_spread"]["margins_in_sd"] = {str(m): round(m / sd, 2) for m in MARGINS}

    # equivalence for every family from the committed SSOTs
    eq = {}
    for d in sorted(p for p in BENCH.iterdir() if (p / "seed_avg_paired.json").is_file()):
        e = json.loads((d / "seed_avg_paired.json").read_text())["dims"]["B_n"]["instruction"]
        ci = e["median_delta_95ci"]
        m_star = max(abs(ci[0]), abs(ci[1]))
        eq[d.name] = {"median_delta": e["median_delta_h_minus_ai"], "ci95": ci, "wilcoxon_p": e["wilcoxon_p"],
                      "significant_05": e["wilcoxon_p"] < 0.05, "m_star": m_star,
                      "equivalent_at": {str(m): m_star <= m for m in MARGINS}}
    out["equivalence_instruction_B_n"] = eq

    # human-side statistics for the variance baseline
    hv = {k: _var_L(t) for k, t in htext.items()}
    ml = {k: st.fmean(sentence_lengths(t)) for k, t in htext.items()}
    ks = [k for k in htext if hv[k] is not None and hm[k]]
    out["variance_baseline"] = {"human_corr_with_mean_sentence_length": {
        "var_L": round(_corr([hv[k] for k in ks], [ml[k] for k in ks]), 2),
        "B_n": round(_corr([hm[k]["B_n"] for k in ks], [ml[k] for k in ks]), 2)}, "legs": {}}

    hflags = [gaming_flags(t) for t in htext.values()]
    hclean = {k for k, g in zip(htext, hflags) if g["ok"]}
    out["guard_rates"] = {"human": {"flagged": sum(not g["ok"] for g in hflags), "n": len(hflags),
                                    "by_flag": dict(collections.Counter(k for g in hflags for k, v in g["flags"].items() if v))},
                          "legs": {}}
    out["mode_contrasts_B_n"], out["floor10_B_n"], out["lengths"] = {}, {}, {}
    out["guard_filtered_B_n"] = {}
    hb10 = {k: _bn_floor10(t) for k, t in htext.items()}

    for f in sorted(a.norm_dir.glob("*.jsonl")):
        mid, rows = f.stem, _load(f)
        bn = _seed_avg(rows, lambda t: (metrics(t) or {}).get("B_n"))
        out["mode_contrasts_B_n"][mid] = {
            f"{x}-{y}": _paired([bn[x][k] - bn[y][k] for k in sorted(set(bn[x]) & set(bn[y]))], rng)
            for x, y in CONTRASTS}   # positive: mode x more dispersed than mode y
        va = _seed_avg(rows, _var_L)
        out["variance_baseline"]["legs"][mid] = {
            m: _paired([hv[k] - va[m][k] for k in va[m] if hv.get(k) is not None], rng) for m in MODES}
        b10 = _seed_avg(rows, _bn_floor10)
        out["floor10_B_n"][mid] = {
            m: _paired([hb10[k] - b10[m][k] for k in b10[m] if hb10.get(k) is not None], rng) for m in MODES}
        flag_n, flag_k, wr, ns = collections.Counter(), collections.Counter(), collections.defaultdict(list), collections.defaultdict(list)
        gb = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            t = r["text"]
            if not t.strip():
                continue
            m = r["eval_prompt_id"]
            flag_n[m] += 1
            ok = gaming_flags(t)["ok"]
            flag_k[m] += not ok
            v = (metrics(t) or {}).get("B_n")
            if ok and v is not None:
                gb[m][str(r["scenario_id"])].append(v)
            wr[m].append(len(t.split()) / hrows[str(r["scenario_id"])]["human_words"])
            ns[m].append(len(sentence_lengths(t)))
        out["guard_rates"]["legs"][mid] = {m: round(flag_k[m] / flag_n[m], 3) for m in MODES}
        out["guard_filtered_B_n"][mid] = {
            m: _paired([hm[k]["B_n"] - st.fmean(v) for k, v in gb[m].items() if k in hclean], rng) for m in MODES}
        out["lengths"][mid] = {m: {"word_ratio_median": round(st.median(wr[m]), 2),
                                   "sentences_median": st.median(ns[m]),
                                   "frac_below_10_sentences": round(sum(n < 10 for n in ns[m]) / len(ns[m]), 3)}
                               for m in MODES}
        print(f"done {mid}", flush=True)

    out["lengths"]["human"] = {"sentences_median": st.median(len(sentence_lengths(t)) for t in htext.values()),
                               "min_sentences": min(len(sentence_lengths(t)) for t in htext.values())}
    a.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
