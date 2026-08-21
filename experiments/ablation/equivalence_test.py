#!/usr/bin/env python3
"""
CI-based equivalence (TOST-style) assessment of the prompt-ceiling parity claims.

Addresses reviewer remark R1-2: the paper's parity claims ("a plain instruction reaches
parity on B_n, Gini, and segmentation entropy") are failures to reject a null, not
demonstrated equivalence. This turns them into an actual equivalence statement using the
committed 95% CIs on the paired median delta: a delta is *statistically equivalent to zero*
at margin m when its two-sided 95% CI lies entirely inside [-m, +m] (the two-one-sided-tests
logic, read off the interval).

Because the equivalence margin m is a judgement about what sentence-length-rhythm difference
is negligible on the bounded [-1, 1] coefficient, we do NOT hard-code one answer: we report,
per (mode, metric), the tightest margin at which equivalence holds (m* = max|CI endpoint|)
and evaluate a couple of candidate margins. m* is the number to argue about; pick m first,
then read equivalence.

NOTE (honesty): this reads the committed summary CIs (bootstrap CI of the paired median).
A fully rigorous TOST on the raw 200 paired differences needs the per-headline scores
(data/*.jsonl on HF scored via seed_avg_paired.py); the margin and the raw-data TOST both
want author sign-off before going into the paper.

Run:
    python3 experiments/ablation/equivalence_test.py
    python3 experiments/ablation/equivalence_test.py --margin 0.05 --mode instruction
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "experiments" / "results" / "benchmark"
DEFAULT_MODEL = "qwen3.6-27b"

# The paper's parity claim is about the dispersion + segmentation marginals under instruction.
PARITY_DIMS = ["B_n", "gini", "seg_entropy"]
CANDIDATE_MARGINS = [0.03, 0.05, 0.08]


def tightest_margin(ci: list[float]) -> float:
    """Smallest symmetric margin m for which [-m, m] contains the whole CI."""
    return max(abs(ci[0]), abs(ci[1]))


def assess(data: dict, mode: str, margins: list[float]) -> dict:
    dims = data["dims"]
    out = {}
    for metric, cells in dims.items():
        if mode not in cells:
            continue
        ci = cells[mode]["median_delta_95ci"]
        m_star = tightest_margin(ci)
        out[metric] = {
            "median_delta": cells[mode]["median_delta_h_minus_ai"],
            "ci95": ci,
            "tightest_equivalence_margin": round(m_star, 4),
            "equivalent_at": {f"{m}": (ci[0] >= -m and ci[1] <= m) for m in margins},
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="instruction")
    ap.add_argument("--margin", type=float, default=None, help="add a specific margin to evaluate")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model dir under experiments/results/benchmark/ (default %(default)s)")
    args = ap.parse_args()
    ssot = BENCH / args.model / "seed_avg_paired.json"
    out = BENCH / args.model / "equivalence.json"
    if not ssot.exists():
        raise SystemExit(f"no SSOT at {ssot}")
    data = json.loads(ssot.read_text())
    margins = CANDIDATE_MARGINS + ([args.margin] if args.margin else [])

    res = assess(data, args.mode, sorted(set(margins)))
    print(f"Equivalence assessment, mode = {args.mode} (delta = human - model, coefficient in [-1,1])")
    print(f"{'metric':<14}{'delta':>8}{'  95% CI':>20}{'  m*':>8}   equivalent at margin")
    for metric, r in res.items():
        ci = r["ci95"]
        flags = "  ".join(f"{m}:{'Y' if ok else 'n'}" for m, ok in r["equivalent_at"].items())
        star = " <- parity dim" if metric in PARITY_DIMS else ""
        print(f"{metric:<14}{r['median_delta']:>8.3f}   [{ci[0]:>6.3f},{ci[1]:>6.3f}]{r['tightest_equivalence_margin']:>8.3f}   {flags}{star}")

    # headline read for the parity dims
    verdict = {d: res[d]["tightest_equivalence_margin"] for d in PARITY_DIMS if d in res}
    out.write_text(json.dumps({"mode": args.mode, "margins": sorted(set(margins)),
                               "by_metric": res, "parity_dim_tightest_margin": verdict}, indent=2))
    print(f"\nParity dims need margins up to {max(verdict.values()):.3f} for equivalence "
          f"({', '.join(f'{k} {v:.3f}' for k, v in verdict.items())}).")
    print(f"Wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
