#!/usr/bin/env python3
"""
Family-wise / FDR correction over the 20 prompt-ceiling significance tests of one model.

Each model's grid holds 20 Wilcoxon tests (5 modes x 4 metrics). This applies Holm-Bonferroni
(controls the family-wise error rate) and Benjamini-Hochberg (controls the false discovery
rate) to the 20 raw p-values from the committed seed_avg_paired.json and reports which effects
survive at alpha=0.05. A test with 0.01 <= p < 0.05 is marked fragile.

Stdlib only. Emits experiments/results/benchmark/<model>/multiplicity.json. The paper reports
BH at q=0.05 within each family's twenty-cell grid, with Holm alongside.

Run:
    python3 experiments/ablation/multiplicity_correction.py --model <model_id>
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "experiments" / "results" / "benchmark"
DEFAULT_MODEL = "qwen3.6-27b"
ALPHA = 0.05


def collect(data: dict) -> list[dict]:
    tests = []
    for metric, cells in data["dims"].items():
        for mode, e in cells.items():
            tests.append({"metric": metric, "mode": mode, "p": float(e["wilcoxon_p"]),
                          "rb": e["matched_rank_biserial"]})
    return tests


def holm(ps: list[float], alpha: float) -> list[bool]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    survive = [False] * m
    for rank, i in enumerate(order):
        thr = alpha / (m - rank)
        if ps[i] <= thr:
            survive[i] = True
        else:
            break  # Holm stops at the first non-rejection
    return survive


def benjamini_hochberg(ps: list[float], alpha: float) -> list[bool]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    survive = [False] * m
    kmax = -1
    for rank, i in enumerate(order, start=1):
        if ps[i] <= (rank / m) * alpha:
            kmax = rank
    for rank, i in enumerate(order, start=1):
        if rank <= kmax:
            survive[i] = True
    return survive


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model dir under experiments/results/benchmark/ (default %(default)s)")
    args = ap.parse_args()
    ssot = BENCH / args.model / "seed_avg_paired.json"
    out = BENCH / args.model / "multiplicity.json"
    if not ssot.exists():
        raise SystemExit(f"no SSOT at {ssot}")
    data = json.loads(ssot.read_text())
    tests = collect(data)
    ps = [t["p"] for t in tests]
    h = holm(ps, ALPHA)
    bh = benjamini_hochberg(ps, ALPHA)
    for t, a, b in zip(tests, h, bh):
        t["raw_sig_05"] = t["p"] < 0.05
        t["holm_survives"] = a
        t["bh_survives"] = b
        t["fragile"] = 0.01 <= t["p"] < 0.05  # significant only at the nominal 0.05 level

    lost_bh = [f"{t['mode']}/{t['metric']} (p={t['p']:.3f})"
               for t in tests if t["raw_sig_05"] and not t["bh_survives"]]
    lost_holm = [f"{t['mode']}/{t['metric']} (p={t['p']:.3f})"
                 for t in tests if t["raw_sig_05"] and not t["holm_survives"]]

    print(f"Multiplicity correction over {len(tests)} tests, alpha={ALPHA}")
    print(f"{'mode/metric':<24}{'p':>10}{'  raw<.05':>9}{'  Holm':>7}{'  BH':>5}")
    for t in sorted(tests, key=lambda x: x["p"]):
        print(f"{t['mode']+'/'+t['metric']:<24}{t['p']:>10.4f}"
              f"{('Y' if t['raw_sig_05'] else 'n'):>9}{('Y' if t['holm_survives'] else 'n'):>7}"
              f"{('Y' if t['bh_survives'] else 'n'):>5}")
    n_raw = sum(t["raw_sig_05"] for t in tests)
    print(f"\nSignificant at raw 0.05: {n_raw}/{len(tests)}; "
          f"survive Holm: {sum(h)}; survive BH: {sum(bh)}.")
    print("Lost under BH: " + (", ".join(lost_bh) if lost_bh else "none (all raw-significant survive BH)"))
    print("Lost under Holm: " + (", ".join(lost_holm) if lost_holm else "none"))

    out.write_text(json.dumps({
        "alpha": ALPHA, "n_tests": len(tests),
        "n_raw_sig": n_raw, "n_holm": sum(h), "n_bh": sum(bh),
        "lost_under_bh": lost_bh, "lost_under_holm": lost_holm,
        "tests": tests,
    }, indent=2))
    print(f"Wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
