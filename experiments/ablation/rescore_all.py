#!/usr/bin/env python3
"""
Re-score helper: regenerate every model's seed_avg_paired SSOT with the fixed metric.

After the R1-5 quote-normalization fix in experiments/burstiness_pp.py, the committed
per-model SSOTs (scored with the old metric) are stale for any model whose generations used
smart quotes (DeepSeek, Grok). This driver re-runs the CANONICAL scorer
(experiments/ablation/seed_avg_paired.py, unchanged) for each model against the frozen human
baseline and writes the corrected SSOT into experiments/results/benchmark/<model_id>/, so the
numbers are apples-to-apples and traceable.

Reuses the canonical scorer verbatim (no reimplementation) and prints a before/after
seg_entropy control delta per model so the effect of the fix is visible.

One author runs this (they have the human baseline + all raw generations). It needs:
  - the human baseline JSONL (ANONYMIZED/paper-dataset :: prompts_200_human.jsonl)
  - each model's raw generations JSONL (bench/results/*.jsonl locally, or from each HF dataset)

Run:
    python3 experiments/ablation/rescore_all.py --human path/to/prompts_200_human.jsonl \\
        qwen3.6-27b=path/to/qwen3.6-27b.jsonl \\
        deepseek-v4-pro=bench/results/deepseek-v4.jsonl \\
        grok-4.3=bench/results/grok-4.3.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "experiments" / "results" / "benchmark"


def _control_seg(ssot_path: Path):
    if not ssot_path.exists():
        return None
    try:
        d = json.loads(ssot_path.read_text())
        return d["dims"]["seg_entropy"]["control"]["median_delta_h_minus_ai"]
    except Exception:
        return None


def rescore(model_id: str, human: Path, gen: Path) -> None:
    before = _control_seg(BENCH / model_id / "seed_avg_paired.json")
    # run the canonical scorer verbatim; it writes seed_avg_paired_<label>.json next to gen
    label = model_id.replace("/", "_")
    subprocess.run([sys.executable, "-m", "experiments.ablation.seed_avg_paired",
                    str(human), str(gen), label], cwd=ROOT, check=True)
    produced = gen.parent / f"seed_avg_paired_{label}.json"
    dest_dir = BENCH / model_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "seed_avg_paired.json"
    dest.write_text(produced.read_text())
    after = _control_seg(dest)
    b = f"{before:+.3f}" if before is not None else "n/a"
    a = f"{after:+.3f}" if after is not None else "n/a"
    print(f"  {model_id}: control seg_entropy delta(human-model)  {b} -> {a}  (wrote {dest.relative_to(ROOT)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True, type=Path, help="human baseline JSONL (prompts_200_human)")
    ap.add_argument("models", nargs="+", help="model_id=path/to/raw_generations.jsonl")
    args = ap.parse_args()
    if not args.human.exists():
        raise SystemExit(f"human baseline not found: {args.human}")
    print("Re-scoring with the quote-normalized metric (fix R1-5):")
    for spec in args.models:
        if "=" not in spec:
            raise SystemExit(f"bad model spec {spec!r}; use model_id=path.jsonl")
        model_id, gen = spec.split("=", 1)
        gen_path = Path(gen)
        if not gen_path.exists():
            print(f"  {model_id}: SKIP (raw generations not found: {gen})")
            continue
        rescore(model_id, args.human, gen_path)
    print("\nDone. Re-run `make bench-table` + the parity/congruence checks; the seg_entropy "
          "cross-model gap should collapse to parity for the API models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
