#!/usr/bin/env python3
"""
PROOF: the cross-model "segmentation-entropy split" is a Unicode-quote metric artifact.

The overnight 3-model expansion reported a large seg_entropy deficit for the two API models
(DeepSeek V4 Pro rank-biserial ~0.9, Grok 4.3 ~0.98 vs Qwen ~0.07) and it was queued to be
named as a headline finding. It is not real. `segmentation_entropy` scores the entropy of the
punctuation distribution using the regex `[.!?,;:\\-—()"\\']`, which counts ONLY straight
quotes. Both API models overwhelmingly emit curly/smart quotes (" " ' '), so all their
attribution quotes are invisible to the metric, collapsing their punctuation onto period+comma
and deflating entropy. The human baseline and the self-hosted vLLM Qwen use straight quotes, so
theirs are counted. Normalizing curly -> straight before scoring closes the entire gap.

Tells that flagged it before this proof: the deficit is present at full strength in the
`control` mode and is nearly mode-invariant (a fixed model/tokenizer property, not a prompting
effect), it hits two independent API families identically, and B_n (the actual burstiness
construct) is normal for all three models.

Reads the local raw generations in bench/results/*.jsonl (no network). Requires the human
baseline to confirm it uses straight quotes (arithmetic already implies it: normalized model
seg_entropy ~= the SSOT-implied human value).

Run:
    python3 experiments/ablation/segentropy_quote_artifact.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # allow direct `python3 experiments/ablation/...` runs

from experiments.burstiness_pp import segmentation_entropy  # noqa: E402
RESULTS = ROOT / "bench" / "results"
BENCH = ROOT / "experiments" / "results" / "benchmark"
CURLY = {"“": '"', "”": '"', "‘": "'", "’": "'"}

SOURCES = {  # raw jsonl -> benchmark model_id (for the SSOT-implied human value)
    "deepseek-v4.jsonl": "deepseek-v4-pro",
    "grok-4.3.jsonl": "grok-4.3",
}


def normalize_quotes(t: str) -> str:
    for a, b in CURLY.items():
        t = t.replace(a, b)
    return t


def control_texts(path: Path, limit: int = 200) -> list[str]:
    out = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("eval_prompt_id") == "control":
            out.append(r.get("text", ""))
        if len(out) >= limit:
            break
    return out


def main() -> int:
    if not RESULTS.exists():
        raise SystemExit(f"No raw generations at {RESULTS}")
    print("seg_entropy (control mode): raw metric vs curly-quote-normalized\n")
    for fname, model_id in SOURCES.items():
        path = RESULTS / fname
        if not path.exists():
            print(f"  {model_id}: {fname} not found, skipping")
            continue
        texts = control_texts(path)
        curly = sum(t.count(c) for t in texts for c in CURLY)
        raw = sum(segmentation_entropy(t) for t in texts) / len(texts)
        fixed = sum(segmentation_entropy(normalize_quotes(t)) for t in texts) / len(texts)
        ssot_f = BENCH / model_id / "seed_avg_paired.json"
        implied_human = None
        if ssot_f.exists():
            c = json.loads(ssot_f.read_text())["dims"]["seg_entropy"]["control"]
            implied_human = raw + c["median_delta_h_minus_ai"]  # delta = human - model
        print(f"  {model_id}: n={len(texts)}  curly-quote chars={curly}")
        print(f"    raw seg_entropy      = {raw:.3f}")
        print(f"    normalized seg_entropy = {fixed:.3f}  (+{fixed - raw:.3f})")
        if implied_human is not None:
            print(f"    SSOT-implied human   = {implied_human:.3f}  ->  after fix the gap is "
                  f"{implied_human - fixed:+.3f} (was {implied_human - raw:+.3f})")
        print()
    print("Verdict: the seg_entropy deficit is a straight-quote-only regex missing smart quotes.")
    print("Fix: normalize curly->straight (or add “”‘’ to _PUNCT_SYMBOLS) before "
          "scoring, then re-score ALL models. Do NOT headline the split. Needs author sign-off "
          "(touches the metric SSOT).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
