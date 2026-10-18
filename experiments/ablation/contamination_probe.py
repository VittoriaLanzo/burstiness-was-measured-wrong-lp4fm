#!/usr/bin/env python3
"""Contamination / memorization probe for the prompt-ceiling benchmark.

The human Wikinews articles may sit in each generator's training data; to the
extent a model reproduces memorized phrasing, the measured authorship gap would
shrink. This probe bounds that account with a measurement.

Method (word-level n-gram overlap):
  For every generation, compute the fraction of its n-grams that appear in
  (a) its paired human article (matched overlap: memorization + topical overlap)
  (b) a fixed mismatched article, id (i+101) mod 200 (topical-register baseline)
Memorization would show as matched >> mismatched, or as individual generations
with runaway matched overlap. Headline-driven topical overlap (names, places)
inflates both sides equally at small n, which is why the mismatched baseline
and the long-n (13-gram) numbers carry the most weight.

Reads bench-format JSONL (control mode by default: the uninstructed condition
is where memorized phrasing would surface most directly). Stdlib only.

Run:
    python3 -m experiments.ablation.contamination_probe --human HUMAN.jsonl \\
        qwen3.6-27b=path.jsonl deepseek-v4-pro=path.jsonl grok-4.3=path.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
NGRAM_SIZES = (8, 13)
_WORD = re.compile(r"[a-z0-9']+")


def ngrams(text: str, n: int) -> set:
    w = _WORD.findall((text or "").lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def overlap(gen: str, ref: str, n: int) -> float:
    """Fraction of the generation's n-grams present in the reference (0..1)."""
    g = ngrams(gen, n)
    if not g:
        return 0.0
    return len(g & ngrams(ref, n)) / len(g)


def probe_model(gens_by_id: dict, human_by_id: dict, n_sizes=NGRAM_SIZES) -> dict:
    ids = sorted(set(gens_by_id) & set(human_by_id), key=int)
    id_list = [i for i in ids]
    out = {}
    for n in n_sizes:
        matched, mismatched = [], []
        worst = ("", 0.0)
        for idx, sid in enumerate(id_list):
            other = id_list[(idx + 101) % len(id_list)]
            for text in gens_by_id[sid]:
                m = overlap(text, human_by_id[sid], n)
                x = overlap(text, human_by_id[other], n)
                matched.append(m)
                mismatched.append(x)
                if m > worst[1]:
                    worst = (sid, m)
        srt = sorted(matched)
        out[f"{n}gram"] = {
            "matched_mean": round(st.fmean(matched), 5),
            "mismatched_mean": round(st.fmean(mismatched), 5),
            "matched_p95": round(srt[int(0.95 * len(srt))], 5),
            "matched_max": round(worst[1], 5),
            "matched_max_id": worst[0],
            "n_gens": len(matched),
            "frac_over_10pct": round(sum(1 for v in matched if v > 0.10) / len(matched), 5),
        }
    return out


def load_bench(path: Path, mode: str) -> dict:
    by_id: dict[str, list[str]] = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("eval_prompt_id") == mode and r.get("text", "").strip():
            by_id.setdefault(str(r["scenario_id"]), []).append(r["text"])
    return by_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True, type=Path)
    ap.add_argument("--mode", default="control")
    ap.add_argument("models", nargs="+", help="model_id=path.jsonl")
    a = ap.parse_args()
    human = {str(r["id"]): r["human_text"] for r in
             (json.loads(l) for l in a.human.open(encoding="utf-8") if l.strip())}
    result = {"mode": a.mode, "ngram_sizes": list(NGRAM_SIZES), "models": {},
              "note": ("Memorization bound. matched = generation vs its paired human "
                       "article; mismatched = vs a fixed other article (topical-register "
                       "baseline). Memorization shows as matched >> mismatched or runaway "
                       "per-generation maxima, especially at 13-grams.")}
    for spec in a.models:
        mid, p = spec.split("=", 1)
        gens = load_bench(Path(p), a.mode)
        result["models"][mid] = probe_model(gens, human)
        r8, r13 = result["models"][mid]["8gram"], result["models"][mid]["13gram"]
        print(f"{mid}: 8-gram matched {r8['matched_mean']:.4f} vs mismatched "
              f"{r8['mismatched_mean']:.4f} (max {r8['matched_max']:.3f}); "
              f"13-gram matched {r13['matched_mean']:.5f} (max {r13['matched_max']:.3f})")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "contamination_probe.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
