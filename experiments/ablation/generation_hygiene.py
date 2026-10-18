#!/usr/bin/env python3
"""
Generation hygiene audit: report formatting in a model's raw generations that the
straight-quote, markdown-free human news baseline does not contain.

Three detectors, applied to control-mode generations: typographic (smart) quotes, markdown
syntax, and a leading title line. Smart quotes are invisible to the segmentation-entropy
regex, which counts straight quotes only, and title lines and markdown scaffolding add
pseudo-sentences and symbols the article prose does not have; the metric normalizes quote
style itself, and normalize_generation.py strips the markdown and title block before scoring.
A model whose rate on any detector exceeds --warn-rate is flagged.

Pure stdlib. The CLI reports per-model rates from the raw generations (bench/results/*.jsonl).

Run:
    python3 experiments/ablation/generation_hygiene.py
    python3 experiments/ablation/generation_hygiene.py --file bench/results/grok-4.3.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "bench" / "results"

_SMART_QUOTES = "“”‘’"
_MD_INLINE = re.compile(r"(\*\*|__|`|\[[^\]]+\]\([^)]+\))")     # bold / code / links
_MD_BLOCK = re.compile(r"(?m)^(#{1,6}\s|\s*[-*]\s+|\s*\d+\.\s+|>\s)")  # headings / bullets / quotes


def has_smart_quotes(text: str) -> bool:
    return any(c in text for c in _SMART_QUOTES)


def has_markdown(text: str) -> bool:
    return bool(_MD_INLINE.search(text) or _MD_BLOCK.search(text))


def has_title_line(text: str) -> bool:
    parts = (text or "").split("\n\n", 1)
    if len(parts) != 2:
        return False
    first = parts[0].strip()
    return bool(first) and first[-1] not in ".!?\"'”’" and "\n" not in first and len(first) < 160


CHECKS = {"smart_quotes": has_smart_quotes, "markdown": has_markdown, "title_line": has_title_line}


def report(texts: list[str]) -> dict:
    n = len(texts) or 1
    return {name: round(sum(fn(t) for t in texts) / n, 3) for name, fn in CHECKS.items()}


def _texts(path: Path, mode: str = "control", limit: int = 200) -> list[str]:
    out = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("eval_prompt_id") == mode:
            out.append(r.get("text", ""))
        if len(out) >= limit:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, help="a single raw generations JSONL")
    ap.add_argument("--warn-rate", type=float, default=0.05,
                    help="flag a model if any anomaly exceeds this fraction (default 0.05)")
    args = ap.parse_args()
    files = [args.file] if args.file else (sorted(RESULTS.glob("*.jsonl")) if RESULTS.exists() else [])
    files = [f for f in files if "dirty" not in f.name]
    if not files:
        print("No raw generations found (bench/results/*.jsonl). Nothing to check.")
        return 0
    print(f"Generation hygiene (control mode). Flagged if rate > {args.warn_rate:.0%}:\n")
    print(f"{'file':<28}{'smart_quotes':>14}{'markdown':>10}{'title_line':>12}")
    flagged = []
    for f in files:
        r = report(_texts(f))
        marks = "".join(f"{('*' if v > args.warn_rate else ' ')}{v:>{w-1}.0%}"
                        for v, w in [(r['smart_quotes'], 14), (r['markdown'], 10), (r['title_line'], 12)])
        print(f"{f.name:<28}{marks}")
        if any(v > args.warn_rate for v in r.values()):
            flagged.append(f.name)
    print()
    if flagged:
        print(f"FLAGGED (formatting the human baseline lacks -> normalize before scoring): {flagged}")
    else:
        print("Clean: all models within the human-baseline formatting envelope.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
