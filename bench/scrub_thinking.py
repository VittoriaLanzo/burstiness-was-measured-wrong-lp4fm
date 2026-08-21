#!/usr/bin/env python3
"""Split a bench output JSONL into clean rows and thinking-truncation casualties.

A cell is DIRTY when the model's reasoning overran max_tokens without ever emitting the
closing </think>, so strip_think had nothing to strip and the recorded text IS truncated
reasoning (v2 run 2026-07-03: 120/3000 cells). The rule is deliberately conservative and
documented: the text starts with a known reasoning scaffold marker, or it is longer than
--max-chars (default 8000; real articles run 1.0-3.5k chars, truncated 6k-token reasoning
runs ~20k). Stdlib only.

    python3 -m bench.scrub_thinking out/qwen3.6-27b.jsonl              # report only
    python3 -m bench.scrub_thinking out/qwen3.6-27b.jsonl --in-place   # move dirty rows out

--in-place moves the dirty rows to <file>.dirty-<utcstamp>.jsonl next to the input (the
CommitScheduler mirrors that sidecar to the HF dataset, so nothing is ever lost) and keeps
only clean rows in the main file, so gen_vllm's append-only resume re-queues exactly the
dirty cells on the next run: a self-healing rerun.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

MARK = re.compile(r"\s*(Thinking Process|\*\*Deconstruct|1\.\s+\*\*)")


def is_dirty(text: str, max_chars: int) -> bool:
    return bool(MARK.match(text)) or len(text) > max_chars


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--in-place", action="store_true", help="move dirty rows to a sidecar; keep clean rows in place")
    ap.add_argument("--max-chars", type=int, default=8000)
    a = ap.parse_args()

    p = Path(a.path)
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    clean = [r for r in rows if not is_dirty(r["text"], a.max_chars)]
    dirty = [r for r in rows if is_dirty(r["text"], a.max_chars)]
    print(f"{p.name}: {len(rows)} rows -> {len(clean)} clean, {len(dirty)} dirty", flush=True)
    for r in dirty[:5]:
        print(f"  dirty: {r['cell_id']} ({len(r['text'])} chars)", flush=True)

    if a.in_place and dirty:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        side = p.with_name(f"{p.stem}.dirty-{ts}.jsonl")
        side.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in dirty) + "\n", encoding="utf-8")
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in clean) + ("\n" if clean else ""),
                     encoding="utf-8")
        print(f"moved {len(dirty)} dirty rows to {side.name}; resume regenerates exactly those cells", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
