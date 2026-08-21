"""Checkpointed, append-only result store.

One JSONL file per model under results/. Each line is one generation cell, keyed by
(prompt_id, iteration) where prompt_id = <mode>__s<brief_id>. Successful cells are flushed
(and fsync'd) immediately, so a crash or a model timing out never loses earlier outputs and a
re-run resumes by skipping cells already on disk. Failures are NOT written, so they are simply
retried on the next run.

The (mode x model x brief x iteration) array is reconstructed by reading these files back;
load_records() returns the raw rows for that.
"""
from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime, timezone

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def _path(model_key):
    return RESULTS_DIR / f"{model_key}.jsonl"


def done_cells(model_key):
    """Cells already generated for this model: (prompt_id, iteration)."""
    p = _path(model_key)
    if not p.exists():
        return set()
    done = set()
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done.add((r["prompt_id"], r["iteration"]))
    return done


def append(model_key, *, eval_prompt_id, scenario_id, prompt_id, iteration, text,
           reasoning=None, usage=None):
    """Append one completed cell and force it to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "model_key": model_key,
        "eval_prompt_id": eval_prompt_id,  # the mode
        "scenario_id": scenario_id,        # the brief / headline id
        "prompt_id": prompt_id,            # <mode>__s<brief_id>
        "iteration": iteration,
        "text": text,
        "reasoning": reasoning,
        "usage": usage,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with _path(model_key).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return rec


def load_records(model_key):
    """All rows for a model (for analysis / array reconstruction)."""
    p = _path(model_key)
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
