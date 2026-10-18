#!/usr/bin/env python3
"""R1-3: is the instruction-mode result robust to the prompt's wording?

The benchmark uses ONE fixed wording per mode; R1-3 notes the mode ordering
could be an artifact of those particular sentences. This harness reruns the
instruction condition under three independent paraphrases of the VOICE+TAIL
pair (same Persona Pattern and the same constraint as the canonical VOICE/TAIL:
no mention of sentence length, rhythm, or any number) on one model family, one
seed, all 200 briefs, and scores each variant with the same paired machinery.

If the instruction-mode deltas agree in sign and rough magnitude across
paraphrases, the mode result is about the CONDITION, not the sentence.

Checkpoint-resumable per variant. Usage:
  python3 -m experiments.ablation.prompt_paraphrase --model deepseek-v4 \
      --outdir <dir> [--limit N] [--workers 8]
Then score each <dir>/variant_<k>.jsonl with experiments.ablation.seed_avg_paired.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bench import prompts
from bench.config import MODELS
from bench.providers import generate

# Three independent paraphrases of (VOICE, TAIL). Persona Pattern preserved;
# like the canonical pair, none names the statistic, sentence length, or rhythm.
VARIANTS = {
    "v1": (
        "You are a veteran newspaper reporter. Write the story the way an experienced "
        "newsroom professional would: specific, grounded in facts, with a voice that reads "
        "as unmistakably human rather than machine-written.",
        "Hold that authentic reporter's voice through to the final sentence.",
    ),
    "v2": (
        "Take the role of a seasoned journalist at a major daily. Produce the kind of article "
        "an editor would run with light touches only: concrete details, a living, natural "
        "voice, nothing stiff or formulaic.",
        "Make sure the writing still sounds fully human in the closing paragraphs, not just the opening.",
    ),
    "v3": (
        "Write as a long-time staff correspondent. The article should carry the texture of "
        "real human reportage: precise facts, natural phrasing, and none of the flat, "
        "mechanical quality of generated text.",
        "Sustain that human texture to the very end of the piece.",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    model = next((m for m in MODELS if m["key"] == a.model), None)
    if not model:
        sys.exit(f"unknown model {a.model}")
    briefs = prompts.load_briefs()
    if a.limit:
        briefs = briefs[: a.limit]
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for vk, (voice, tail) in VARIANTS.items():
        out = outdir / f"variant_{vk}.jsonl"
        done = set()
        if out.exists():
            for line in out.read_text().splitlines():
                if line.strip():
                    done.add(json.loads(line)["scenario_id"])
        todo = [b for b in briefs if b["id"] not in done]
        print(f"[{vk}] {len(briefs)} briefs, {len(done)} done, {len(todo)} to run")
        lock = threading.Lock()
        fails = 0

        def one(b, voice=voice, tail=tail, vk=vk):
            user = f"{voice}\n\n{b['brief']}\n\n{tail}"
            text, _r, usage = generate(model, "", user)
            return {"eval_prompt_id": f"instruction_{vk}", "scenario_id": b["id"],
                    "iteration": 0, "model_key": a.model, "text": (text or "").strip(),
                    "usage": usage}

        with out.open("a") as fh, ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(one, b): b["id"] for b in todo}
            for fut in as_completed(futs):
                try:
                    row = fut.result()
                except Exception as e:  # noqa: BLE001
                    fails += 1
                    print(f"FAIL {futs[fut]}: {e}", file=sys.stderr)
                    if fails >= 5:
                        sys.exit(f"[{vk}] too many failures; rerun to resume")
                    continue
                with lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
        n = sum(1 for line in out.read_text().splitlines() if line.strip())
        print(f"[{vk}] complete: {n}/{len(briefs)} -> {out}")


if __name__ == "__main__":
    main()
