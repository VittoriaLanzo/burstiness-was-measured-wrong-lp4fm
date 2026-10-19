#!/usr/bin/env python3
"""Builder for the era-controlled Wikinews human-vs-AI ablation corpus.

Pairs with `clean_wikinews.py` (the wikitext -> prose cleaner): this script selects which
articles enter the corpus and validates reproduction. Two jobs:

  --validate  Regenerate the human side of the existing pairs from the public source and
              assert byte-for-byte reproduction (a 100/100 regression check that the
              committed cleaner is the one that built the frozen corpus).
  --select    Select a new, disjoint batch of human articles with the same method, for
              expanding the corpus (for example 100 -> 200). Emits the human side plus a
              fairness report; the AI side is generated separately.

Selection method:
  * Source: malteos/wikinews, English config, raw wikitext (`raw_text`).
  * Window: article dated 2016-2021 (pre-model).
  * Clean to plain prose via `clean_wikinews.clean`; filter only by length and format,
    never by style: keep 120..500 words and >= 10 sentences.
  * Stratify evenly across the six years (100 -> 17/17/17/17/16/16).
  * Word count is whitespace-split; sentence count splits on [.!?] followed by whitespace
    (both rules reproduce the frozen human_words / human_n_sentences 100/100).

Deps: mwparserfromhell (via clean_wikinews), huggingface_hub, pandas. The HF token is read
from the standard cache (`huggingface-cli login`); the ablation dataset is private.

Usage:
  python -m experiments.ablation.build_ablation_corpus --validate
  python -m experiments.ablation.build_ablation_corpus --select --n 100 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path
from experiments.ablation.clean_wikinews import clean  # noqa: E402

# --- configuration -----------------------------------------------------------
SOURCE_REPO = "malteos/wikinews"
ABLATION_REPO = "burstiness-engine/burstiness-engine-ablation"
YEARS = list(range(2016, 2022))
MIN_WORDS, MAX_WORDS = 120, 500
MIN_SENTENCES = 10
DATA_DIR = Path(__file__).resolve().parent / "data"


# --- counting rules (reproduce the frozen human_words / human_n_sentences 100/100) ---
def words(t: str) -> int:
    return len(t.split())


def n_sentences(t: str) -> int:
    t = t.strip()
    return 0 if not t else len([s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()])


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def eligible(word_count: int, sent_count: int) -> bool:
    return MIN_WORDS <= word_count <= MAX_WORDS and sent_count >= MIN_SENTENCES


# --- data loading ------------------------------------------------------------
def load_pool():
    import pandas as pd
    from huggingface_hub import hf_hub_download

    frames = []
    for y in YEARS:
        p = hf_hub_download(SOURCE_REPO, f"en/{y}-00000-of-00001.parquet", repo_type="dataset")
        frames.append(pd.read_parquet(
            p, columns=["wiki_page_id", "article_year", "article_timestamp", "title", "url", "raw_text"]))
    df = pd.concat(frames, ignore_index=True)
    df = df[df.article_year.isin(YEARS)].drop_duplicates("wiki_page_id").reset_index(drop=True)
    return df


def load_existing():
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(ABLATION_REPO, "burstiness_pairs.jsonl", repo_type="dataset")
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


# --- modes -------------------------------------------------------------------
def do_validate() -> int:
    df = load_pool()
    rows = load_existing()
    raw_by_key = {}
    for _, r in df.iterrows():
        raw_by_key.setdefault((norm(r.title), str(r.article_timestamp)[:10]), r.raw_text)
        raw_by_key.setdefault(norm(r.title), r.raw_text)

    exact = wok = sok = miss = 0
    fails = []
    for r in rows:
        raw = raw_by_key.get((norm(r["headline"]), r["date"])) or raw_by_key.get(norm(r["headline"]))
        if raw is None:
            miss += 1
            continue
        got = re.sub(r"\s+", " ", clean(raw)).strip()
        want = re.sub(r"\s+", " ", r["human_text"]).strip()
        if got == want:
            exact += 1
        else:
            fails.append(r["id"])
        wok += words(r["human_text"]) == r["human_words"]
        sok += n_sentences(r["human_text"]) == r["human_n_sentences"]

    n = len(rows)
    print(f"human_text exact reproduction: {exact}/{n}  (no-source: {miss})")
    print(f"word-count rule matches stored: {wok}/{n}")
    print(f"sentence-count rule matches stored: {sok}/{n}")
    if fails:
        print(f"NON-REPRODUCED ids: {fails}")
    ok = exact == n and wok == n and sok == n
    print("VALIDATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _quotas(n: int, k: int) -> list[int]:
    """Even split of n across k buckets; the first n%k buckets get one extra (100 -> 17,17,17,17,16,16)."""
    base, extra = divmod(n, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def do_select(n: int, seed: int) -> int:
    import random

    df = load_pool()
    rows = load_existing()
    used = {norm(r["headline"]) for r in rows}

    # build the eligible, unused candidate pool (clean once; log any cleaner failures)
    cands = []
    errors = 0
    for _, r in df.iterrows():
        if norm(r.title) in used:
            continue
        try:
            body = clean(r.raw_text)
        except Exception as e:  # a proper parser can still trip on malformed markup
            errors += 1
            print(f"  clean() failed on page {r.wiki_page_id} ({r.title!r}): {type(e).__name__}", file=sys.stderr)
            continue
        w, s = words(body), n_sentences(body)
        if eligible(w, s):
            cands.append({"wiki_page_id": int(r.wiki_page_id), "year": int(r.article_year),
                          "date": str(r.article_timestamp)[:10], "headline": r.title, "url": r.url,
                          "human_text": body, "human_words": w, "human_n_sentences": s})
    print(f"cleaner failures skipped: {errors}")

    by_year = {y: sorted([c for c in cands if c["year"] == y], key=lambda c: c["wiki_page_id"]) for y in YEARS}
    quotas = _quotas(n, len(YEARS))
    rng = random.Random(seed)
    selected = []
    for y, q in zip(YEARS, quotas):
        avail = by_year[y]
        if len(avail) < q:
            print(f"  WARNING year {y}: wanted {q}, only {len(avail)} eligible; taking all", file=sys.stderr)
        selected.extend(rng.sample(avail, min(q, len(avail))))

    for i, c in enumerate(selected):
        c["id"] = len(rows) + i  # continue ids after the existing corpus (100..)

    _write_outputs(selected, cands, n, seed)
    _fairness_report(selected, cands)
    return 0


def _write_outputs(selected, cands, n, seed):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"new_{n}_seed{seed}"
    jl = DATA_DIR / f"{stem}_human.jsonl"
    with jl.open("w", encoding="utf-8") as f:
        for c in selected:
            f.write(json.dumps({k: c[k] for k in
                    ("id", "headline", "date", "human_text", "human_words", "human_n_sentences")},
                    ensure_ascii=False) + "\n")
    csv_path = DATA_DIR / f"{stem}_headlines.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["id", "date", "year", "headline", "human_words", "human_n_sentences", "url"])
        for c in selected:
            wtr.writerow([c["id"], c["date"], c["year"], c["headline"], c["human_words"], c["human_n_sentences"], c["url"]])
    print(f"\nwrote {len(selected)} human articles -> {jl}")
    print(f"wrote headline list -> {csv_path}")


def _fairness_report(selected, cands):
    """Check that the selection is not drawn from the length tail: compare the selected
    word-count distribution against the full eligible pool and report percentile spread."""
    pool_w = sorted(c["human_words"] for c in cands)
    sel_w = sorted(c["human_words"] for c in selected)

    def q(a, p):
        return a[min(len(a) - 1, int(p * len(a)))]

    def line(a, label):
        return (f"  {label:<9} n={len(a):<4} min {a[0]:>3} p10 {q(a,.10):>3} p25 {q(a,.25):>3} "
                f"median {int(st.median(a)):>3} p75 {q(a,.75):>3} p90 {q(a,.90):>3} max {a[-1]:>3}")

    def pctile(v):
        return 100.0 * sum(1 for x in pool_w if x < v) / len(pool_w)

    pcts = sorted(pctile(c["human_words"]) for c in selected)
    top = sum(1 for p in pcts if p >= 90)
    bot = sum(1 for p in pcts if p <= 10)

    print("\n--- fairness: selected vs eligible pool (word count) ---")
    print(line(pool_w, "POOL"))
    print(line(sel_w, "SELECTED"))
    print(f"  selected percentile within pool: min {pcts[0]:.0f} median {st.median(pcts):.0f} max {pcts[-1]:.0f}"
          f"  | top decile {top}/{len(selected)}  bottom decile {bot}/{len(selected)}")
    print("  (uniform spread ~ representative middle sample; clustered high/low ~ tail draw)")
    print("\n--- per-year counts (eligible-unused pool vs selected) ---")
    for y in YEARS:
        print(f"  {y}: pool {sum(1 for c in cands if c['year']==y):>3}  selected {sum(1 for c in selected if c['year']==y):>2}")

    report = {"n_selected": len(selected), "pool_n": len(cands),
              "pool_word_dist": {"min": pool_w[0], "median": st.median(pool_w), "max": pool_w[-1]},
              "selected_word_dist": {"min": sel_w[0], "median": st.median(sel_w), "max": sel_w[-1]},
              "selected_pool_percentile": {"min": round(pcts[0], 1), "median": round(st.median(pcts), 1),
                                            "max": round(pcts[-1], 1), "top_decile": top, "bottom_decile": bot},
              "per_year": {y: {"pool": sum(1 for c in cands if c["year"] == y),
                               "selected": sum(1 for c in selected if c["year"] == y)} for y in YEARS}}
    (DATA_DIR / "new_selection_fairness.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote fairness report -> {DATA_DIR / 'new_selection_fairness.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true", help="regenerate the existing 100 and assert 100/100")
    g.add_argument("--select", action="store_true", help="select a new disjoint batch of human articles")
    ap.add_argument("--n", type=int, default=100, help="number of new articles to select (default 100)")
    ap.add_argument("--seed", type=int, default=0, help="selection RNG seed (default 0)")
    args = ap.parse_args()
    if args.validate:
        return do_validate()
    return do_select(args.n, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
