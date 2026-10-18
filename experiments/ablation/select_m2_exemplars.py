#!/usr/bin/env python3
"""Select and freeze the 3 few-shot exemplars for the prompt-ceiling bench.

The exemplars teach rhythm, not content: they are clean human news reports (not
interviews / opinion / photo galleries) that are
  (1) disjoint from the 200 reference headlines (never in the target set),
  (2) representative of human rhythm: nearest the median B_n of the pool (B_n is used only to
      exclude degenerate tails, never to select high, so selection does not leak the measured
      axis into the few-shot condition), and
  (3) topically dissimilar from every target and from each other, so no content leaks.

Method basis: few-shot / in-context learning (Brown 2020, arXiv:2005.14165) and demonstration
format/selection (Min 2022, arXiv:2202.12837); the no-metric-leak rule (median-centered, not
burstiness-selected) is the adaptation.

Reuses the Wikinews cleaner and length filter from build_ablation_corpus; scores B_n with
burstiness_pp; checks topical overlap with TF-IDF cosine over headlines (sklearn).
gaming_flags does not gate here: it is a generated-text guard and is false-positive prone on
human prose, so it is only counted, as a calibration signal. The scored pool is cached
(gitignored) so re-picking needs no re-download. Writes the frozen triple and its provenance
to bench/m2_exemplars.json, which prompts.py loads.

    python -m experiments.ablation.select_m2_exemplars           # full: download, score, pick
    python -m experiments.ablation.select_m2_exemplars --repick  # re-pick from the cache only
    python -m experiments.ablation.select_m2_exemplars --demo    # offline self-check
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path
from experiments.ablation import build_ablation_corpus as bac  # noqa: E402
from experiments.ablation.clean_wikinews import clean  # noqa: E402
from experiments.burstiness_pp import burstiness_B_corrected, gaming_flags  # noqa: E402
from experiments.burstiness_metrics import sentence_lengths  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / "m2_exemplars.json"
CACHE = bac.DATA_DIR / "m2_pool_cache.jsonl"  # gitignored (experiments/ablation/data/)
PAPER_REPO, PAPER_FILE = "ANONYMIZED/paper-dataset", "prompts_200_human.jsonl"
PCTILE_LO, PCTILE_HI = 0.10, 0.90  # keep the representative middle; exclude only degenerate tails
LEAK_THRESHOLD = 0.40              # max TF-IDF headline cosine allowed (vs targets, and between picks)
# straight news reports only: drop interviews, opinion, galleries, obituaries, self-referential Wikinews
GENRE_EXCLUDE = re.compile(
    r"\binterviews?\b|\bin (pictures|photos)\b|\bobituar|\bgallery\b|"
    r"\b(have your|your) say\b|\bopinion\b|wikinews", re.I)


def _b_n(text):
    return burstiness_B_corrected([float(x) for x in sentence_lengths(text)])


def _quantile(sorted_vals, p):
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]


def load_targets():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(PAPER_REPO, PAPER_FILE, repo_type="dataset")
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def build_pool(exclude_headlines):
    """Eligible disjoint straight-report Wikinews articles, scored by B_n.

    gaming_flags is counted (human false-positive signal) but not used to exclude.
    """
    df = bac.load_pool()
    pool, errors, special, flagged = [], 0, 0, 0
    for _, r in df.iterrows():
        title = str(r.title)
        if bac.norm(title) in exclude_headlines:
            continue
        if GENRE_EXCLUDE.search(title):
            special += 1
            continue
        try:
            body = clean(r.raw_text)
        except Exception:
            errors += 1
            continue
        w, s = bac.words(body), bac.n_sentences(body)
        if not bac.eligible(w, s):
            continue
        b = _b_n(body)
        if b is None:
            continue
        if not gaming_flags(body)["gate_ok"]:
            flagged += 1  # counted only; gaming_flags is a generated-text guard, FP-prone on human prose
        pool.append({"wiki_page_id": int(r.wiki_page_id), "year": int(r.article_year),
                     "headline": title, "url": str(r.url), "text": body,
                     "words": w, "n_sentences": s, "B_n": round(b, 4)})
    fp = 100.0 * flagged / max(1, len(pool))
    print(f"straight-report disjoint pool: {len(pool)}  (special-format excluded: {special}, "
          f"cleaner failures: {errors})")
    print(f"gaming_flags human false-positive rate (NOT used to gate): {flagged}/{len(pool)} = {fp:.0f}%")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w", encoding="utf-8") as f:
        for c in pool:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return pool


def load_cache():
    if not CACHE.exists():
        raise SystemExit(f"no pool cache at {CACHE}; run without --repick first")
    return [json.loads(l) for l in CACHE.read_text(encoding="utf-8").splitlines() if l.strip()]


def leakage_scores(cand_headlines, target_headlines):
    """Max TF-IDF cosine of each candidate headline against all target headlines."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    X = TfidfVectorizer(stop_words="english").fit_transform(target_headlines + cand_headlines)
    n = len(target_headlines)
    return cosine_similarity(X[n:], X[:n]).max(axis=1)


def pick_three(pool, targets, threshold=LEAK_THRESHOLD):
    """3 representative, leakage-clean, mutually-distinct straight reports near the band center."""
    bns = sorted(c["B_n"] for c in pool)
    lo, hi = _quantile(bns, PCTILE_LO), _quantile(bns, PCTILE_HI)
    center = _quantile(bns, 0.50)  # the median: representative exemplars, never selected for high B_n
    band = [c for c in pool if lo <= c["B_n"] <= hi]
    sims = leakage_scores([c["headline"] for c in band], [t["headline"] for t in targets])
    clean_band = [c for c, s in zip(band, sims) if s < threshold]
    clean_band.sort(key=lambda c: abs(c["B_n"] - center))  # most representative (band center) first
    picks = []
    for c in clean_band:
        if len(picks) == 3:
            break
        if picks and leakage_scores([c["headline"]], [p["headline"] for p in picks]).max() >= threshold:
            continue  # too similar to an already-picked exemplar (enforce topic diversity)
        picks.append(c)
    if len(picks) < 3:
        raise SystemExit(f"only {len(picks)} clean topic-diverse candidates; loosen band/threshold")
    a, b, c = sorted(picks, key=lambda x: x["B_n"], reverse=True)  # a>b>c
    return [a, c, b]  # strongest B_n first and last, weakest in the middle (primacy/recency)


def write_exemplars(picks, pool_n, threshold):
    keys = ("wiki_page_id", "headline", "year", "url", "words", "n_sentences", "B_n", "text")
    payload = {
        "exemplars": [{k: p[k] for k in keys} for p in picks],
        "provenance": {"pool_n": pool_n, "pctile_band": [PCTILE_LO, PCTILE_HI],
                       "select": "representative straight reports nearest the median B_n; B_n excludes only degenerate tails, never selects high (no metric leak)",
                       "genre_filter": "interviews/opinion/galleries/obituaries/self-ref excluded",
                       "leak_metric": "tfidf_cosine_headline", "leak_threshold": threshold,
                       "order": "strongest B_n first and last (primacy/recency), weakest middle",
                       "source": "malteos/wikinews via build_ablation_corpus", "disjoint_from": PAPER_REPO},
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote 3 exemplars -> {OUT}")
    for p in picks:
        print(f"  B_n={p['B_n']:+.3f}  y{p['year']}  n={p['n_sentences']}  {p['headline'][:70]}")


def demo():
    s = leakage_scores(["cat dog fish"], ["cat dog fish", "quantum senate policy"])
    assert s[0] > 0.9, s
    assert leakage_scores(["banana republic vote"], ["quantum senate policy"])[0] < 0.2
    pool = [{"B_n": (i - 50) / 100, "year": 2016 + (i % 6), "headline": f"alpha{i} bravo{i} charlie{i}",
             "wiki_page_id": i, "url": "", "text": "x", "words": 200, "n_sentences": 12} for i in range(100)]
    picks = pick_three(pool, [{"headline": "zeta omega sigma unrelated matter"}])
    assert len(picks) == 3
    assert picks[0]["B_n"] >= picks[1]["B_n"] <= picks[2]["B_n"], [p["B_n"] for p in picks]  # weakest middle
    assert len({p["wiki_page_id"] for p in picks}) == 3
    print("select_m2 self-check PASS: representative band + leakage + topic-diverse + ordering.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repick", action="store_true", help="re-pick from the cached pool (no download)")
    ap.add_argument("--threshold", type=float, default=LEAK_THRESHOLD, help="max leakage cosine")
    ap.add_argument("--demo", action="store_true", help="offline self-check")
    a = ap.parse_args()
    if a.demo:
        return demo()
    targets = load_targets()
    exclude = {bac.norm(t["headline"]) for t in targets}
    pool = load_cache() if a.repick else build_pool(exclude)
    write_exemplars(pick_three(pool, targets, a.threshold), len(pool), a.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
