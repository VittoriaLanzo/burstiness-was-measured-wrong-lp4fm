#!/usr/bin/env python3
"""Point-process burstiness of text rhythm (stdlib-only).

Sentence boundaries are treated as a point process in word-time, so the sentence-length
sequence L = (l_1 ... l_n) is the sequence of inter-event intervals, and the
complex-systems burstiness measures apply to it.

The scalars, all model-free (computed from sentence lengths alone):

    B   = (sigma - mu) / (sigma + mu)          Goh & Barabasi (EPL 2008)
          bounded in [-1, 1]:
            B -> -1  perfectly regular (a metronome; flat rhythm)
            B  =  0  Poisson / memoryless (sigma == mu)
            B -> +1  maximally bursty (clumps and lulls)

    B_n = finite-size-corrected B               Kim & Jo (Phys Rev E 2016)
          corrects the small-n bias of B; reduces to B as n -> inf. This is the
          primary statistic for text, where n (sentences per document) is small.

    M   = lag-1 autocorrelation of consecutive intervals   Goh & Barabasi
          the "memory" / clumping signal: whether a short sentence tends to follow
          a short one. var(L) and kurt(L) are order-invariant and cannot see this.

Raw var(L) is confounded by absolute sentence length (a genre/length artifact); B and B_n
are scale-free ratios, so they factor that out.

B_n follows Kim & Jo (2016): with r = sigma/mu over n intervals,
    B_n = (sqrt(n+1) r - sqrt(n-1)) / ((sqrt(n+1) - 2) r + sqrt(n-1)),
which is -1 at r = 0, 0 at the finite-n Poisson value r = sqrt((n-1)/(n+1)), and 1 at
the largest r attainable with n intervals, sqrt(n-1). It depends on n as well as on r.

CLI:
    python -m experiments.burstiness_pp --demo            # self-check
    python -m experiments.burstiness_pp --text "..."      # B, B_n, M of a doc

Design
    burstiness     B = [B_n, M]          B_n = Goh-Barabasi/Kim-Jo (primary), Gini = robustness check
    segmentation   S = [punct_entropy]   separate axis (segmentation_entropy here; see burstiness_metrics)
    predictability P = [mean_surprisal, fluc_dabs]   kept outside the construct, not scored (not in this module)
    diagnostics    var_L, mean_len
    Goodhart guard gaming_flags()        flag degenerate hits
    multi-scale    multiscale_burstiness()

Limitations
    - genre/register relativity: B has no absolute meaning; a terse human register reads
      "flat", so a target B must be genre-conditioned.
    - short-document floor: fewer than RELIABILITY_FLOOR sentences gives noisy
      per-document estimates; flagged via burstiness_pp()["reliable"].
    - syntactic complexity is unmeasured (length and punctuation only); see
      syntactic_complexity().
    - long-range temporality: only lag-1 M and the coarse multiscale profile; a DFA/Hurst
      exponent needs longer series than the generations here provide.
    - construct validity (perception): whether B_n tracks perceived rhythm is not
      established by statistical separation; it needs a perception study (not in this repo).
"""
from __future__ import annotations

import argparse
import json
import math
from statistics import fmean, pstdev
from typing import List, Optional, Sequence

# Short-document floor: below this many sentences, a single document's B_n / M estimates
# are noisy. Flagged at runtime via burstiness_pp()["reliable"] and gaming_flags()["gate_ok"];
# the estimates are not adjusted.
RELIABILITY_FLOOR = 10

# ---------------------------------------------------------------------------
# Core measures (operate directly on a sequence of inter-event intervals)
# ---------------------------------------------------------------------------


def burstiness_B(intervals: Sequence[float]) -> Optional[float]:
    """Goh-Barabasi burstiness coefficient B = (sigma - mu) / (sigma + mu).

    Returns None for fewer than two intervals (dispersion undefined). Returns
    -1.0 for a perfectly regular sequence (sigma == 0, the metronome).
    """
    if len(intervals) < 2:
        return None
    mu = fmean(intervals)
    sigma = pstdev(intervals)
    denom = sigma + mu
    if denom <= 0:
        return None
    return (sigma - mu) / denom


def burstiness_B_corrected(intervals: Sequence[float]) -> Optional[float]:
    """Finite-size-corrected burstiness B_n (Kim & Jo 2016).

    B_n = (sqrt(n+1)*r - sqrt(n-1)) / ((sqrt(n+1) - 2)*r + sqrt(n-1)),
    with r = sigma/mu and n = number of intervals. Reduces to (r-1)/(r+1),
    i.e. the asymptotic B, as n -> inf. Removes the systematic small-n bias
    that makes the raw estimator unusable on short documents.
    """
    n = len(intervals)
    if n < 2:
        return None
    mu = fmean(intervals)
    if mu <= 0:
        return None
    r = pstdev(intervals) / mu
    sp, sm = math.sqrt(n + 1), math.sqrt(n - 1)
    denom = (sp - 2.0) * r + sm
    if denom == 0:
        return None
    return (sp * r - sm) / denom


def memory_M(intervals: Sequence[float]) -> Optional[float]:
    """Goh-Barabasi memory coefficient: lag-1 autocorrelation of intervals.

    M = mean_i [ (tau_i - m1)(tau_{i+1} - m2) / (s1 s2) ], where (m1, s1) are
    the mean/std of tau_1..tau_{n-1} and (m2, s2) of tau_2..tau_n. Positive M:
    similar-length sentences cluster (clumping). Negative M: lengths alternate.
    Returns None when undefined (fewer than 3 intervals, or a flat run).

    Lag-1 autocorrelation is biased by about -1/(n-1) at small n; memory_M_corrected adds
    that term back for cross-document comparison at differing n. The guards in gaming_flags
    use raw M.
    """
    n = len(intervals)
    if n < 3:
        return None
    a = list(intervals[:-1])
    b = list(intervals[1:])
    m1, m2 = fmean(a), fmean(b)
    s1, s2 = pstdev(a), pstdev(b)
    if s1 <= 0 or s2 <= 0:
        return None
    cov = fmean((x - m1) * (y - m2) for x, y in zip(a, b))
    return cov / (s1 * s2)


def memory_M_corrected(intervals: Sequence[float]) -> Optional[float]:
    """memory_M plus 1/(n-1), n the number of intervals: the M the benchmark scores."""
    m = memory_M(intervals)
    return None if m is None else m + 1.0 / (len(intervals) - 1)


# ---------------------------------------------------------------------------
# Goodhart guard: flag texts that hit the metric via degenerate patterns
# ---------------------------------------------------------------------------


def _trigram_diversity(text: str) -> float:
    """Distinct word-trigram ratio (catches verbatim repetition / copy-paste)."""
    w = (text or "").split()
    if len(w) < 4:
        return 1.0
    tris = list(zip(w, w[1:], w[2:]))
    return len(set(tris)) / len(tris)


def _runon_ratio(text: str) -> float:
    """Internal clause delimiters (, ; :) per sentence terminal (. ! ?).

    A high ratio means clauses are hidden inside semicolon/comma run-ons, which spoofs
    the period-based segmenter (a 60-word `;`-chain counts as one long "sentence").
    """
    import re as _re
    internal = len(_re.findall(r"[,;:]", text or ""))
    terminal = len(_re.findall(r"[.!?]+", text or "")) or 1
    return internal / terminal


def gaming_flags(text, *, loo_delta=0.25, m_floor=-0.5, m_ceiling=0.7,
                 distinct_len_min=0.5, trigram_min=0.5, frag_frac_max=0.3,
                 punct_rate_max=0.25, runon_max=8.0, finescale_drop=0.4):
    """Goodhart guards: flag texts that hit the metric via degenerate patterns.

    Returns {"ok", "gate_ok", "flags", "detail"}. ``ok`` = no gaming pattern; ``gate_ok``
    = ok and reliable (n >= RELIABILITY_FLOOR). A use that selects text by B_n should require
    gate_ok and a separate quality gate: B_n alone is forgeable by re-punctuation
    (re-pointing periods swings B_n by more than 1 with zero word change), so neither B_n
    nor this guard should select text on its own.

    Flags:
      outlier_driven    : dropping the single longest sentence swings B_n > loo_delta
      low_len_diversity : distinct sentence-length values / n < distinct_len_min, n>=8
                          (blocky / bimodal / period-k / repeated patterns)
      repetitive        : distinct word-trigram ratio < trigram_min (copy-paste)
      alternating       : lag-1 M < m_floor (short/long alternation)
      over_clumped      : lag-1 M > m_ceiling (all-short-then-all-long blocks)
      fine_scale_only   : B_n@1 > 0 but drops by > finescale_drop at scale 2
                          (period-k: fine-scale dispersion with no long-range structure)
      fragment_spam     : >= frag_frac_max of sentences are < 3 words
      punct_gamed       : punctuation marks per word > punct_rate_max
      runon             : internal (,;:) per terminal (.!?) > runon_max

    The thresholds are not calibrated against the human reference corpus; the human
    false-positive rate is reported by review_analyses.py (guard_rates) and
    select_m2_exemplars.py.
    """
    from experiments.burstiness_metrics import _PUNCT_SYMBOLS, sentence_lengths

    Li = [int(x) for x in sentence_lengths(text)]
    L = [float(x) for x in Li]
    n = len(L)
    if n < 3:
        return {"ok": False, "gate_ok": False, "flags": {"too_short": True}, "detail": {"n": n}}

    B_full = burstiness_B_corrected(L)
    j = max(range(n), key=lambda i: L[i])
    B_loo = burstiness_B_corrected(L[:j] + L[j + 1:])
    loo = abs(B_full - B_loo) if (B_full is not None and B_loo is not None) else 0.0
    M = memory_M(L)
    distinct_ratio = len(set(Li)) / n
    trig = _trigram_diversity(text)
    frag_frac = sum(1 for x in Li if x < 3) / n
    words = sum(L) or 1.0
    punct_rate = len(_PUNCT_SYMBOLS.findall(text or "")) / words
    runon = _runon_ratio(text)
    ms = multiscale_burstiness(text)
    b1, b2 = ms.get("B_n@1"), ms.get("B_n@2")
    fine_scale_only = (b1 is not None and b2 is not None and b1 > 0.0
                       and (b1 - b2) > finescale_drop)

    flags = {
        "outlier_driven": loo > loo_delta,
        "low_len_diversity": n >= 8 and distinct_ratio < distinct_len_min,
        "repetitive": trig < trigram_min,
        "alternating": M is not None and M < m_floor,
        "over_clumped": M is not None and M > m_ceiling,
        "fine_scale_only": fine_scale_only,
        "fragment_spam": frag_frac >= frag_frac_max,
        "punct_gamed": punct_rate > punct_rate_max,
        "runon": runon > runon_max,
    }
    ok = not any(flags.values())
    return {
        "ok": ok,
        "gate_ok": ok and n >= RELIABILITY_FLOOR,
        "flags": flags,
        "detail": {"n": n, "loo_dB": round(loo, 3),
                   "M": None if M is None else round(M, 3),
                   "distinct_ratio": round(distinct_ratio, 3), "trigram_div": round(trig, 3),
                   "frag_frac": round(frag_frac, 3), "punct_rate": round(punct_rate, 3),
                   "runon": round(runon, 2)},
    }


# ---------------------------------------------------------------------------
# Robustness check: Gini coefficient of sentence lengths
# ---------------------------------------------------------------------------


def gini(intervals: Sequence[float]) -> float:
    """Gini coefficient of sentence lengths, the robustness check for B_n.

    Bounded [0,1], aggregates spread across all pairwise differences, so it is not
    hostage to a single outlier the way B_n (mean/std based) can be. Reported beside
    B_n; the two agree at r ~ 0.95, so concordance shows the burstiness result is not
    an estimator artifact. Reference: Gini (1912).
    """
    n = len(intervals)
    if n < 2:
        return 0.0
    m = fmean(intervals)
    if m == 0:
        return 0.0
    # sample (unbiased) Gini: divide by n*(n-1), not n^2 (the n^2 form underestimates
    # by (n-1)/n, an n-dependent bias that confounds cross-document comparison).
    return sum(abs(a - b) for a in intervals for b in intervals) / (2.0 * n * (n - 1) * m)


_TYPO_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})


def segmentation_entropy(text: str) -> float:
    """Miller-Madow-corrected Shannon entropy (bits) of the punctuation distribution.

    The Miller-Madow term corrects the plug-in entropy's downward bias (about (K-1)/(2N)
    nats), which otherwise favors the side with fewer punctuation tokens. This is the
    segmentation-diversity axis, separate from burstiness. The full-marginal entropy can be
    inflated by sprinkling rare non-terminal symbols on otherwise flat text; a terminal-only
    ('.!?') variant, or one conditioned on clause placement, would resist that and is not
    implemented.
    """
    from experiments.burstiness_metrics import _PUNCT_SYMBOLS

    # Typographic quotes are mapped to ASCII so quote style (curly vs straight) is not
    # scored as segmentation diversity: _PUNCT_SYMBOLS contains straight quotes only, so
    # without the mapping a text with curly quotes has all its quotation marks dropped from
    # the count, deflating its entropy relative to a straight-quote baseline (see
    # experiments/ablation/segentropy_quote_artifact.py).
    counts = {}
    for ch in _PUNCT_SYMBOLS.findall((text or "").translate(_TYPO_QUOTES)):
        counts[ch] = counts.get(ch, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return h + (len(counts) - 1) / (2.0 * total * math.log(2))  # Miller-Madow, in bits


# ---------------------------------------------------------------------------
# Multi-scale temporality
# ---------------------------------------------------------------------------


def _coarse_grain(L: Sequence[float], w: int) -> List[float]:
    """Non-overlapping window means: coarsen the length series to scale w."""
    return [fmean(L[i:i + w]) for i in range(0, len(L) - w + 1, w)]


def multiscale_burstiness(text: str, scales: Sequence[int] = (1, 2, 4)) -> dict:
    """Burstiness B_n at multiple scales (sentence, then coarse-grained windows).

    A long-range-bursty document keeps dispersion at coarse scales; one whose
    variation is only fine-scale (e.g. mechanical alternation) washes out toward
    B_n = -1 at scale >= 2. Returns {"B_n@1":.., "B_n@2":.., "B_n@4":..}, None where
    fewer than 2 windows exist. This is the multi-scale measure that is estimable at
    generation lengths (about 30 sentences); a formal long-range exponent (DFA / Hurst,
    the 1/f signature used in the Drozdz/Kwapien stylometry work) needs series of 50+
    sentences to be stable and is not computed here.
    """
    from experiments.burstiness_metrics import sentence_lengths
    L = [float(x) for x in sentence_lengths(text)]
    out = {}
    for s in scales:
        cg = list(L) if s == 1 else _coarse_grain(L, s)
        out[f"B_n@{s}"] = burstiness_B_corrected(cg) if len(cg) >= 2 else None
    return out


# ---------------------------------------------------------------------------
# Syntactic complexity (not implemented)
# ---------------------------------------------------------------------------


def syntactic_complexity(text: str):
    """Placeholder for a syntactic-complexity dimension; not implemented.

    The measures above cover sentence-length rhythm and punctuation, not clause
    structure, subordination or parse-tree shape, which would need a parser and so
    would not be model-free. Raises NotImplementedError rather than returning a
    placeholder value, so nothing treats it as implemented.
    """
    raise NotImplementedError("syntactic complexity is not implemented")


# ---------------------------------------------------------------------------
# Text entry point
# ---------------------------------------------------------------------------


def burstiness_pp(text: str) -> dict:
    """Structural burstiness of a document (model-free; reuses the repo segmenter).

    Burstiness B = [B_n (= B_corrected), M]; Gini is the robustness check; multiscale
    gives the temporal profile. Segmentation diversity is a separate axis
    (segmentation_entropy); predictability (mean_surprisal, fluc) is kept outside the
    construct, not computed here; var_L is a diagnostic. ``reliable`` is False below
    RELIABILITY_FLOOR sentences.
    """
    from experiments.burstiness_metrics import sentence_lengths

    L = [float(x) for x in sentence_lengths(text)]
    n = len(L)
    return {
        "B_n": burstiness_B_corrected(L),     # primary dispersion (Goh-Barabasi/Kim-Jo)
        "M": memory_M_corrected(L),           # lag-1 clumping, bias-corrected
        "gini": gini(L),                      # robustness check (Gini 1912)
        "multiscale": multiscale_burstiness(text),
        "n_sentences": n,
        "reliable": n >= RELIABILITY_FLOOR,   # short-doc floor guard
        "sentence_lengths": [int(x) for x in L],
    }


# ---------------------------------------------------------------------------
# Self-check: B must order metronome < poisson < bursty, and B(metronome) == -1
# ---------------------------------------------------------------------------


def demo() -> int:
    metronome = [10.0] * 30                       # sigma = 0  -> B = -1
    # sigma ~= mu  -> B ~= 0 : a sequence whose std approximately equals its mean
    poisson = [1, 1, 2, 1, 3, 1, 1, 5, 2, 1, 4, 1, 1, 2, 8, 1, 1, 3, 1, 6,
               1, 2, 1, 1, 4, 1, 1, 2, 1, 9]
    poisson = [float(x) for x in poisson]
    bursty = [2.0] * 27 + [120.0, 140.0, 160.0]   # long lulls, rare huge bursts

    b_met = burstiness_B(metronome)
    b_poi = burstiness_B(poisson)
    b_bur = burstiness_B(bursty)

    print(f"{'sequence':<12}{'B':>10}{'B_n':>10}{'M':>10}")
    print("-" * 42)
    for name, seq in [("metronome", metronome), ("poisson", poisson), ("bursty", bursty)]:
        b = burstiness_B(seq)
        bn = burstiness_B_corrected(seq)
        m = memory_M(seq)
        fmt = lambda v: f"{v:>10.3f}" if v is not None else f"{'None':>10}"  # noqa: E731
        print(f"{name:<12}{fmt(b)}{fmt(bn)}{fmt(m)}")

    # ordering + bounds. NOTE: at small n, B is capped at a CV of sqrt(n-1), so a
    # short bursty doc cannot reach +1 (n=30 caps a single spike near 0.69). This
    # finite-n ceiling is the bias Kim & Jo's B_n corrects, hence B_n > B here.
    assert b_met is not None and abs(b_met - (-1.0)) < 1e-9, f"metronome B != -1: {b_met}"
    assert b_poi is not None and abs(b_poi) < 0.25, f"poisson B not near 0: {b_poi}"
    assert b_bur is not None and b_bur > 0.4, f"bursty B not clearly positive: {b_bur}"
    assert b_met < b_poi < b_bur, "B failed to order metronome < poisson < bursty"
    assert burstiness_B_corrected(bursty) > b_bur, "B_n should lift the small-n bias"
    for seq in (metronome, poisson, bursty):
        b = burstiness_B(seq)
        assert -1.0 - 1e-9 <= b <= 1.0 + 1e-9, f"B out of [-1,1]: {b}"

    # memory sign: a clumped (sorted) sequence has M>0; an alternating one M<0
    clumped = [1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0]
    alternating = [1.0, 9.0] * 6
    assert memory_M(clumped) > 0, "clumped sequence should have positive memory"
    assert memory_M(alternating) < 0, "alternating sequence should have negative memory"

    # Goodhart guard: synthetic length patterns via mk(); natural text is real prose.
    def mk(lengths):
        return " ".join(" ".join(["word"] * int(k)) + "." for k in lengths)
    g_natural = (
        "The market opened at dawn. Vendors stacked crates of fruit and called their prices. "
        "The smell of bread drifted from the corner bakery. A boy chased a dog between the stalls. "
        "Two women haggled over a basket of plums. The fishmonger laid out his catch on crushed ice. "
        "Coins changed hands. Somewhere a radio played an old song. By noon the crowd had thinned, "
        "and the vendors began to pack away what they had not sold. The square grew quiet again. "
        "A cat slept in a patch of sun. It was an ordinary morning, much like the one before it."
    )
    assert gaming_flags(g_natural)["gate_ok"], gaming_flags(g_natural)
    # each degenerate pattern raises its named flag:
    assert gaming_flags(mk([4] * 12 + [120]))["flags"]["outlier_driven"]
    assert gaming_flags(mk([3] * 24 + [90] * 3))["flags"]["low_len_diversity"]   # 2+ equal outliers
    assert gaming_flags(mk([3, 3, 80] * 6))["flags"]["low_len_diversity"]        # period-3 cycle
    assert gaming_flags(mk([4, 28] * 8))["flags"]["alternating"]                 # 2-cycle
    assert gaming_flags(mk([4] * 8 + [28] * 8))["flags"]["over_clumped"]         # block clumping
    assert gaming_flags(mk([1, 1, 2] * 4 + [20, 20]))["flags"]["fragment_spam"]  # >=30% fragments
    g_runon = " ".join(["one, two, three, four, five, six, seven, eight, nine, ten." for _ in range(3)])
    assert gaming_flags(g_runon)["flags"]["runon"], gaming_flags(g_runon)
    for atk in (mk([3] * 24 + [90] * 3), mk([3, 3, 80] * 6), mk([4, 28] * 8)):
        assert not gaming_flags(atk)["ok"]
    assert abs(gini([5.0, 5.0, 30.0, 30.0]) - 0.476) < 0.01   # sample-Gini denominator n*(n-1)
    assert segmentation_entropy("A. B! C? D, e; f.") >= 0.0   # Miller-Madow corrected, finite
    # quote style must not change segmentation entropy
    assert (segmentation_entropy('He said, "yes," and left. It ended.')
            == segmentation_entropy('He said, “yes,” and left. It ended.'))

    # multi-scale: long-range-bursty keeps coarse-scale dispersion; alternation washes out
    ms_blocky = multiscale_burstiness(mk([5, 5, 5, 5, 30, 30, 30, 30] * 2))
    ms_altern = multiscale_burstiness(mk([5, 30] * 8))
    assert ms_blocky["B_n@2"] > ms_altern["B_n@2"], (ms_blocky, ms_altern)

    # short-doc floor flag + Gini robustness check present
    assert burstiness_pp(mk([8] * 3 + [12] * 9))["reliable"] is True
    assert burstiness_pp(mk([8, 12, 6, 14, 9]))["reliable"] is False
    assert 0.0 <= gini([5.0, 5.0, 30.0, 30.0]) <= 1.0

    # syntactic complexity raises rather than returning a placeholder value
    try:
        syntactic_complexity("x")
        raise AssertionError("syntactic_complexity should raise NotImplementedError")
    except NotImplementedError:
        pass

    print("\nself-check PASS: B orders metronome<poisson<bursty, B(metronome)=-1, M sign "
          "correct; guard flags gaming; multiscale separates long-range vs alternation; "
          "reliability floor + Gini + syntactic-complexity guard wired.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Point-process burstiness (B, B_n, M).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--text", help="document text to analyze")
    g.add_argument("--demo", action="store_true", help="run the self-check")
    args = p.parse_args(argv)
    if args.demo or (args.text is None):
        return demo()
    print(json.dumps(burstiness_pp(args.text), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
