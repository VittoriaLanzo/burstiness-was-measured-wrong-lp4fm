#!/usr/bin/env python3
"""Sentence segmentation, punctuation helpers, and the raw-moment baseline (stdlib-only).

sentence_lengths() and _PUNCT_SYMBOLS are the segmenter every metric in this repo uses
(experiments/burstiness_pp.py imports them). The rest of the module is the earlier raw-moment
vector, kept as the diagnostic baseline the paper compares against; var(L) here is the
length-confounded statistic that B_n replaces. It is not the paper's construct.

The earlier vector:

    B = [ var(L), kurt(L), fluc(S), punct_entropy ]

where
    L            = the sequence of sentence lengths (in whitespace tokens)
    var(L)       = variance of sentence length        -- spread of the rhythm
    kurt(L)      = excess kurtosis of sentence length  -- tail / burst behavior
    fluc(S)      = surprisal fluctuation across the doc (DivEye-style, 2509.18880)
    punct_entropy= Shannon entropy of the punctuation / segmentation pattern
                   (timing-free proxy; see Onegin symbolic time-series 2604.20221)

Human writing has rhythm: sentence length and complexity vary in bursts.
LLM text is rhythmically flat ("average prosody" for text). var(L), kurt(L)
and fluc(S) all collapse toward small values for flat text. punct_entropy
collapses when segmentation becomes monotone (every sentence ends in '.').

fluc(S) here is a model-free proxy (stdev of sentence-length first-differences / mean
length); the real DivEye quantity needs per-token surprisal from a model, which this module
deliberately does not load.

CLI:
    python3 experiments/burstiness_metrics.py --text "Some text here."
    python3 experiments/burstiness_metrics.py --file path/to/doc.txt
    python3 experiments/burstiness_metrics.py --demo
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from typing import Dict, List

# ---------------------------------------------------------------------------
# Sentence + token segmentation
# ---------------------------------------------------------------------------

# A small abbreviation set so a period after "Mr" / "e.g" does not falsely
# split a sentence. This is intentionally lightweight (stdlib-only, no NLTK).
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "fig", "no", "vol", "cf", "al", "inc", "ltd", "co",
}

# Split candidates: sequences of sentence-final punctuation [.!?] optionally
# followed by closing quotes/brackets, then whitespace before the next char.
_SENT_SPLIT = re.compile(r'([.!?]+["\')\]]?)\s+')

# Symbols counted for the punctuation/segmentation entropy.
_PUNCT_SYMBOLS = re.compile(r'[.!?,;:\-\u2014()"\']')

# Typographic quotes are mapped to ASCII before segmentation. The split pattern's
# closing-quote class is ASCII-only, so without the mapping a sentence ending in a curly
# closing quote (.” followed by a space) is not split from the next one. The human
# baseline uses straight quotes while several API models emit curly ones, so the mapping
# keeps segmentation identical on both sides.
_TYPO_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})


def sentence_lengths(text: str) -> List[int]:
    """Return the list of sentence lengths (whitespace-token counts).

    Sentences are split on runs of ``[.!?]`` followed by whitespace, with a
    basic abbreviation guard so common abbreviations (Mr., e.g.) do not split
    a sentence. Tokens are whitespace-delimited runs. Empty/whitespace-only
    input returns ``[]``; a single unterminated sentence returns a one-element
    list so callers never crash on degenerate input.

    Args:
        text: raw document text.

    Returns:
        list[int]: token count for each detected sentence (empty if no text).
    """
    if not text or not text.strip():
        return []
    text = text.translate(_TYPO_QUOTES)

    sentences: List[str] = []
    last = 0
    for m in _SENT_SPLIT.finditer(text):
        end = m.end()
        candidate = text[last:end].strip()
        # Abbreviation guard: if the chunk ends on a known abbreviation, do not
        # treat this period as a sentence boundary -- glue it to the next chunk.
        bare = candidate.rstrip('.!?"\')]').strip()
        last_word = bare.split()[-1].lower() if bare.split() else ""
        if last_word in _ABBREVIATIONS:
            continue
        if candidate:
            sentences.append(candidate)
        last = end
    tail = text[last:].strip()
    if tail:
        sentences.append(tail)

    lengths = [len(s.split()) for s in sentences if s.split()]
    return lengths


# ---------------------------------------------------------------------------
# Individual metric components
# ---------------------------------------------------------------------------

def var_L(lengths: List[int]) -> float:
    """Population variance of sentence length.

    Variance is the most basic burstiness signal: flat LLM rhythm collapses
    var(L) toward 0; human bursts inflate it. Returns 0.0 for fewer than two
    sentences (no spread is defined).
    """
    if len(lengths) < 2:
        return 0.0
    return statistics.pvariance(lengths)


def kurt_L(lengths: List[int]) -> float:
    """Excess kurtosis of sentence length (tail / burst behavior).

    Excess kurtosis = (m4 / m2**2) - 3, where m2/m4 are the 2nd/4th central
    moments. Positive excess kurtosis = heavy tails (occasional very long or
    very short bursts), the hallmark of human rhythm. Returns 0.0 when fewer
    than two sentences exist or variance is ~0 (kurtosis undefined for a
    degenerate distribution).
    """
    n = len(lengths)
    if n < 2:
        return 0.0
    mean = statistics.fmean(lengths)
    m2 = sum((x - mean) ** 2 for x in lengths) / n
    if m2 == 0:
        return 0.0
    m4 = sum((x - mean) ** 4 for x in lengths) / n
    return (m4 / (m2 ** 2)) - 3.0


def fluc_S_proxy(lengths: List[int]) -> float:
    """Model-free proxy for DivEye-style surprisal fluctuation fluc(S).

    The real fluc(S) (DivEye, arXiv:2509.18880) is the fluctuation of per-token
    surprisal (negative log-prob) across the document, and requires a language
    model. This module is stdlib-only, so the proxy is computed from the rhythm
    signal already available: the standard deviation of the first-differences of
    sentence length, normalized by mean sentence length.

        d_i = L_{i+1} - L_i      (how much the rhythm jumps between sentences)
        fluc_S_proxy = stdev(d) / mean(L)

    This measures how much the local rhythm changes, the same intuition surprisal
    fluctuation captures at the token scale. It is a proxy (see the module
    docstring). Returns 0.0 with fewer than two sentences (no first-difference is
    defined).
    """
    if len(lengths) < 2:
        return 0.0
    diffs = [lengths[i + 1] - lengths[i] for i in range(len(lengths) - 1)]
    if len(diffs) < 2:
        # A single difference has no spread; report magnitude/mean instead.
        mean_len = statistics.fmean(lengths) or 1.0
        return abs(diffs[0]) / mean_len
    spread = statistics.pstdev(diffs)
    mean_len = statistics.fmean(lengths) or 1.0
    return spread / mean_len


def punct_entropy(text: str) -> float:
    """Shannon entropy (bits) over punctuation / segmentation symbols.

    This is the timing-free segmentation proxy (cf. the Onegin symbolic time-series work,
    arXiv:2604.20221). Human prose mixes commas,
    semicolons, dashes, question and exclamation marks; flat LLM prose tends
    toward a monotone '.'-only segmentation. We count the distribution of
    punctuation symbols and return its Shannon entropy:

        H = -sum_i p_i * log2(p_i)

    A monotone-punctuation document has H -> 0; a varied one has higher H.
    Returns 0.0 when no punctuation symbols are present.
    """
    counts: Dict[str, int] = {}
    for ch in _PUNCT_SYMBOLS.findall(text or ""):
        counts[ch] = counts.get(ch, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# Composite vector
# ---------------------------------------------------------------------------

def burstiness_vector(text: str, surprisal_fn=None) -> dict:
    """Compute the full burstiness vector B for a document.

    Returns a dict with the four B components plus diagnostics:

        {
          "var_L": float,
          "kurt_L": float,
          "fluc_S": float,          # real if surprisal_fn given, else == fluc_S_proxy
          "fluc_S_proxy": float,
          "punct_entropy": float,
          "n_sentences": int,
          "sentence_lengths": list[int],
          "fluc_S_is_proxy": bool,
          "fluc_S_note": str,
        }

    The ``var_L`` / ``kurt_L`` / ``fluc_S`` / ``punct_entropy`` keys are the
    raw-moment burstiness vector B = [var(L), kurt(L), fluc(S), punct_entropy].
    The quality gate (2504.12522) is applied outside this function: B is only
    counted for a generation that has already passed a coherence threshold, so
    high-variance gibberish never wins.

    ``surprisal_fn``: optional callable ``str -> float`` returning the real
    DivEye-style surprisal fluctuation of the document. When given, ``fluc_S``
    holds that real value and ``fluc_S_is_proxy`` is False. When omitted, the
    function stays pure standard library and ``fluc_S`` falls back to the
    model-free proxy. Any callable str -> float can be passed (none ships with this repo).
    """
    lengths = sentence_lengths(text)
    proxy = fluc_S_proxy(lengths)
    if surprisal_fn is not None:
        real = float(surprisal_fn(text))
        fluc, is_proxy, note = real, False, (
            "Real per-token surprisal fluctuation from a language model "
            "(DivEye-style, 2509.18880). Proxy retained in fluc_S_proxy."
        )
    else:
        fluc, is_proxy, note = proxy, True, (
            "Model-free proxy (stdev of sentence-length first-differences / "
            "mean length). Real DivEye fluc(S) needs per-token surprisal from "
            "a model; pass surprisal_fn."
        )
    return {
        "var_L": var_L(lengths),
        "kurt_L": kurt_L(lengths),
        "fluc_S": fluc,
        "fluc_S_proxy": proxy,
        "punct_entropy": punct_entropy(text),
        "n_sentences": len(lengths),
        "sentence_lengths": lengths,
        "fluc_S_is_proxy": is_proxy,
        "fluc_S_note": note,
    }


# ---------------------------------------------------------------------------
# Demo samples (human-like high variance vs LLM-like flat)
# ---------------------------------------------------------------------------

# High-variance, human-like: short punchy sentences mixed with long winding
# ones, varied punctuation (dashes, semicolons, question marks).
_DEMO_HUMAN = (
    "Rain. It came down hard, harder than anyone in the small fishing town "
    "of Abermaw had seen in a generation, and the old men gathered at the "
    "harbor wall to watch the boats strain against their ropes. Nobody spoke. "
    "Then Maggie laughed -- a sharp, surprised sound -- because the dog had "
    "leapt straight into the churning grey water after a gull, and now it "
    "paddled back, triumphant and ridiculous, a feather still clamped in its "
    "jaws; the men roared. Was it foolish? Of course. But it was the first "
    "joy in weeks."
)

# Flat, LLM-like: uniform medium-length declarative sentences, monotone
# '.'-only punctuation, no rhythmic bursts.
_DEMO_LLM = (
    "The weather in the town was very rainy on that particular day. The local "
    "residents gathered near the harbor to observe the situation closely. The "
    "boats were tied securely to prevent any potential damage there. A dog "
    "decided to jump into the water to chase after a bird. The dog then "
    "returned to the shore while holding a feather in place. The people who "
    "were present found the entire scene to be quite amusing. It was a "
    "memorable moment for everyone in the community that day."
)


def _print_demo() -> None:
    """Print the high-variance vs flat vectors side by side for contrast."""
    human = burstiness_vector(_DEMO_HUMAN)
    llm = burstiness_vector(_DEMO_LLM)
    keys = ["var_L", "kurt_L", "fluc_S_proxy", "punct_entropy", "n_sentences"]

    print("Burstiness vector B -- demo contrast")
    print("(fluc_S is a model-free proxy; see --help)")
    print()
    header = f"{'metric':<16}{'human-like':>16}{'LLM-like flat':>16}"
    print(header)
    print("-" * len(header))
    for k in keys:
        hv = human[k]
        lv = llm[k]
        hs = f"{hv:.4f}" if isinstance(hv, float) else str(hv)
        ls = f"{lv:.4f}" if isinstance(lv, float) else str(lv)
        print(f"{k:<16}{hs:>16}{ls:>16}")
    print()
    print("Full JSON vectors:")
    print("human-like:", json.dumps(human))
    print("LLM-like:  ", json.dumps(llm))
    print()
    verdict = "PASS" if human["var_L"] > llm["var_L"] else "FAIL"
    print(f"Contrast check (human var_L > flat var_L): {verdict} "
          f"({human['var_L']:.4f} > {llm['var_L']:.4f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute the burstiness vector B = [var(L), kurt(L), "
                    "fluc(S) proxy, punct_entropy] over a text document.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", help="literal text to analyze")
    group.add_argument("--file", help="path to a UTF-8 text file to analyze")
    group.add_argument("--demo", action="store_true",
                       help="run the built-in human-like vs flat contrast demo")
    args = parser.parse_args(argv)

    if args.demo:
        _print_demo()
        return 0

    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            text = fh.read()
    elif args.text is not None:
        text = args.text
    else:
        parser.error("provide one of --text, --file, or --demo")
        return 2  # unreachable; parser.error exits

    print(json.dumps(burstiness_vector(text), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
