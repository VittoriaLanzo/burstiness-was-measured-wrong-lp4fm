"""Single source of truth for the benchmark prompts: the 200 titles and the 5 modes.

The wording is finalized (PROMPTS_FINALIZED = True): the 5 modes are locked, and each mode
is a cited method (provenance map below). Real runs proceed without --force; they need API
keys and SDKs (see README).

The benchmark fetches the 200 news headlines (frozen HF dataset) and generates an article
under each of 5 non-overlapping prompting modes. A mode is a function (gen, brief) -> final
article text, where gen(user) is one model completion; most modes make one call, self_refine
makes three. Every prompt lives in this module and nowhere else, so the generation lanes
cannot drift apart.

# --- LOCKED CONDITION SET: mechanism coverage + provenance ------------------
# The 5 modes cover the 5 orthogonal mechanisms a prompt can use to move style, spanning all 6
# families of The Prompt Report (Schulhoff et al. 2024, arXiv 2406.06608). One representative
# each; coverage is at the MECHANISM level -- the fairness guarantee (no cherry-picked
# techniques). Where a canonical template exists it is adopted; where only the method exists the
# wording is adapted to news register. Every mode is a cited, recognized technique:
#   control      void baseline (unprompted floor; no technique)
#   instruction  M1 tell:        Persona Pattern (White et al. 2023,  arXiv 2302.11382)
#   fewshot      M2 show:        few-shot / ICL  (Brown 2020 arXiv 2005.14165; Min 2022 arXiv 2202.12837)
#   plan         M3 plan-before: plan-then-write (Wang et al. 2023 Plan-and-Solve, arXiv 2305.04091)
#   self_refine  M4 fix-after:   Self-Refine     (Madaan et al. 2023, arXiv 2303.17651)
# best_of_N (M5 select) is NOT a mode: picking the highest-B_n of N optimizes the metric
# (Goodhart). Report it only as a labeled, compute-matched upper bound.
# Design support: Lost-in-the-Middle (Liu et al. 2024, arXiv 2307.03172) -> the TAIL re-anchor and
# the exemplar ordering; Huang et al. 2024 (arXiv 2310.01798) -> the M4 self-leniency guard.
# METRIC-FREE WORDING: no prompt names the statistic or gives a target on it (no B_n /
# dispersion, no "vary sentence length", no clumping/M, no numbers). The few-shot and
# self-refine prompts do describe the style in plain words (rhythm, pacing, cadence), as a
# user asking for human-sounding prose would; the self-check below enforces the stricter rule only. The metric is
# model-free point-process burstiness B_n (Goh-Barabasi 2008 / Kim-Jo 2016); human news prose sits
# at B_n ~ -0.4 and B_n is genre-relative, so the voice is a news journalist, not an essayist.
# Decoding is held constant across modes (temp 0.7 where the API allows; see providers).
"""
from __future__ import annotations

import json
import pathlib

# The 5 modes are locked; each is grounded in a cited method (see the map above).
PROMPTS_FINALIZED = True

# --- data: the 200 titles ---------------------------------------------------
HF_DATASET = "ANONYMIZED/paper-dataset"
HF_FILE = "prompts_200_human.jsonl"
HF_REVISION = None  # pin a commit sha here for the frozen run (reproducibility)
ROUND_TO = 25       # round the head-to-head length hint so it reads natural ("about 275 words")

BRIEF = ("Write a news article with the following headline:\n\n"
         "{headline}\n\nAim for about {target_words} words.")


def load_briefs():
    """The 200 neutral briefs: titles from the frozen HF dataset, head-to-head length.

    target_words = the paired human article's whitespace word count (the metric's own rule),
    rounded to nearest ROUND_TO. The brief carries only task + genre + length: no style words.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_DATASET, HF_FILE, repo_type="dataset", revision=HF_REVISION)
    briefs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            tw = max(ROUND_TO, round(r["human_words"] / ROUND_TO) * ROUND_TO)
            briefs.append({"id": r["id"], "brief": BRIEF.format(headline=r["headline"], target_words=tw)})
    return briefs


# --- prompt text (metric-free; news register) ---------------------------------
VOICE = (  # Persona Pattern (White et al. 2023, arXiv:2302.11382); metric-free by construction
    "Act as a professional news journalist. Provide the article a seasoned staff reporter "
    "would write: concrete and specific, in a natural human voice, never flat or mechanical."
)
# TAIL re-anchors the directive near the end: instruction adherence decays over long outputs
# (Liu et al. 2024, Lost in the Middle, arXiv:2307.03172).
TAIL = "Keep that natural, human quality all the way through, including the closing paragraphs."
# Plan-then-write / decomposition (Wang et al. 2023 Plan-and-Solve, arXiv:2305.04091); adapted
# to journalism's 5 W's / inverted pyramid because no news-specific plan template exists.
PLAN = (
    "Before you write, plan the article the way a reporter does: pick the lede (the single most "
    "important fact), then lay out the who, what, when, where, why, and how, and the order to "
    "present them, most important first. Work that outline out for yourself, then write the "
    "article to it. Do not show your planning; return only the finished article."
)
FEWSHOT_LEAD = (
    "Study the voice, pacing, and rhythm of the examples below, then write in the same spirit. "
    "Borrow only their style, never their facts or wording.\n\n<examples>\n"
)
FEWSHOT_TAIL = "\n</examples>"


def _load_exemplars():
    """The frozen few-shot triple from bench/m2_exemplars.json (written by select_m2_exemplars).

    The exemplars are representative straight news reports, selected by register/topic and
    centered on the median of the human B_n distribution, not chosen for high burstiness, since
    selecting shots by the measured axis would leak it into the few-shot condition (few-shot
    basis: Brown 2020 arXiv:2005.14165, Min 2022 arXiv:2202.12837). Returns "" when the file
    does not exist, which leaves the fewshot mode without exemplars.
    """
    p = pathlib.Path(__file__).parent / "m2_exemplars.json"
    if not p.exists():
        return ""
    data = json.loads(p.read_text(encoding="utf-8"))
    return "".join(f"<example>\n{e['text']}\n</example>\n" for e in data["exemplars"])


EXEMPLARS = _load_exemplars()
# Feedback step (Madaan et al. 2023 Self-Refine, arXiv:2303.17651): the init->feedback->refine
# structure with qualitative feedback. The reference implementation's numeric 10-trait rubric is
# dropped because it would introduce numeric targets. Requiring three named weaknesses even on a
# strong draft is the self-leniency guard (cf. Huang et al. 2024, arXiv:2310.01798).
FEEDBACK = (
    "You are a demanding editor. Critique the draft below only on how naturally human its prose "
    "reads: its rhythm, pacing, and flow. Name at least THREE specific passages that read as "
    "flat, monotonous, or mechanical. For each, quote the exact sentence(s) and say concretely "
    "what to change to give it a more human cadence. Do not rewrite the draft; give only the "
    "critique. Even if the writing is already strong, still name the three weakest spots.\n\n"
    "DRAFT:\n{draft}"
)
REWRITE = (
    "Here is your draft and an editor's critique.\n\nDRAFT:\n{draft}\n\nCRITIQUE:\n{feedback}\n\n"
    "Rewrite the article in full, applying every point of the critique. This is a substantial "
    "rewrite, not a light edit: change how the sentences move wherever the critique flagged "
    "flatness, while preserving the meaning, the facts, and the target length. Return only the "
    "rewritten article."
)


# --- modes: (gen, brief) -> final article text ------------------------------
def _wrap(body, brief):
    """The shared VOICE skeleton; body is the mode-specific middle (empty for plain instruction)."""
    return f"{VOICE}\n\n{body}\n\n{brief}\n\n{TAIL}" if body else f"{VOICE}\n\n{brief}\n\n{TAIL}"


def control(gen, brief):
    return gen(brief)


def instruction(gen, brief):
    return gen(_wrap("", brief))


def fewshot(gen, brief):
    return gen(_wrap(FEWSHOT_LEAD + EXEMPLARS + FEWSHOT_TAIL, brief))


def plan(gen, brief):
    return gen(_wrap(PLAN, brief))


def self_refine(gen, brief):
    draft = gen(_wrap("", brief))  # identical to the instruction draft, on purpose
    fb = gen(FEEDBACK.replace("{draft}", draft))
    return gen(REWRITE.replace("{draft}", draft).replace("{feedback}", fb))


MODES = {
    "control": control,
    "instruction": instruction,
    "fewshot": fewshot,
    "plan": plan,
    "self_refine": self_refine,
}


def preview(mode, brief):
    """The first user prompt `mode` would send, for --dry-run inspection (no model calls)."""
    captured = {}

    def gen(user):
        captured.setdefault("first", user)
        return "[draft placeholder]"

    MODES[mode](gen, brief)
    return captured["first"]


def _selfcheck():
    brief = BRIEF.format(headline="Council approves new bridge after decade of delays", target_words=300)
    assert preview("control", brief) == brief
    for mode in MODES:
        assert "Write a news article" in preview(mode, brief), mode
    # self_refine must make exactly three model calls (draft, critique, rewrite)
    calls = []
    self_refine(lambda u: calls.append(u) or "x", brief)
    assert len(calls) == 3, f"self_refine should make 3 calls, made {len(calls)}"
    # metric-free guard: no statistic or measured-axis term in any prompt text (plain style
    # words such as "rhythm" or "pacing" are allowed and do appear: see the comment at the top)
    forbidden = ["sentence length", "sentence-length", "variance", "std", "b_n",
                 "clumping", "standard deviation", "burstiness"]
    blob = " ".join([VOICE, TAIL, PLAN, FEEDBACK, REWRITE, FEWSHOT_LEAD, BRIEF]).lower()
    for w in forbidden:
        assert w not in blob, f"prompt text leaks metric term: {w!r}"
    print("prompts self-check PASS: 5 modes, self_refine=3 calls, metric-free text.")


if __name__ == "__main__":
    _selfcheck()
