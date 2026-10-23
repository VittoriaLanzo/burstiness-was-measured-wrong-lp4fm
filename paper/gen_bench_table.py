#!/usr/bin/env python3
"""Write every benchmark number the LP4FM paper prints, from the committed result JSONs.

Outputs go to paper/lp4fm/generated/ and main.tex \\input's them:
  numbers.tex            \\val{key} macros for every number quoted in the prose
  tab_ninefam.tex        Table 1: B_n, nine models x five modes, plus the dagger-marked re-run row
  tab_family_grids.tex   Appendix B: one four-dimension grid per leg (Qwen first)
  tab_equivalence.tex    instruction-mode significance and equivalence, every leg
  tab_contrasts.tex      direct mode-vs-mode paired tests within a leg
  tab_variance.tex       raw var(L) beside B_n
  tab_audit.tex          segmenter-fix and ten-sentence-floor sensitivity, guard rates, lengths
  tab_guard.tex          B_n per model and mode, main scoring beside guard-filtered scoring

Nothing is hand-typed: rerun after any re-score and rebuild the paper.

    python3 paper/gen_bench_table.py
"""
from __future__ import annotations

import json
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
BENCH = RES / "benchmark"
OUT = ROOT / "paper" / "lp4fm" / "generated"

# model_id -> (macro key, display name); table order = control-mode gap, Grok pair last
FAMILIES = {
    "gemini-3.1-pro": ("gemini", "Gemini 3.1 Pro"),
    "glm-5.2": ("glm", "GLM-5.2"),
    "opus-4.8": ("opus", "Claude Opus 4.8"),
    "qwen3.6-27b": ("qwen", "Qwen3.6-27B"),
    "kimi-k2.6": ("kimi", "Kimi K2.6"),
    "gemma-3-27b": ("gemma", "Gemma 3 27B"),
    "deepseek-v4-pro": ("deepseek", "DeepSeek V4 Pro"),
    "deepseek-v4-pro-1018": ("deepseek1018", "DeepSeek V4 Pro$^{\\dagger}$"),
    "grok-4.3": ("grok43", "Grok 4.3"),
    "grok-4.5": ("grok45", "Grok 4.5"),
}
GROK_PAIR = ["grok-4.3", "grok-4.5"]
# Later-snapshot replications of a leg already in the benchmark: in every table and figure,
# outside the nine-model counts.
REPLICATES = {"deepseek-v4-pro-1018": "deepseek-v4-pro"}
# Each leg's generations in burstiness-engine/sentence-length-dispersion, as keyed in
# experiments/results/generation_hygiene.json.
HYGIENE_FILES = {
    "gemini-3.1-pro": "gemini31-news-gen/gemini-3.1-pro.jsonl", "glm-5.2": "glm52-news-gen/glm-5.2.jsonl",
    "opus-4.8": "opus48-news-gen/opus-4.8.jsonl", "qwen3.6-27b": "qwen36-news-gen/qwen3.6-27b.jsonl",
    "kimi-k2.6": "kimi26-news-gen/kimi-k2.6.jsonl", "gemma-3-27b": "gemma3-news-gen/gemma-3-27b.jsonl",
    "deepseek-v4-pro": "deepseek-v4-news-gen/deepseek-v4.jsonl",
    "deepseek-v4-pro-1018": "deepseek-v4-news-gen/benchmark-2026-10-18/deepseek-v4.jsonl",
    "grok-4.3": "grok43-news-gen/grok-4.3.jsonl", "grok-4.5": "grok45-news-gen/grok-4.5.jsonl",
}
MODELS = [m for m in FAMILIES if m not in REPLICATES]
MODES = [("control", "Control"), ("instruction", "Instruction"), ("fewshot", "Few-shot"),
         ("plan", "Plan"), ("self_refine", "Self-refine")]
DIMS = [("B_n", "Bn", "$B_n$"), ("gini", "gini", "Gini"), ("seg_entropy", "seg", "Seg.\\ entropy"),
        ("M", "M", "Memory $M$")]
MARGINS = [0.03, 0.05, 0.08]
WORDS = "none one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
        "sixteen seventeen eighteen nineteen twenty".split()
TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}

# Family-specific caption notes for the appendix grids (facts about the run, not results).
GRID_NOTES = {
    "glm-5.2": "Silent-refusal empty cells are excluded by the seed-averaging protocol (Section~\\ref{sec:ninefam}).",
    "kimi-k2.6": "Provider content-filter refusals are excluded by the seed-averaging protocol (Section~\\ref{sec:ninefam}).",
    "opus-4.8": "Ran at effort high with a 32{,}000-token cap (Section~\\ref{sec:ninefam}).",
    "deepseek-v4-pro": "The leg's cap was raised from 16{,}000 to 32{,}000 tokens early in the run (Appendix~\\ref{sec:caps}).",
    "deepseek-v4-pro-1018": "The DeepSeek V4 Pro leg re-run on 18 October 2026 against the snapshot the API alias then "
                            "served, with the same prompts and briefs (Appendix~\\ref{sec:protocoldetail}); reported beside the "
                            "nine models and outside their counts.",
}


def word(n: int) -> str:
    if n < 21:
        return WORDS[n]
    t, u = divmod(n, 10)
    return TENS[t * 10] + ("" if u == 0 else "-" + WORDS[u])


def of_total(n: int, total: int) -> str:
    """'every' when all of them, else 'six of eight'."""
    if n == total == 9:
        return "every"
    return f"{word(n)} of {word(total)}"


def models_phrase(n: int) -> str:
    return {0: "no model", 1: "one model"}.get(n, f"{word(n)} models")


def join_names(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def f3(x):
    s = f"{x:.3f}"
    return "0.000" if s == "-0.000" else s


def _f3out(x, rounding):
    s = str(Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=rounding))
    return "0.000" if s == "-0.000" else s


def f3lo(x):
    """Lower interval bound printed to three decimals, rounded outward (down)."""
    return _f3out(x, ROUND_FLOOR)


def f3hi(x):
    """Upper interval bound, or a tightest margin, printed to three decimals, rounded outward (up)."""
    return _f3out(x, ROUND_CEILING)


def peq(p: float) -> str:
    """p as the paper prints it, math mode without $: p=0.002, p=3\\times10^{-5}, p<10^{-5}."""
    if p < 1e-5:
        return "p<10^{-5}"
    if p < 1e-3:
        m, e = f"{p:.0e}".split("e")
        return f"p={m}\\times10^{{{int(e)}}}"
    if p < 0.01:
        return f"p={p:.3f}"
    return f"p={p:.2f}" if round(p, 2) != 0.05 else f"p={p:.3f}"


def stars(p):
    return "^{**}" if p < 0.01 else ("^{*}" if p < 0.05 else "")


def cell(e):
    return f"${f3(e['median_delta_h_minus_ai'])}\\;({e['matched_rank_biserial']:.2f}){stars(e['wilcoxon_p'])}$"


def landing(e):
    """flat / ns / past for one median delta (human - model) and its test."""
    if e["wilcoxon_p"] >= 0.05:
        return "ns"
    return "flat" if e["median_delta_h_minus_ai"] > 0 else "past"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


class Vals(dict):
    def put(self, key, v):
        assert key not in self, f"duplicate macro key {key}"
        self[key] = str(v)

    def count(self, key, n):
        """digits, word, and capitalised word for one count."""
        self.put(key, n)
        self.put(f"{key}.w", word(n))
        self.put(f"{key}.W", word(n).capitalize())

    def stat(self, key, e, delta="median_delta_h_minus_ai"):
        d = e[delta]
        self.put(f"{key}.d", f3(d))
        self.put(f"{key}.ad", f3(abs(d)))
        self.put(f"{key}.rb", f"{e['matched_rank_biserial']:.2f}")
        self.put(f"{key}.p", peq(e["wilcoxon_p"]))
        ci = e.get("median_delta_95ci")
        if ci:
            m_star = max(abs(ci[0]), abs(ci[1]))
            self.put(f"{key}.ci", f"[{f3lo(ci[0])}, {f3hi(ci[1])}]")
            self.put(f"{key}.mstar", f3hi(m_star))
            self.put(f"{key}.hi4", f"{ci[1]:.4f}")
            cleared = [m for m in MARGINS if m_star <= m]
            self.put(f"{key}.eqphrase", f"equivalent at the ${cleared[0]}$ margin" if cleared
                     else "equivalent at none of the pre-specified margins")
        self.put(f"{key}.n", e["n_pairs"])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ssot = {m: load(BENCH / m / "seed_avg_paired.json") for m in FAMILIES}
    mult = {m: load(BENCH / m / "multiplicity.json") for m in FAMILIES}
    review = load(RES / "review_analyses.json")
    hygiene = load(RES / "generation_hygiene.json")
    v = Vals()

    # --- generation hygiene: share of control-mode generations with each formatting feature --
    hyg = {m: hygiene[HYGIENE_FILES[m]] for m in FAMILIES}
    for m, (k, _) in FAMILIES.items():
        for feat, fk in (("markdown", "md"), ("title_line", "title"), ("smart_quotes", "smart")):
            v.put(f"hyg.{k}.{fk}", f"{round(100 * hyg[m][feat])}\\%")
    # Section 4.2 and Appendix D.6: every model flagged on both features (above the guard's 5% warn
    # rate), DeepSeek V4 Pro the least formatted of the nine, Gemini 3.1 Pro at 100% on both.
    assert all(min(hyg[m]["markdown"], hyg[m]["title_line"]) > 0.05 for m in MODELS)
    for feat in ("markdown", "title_line"):
        assert min(MODELS, key=lambda m: hyg[m][feat]) == "deepseek-v4-pro", feat
    assert hyg["gemini-3.1-pro"]["markdown"] == hyg["gemini-3.1-pro"]["title_line"] == 1.0

    # --- per-cell stats -------------------------------------------------------------------
    for m, (k, _) in FAMILIES.items():
        for dim, dk, _ in DIMS:
            for mode, _ in MODES:
                v.stat(f"{k}.{dk}.{mode}", ssot[m]["dims"][dim][mode])
        cons = ssot[m].get("consistency_B_n_rank_biserial", {})
        for mode, its in cons.items():
            vals = [x for x in its.values() if x == x]
            v.put(f"{k}.cons.{mode}.min", f"{min(vals):.2f}")
            v.put(f"{k}.cons.{mode}.max", f"{max(vals):.2f}")
        v.put(f"{k}.npairs", ssot[m]["dims"]["B_n"]["control"]["n_pairs"])
    raw = load(BENCH / "qwen3.6-27b" / "seed_avg_paired.raw-archive.json")
    for dim, dk, _ in DIMS:
        for mode, _ in MODES:
            v.stat(f"qwenraw.{dk}.{mode}", raw["dims"][dim][mode])
    for mode, its in raw["consistency_B_n_rank_biserial"].items():
        v.put(f"qwenraw.cons.{mode}.min", f"{min(its.values()):.2f}")
        v.put(f"qwenraw.cons.{mode}.max", f"{max(its.values()):.2f}")

    # --- census over the 45 B_n cells -----------------------------------------------------
    cells = {(m, mode): landing(ssot[m]["dims"]["B_n"][mode]) for m in FAMILIES for mode, _ in MODES}
    for cat in ("flat", "ns", "past"):
        v.count(f"census.{cat}", sum(1 for (m, _), c in cells.items() if m in MODELS and c == cat))
    inst = {m: cells[(m, "instruction")] for m in MODELS}
    ctrl = {m: cells[(m, "control")] for m in MODELS}
    for cat in ("flat", "ns", "past"):
        v.count(f"inst.{cat}", sum(1 for c in inst.values() if c == cat))
        v.count(f"ctrl.{cat}", sum(1 for c in ctrl.values() if c == cat))
    fs_flat = sum(1 for m in MODELS if cells[(m, "fewshot")] == "flat")
    v.put("fewshot.flat.w", word(fs_flat))
    v.put("fewshot.flat.W", word(fs_flat).capitalize())
    v.put("sr.past.w", word(sum(1 for m in MODELS if cells[(m, "self_refine")] == "past")))
    ctrl_d = [ssot[m]["dims"]["B_n"]["control"]["median_delta_h_minus_ai"] for m in MODELS if ctrl[m] == "flat"]
    v.put("ctrl.d.min", f3(min(ctrl_d)))
    v.put("ctrl.d.max", f3(max(ctrl_d)))
    anypast = [FAMILIES[m][1] for m in MODELS if any(cells[(m, mode)] == "past" for mode, _ in MODES)]
    v.put("anypast.names", join_names(anypast) if anypast else "no model")
    v.put("anypast.n", len(anypast))
    npairs = [ssot[m]["dims"]["B_n"][mode]["n_pairs"] for m in MODELS for mode, _ in MODES]
    v.put("npairs.min", min(npairs))
    v.put("npairs.max", max(npairs))
    for m, (k, _) in FAMILIES.items():
        per = [ssot[m]["dims"]["B_n"][mode]["n_pairs"] for mode, _ in MODES]
        v.put(f"{k}.npairs.min", min(per))
        v.put(f"{k}.npairs.max", max(per))
        v.put(f"{k}.npairs.range", f"{min(per)}" if min(per) == max(per) else f"{min(per)}--{max(per)}")
    short = [FAMILIES[m][1] for m in MODELS if min(ssot[m]["dims"]["B_n"][mode]["n_pairs"] for mode, _ in MODES) < 200]
    v.put("npairs.short.names", join_names(short) if short else "no model")
    sd_h = review["human_spread"]["sd"]
    v.put("ctrl.sd.lo", f"{min(ctrl_d) / sd_h:.1f}")
    v.put("ctrl.sd.hi", f"{max(ctrl_d) / sd_h:.1f}")
    for mg in MARGINS:
        n_eq = sum(1 for m in MODELS
                   if max(map(abs, ssot[m]["dims"]["B_n"]["instruction"]["median_delta_95ci"])) <= mg)
        v.put(f"inst.eq{str(mg)[2:]}.w", word(n_eq))
        n_eq_sig = sum(1 for m in MODELS
                       if max(map(abs, ssot[m]["dims"]["B_n"]["instruction"]["median_delta_95ci"])) <= mg
                       and ssot[m]["dims"]["B_n"]["instruction"]["wilcoxon_p"] < 0.05)
        v.put(f"inst.eq{str(mg)[2:]}.sig.w", word(n_eq_sig))
    rb_top = sum(1 for m in MODELS if max(MODES, key=lambda md: ssot[m]["dims"]["B_n"][md[0]]["matched_rank_biserial"])[0] == "fewshot")
    med_top = sum(1 for m in MODELS if max(MODES, key=lambda md: ssot[m]["dims"]["B_n"][md[0]]["median_delta_h_minus_ai"])[0] == "fewshot")
    fs_gt_ctrl = sum(1 for m in MODELS if ssot[m]["dims"]["B_n"]["fewshot"]["median_delta_h_minus_ai"]
                     > ssot[m]["dims"]["B_n"]["control"]["median_delta_h_minus_ai"])
    v.put("fewshot.rbtop.w", word(rb_top))
    v.put("fewshot.medtop.w", word(med_top))
    v.put("fewshot.gtctrl.w", word(fs_gt_ctrl))

    # --- multiplicity ---------------------------------------------------------------------
    v.put("sig.raw", sum(mult[m]["n_raw_sig"] for m in MODELS))
    v.put("sig.bh", sum(mult[m]["n_bh"] for m in MODELS))
    v.put("sig.holm", sum(mult[m]["n_holm"] for m in MODELS))
    lost = [(m, t) for m in MODELS for t in mult[m]["tests"] if t["raw_sig_05"] and not t["bh_survives"]]
    v.put("sig.lostbh", len(lost))
    v.put("sig.lostbh.w", word(len(lost)))
    v.put("sig.lostbh.onBn", sum(1 for _, t in lost if t["metric"] == "B_n"))
    v.put("sig.lostbh.onBn.w", word(sum(1 for _, t in lost if t["metric"] == "B_n")))
    dimname = {d: l.replace("\\", "").replace("$", "").replace("Seg. entropy", "segmentation entropy").replace("Memory M", "memory") for d, _, l in DIMS}
    v.put("sig.lostbh.names", join_names([f"{FAMILIES[m][1]}'s {t['mode'].replace('_', '-')}-mode {dimname[t['metric']]}" for m, t in lost]) if lost else "none")
    v.put("fs.Mover.w", word(sum(1 for m in MODELS if ssot[m]["dims"]["M"]["fewshot"]["median_delta_h_minus_ai"] < 0
                                 and ssot[m]["dims"]["M"]["fewshot"]["wilcoxon_p"] < 0.05)))
    for m, k in (("grok-4.5", "grok45"), ("deepseek-v4-pro", "deepseek")):
        se = ssot[m]["dims"]["seg_entropy"]
        v.put(f"{k}.segabove.w", word(sum(1 for mode, _ in MODES if se[mode]["median_delta_h_minus_ai"] < 0 and se[mode]["wilcoxon_p"] < 0.05)))
    past_both = sum(1 for m in MODELS for mode, _ in MODES if cells[(m, mode)] == "past" and any(
        t["metric"] == "B_n" and t["mode"] == mode and t["bh_survives"] and t["holm_survives"] for t in mult[m]["tests"]))
    v.put("past.bothcorr.w", word(past_both))
    n_past = sum(1 for (m, _), c in cells.items() if m in MODELS and c == "past")
    fell = [f"{FAMILIES[m][1]}'s {dict(MODES)[mode].lower()} crossing" for m in MODELS for mode, _ in MODES
            if cells[(m, mode)] == "past" and not any(t["metric"] == "B_n" and t["mode"] == mode and t["bh_survives"]
                                                      and t["holm_survives"] for t in mult[m]["tests"])]
    v.put("past.corr.phrase", (f"all {word(n_past)} survive both corrections" if not fell else
                                f"{word(past_both)} survive both corrections; {join_names(fell)} "
                                f"{'falls' if len(fell) == 1 else 'fall'} under Holm"))

    # --- paraphrase probe -----------------------------------------------------------------
    para = RES / "paraphrase_robustness"
    ns = []
    for i in (1, 2, 3):
        e = load(para / f"seed_avg_paired_para-v{i}.json")["dims"]["B_n"][f"instruction_v{i}"]
        v.stat(f"para.v{i}", e)
        ns.append(e["n_pairs"])
    v.put("para.n.min", min(ns))
    v.put("para.n.max", max(ns))
    pd = [load(para / f"seed_avg_paired_para-v{i}.json")["dims"]["B_n"][f"instruction_v{i}"]["median_delta_h_minus_ai"] for i in (1, 2, 3)]
    v.put("para.d.lo", f3(min(pd)))
    v.put("para.d.hi", f3(max(pd)))
    pp = [load(para / f"seed_avg_paired_para-v{i}.json")["dims"]["B_n"][f"instruction_v{i}"]["wilcoxon_p"] for i in (1, 2, 3)]
    v.put("para.past.w", word(sum(1 for p, d in zip(pp, pd) if p < 0.05 and d < 0)))
    v.put("para.ns.w", word(sum(1 for p in pp if p >= 0.05)))
    pm = [max(map(abs, load(para / f"seed_avg_paired_para-v{i}.json")["dims"]["B_n"][f"instruction_v{i}"]["median_delta_95ci"])) for i in (1, 2, 3)]
    v.put("para.mstar.lo", f3(min(pm)))
    v.put("para.mstar.hi", f3(max(pm)))
    cleared = [mg for mg in MARGINS if max(pm) <= mg]
    v.put("para.eqphrase", f"clear only the ${cleared[0]}$ margin" if cleared and cleared[0] == MARGINS[-1]
          else (f"clear the ${cleared[0]}$ margin" if cleared else "clear none of the pre-specified margins"))
    v.put("para.ad.lo", f3(min(abs(x) for x in pd)))
    v.put("para.ad.hi", f3(max(abs(x) for x in pd)))
    canon = ssot["deepseek-v4-pro-1018"]["dims"]["B_n"]["instruction"]["median_delta_h_minus_ai"]
    v.put("para.shift.max", f3(max(abs(x - canon) for x in pd)))
    v.put("para.shift.min", f3(min(abs(x - canon) for x in pd)))
    # paired contrast paraphrase minus canonical wording on the same headlines (positive: more dispersed)
    pc = review.get("paraphrase_contrasts_B_n", {})
    pcs = [pc[f"instruction_v{i}"] for i in (1, 2, 3) if f"instruction_v{i}" in pc]
    for i, e in enumerate(pcs, 1):
        v.stat(f"para.v{i}.con", e, delta="median_delta")
    if pcs:
        v.put("para.con.d.min", f3(min(e["median_delta"] for e in pcs)))
        v.put("para.con.d.max", f3(max(e["median_delta"] for e in pcs)))
        v.put("para.con.p.max", peq(max(e["wilcoxon_p"] for e in pcs)))
        v.put("para.con.sig.w", word(sum(1 for e in pcs if e["wilcoxon_p"] < 0.05)))
        v.put("para.mstar.max", f3hi(max(max(map(abs, e["median_delta_95ci"])) for e in
                                        [load(para / f"seed_avg_paired_para-v{i}.json")["dims"]["B_n"][f"instruction_v{i}"] for i in (1, 2, 3)])))

    # --- review analyses ------------------------------------------------------------------
    hs = review["human_spread"]
    v.put("human.median", f3(hs["median"]))
    v.put("human.sd", f3(hs["sd"]))
    v.put("human.iqr", f"[{f3(hs['iqr'][0])}, {f3(hs['iqr'][1])}]")
    for mg, x in hs["margins_in_sd"].items():
        v.put(f"human.margin{mg}.sd", f"{x:.2f}")
    vb = review["variance_baseline"]["human_corr_with_mean_sentence_length"]
    v.put("var.corr.varL", f"{vb['var_L']:.2f}")
    v.put("var.corr.Bn", f"{vb['B_n']:.2f}")
    g = review["guard_rates"]
    v.put("guard.human.k", g["human"]["flagged"])
    v.put("guard.human.pct", round(100 * g["human"]["flagged"] / g["human"]["n"]))
    rates = [r for leg in g["legs"].values() for r in leg.values()]
    v.put("guard.legs.min", round(100 * min(rates)))
    v.put("guard.legs.max", round(100 * max(rates)))
    lens = {m: x for m, x in review["lengths"].items() if m != "human"}
    ratios = [c["word_ratio_median"] for x in lens.values() for c in x.values()]
    v.put("len.ratio.min", f"{min(ratios):.2f}")
    v.put("len.ratio.max", f"{max(ratios):.2f}")
    below = [c["frac_below_10_sentences"] for x in lens.values() for c in x.values()]
    v.put("len.below10.max", round(100 * max(below)))
    con = review["mode_contrasts_B_n"]
    for m, (k, _) in FAMILIES.items():
        for name, e in con.get(m, {}).items():
            v.stat(f"con.{k}.{name}", e, delta="median_delta")
    # contrast = first mode minus second, in the model's own B_n: negative = first mode flatter
    def sig(m, name, sign):
        e = con.get(m, {}).get(name)
        return e is not None and e["wilcoxon_p"] < 0.05 and (e["median_delta"] * sign > 0)
    n_con = len([m for m in MODELS if m in con])
    v.count("con.n", n_con)
    v.put("con.widen.phrase", models_phrase(sum(sig(m, "instruction-control", -1) for m in MODELS)))
    v.put("con.narrow.w", word(sum(sig(m, "instruction-control", +1) and inst[m] == "flat" for m in MODELS)))
    n_fslt = sum(sig(m, "fewshot-instruction", -1) for m in MODELS)
    v.put("con.fslt.w", of_total(n_fslt, n_con))
    v.put("con.fslt.phrase", ("every model" if n_fslt == n_con == 9 else
                               "every model with a contrast" if n_fslt == n_con else
                               f"{word(n_fslt)} of the {word(n_con)} models with a contrast"))
    v.put("con.fsltctrl.w", word(sum(sig(m, "fewshot-control", -1) for m in MODELS)))
    v.put("con.fsgtctrl.w", word(sum(sig(m, "fewshot-control", +1) for m in MODELS)))
    v.put("con.fsnsctrl.w", word(sum(1 for m in MODELS if m in con and not sig(m, "fewshot-control", -1) and not sig(m, "fewshot-control", +1))))
    # replication of a leg at a later snapshot: delta = replicate minus original, in B_n
    for rep_m, e in review.get("replication_contrasts_B_n", {}).items():
        k = FAMILIES[rep_m][0]
        for mode, _ in MODES:
            v.stat(f"rep.{k}.{mode}", e[mode], delta="median_delta")
        v.put(f"rep.{k}.sig.w", word(sum(1 for mode, _ in MODES if e[mode]["wilcoxon_p"] < 0.05)))
        v.put(f"rep.{k}.d.min", f3(min(e[mode]["median_delta"] for mode, _ in MODES)))
        v.put(f"rep.{k}.d.max", f3(max(e[mode]["median_delta"] for mode, _ in MODES)))
        v.put(f"rep.{k}.p.max", peq(max(e[mode]["wilcoxon_p"] for mode, _ in MODES)))
    gf = review.get("guard_filtered_B_n", {})
    g_changed = [(m, mode) for m in MODELS for mode, _ in MODES if m in gf and landing(
        {"wilcoxon_p": gf[m][mode]["wilcoxon_p"], "median_delta_h_minus_ai": gf[m][mode]["median_delta"]}) != cells[(m, mode)]]
    v.put("guard.changed.w", word(len(g_changed)))
    cellname = lambda m, mode: f"{FAMILIES[m][1]}'s {mode.replace('_', '-')} cell"  # noqa: E731
    v.put("guard.changed.names", join_names([cellname(m, mode) for m, mode in g_changed]) if g_changed else "none")
    v.put("guard.n", sum(1 for m in MODELS if m in gf) * len(MODES))
    v.put("floor10.n", sum(1 for m in MODELS if m in review["floor10_B_n"]) * len(MODES))
    hs_med = review["lengths"]["human"]["sentences_median"]
    v.put("human.sent.median", int(hs_med) if float(hs_med).is_integer() else hs_med)
    v.put("human.sent.min", review["lengths"]["human"]["min_sentences"])
    for m, (k, _) in FAMILIES.items():
        for mode, _ in MODES:
            if m in gf:
                v.stat(f"guardf.{k}.{mode}", gf[m][mode], delta="median_delta")
    f10 = review["floor10_B_n"]
    changed = [(m, mode) for m in MODELS for mode, _ in MODES
               if m in f10 and landing({"wilcoxon_p": f10[m][mode]["wilcoxon_p"],
                                        "median_delta_h_minus_ai": f10[m][mode]["median_delta"]}) != cells[(m, mode)]]
    v.put("floor10.changed", len(changed))
    v.put("floor10.changed.w", word(len(changed)))
    v.put("floor10.changed.names", join_names([cellname(m, mode) for m, mode in changed]) if changed else "none")
    for m, (k, _) in FAMILIES.items():
        for mode, _ in MODES:
            if m in f10:
                v.stat(f"floor10.{k}.{mode}", f10[m][mode], delta="median_delta")

    # --- segmenter-fix audit (needs the archived pre-fix SSOTs) ---------------------------
    pre = {m: BENCH / m / "seed_avg_paired.pre-segfix.json" for m in MODELS}
    if all(p.exists() for p in pre.values()):
        old = {m: load(p) for m, p in pre.items()}
        moved = [(m, mode) for m in MODELS for mode, _ in MODES
                 if landing(old[m]["dims"]["B_n"][mode]) != cells[(m, mode)]]
        shifts = [ssot[m]["dims"]["B_n"][mode]["median_delta_h_minus_ai"] - old[m]["dims"]["B_n"][mode]["median_delta_h_minus_ai"]
                  for m in MODELS for mode, _ in MODES]
        v.put("segfix.moved", len(moved))
        v.put("segfix.moved.w", word(len(moved)))
        v.put("segfix.moved.names", join_names([cellname(m, mode) for m, mode in moved]) if moved else "none")
        for m in MODELS:
            for mode, _ in MODES:
                v.stat(f"pre.{FAMILIES[m][0]}.{mode}", old[m]["dims"]["B_n"][mode])
        v.put("segfix.shift.min", f3(min(shifts)))
        v.put("segfix.shift.max", f3(max(shifts)))
    else:
        old = None
        for k in ("segfix.moved", "segfix.moved.w", "segfix.moved.names", "segfix.shift.min", "segfix.shift.max"):
            v.put(k, "??")   # placeholders until every family has a seed_avg_paired.pre-segfix.json
        print("WARNING: no pre-segfix SSOTs; segfix macros are placeholders")

    # --- write ----------------------------------------------------------------------------
    lines = ["% Generated by paper/gen_bench_table.py -- do not hand-edit.",
             "\\makeatletter",
             "\\newcommand{\\val}[1]{\\ifcsname val@#1\\endcsname\\csname val@#1\\endcsname"
             "\\else\\PackageError{numbers}{Undefined value #1}{Rerun paper/gen_bench_table.py}\\fi}"]
    lines += [f"\\expandafter\\def\\csname val@{k}\\endcsname{{{x}}}" for k, x in sorted(v.items())]
    lines.append("\\makeatother")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_tables(ssot, mult, review, old)
    print(f"wrote {len(v)} values and 7 tables to {OUT.relative_to(ROOT)}")
    return 0


def write_tables(ssot, mult, review, old):
    def row(m, fn):
        return f"{FAMILIES[m][1]} & " + " & ".join(fn(m, mode) for mode, _ in MODES) + " \\\\"

    head = "Model & " + " & ".join(lbl for _, lbl in MODES) + " \\\\"
    main = [m for m in FAMILIES if m not in GROK_PAIR]
    body = [row(m, lambda m, md: cell(ssot[m]["dims"]["B_n"][md])) for m in main]
    pair = [row(m, lambda m, md: cell(ssot[m]["dims"]["B_n"][md])) for m in GROK_PAIR]
    (OUT / "tab_ninefam.tex").write_text("\n".join(
        ["% Generated by paper/gen_bench_table.py -- do not hand-edit.", "\\begin{tabular}{lccccc}", "\\toprule", head, "\\midrule",
         *body, "\\midrule", *pair, "\\bottomrule", "\\end{tabular}"]) + "\n", encoding="utf-8")

    grids = []
    for m in ["qwen3.6-27b"] + [x for x in FAMILIES if x != "qwen3.6-27b"]:
        mu, name = mult[m], FAMILIES[m][1]
        lost = [f"{t['mode'].replace('_', '-')} {dict((d, l) for d, _, l in DIMS)[t['metric']]}" for t in mu["tests"]
                if t["raw_sig_05"] and not t["bh_survives"]]
        census = (f"Within-model census: ${mu['n_raw_sig']}$ of $20$ cells per-cell significant; "
                  f"${mu['n_bh']}$ survive Benjamini--Hochberg at $q=0.05$ and ${mu['n_holm']}$ survive Holm"
                  + (f" (lost under Benjamini--Hochberg: {', '.join(lost)})" if lost else "") + ".")
        per = [ssot[m]["dims"]["B_n"][mode]["n_pairs"] for mode, _ in MODES]
        n = (f"$n={per[0]}$ headlines" if min(per) == max(per) else
             "$n=" + "/".join(map(str, per)) + "$ headlines per mode, in Table~\\ref{tab:ninefam} order")
        conv = ("Seed-averaged Wilcoxon signed-rank paired tests over the matched headlines; median "
                "$\\Delta =$ human $-$ model, so positive means the model sits \\emph{below} the human article "
                "on that dimension. Matched rank-biserial in parentheses; $^{*}p<0.05$, $^{**}p<0.01$ "
                "(per cell, two-sided)." if m == "qwen3.6-27b" else
                "Design and conventions as in Table~\\ref{tab:promptceiling}.")
        cap = f"{name}, full grid ({n}). {conv} {census} {GRID_NOTES.get(m, '')}".strip()
        label = "tab:promptceiling" if m == "qwen3.6-27b" else "tab:grid-" + FAMILIES[m][0]
        rows = [f"{lbl} & " + " & ".join(cell(ssot[m]["dims"][d][mode]) for d, _, _ in DIMS) + " \\\\"
                for mode, lbl in MODES]
        grids += ["\\begin{table}[htbp]", "\\centering", "\\small", f"\\caption{{{cap}}}", f"\\label{{{label}}}",
                  "\\begin{tabular}{lcccc}", "\\toprule",
                  "Mode & " + " & ".join(l for _, _, l in DIMS) + " \\\\", "\\midrule", *rows,
                  "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (OUT / "tab_family_grids.tex").write_text("% Generated by paper/gen_bench_table.py -- do not hand-edit.\n"
                                              + "\n".join(grids), encoding="utf-8")

    eq = review["equivalence_instruction_B_n"]
    yes = lambda b: "\\checkmark" if b else "--"  # noqa: E731
    rows = []
    for m in FAMILIES:
        e = eq[m]
        rows.append(f"{FAMILIES[m][1]} & ${f3(e['median_delta'])}$ & $[{f3lo(e['ci95'][0])}, {f3hi(e['ci95'][1])}]$ & "
                    f"${peq(e['wilcoxon_p'])}$ & ${f3hi(e['m_star'])}$ & "
                    + " & ".join(yes(e["equivalent_at"][str(mg)]) for mg in MARGINS) + " \\\\")
    (OUT / "tab_equivalence.tex").write_text("\n".join(
        ["% Generated by paper/gen_bench_table.py -- do not hand-edit.", "\\begin{tabular}{lccccccc}", "\\toprule",
         "Model & median $\\Delta B_n$ & 95\\% CI & Wilcoxon & $m^{*}$ & $0.03$ & $0.05$ & $0.08$ \\\\",
         "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]) + "\n", encoding="utf-8")

    con = review["mode_contrasts_B_n"]
    pairs = [("instruction-control", "Instr.\\ $-$ control"), ("fewshot-control", "Few-shot $-$ control"),
             ("fewshot-instruction", "Few-shot $-$ instr.")]
    ccell = lambda e: f"${f3(e['median_delta'])}\\;({e['matched_rank_biserial']:.2f}){stars(e['wilcoxon_p'])}$"  # noqa: E731
    rows = [f"{FAMILIES[m][1]} & " + " & ".join(ccell(con[m][k]) for k, _ in pairs) + " \\\\" for m in FAMILIES if m in con]
    (OUT / "tab_contrasts.tex").write_text("\n".join(
        ["% Generated by paper/gen_bench_table.py -- do not hand-edit.", "\\begin{tabular}{lccc}", "\\toprule",
         "Model & " + " & ".join(l for _, l in pairs) + " \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]) + "\n",
        encoding="utf-8")

    var = review["variance_baseline"]["legs"]
    vcell = lambda e: f"${e['matched_rank_biserial']:.2f}{stars(e['wilcoxon_p'])}$"  # noqa: E731
    rows = []
    for m in FAMILIES:
        if m not in var:
            continue
        bn = " & ".join(vcell({"matched_rank_biserial": ssot[m]["dims"]["B_n"][md]["matched_rank_biserial"],
                               "wilcoxon_p": ssot[m]["dims"]["B_n"][md]["wilcoxon_p"]}) for md in ("control", "instruction"))
        vl = " & ".join(vcell(var[m][md]) for md in ("control", "instruction"))
        rows.append(f"{FAMILIES[m][1]} & {bn} & {vl} \\\\")
    (OUT / "tab_variance.tex").write_text("\n".join(
        ["% Generated by paper/gen_bench_table.py -- do not hand-edit.", "\\begin{tabular}{lcccc}", "\\toprule",
         " & \\multicolumn{2}{c}{$B_n$} & \\multicolumn{2}{c}{raw $\\mathrm{var}(L)$} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Model & Control & Instruction & Control & Instruction \\\\",
         "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]) + "\n", encoding="utf-8")

    gf = review.get("guard_filtered_B_n", {})
    gcell = lambda a, b: f"${f3(a['median_delta_h_minus_ai'])}{stars(a['wilcoxon_p'])}$ / ${f3(b['median_delta'])}{stars(b['wilcoxon_p'])}$"  # noqa: E731
    rows = [f"{FAMILIES[m][1]} & " + " & ".join(gcell(ssot[m]["dims"]["B_n"][md], gf[m][md]) for md, _ in MODES) + " \\\\"
            for m in FAMILIES if m in gf]
    (OUT / "tab_guard.tex").write_text("\n".join(
        ["% Generated by paper/gen_bench_table.py -- do not hand-edit.", "\\begin{tabular}{lccccc}", "\\toprule",
         "Model & " + " & ".join(lbl for _, lbl in MODES) + " \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]) + "\n",
        encoding="utf-8")

    f10, g, lens = review["floor10_B_n"], review["guard_rates"]["legs"], review["lengths"]
    rows = []
    for m in FAMILIES:
        if m not in g:
            continue
        cur = [landing(ssot[m]["dims"]["B_n"][md]) for md, _ in MODES]
        fl = [landing({"wilcoxon_p": f10[m][md]["wilcoxon_p"], "median_delta_h_minus_ai": f10[m][md]["median_delta"]})
              for md, _ in MODES]
        moved_f = sum(a != b for a, b in zip(cur, fl))
        moved_s = (f"${sum(landing(old[m]['dims']['B_n'][md]) != c for (md, _), c in zip(MODES, cur))}$"
                   if old and m in old else "--")
        r = list(g[m].values())
        lr = [c["word_ratio_median"] for c in lens[m].values()]
        b10 = max(c["frac_below_10_sentences"] for c in lens[m].values())
        sm = [c["sentences_median"] for c in lens[m].values()]
        rows.append(f"{FAMILIES[m][1]} & {moved_s} & ${moved_f}$ & ${round(100 * min(r))}$--${round(100 * max(r))}\\%$ & "
                    f"${min(lr):.2f}$--${max(lr):.2f}$ & ${min(sm):g}$--${max(sm):g}$ & ${round(100 * b10)}\\%$ \\\\")
    (OUT / "tab_audit.tex").write_text("\n".join(
        ["% Generated by paper/gen_bench_table.py -- do not hand-edit.", "\\begin{tabular}{lcccccc}", "\\toprule",
         " & \\multicolumn{2}{c}{$B_n$ cells changing landing} & & & & \\\\", "\\cmidrule(lr){2-3}",
         "Model & segmenter fix & 10-sentence floor & guard flags & word ratio & sentences & $<10$ sentences \\\\",
         "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
