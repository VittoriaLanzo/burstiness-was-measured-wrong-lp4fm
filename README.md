# Burstiness Was Measured Wrong, and Prompting Cannot Aim It: code and evaluation artifacts

Private artifact repository for the paper "Burstiness Was Measured Wrong, and
Prompting Cannot Aim It" (TAE @ NeurIPS 2026 submission). Every script and
evaluation artifact the paper relies on, in source-repository layout, so the
scripts run from this directory as the working root. The anonymized review-time
supplementary zips are generated from these same files by
paper/tae/make_supplementary.py in the main research repo.

## Paper source

- paper/tae/main.tex: the unified LaTeX source of the submission (single assembled
  file), with paper/tae/references.bib, paper/tae/neurips_2026.sty, and
  paper/tae/build.sh (tectonic) alongside.

## Code (mirrors repository paths)

- bench/prompts.py: the five prompting modes, their composition rule, and a
  self-check that enforces the three-call self-refine structure and metric-free
  prompt text (Appendix A). bench/m2_exemplars.json holds the three few-shot
  exemplar articles.
- bench/run.py, bench/run_batch.py, bench/providers.py, bench/config.py,
  bench/storage.py: the API generation harness. bench/gen_vllm.py is the local
  open-weights lane (Qwen, Gemma). bench/scrub_thinking.py strips reasoning traces
  before scoring.
- experiments/burstiness_pp.py: the metric (B_n under the finite-n correction, the
  Gini robustness check, Miller-Madow segmentation entropy, memory coefficient M),
  the Goodhart guard, the multiscale profile, and a self-check. Standard library
  only; reproduced in Appendix C. experiments/burstiness_metrics.py carries the
  sentence segmentation and punctuation helpers it imports.
- experiments/ablation/clean_wikinews.py: deterministic corpus parse from Wikinews
  wikitext (Section 4.1). build_ablation_corpus.py assembles the paired corpus.
  select_m2_exemplars.py applies the median-rule exemplar selection.
- experiments/ablation/normalize_generation.py: the uniform normalization applied to
  every leg before scoring (Section 4.2), with test_normalize_generation.py.
- experiments/ablation/seed_avg_paired.py (with paired_test.py): the seed-averaged
  Wilcoxon protocol behind every reported delta (Section 4.3).
- experiments/ablation/equivalence_test.py: CI-bounded equivalence margins from the
  committed per-family CIs (Section 4.3).
- experiments/ablation/multiplicity_correction.py: the BH and Holm census
  (Section 4.3).
- experiments/ablation/contamination_probe.py: the n-gram overlap probe
  (Limitations), with test_contamination_probe.py.
- experiments/ablation/generation_hygiene.py: the markdown and title-echo audit that
  motivated normalization (Section 4.2).
- experiments/ablation/segentropy_quote_artifact.py: the smart-quote artifact audit
  behind the uniform ASCII quote normalization (Section 3.2).
- experiments/ablation/prompt_paraphrase.py: the instruction-wording robustness
  probe (Limitations).
- experiments/ablation/rescore_all.py: rescores every leg raw and normalized.
- paper/gen_bench_table.py: regenerates Table 1 and Figure 1 from the released
  evals; no hand-transcribed values.

## Data

- experiments/results/benchmark/<family>/seed_avg_paired.json: the compiled scored
  evaluation per family (per metric and mode: n_pairs, median paired delta,
  bootstrap 95% CI, Wilcoxon z and p, matched rank-biserial, paired dz).
- experiments/results/benchmark/<family>/meta.json: generation metadata (decoding,
  corpus, design, exclusions, provenance).
- experiments/results/benchmark/<family>/multiplicity.json: the BH and Holm census.
- experiments/results/benchmark/<family>/equivalence.json: compiled instruction-mode
  equivalence margins; regenerable for any family from seed_avg_paired.json via
  equivalence_test.py.
- experiments/results/benchmark/qwen3.6-27b/seed_avg_paired.raw-archive.json: the
  audited leg scored under the raw protocol (Section 4.2 reversal).
- experiments/results/paraphrase_robustness/: the DeepSeek paraphrase probe evals.
- experiments/results/contamination_probe.json: the n-gram probe output.
- hf-datasets.zip: mirror of the HuggingFace datasets the pipeline consumes,
  masked ids restored by path. Contains the frozen human baseline
  (prompts_200_human.jsonl, 200 paired Wikinews articles), the ablation-corpus
  pairs, and raw generations for eight of the nine benchmark legs (released
  post-scrub revisions). Claude Opus 4.8 raw generations were never mirrored to
  a dataset release; that leg ships as its compiled scored evaluation only.
  See the README inside the zip.
- research-db.zip: the SQLite research corpus behind the literature curation
  (papers with verified arXiv ids, queries, findings, learnings, session
  provenance), with its seed script and schemas. Cell-scrubbed and VACUUMed;
  author names masked as Author A / Author B. See the README inside the zip.

The human-side corpus also regenerates deterministically from Wikinews wikitext
via clean_wikinews.py; normalized generation variants regenerate from the raw
files via normalize_generation.py.

## Quick verification (run from this directory)

    python -m experiments.burstiness_pp --demo        # metric self-check
    python bench/prompts.py                           # prompt composition self-check
    python experiments/ablation/equivalence_test.py --model kimi-k2.6
    python paper/gen_bench_table.py                   # re-emit Table 1 / Figure 1

Scripts with package imports (burstiness_pp, seed_avg_paired) run module-style with
python -m from this root. Requirements: Python 3.10+. The metric and all scoring are
standard library. Generation needs bench/requirements.txt; the corpus rebuild needs
mwparserfromhell.
