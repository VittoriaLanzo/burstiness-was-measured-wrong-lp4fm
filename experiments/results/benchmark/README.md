# Prompt-ceiling benchmark results (model-keyed)

Single source of truth for the numbers in the paper's prompt-ceiling section
(Table 1, Figure 1, and the per-family grids). One directory per generator
model. The paper's tables and figures are generated from these files by
`paper/gen_bench_table.py` and `paper/lp4fm/make_lp4fm_figures.py`, never
hand-typed.

## Layout

```
experiments/results/benchmark/
  <model_id>/
    seed_avg_paired.json   # compiled scored eval: per (metric, mode) paired stats
    meta.json              # model, decoding, corpus, design, exclusions, provenance
    multiplicity.json      # BH and Holm census over the 20 tests (multiplicity_correction.py)
    equivalence.json       # instruction-mode equivalence margins (equivalence_test.py)
    seed_avg_paired.pre-segfix.json   # the same eval scored with the released segmenter
  qwen3.6-27b/
    seed_avg_paired.raw-archive.json   # the audited leg scored under the raw protocol
```

Models:

| model_id        | family                 | openness     |
|-----------------|------------------------|--------------|
| qwen3.6-27b     | Qwen                   | open-weights |
| gemma-3-27b     | Gemma (Google)         | open-weights |
| deepseek-v4-pro | DeepSeek               | api          |
| grok-4.3        | Grok (xAI)             | api          |
| grok-4.5        | Grok (xAI)             | api          |
| gemini-3.1-pro  | Gemini (Google)        | api          |
| glm-5.2         | GLM (Zhipu)            | api          |
| kimi-k2.6       | Kimi (Moonshot)        | api          |
| opus-4.8        | Claude Opus (Anthropic)| api          |
| deepseek-v4-pro-1018 | DeepSeek, 18 October 2026 snapshot | api |

`deepseek-v4-pro-1018` regenerated the DeepSeek leg on 18 October 2026 through the
same API alias with the same prompts and briefs (the paper's dagger-marked row). It is in
every table and figure and outside the nine-model counts; `review_analyses.json`
carries its paired contrast against the July leg (`replication_contrasts_B_n`).

The two open-weights legs were generated locally with `bench/gen_vllm.py`; the
api legs with `bench/run.py` (`opus-4.8` with `bench/run_batch.py`). Every leg
is scored under the normalized protocol (`normalize_generation.py`, then
`seed_avg_paired.py`); `qwen3.6-27b` additionally carries its raw-protocol
scoring as `seed_avg_paired.raw-archive.json`. `seed_avg_paired.pre-segfix.json` (the nine
models' legs; the re-run leg postdates the fix) is the evaluation as reviewed, scored before the segmenter mapped typographic
closing quotes to ASCII; the paper's segmentation audit compares the two.

## seed_avg_paired.json schema

```
label:  str                       # short model tag
dims:   { B_n | M | gini | seg_entropy :
            { control|instruction|fewshot|plan|self_refine :
                { n_pairs, median_delta_h_minus_ai, median_delta_95ci[2],
                  wilcoxon_z, wilcoxon_p, matched_rank_biserial, paired_dz } } }
consistency_B_n_rank_biserial: { mode : { it0, it1, it2 } }   # per-seed stability
note:   str                       # analysis note (seed-averaged, delta = human - model)
```

`delta = human - model` (positive = model below human on that dimension, the
flatness direction). Significance stars in the paper: `*` p<0.05, `**` p<0.01.
`median_delta_95ci` is stored unrounded; the equivalence margins are compared
with these values, and the paper prints the bounds rounded outward to three
decimals. `M` carries the +1/(n-1) small-sample term (`memory_M_corrected`).

## Adding a model

1. Run the benchmark for the new model (200 Wikinews headlines x 5 modes x 3 seeds),
   normalize the generations with `experiments/ablation/normalize_generation.py`, and
   score them with `experiments/ablation/seed_avg_paired.py` against
   `prompts_200_human.jsonl`.
2. Publish the raw generations and the scored eval to a HF dataset.
3. Create `experiments/results/benchmark/<model_id>/` with `seed_avg_paired.json`
   and `meta.json` (copy an existing meta.json and edit), then run
   `multiplicity_correction.py --model <model_id>` and
   `equivalence_test.py --model <model_id>` for the other two files.
4. Add the model to `FAMILIES` in `paper/gen_bench_table.py` and to `MODEL_IDS` in
   `paper/lp4fm/make_lp4fm_figures.py`, run both, then rebuild the paper.

## Provenance

Raw generations are not in git; the small scored `seed_avg_paired.json` is
carried here so every published number traces to a committed artifact. Each
model's `meta.json` gives the dataset refs; the
top-level README describes the `hf-datasets.zip` mirror, which holds the raw
generations of all nine legs.
