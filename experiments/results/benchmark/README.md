# Prompt-ceiling benchmark results (model-keyed)

Single source of truth for the numbers in the paper's prompt-ceiling section
(Table 1 + Figure 1). One directory per generator model. The paper's table and
figure are generated from these files, never hand-typed.

## Layout

```
experiments/results/benchmark/
  <model_id>/
    seed_avg_paired.json   # compiled scored eval: per (metric, mode) paired stats
    meta.json              # model, decoding, corpus, design, HF artifact refs
```

Current models:

| model_id        | family        | openness     | run date   | HF dataset                            |
|-----------------|---------------|--------------|------------|---------------------------------------|
| qwen3.6-27b     | Qwen          | open-weights | 2026-07-03 | ANONYMIZED/qwen36-news-gen     |
| deepseek-v4-pro | DeepSeek      | api          | 2026-07-07 | ANONYMIZED/deepseek-v4-news-gen |
| grok-4.3        | Grok (xAI)    | api          | 2026-07-07 | ANONYMIZED/grok43-news-gen     |

qwen3.6-27b is the paper-integrated leg; deepseek-v4-pro and grok-4.3 carry committed
results awaiting paper integration (behind the seg-entropy quote-artifact audit, PR #21).

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

## Adding a model (the drop-in contract)

1. Run the benchmark for the new model (200 Wikinews headlines x 5 modes x 3 seeds)
   and score it with `experiments/ablation/seed_avg_paired.py` against
   `prompts_200_human.jsonl`.
2. Publish the raw generations + scored eval to a HF dataset (or the shared one).
3. Create `experiments/results/benchmark/<model_id>/` with `seed_avg_paired.json`
   and `meta.json` (copy an existing meta.json and edit).
4. Run `python3 paper/gen_bench_table.py` to regenerate the table + figure, then
   rebuild the paper. No `main.tex` edits needed to add a model.

## Provenance

Raw generations (6.2 MB `data/qwen3.6-27b.jsonl`) live in the HF dataset, not git;
the small scored `seed_avg_paired.json` is carried here so every published number
traces to a committed artifact. See each model's `meta.json` for the exact HF refs.
