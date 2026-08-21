# Era-controlled Wikinews human-vs-AI ablation

Reproduces the validation tables in the paper (`paper/main.tex`, `\section{Validating the
Definition}`, `sec:ablation`). It tests the model-free burstiness definition on a *matched*
corpus that holds era, topic, and register fixed so authorship is the only free variable:
100 English Wikinews articles (CC BY 2.5, 2016-2021, confidently pre-model) versus 100 blind
language-model articles written on the same headlines and matched in length. This removes the
century / editing / genre confounds of comparing machine text against 19th-century literature.

## Data (not in this repo)

The corpus is the private HuggingFace dataset `ANONYMIZED-ablation-corpus`.
It ships:

- `burstiness_pairs.jsonl` — one row per headline: `{headline, human_text, ai_text}`
- `human_wikinews.txt`, `ai_wikinews.txt` — `@@@`-delimited flat corpora (same texts, unpaired)

It is kept off the repo on purpose: it is large, and the paper's claim is the *definition*, not
the corpus. Download it into `experiments/ablation/data/` before running (or point the scripts at
it; see below).

## Reproduce the paper tables

Run from the repo root. All measures are model-free (no GPU, stdlib only).

```bash
# tab:wikinews  — unpaired AUC / rank-biserial / Glass / BH-q
python -m experiments.ablation.compare_stats data/human_wikinews.txt data/ai_wikinews.txt wikinews

# tab:wikinews-paired — matched-headline Wilcoxon signed-rank
BURST_PAIRS=data/burstiness_pairs.jsonl python -m experiments.ablation.paired_test

# bench generations vs human, multi-seed: per headline the metric vector is averaged across
# the bench's seeds FIRST, then one paired Wilcoxon on independent headline pairs (pooling
# seeds as separate pairs would pseudo-replicate the human docs). Used for the qwen3.6-27b
# prompt-ceiling readout (results archived on the HF dataset under eval/).
python -m experiments.ablation.seed_avg_paired prompts_200_human.jsonl qwen3.6-27b.jsonl qwen27b

# within-document length test (B_n gap vs truncation length k)
python -m experiments.ablation.length_test data/human_wikinews.txt data/ai_wikinews.txt
```

The committed `*_results.json` / `compare_stats_wikinews.json` are the exact outputs the paper
cites; the scripts are seeded (`random.seed(0)`), so rerunning reproduces them.

## What is and isn't here

Shipped: the three analysis scripts and their result JSONs. The measure itself lives one level up
in `experiments/burstiness_pp.py` (`burstiness_B_corrected`, `gini`, `memory_M`,
`segmentation_entropy`). Corpus construction (Wikinews scrape, blind same-headline generation) is
documented in the HF dataset card, not duplicated here.
