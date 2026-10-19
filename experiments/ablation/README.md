# Era-controlled Wikinews human-vs-AI ablation

Corpus construction, normalization, scoring, and audit scripts for the paired Wikinews
protocol. The design holds era, topic, and register fixed so authorship is the only free
variable: English Wikinews articles (CC BY 2.5, 2016-2021, pre-model) versus language-model
articles written on the same headlines and matched in length. This removes the century,
editing, and genre confounds of comparing machine text against older literature.

Two corpora share the construction:

- the 100-pair ablation corpus (`burstiness_pairs.jsonl`, private HF dataset
  `burstiness-engine/burstiness-engine-ablation`, mirrored in hf-datasets.zip), scored by `paired_test.py`;
- the 200-headline benchmark reference set (`prompts_200_human.jsonl`, HF dataset
  `burstiness-engine/paper-dataset`), against which every benchmark leg is scored by
  `seed_avg_paired.py`.

## Data (not in this repo)

The ablation corpus ships:

- `burstiness_pairs.jsonl` — one row per headline: `{id, headline, date, human_text,
  human_words, human_n_sentences, ai_text}`
- `human_wikinews.txt`, `ai_wikinews.txt` — `@@@`-delimited flat corpora (same texts, unpaired)

Download it into `experiments/ablation/data/`, or point `BURST_PAIRS` at it. The human side
regenerates from the public `malteos/wikinews` wikitext with `clean_wikinews.py`;
`build_ablation_corpus.py --validate` checks that reproduction (100/100).

## Scripts

Run from the repo root. All measures are model-free (no GPU, stdlib only); corpus construction
needs `mwparserfromhell`, `huggingface_hub`, `pandas`, and `select_m2_exemplars.py` needs
`scikit-learn`.

| script | job |
|---|---|
| `clean_wikinews.py` | wikitext -> plain article body (deterministic MediaWiki parse) |
| `build_ablation_corpus.py` | `--validate` the frozen human side; `--select` a new disjoint batch |
| `select_m2_exemplars.py` | the median-rule selection of the three few-shot exemplars (`bench/m2_exemplars.json`) |
| `normalize_generation.py` | uniform normalization of every generated leg before scoring (with `test_normalize_generation.py`) |
| `paired_test.py` | matched-headline Wilcoxon on the ablation corpus; also the `metrics()` / `wilcoxon()` used by every other scorer |
| `seed_avg_paired.py` | seed-averaged paired test of a benchmark leg against the human reference set |
| `rescore_all.py` | re-runs `seed_avg_paired.py` for every leg and writes the committed results |
| `equivalence_test.py` | CI-bounded equivalence margins from a leg's committed CIs |
| `multiplicity_correction.py` | BH and Holm census over a leg's 20 tests |
| `review_analyses.py` | human spread, mode contrasts, variance baseline, guard rates, floor sensitivity, lengths |
| `generation_hygiene.py` | smart-quote, markdown, and title-line rates in raw generations |
| `segentropy_quote_artifact.py` | size of the typographic-quote effect on segmentation entropy |
| `contamination_probe.py` | n-gram overlap probe (with `test_contamination_probe.py`) |
| `prompt_paraphrase.py` | instruction-wording robustness probe |

```bash
# matched-headline Wilcoxon signed-rank on the ablation corpus
BURST_PAIRS=data/burstiness_pairs.jsonl python -m experiments.ablation.paired_test

# bench generations vs human, multi-seed: per headline the metric vector is averaged across
# the bench's seeds first, then one paired Wilcoxon on independent headline pairs (pooling
# seeds as separate pairs would pseudo-replicate the human docs)
python -m experiments.ablation.seed_avg_paired prompts_200_human.jsonl qwen3.6-27b.jsonl qwen27b
```

The scripts are seeded (`random.seed(0)`), so rerunning reproduces the committed JSONs.

The measure itself lives one level up in `experiments/burstiness_pp.py`
(`burstiness_B_corrected`, `gini`, `memory_M`, `segmentation_entropy`), with the sentence
segmenter in `experiments/burstiness_metrics.py`.
