# Running SERE — from data prep to metrics

This documents the actual working pipeline as set up in this checkout, including the
fixes applied to the original repo (see "Deviations from upstream" at the bottom).

## 0. One-time environment setup

```bash
cd /home/jovyan/project/litt/SERE
uv sync                      # installs deps from pyproject.toml/uv.lock into .venv
uv run python -m spacy download en_core_web_sm
```

Set your OpenRouter key (the project reads it from the shell environment, not from a file):

```bash
read -s OPENROUTER_API_KEY && export OPENROUTER_API_KEY
```

`env.env` controls which model is used and is already configured:

```
GPT_4O_MINI_NAME='openai/gpt-4o-mini'   # any OpenRouter model slug
GPT_4O_ENCODING_NAME='o200k_base'        # tiktoken encoding, cosmetic token-cost only
```

To compare a different model, just edit `GPT_4O_MINI_NAME` in `env.env`.

## 1. Data prep — CPATT → SERE jsonl format

Upstream SERE expects `dataset/<NAME>/full_{train,test}.jsonl` built from CPATT's CSVs via
`CPATT_dataset_format.ipynb`. That dataset directory does not ship with the repo — it was
generated once already and lives at `dataset/CTB/`, `dataset/ESC - inter/`, `dataset/ESC - intra/`.

To regenerate from scratch (e.g. against a different copy of CPATT), reuse the notebook's
`format_dataset()` logic against `/home/jovyan/project/litt/Code4CPATT/data/*.csv`. Output:

- `dataset/CTB/full_{test,train}.jsonl` — 142 / 568 rows
- `dataset/ESC - inter/full_{test,train}.jsonl` — 1424 / 5916 rows
- `dataset/ESC - intra/full_{test,train}.jsonl` — 571 / 2067 rows

Each row has `input_text`, `source`, `target`, `ground` (1=causal/0=not), `unique_id`.

## 2. Preprocessing — `utils/dataset_preprocess.py`

This step is required before inference; it adds the fields `sere.py` looks up by key.
Run from the SERE root:

```bash
uv run python utils/dataset_preprocess.py
```

`main()` at the bottom of the file picks the dataset (`main('CTB')` by default — uncomment
`main('ESC - inter')` / `main('ESC - intra')` to switch). It performs, in order:

1. `add_pattern_pos` — one LLM call per **positive** train example (CTB: ~252 calls) to label
   the causal pattern (Direct/Coreference/Collider/Fork/Chain).
2. `add_pattern_inference` — one LLM call per **all** test examples (CTB: 142 calls) to infer
   the same pattern for unlabeled pairs.
3. `add_conceptnet_node` / `add_concept_path` — **stubbed** in this checkout (see deviations
   below): instantly writes empty placeholders instead of querying a live ConceptNet/Neo4j
   instance, since no Neo4j deployment exists in this environment.
4. `add_filtered_weighted_unified_examples` — pure local compute. Randomly subsamples the
   test set down to **50 items** (this is original SERE behavior, not something we added) and
   attaches the `filtered_weighted_unified_examples_<model>_structured_pattern` field that
   `sere.py` needs.

This rewrites the jsonl files in place, so re-running is idempotent but redoes the LLM calls.

## 3. Inference — `sere.py`

```bash
uv run python sere.py
```

Reads `dataset/<NAME>/full_{test,train}.jsonl`, retrieves `K=2` few-shot examples per test
item via `utils/retrievers.py` (1 positive + 1 negative, chosen from the 10 candidates that
step 4 above selected), builds the prompt via `utils/prompt.py`, and calls the LLM once per
**sampled** test item (50 calls for CTB). Writes:

```
dataset/<NAME>/results/weighted_unified.jsonl
```

To switch dataset, edit `DATASET_NAME` at the bottom of `sere.py` to match whatever you ran
preprocessing on (`'CTB'`, `'ESC - inter'`, or `'ESC - intra'`).

To inspect the actual few-shot prompt sent for a given prediction (not stored in the output
file), see the snippet in this repo's chat history — it reconstructs it via
`WeightedUnifiedRetriever.retrieve_samples()` + `predict_by_structured_examples_prompt()`.

## 4. Metrics

No scoring script ships with the repo. The results file has `ground` (truth) and `pred`
(0/1) per row — feed directly to sklearn:

```bash
uv run python - <<'EOF'
import json
from sklearn.metrics import classification_report

with open('dataset/CTB/results/weighted_unified.jsonl') as f:
    rows = [json.loads(l) for l in f]

y_true = [r['ground'] for r in rows]
y_pred = [r['pred']   for r in rows]

print(classification_report(y_true, y_pred, target_names=['No-Causal', 'Causal'], digits=4))
EOF
```

## Deviations from upstream

Kept deliberately minimal — only what was needed to actually run the repo in this
environment, nothing about the method's logic was changed:

- `env.env`: added `GPT_4O_ENCODING_NAME` (was missing, caused an `AssertionError` on the
  first LLM call since `num_tokens_from_string` requires exactly one of encoding/model name).
- `utils/llm.py`: swapped the OpenAI SDK's direct `api_key` for `OPENROUTER_API_KEY` +
  `base_url='https://openrouter.ai/api/v1'` (OpenRouter is OpenAI-API-compatible), and made
  `get_llm()` accept any model name instead of hardcoding a single match. Token counting is
  wrapped in try/except since `tiktoken` doesn't recognize non-OpenAI model names — this only
  affects a cosmetic `token_cost` return value that's never used downstream.
- `sere.py` / `utils/dataset_preprocess.py`: fixed Windows-style `\` path separators to `/`.
- `sere.py`: added `os.makedirs(...)` before writing results (the `results/` dir never existed).
- `utils/dataset_preprocess.py`: `add_conceptnet_node` and `add_concept_path` are stubbed to
  write empty placeholders and return early, instead of querying ConceptNet via Neo4j (no
  Neo4j deployment available; would otherwise need a ~2GB ConceptNet assertions download, a
  Neo4j install, and a GPU embedding pass over millions of nodes). This means the
  `conceptnetpath_weight` term in the retrieval scoring is effectively always 0 — retrieval
  falls back to pure syntactic similarity. This is a real ablation versus the paper, not a
  cosmetic fix; flag it if comparing numbers directly to published SERE results.
