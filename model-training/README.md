# Model Training Data Pipeline (XES3G5M)

This folder contains the high-level data preprocessing workflow for building a question-level Deep Knowledge Tracing (DKT) dataset from XES3G5M.

## What this pipeline produces

The pipeline builds a versioned processed dataset at:

- `data/processed/dkt_qlevel_v1/`

It includes:

- Per-fold feature files (`fold_0` ... `fold_4`)
- Per-fold sequence window files for training/validation (and optional test splits)
- `manifest.json` (schema + version + config hash)
- `feature_summary.json` (fold-level row counts and vocab sizes)
- `semantic_embedding_tables.json` (question content/analysis embedding lookup tables)

Reports are written to:

- `data/reports/data_quality_qlevel_v1.json`
- `data/reports/ablation_qlevel_v1.json`
- `data/reports/pipeline_run_summary_qlevel_v1.json`
- `data/reports/output_validation_summary_qlevel_v1.json`

Interim (pre-processed but not final) artifacts are in:

- `data/interim/`

## Current data structure (high level)

- **Raw source data**: `XES3G5M/`
  - Uses the provided question-level sequence CSVs and metadata.
- **Interim data**: `data/interim/`
  - Long-form cleaned interaction data.
- **Processed data**: `data/processed/dkt_qlevel_v1/`
  - Folded features and fixed-length sequence windows.
- **Reports**: `data/reports/`
  - Data quality checks and baseline/ablation summaries.

## How to run the pipeline

From this folder (`model-training`):

Train/valid only:

```bash
python3 -m src.data.run_qlevel_pipeline --raw-root XES3G5M --output-root data
```

Train/valid + question-level test preprocessing:

```bash
python3 -m src.data.run_qlevel_pipeline --raw-root XES3G5M --output-root data --include-test-files
```

Optional flags:

- `--window-size` (default: `200`)
- `--stride` (default: `50`)
- `--min-user-history` (default: `3`)
- `--embedding-projection-dim` (default: `32`)
- `--smoothing-alpha` (default: `5.0`)
- `--seed` (default: `42`)
- `--include-test-files` (also preprocesses `test_window_sequences_quelevel.csv` and `test_quelevel.csv`)

## Expected outputs after a successful run

- `data/processed/dkt_qlevel_v1/manifest.json`
- `data/processed/dkt_qlevel_v1/feature_summary.json`
- `data/processed/dkt_qlevel_v1/fold_*/train_windows.jsonl`
- `data/processed/dkt_qlevel_v1/fold_*/valid_windows.jsonl`
- `data/processed/dkt_qlevel_v1/fold_*/test_window_windows.jsonl` (if `--include-test-files`)
- `data/processed/dkt_qlevel_v1/fold_*/test_quelevel_windows.jsonl` (if `--include-test-files`)
- `data/reports/data_quality_qlevel_v1.json`
- `data/reports/ablation_qlevel_v1.json`

## Notes

- This is a **question-level** pipeline optimized first for **next-step correctness prediction**.
- Outputs are versioned under `dkt_qlevel_v1` to keep experiments reproducible.
