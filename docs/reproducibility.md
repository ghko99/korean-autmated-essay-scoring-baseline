# Reproducibility Notes

Use this checklist before comparing Korean AES baseline runs.

## Environment

- Python version and package versions.
- CUDA version and GPU type.
- KoBERT model revision.
- Dataset split names and preprocessing revision.

## Run Metadata

Record the following with each `kobert_gru.py` run:

- Git commit.
- `config.py` values.
- Random seed.
- Batch size, learning rate, epoch count, and early stopping settings if used.
- Checkpoint output path.

## Comparison Discipline

Compare metrics only when the same split and preprocessing path were used. If embeddings are regenerated, keep the embedding configuration with the result summary.
