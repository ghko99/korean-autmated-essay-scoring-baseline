# Korean Automated Essay Scoring Baseline

Baseline training code for Korean automated essay scoring experiments.

## Contents

- `kobert_gru.py`: KoBERT/GRU-based training script.
- `config.py`: shared experiment configuration.
- `embedding.py`: embedding utilities used by the baseline model.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Check dataset paths and configuration values, then run the baseline training script:

```bash
python kobert_gru.py
```

Large datasets and generated checkpoints should stay outside version control.
