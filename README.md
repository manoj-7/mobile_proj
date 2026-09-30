# Mobile Price Range — Production-ready scaffold

This repository contains a production-ready scaffold for the Kaggle "Mobile Price Range" prediction task. It includes:

- `pyproject.toml` for dependency metadata
- `src/mobile_price_predictor` package with training and prediction entrypoints
- `Dockerfile` for container builds
- GitHub Actions CI workflow for lint/test/build
- A starter exploration notebook in `notebooks/`

Usage

Install dependencies (UV or pip):

If you use UV as your package manager:

```bash
uv install
```

Or with pip (fallback):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Run training (example):

```bash
python -m mobile_price_predictor.train --data data/train.csv --output models/rf.joblib
```

Build Docker image:

```bash
docker build -t mobile-price-range:latest .
```

Production-ready run examples

- Train and save with a date-stamped model (atomic save + metadata):

```bash
python -m mobile_price_predictor.train --data data/train.csv --output models/rf.joblib --date-version
```

- Train into models/ directory and let the tool pick a name:

```bash
python -m mobile_price_predictor.train --data data/train.csv --output models/ --date-version
```

- Use the installed CLI (after `pip install -e .`) and pick the latest model automatically:

```bash
# train (console script)
train --data data/train.csv --output models/ --date-version

# predict using the latest model in models/
predict --latest --input data/test.csv --output out/preds.csv
```

Notes:
- Trained models are saved atomically and accompanied by a small JSON metadata file (created_at, accuracy, params).
- Use `--date-format` for alternate timestamping (example: `--date-format "%Y%m%d-%H%M"`).
- The `--latest` flag in `predict` looks for the newest `*.joblib` file in `models/`.
