# Mobile Price Predictor

This repo contains a small production-ready scaffold for training and serving a mobile price-range model.

Quick start

1. Create and activate a virtualenv (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

2. Train a model (atomic save + metadata):

```bash
# use a friendly base name and version; date will be appended
python -m mobile_price_predictor.train --data data/train.csv --output models/ --name predictor --model-version 1 --date-version
```

3. Run the Streamlit UI:

```bash
python -m streamlit run src/mobile_price_predictor/ui.py
```

API (required)

This project now requires the backend API for the UI to function — the UI no longer loads models locally. You must configure the API URL either in `config.yaml` or via the `MOBILE_API_URL` environment variable.

Add to `config.yaml`:

```yaml
api:
	url: http://localhost:8000
```

Or set environment variable (PowerShell):

```powershell
$env:MOBILE_API_URL = "http://localhost:8000"
streamlit run src/mobile_price_predictor/ui.py
```

See `API_CONTRACT.md` for the full backend contract (endpoints, request/response schemas, examples, and error codes).

4. Run batch predictions from CSV:

```bash
python -m mobile_price_predictor.predict --model models/rf_YYYYMMDD.joblib --input data/test.csv --output out/preds.csv
# or use the newest model in models/:
python -m mobile_price_predictor.predict --latest --input data/test.csv --output out/preds.csv
```

Notes

- Models are saved atomically and accompanied by a small JSON metadata file with creation time and metrics. Filenames are constructed from `--name` and `--model-version` (e.g. `predictor_v1_20260930.joblib`).
- Use `--date-format` with training to customize the timestamp (default `%Y%m%d`).
- The project contains console scripts after `pip install -e .`: `train` and `predict`.
- Do not commit `.venv/` to git; a `.gitignore` file is included that ignores `.venv/` and common Python artifacts.

Docker

```bash
docker build -t mobile-price-range:latest .
```

If you want, I can commit this README for you or add CI checks that validate training and model-save behavior.

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
