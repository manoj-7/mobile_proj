import argparse
from pathlib import Path
import joblib
import logging

import pandas as pd

from mobile_price_predictor.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=False, help="Path to trained model (joblib)")
    parser.add_argument("--latest", action="store_true", help="Use the latest model file from models/ directory")
    parser.add_argument("--input", required=False, help="Path to CSV input features")
    parser.add_argument("--output", required=False, help="Path to save CSV predictions")
    parser.add_argument("--config", required=False, help="Path to YAML config file for default paths")
    args = parser.parse_args(argv)

    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    model_path = args.model or (cfg.get("paths") or {}).get("model")
    input_path = args.input or (cfg.get("data") or {}).get("test")
    output_path = args.output or (cfg.get("paths") or {}).get("predictions")

    if args.latest and not model_path:
        models_dir = Path("models")
        if not models_dir.exists():
            raise SystemExit("No models/ directory found to select latest model")
        candidates = list(models_dir.glob("*.joblib"))
        if not candidates:
            raise SystemExit("No .joblib models found in models/ to select latest")
        model_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
        logger.info("Selected latest model: %s", model_path)

    if not model_path:
        raise SystemExit("Model path must be provided via --model, --latest, or config 'paths.model'")
    if not input_path:
        raise SystemExit("Input CSV path must be provided via --input or config 'data.test'")
    if not output_path:
        raise SystemExit("Output CSV path must be provided via --output or config 'paths.predictions'")

    model = joblib.load(model_path)
    df = pd.read_csv(input_path)
    X = df.select_dtypes(include=["number"]).fillna(0)
    preds = model.predict(X)
    out_df = pd.DataFrame({"prediction": preds})
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path
import joblib
import logging
import sys

import pandas as pd

from mobile_price_predictor.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=False, help="Path to trained model (joblib)")
    parser.add_argument("--latest", action="store_true", help="Use the latest model file from models/ directory")
    parser.add_argument("--input", required=False, help="Path to CSV input features")
    parser.add_argument("--output", required=False, help="Path to save CSV predictions")
    parser.add_argument("--config", required=False, help="Path to YAML config file for default paths")
    args = parser.parse_args(argv)

    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    model_path = args.model or (cfg.get("paths") or {}).get("model")
    input_path = args.input or (cfg.get("data") or {}).get("test")
    output_path = args.output or (cfg.get("paths") or {}).get("predictions")

    # If --latest requested and no explicit model, pick newest .joblib in models/
    if args.latest and not model_path:
        models_dir = Path("models")
        if not models_dir.exists():
            raise SystemExit("No models/ directory found to select latest model")
        candidates = list(models_dir.glob("*.joblib"))
        if not candidates:
            raise SystemExit("No .joblib models found in models/ to select latest")
        model_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
        logger.info("Selected latest model: %s", model_path)

    if not model_path:
        raise SystemExit("Model path must be provided via --model or config 'paths.model'")
    if not input_path:
        raise SystemExit("Input CSV path must be provided via --input or config 'data.test'")
    if not output_path:
        raise SystemExit("Output CSV path must be provided via --output or config 'paths.predictions'")

    model = joblib.load(model_path)
    df = pd.read_csv(input_path)
    X = df.select_dtypes(include=["number"]).fillna(0)
    preds = model.predict(X)
    out_df = pd.DataFrame({"prediction": preds})
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
