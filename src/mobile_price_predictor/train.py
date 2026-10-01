import argparse
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
import json
import tempfile
from datetime import datetime
import os

from mobile_price_predictor.data import load_data
from mobile_price_predictor.config import load_config
from mobile_price_predictor.split import split_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=False, help="Path to CSV with training data (or single combined file if --input-all used)")
    parser.add_argument("--input-all", required=False, help="Path to single CSV file containing all rows; will be auto-split into train/test")
    parser.add_argument("--test-size", type=float, default=0.2, help="If --input-all used, fraction to reserve for test set")
    parser.add_argument("--output", required=False, help="Path to save trained model (joblib)")
    parser.add_argument("--name", required=False, help="Base name for the saved model (e.g. predictor)")
    parser.add_argument("--model-version", required=False, help="Model version string (e.g. 1 or 1.0)")
    parser.add_argument("--date-version", action="store_true", help="Append date to model filename (YYYYMMDD by default)")
    parser.add_argument("--date-format", required=False, default="%Y%m%d", help="Date format to append when --date-version is used")
    parser.add_argument("--config", required=False, help="Path to YAML config file for default paths")
    args = parser.parse_args(argv)

    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    data_path = args.data or (cfg.get("data") or {}).get("train")
    output_path = args.output or (cfg.get("paths") or {}).get("model")

    input_all = args.input_all or (cfg.get("data") or {}).get("all")
    if input_all:
        train_out = (cfg.get("data") or {}).get("train") or "data/train.csv"
        test_out = (cfg.get("data") or {}).get("test") or "data/test.csv"
        logger.info("Auto-splitting %s -> %s + %s (test_size=%s)", input_all, train_out, test_out, args.test_size)
        split_file(input_all, train_out, test_out, test_size=args.test_size, random_state=42)
        data_path = train_out

    if not data_path:
        raise SystemExit("Training CSV path must be provided via --data or config 'data.train'")
    if not output_path:
        raise SystemExit("Output model path must be provided via --output or config 'paths.model'")

    df = load_data(data_path)
    if "price_range" not in df.columns:
        raise SystemExit("Training CSV must contain 'price_range' target column")

    X = df.drop(columns=["price_range"]).select_dtypes(include=["number"]).fillna(0)
    y = df["price_range"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    logger.info("Validation accuracy: %.4f", acc)

    out = Path(output_path)

    # Decide base name
    base_name = None
    if getattr(args, "name", None):
        base_name = args.name
        if getattr(args, "model_version", None):
            base_name = f"{base_name}_v{args.model_version}"
    else:
        if out.suffix:
            base_name = out.stem

    is_dir_target = str(output_path).endswith(os.sep) or (out.exists() and out.is_dir())
    suffix = out.suffix or ".joblib"

    if base_name:
        filename = base_name
        if args.date_version:
            date_str = datetime.now().strftime(args.date_format or "%Y%m%d")
            filename = f"{filename}_{date_str}"
        filename = f"{filename}{suffix}"
        if is_dir_target:
            out_dir = out if out.exists() and out.is_dir() else Path(output_path)
            out = out_dir / filename
        else:
            out = out.with_name(filename)
    else:
        if args.date_version:
            date_str = datetime.now().strftime(args.date_format or "%Y%m%d")
            if is_dir_target:
                out_dir = out if out.exists() and out.is_dir() else Path(output_path)
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / f"model_{date_str}{suffix}"
            else:
                stem = out.stem
                out = out.with_name(f"{stem}_{date_str}{suffix}")

    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.NamedTemporaryFile(dir=str(out.parent), delete=False) as tf:
            tmp_path = Path(tf.name)
        joblib.dump(model, tmp_path)
        os.replace(tmp_path, out)

        meta = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "accuracy": float(acc),
            "n_estimators": 200,
            "model_path": str(out),
        }
        meta_path = out.with_suffix(out.suffix + ".json")
        with tempfile.NamedTemporaryFile(mode="w", dir=str(meta_path.parent), delete=False, encoding="utf-8") as mtf:
            mtf.write(json.dumps(meta, indent=2))
            tmp_meta = Path(mtf.name)
        os.replace(tmp_meta, meta_path)

        logger.info("Saved model to %s", out)
        logger.info("Wrote metadata to %s", meta_path)
    except Exception:
        logger.exception("Failed to save model")
        raise


if __name__ == "__main__":
    main()
