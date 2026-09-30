import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from mobile_price_predictor.config import load_config


def split_file(input_path: str, train_out: str, test_out: str, test_size: float = 0.2, random_state: Optional[int] = 42):
    df = pd.read_csv(input_path)
    # if there's a target column use stratify
    target = df.columns.intersection(["price_range", "target"]).tolist()
    stratify = None
    if target:
        stratify = df[target[0]]

    train = df.sample(frac=1 - test_size, random_state=random_state)
    test = df.drop(train.index)

    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    Path(test_out).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_out, index=False)
    test.to_csv(test_out, index=False)
    print(f"Wrote train -> {train_out} ({len(train)}) and test -> {test_out} ({len(test)})")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, help="Single CSV file with all data")
    parser.add_argument("--train-out", required=False, help="Path to write train CSV")
    parser.add_argument("--test-out", required=False, help="Path to write test CSV")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test set (0-1)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--config", required=False, help="Optional YAML config to read defaults")
    args = parser.parse_args(argv)

    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    # Accept single-file config via data.all as convenience
    input_path = args.input or (cfg.get("data") or {}).get("train") or (cfg.get("data") or {}).get("all")
    train_out = args.train_out or (cfg.get("data") or {}).get("train")
    test_out = args.test_out or (cfg.get("data") or {}).get("test")

    if not input_path:
        raise SystemExit("--input or config.data.train must be provided")
    if not train_out or not test_out:
        raise SystemExit("--train-out and --test-out or config paths must be provided")

    split_file(input_path, train_out, test_out, test_size=args.test_size, random_state=args.random_state)


if __name__ == "__main__":
    main()
