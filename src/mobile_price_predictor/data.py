import pandas as pd
from pathlib import Path


def load_data(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    return pd.read_csv(p)
