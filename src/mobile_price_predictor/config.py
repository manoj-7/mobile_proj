import yaml
from pathlib import Path


def load_config(path: str | Path):
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
