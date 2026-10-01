from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import joblib
import pandas as pd
import json
import threading

app = FastAPI(title="Mobile Price Predictor API")

MODELS_DIR = Path("models")

# simple in-memory model registry
_models: Dict[str, Any] = {}
_models_lock = threading.Lock()

PRICE_LABELS = {
    0: "Low",
    1: "Medium",
    2: "High",
    3: "Very High",
}


class PredictRequest(BaseModel):
    model: Optional[str] = None
    features: Dict[str, Any]


def _load_models():
    with _models_lock:
        _models.clear()
        if MODELS_DIR.exists():
            for p in MODELS_DIR.glob("*.joblib"):
                try:
                    m = joblib.load(p)
                    _models[p.name] = {"model": m, "path": str(p)}
                except Exception:
                    # skip problematic files
                    continue


@app.on_event("startup")
def startup():
    _load_models()


@app.get("/models", response_model=List[str])
def list_models():
    return list(_models.keys())


@app.post("/reload")
def reload_models():
    _load_models()
    return {"loaded": len(_models)}


@app.post("/predict")
def predict(req: PredictRequest):
    if not _models:
        raise HTTPException(status_code=404, detail="No models available")
    # choose model
    if req.model:
        key = req.model
        if key not in _models:
            raise HTTPException(status_code=404, detail=f"Model {key} not found")
        mobj = _models[key]["model"]
    else:
        # pick most recently modified
        keys = sorted(_models.keys(), key=lambda k: Path(_models[k]["path"]).stat().st_mtime, reverse=True)
        mobj = _models[keys[0]]["model"]

    # build DataFrame with a single row
    try:
        df = pd.DataFrame([req.features])

        # If the model exposes expected feature names, align input to them
        expected = None
        if hasattr(mobj, "feature_names_in_"):
            try:
                expected = list(mobj.feature_names_in_)
            except Exception:
                expected = None

        if expected is not None:
            # ensure all expected columns exist, fill missing with zeros
            for c in expected:
                if c not in df.columns:
                    df[c] = 0
            # drop unexpected columns
            unexpected = [c for c in df.columns if c not in expected]
            if unexpected:
                df = df.drop(columns=unexpected)
            # reorder columns to expected order
            df = df[expected]
            # coerce numeric where possible
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            X = df
        else:
            # fallback: use numeric columns only
            X = df.select_dtypes(include=["number"]).fillna(0)

        preds = mobj.predict(X)
        # Map numeric predictions to human-friendly labels when possible
        try:
            labels = [PRICE_LABELS.get(int(p), str(p)) for p in preds.tolist()]
        except Exception:
            labels = [str(p) for p in preds.tolist()]
        return {"predictions": preds.tolist(), "labels": labels}
    except Exception as exc:
        # Return a clearer error for bad input
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
