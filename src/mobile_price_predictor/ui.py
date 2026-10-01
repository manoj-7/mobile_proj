import glob
from pathlib import Path

import joblib
import pandas as pd
import re
import streamlit as st
import requests
import os

from mobile_price_predictor.config import load_config
import time


DESCRIPTIONS = {
    "battery_power": "Battery power in mAh",
    "blue": "Has Bluetooth (0/1)",
    "clock_speed": "Processor clock speed (GHz)",
    "dual_sim": "Supports dual SIM (0/1)",
    "fc": "Front camera megapixels",
    "four_g": "Supports 4G (0/1)",
    "int_memory": "Internal memory (GB)",
    "m_dep": "Mobile depth (cm)",
    "mobile_wt": "Mobile weight (grams)",
    "n_cores": "Number of processor cores",
    "pc": "Primary camera megapixels",
    "px_height": "Pixel resolution height",
    "px_width": "Pixel resolution width",
    "ram": "RAM (MB)",
    "sc_h": "Screen height (cm)",
    "sc_w": "Screen width (cm)",
    "talk_time": "Battery talk time (hours)",
    "three_g": "Supports 3G (0/1)",
    "touch_screen": "Has touch screen (0/1)",
    "wifi": "Has WiFi (0/1)",
    "id": "Row identifier (not a feature)",
    "price_range": "Target price range (0-3)",
}

PRICE_LABELS = {
    0: "Low",
    1: "Medium",
    2: "High",
    3: "Very High",
}


def get_config():
    p = Path("config.yaml")
    if p.exists():
        return load_config(str(p))
    return {}


def list_models(cfg):
    models = sorted(glob.glob("models/*.joblib"))
    cfg_model = (cfg.get("paths") or {}).get("model")
    if cfg_model and cfg_model not in models and Path(cfg_model).exists():
        models.insert(0, cfg_model)
    return models


def load_sample_df(cfg):
    test_path = (cfg.get("data") or {}).get("test") or "data/test.csv"
    p = Path(test_path)
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(ttl=300)
def get_sample_df_cached(path: str):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return None


def feature_display(col: str) -> str:
    desc = DESCRIPTIONS.get(col, "")
    if desc:
        desc_clean = re.sub(r"\(\s*0\s*/?\s*1\s*\)", "", desc)
        desc_clean = desc_clean.strip()
        return f"{desc_clean} ({col})"
    return col


def feature_label(col: str) -> str:
    desc = DESCRIPTIONS.get(col, "")
    if desc:
        return re.sub(r"\(\s*0\s*/?\s*1\s*\)", "", desc).strip()
    return col


def available_features(sample_df: pd.DataFrame | None):
    if sample_df is not None:
        cols = [c for c in sample_df.columns.tolist() if c != "price_range"]
    else:
        cols = [c for c in DESCRIPTIONS.keys() if c != "price_range"]
    cols = [c for c in cols if c != "id"]
    return cols


def main():
    st.set_page_config(page_title="Mobile Price Predictor", layout="wide")
    st.title("Mobile Price Predictor — Inference UI (API-backed)")
    cfg = get_config()

    # Require API URL — no local fallback allowed
    api_url = os.environ.get("MOBILE_API_URL") or (cfg.get("api") or {}).get("url")
    if not api_url:
        st.error("API URL is not configured. Set MOBILE_API_URL env var or add 'api.url' in config.yaml.")
        return

    st.info(f"Using backend API at {api_url}")

    # fetch model names from API (cached in session_state; refresh on demand)
    if "api_models" not in st.session_state:
        st.session_state.api_models = None
    col_refresh = st.columns([1, 4])
    with col_refresh[0]:
        if st.button("Refresh models", key="refresh_api"):
            st.session_state.api_models = None
    if st.session_state.api_models is None:
        try:
            resp = requests.get(f"{api_url.rstrip('/')}/models", timeout=3)
            resp.raise_for_status()
            st.session_state.api_models = resp.json()
        except Exception as exc:
            st.error(f"Failed to query models from API: {exc}")
            st.session_state.api_models = []

    api_models = st.session_state.api_models or []
    model_choice = st.selectbox("Select model file", options=[""] + api_models, key="select_model_api")

    test_path = (cfg.get("data") or {}).get("test") or "data/test.csv"
    sample_df = get_sample_df_cached(test_path)

    st.markdown("---")
    st.write("Select features to use in prediction and enter values below.")

    # prefer model-provided features in the future; for now use sample data or descriptions
    cols = available_features(sample_df)
    cols = [c for c in cols if c != "id"]
    display_options = [feature_display(c) for c in cols]
    selection = st.multiselect("Choose features", options=display_options, key="choose_features_api")

    selected_cols = []
    for disp in selection:
        if disp.endswith(")") and "(" in disp:
            col = disp.split("(")[-1].rstrip(")")
        else:
            col = disp
        if col in cols:
            selected_cols.append(col)

    st.header("Enter feature values")
    if selected_cols:
        st.write("Selected features:")
        for c in selected_cols:
            st.write(f"- {feature_label(c)}")
    else:
        st.write("No features selected")

    inputs = {}
    if st.button("Reset fields", key="reset_api"):
        for c in selected_cols:
            key = f"inp_{c}"
            if key in st.session_state:
                st.session_state[key] = ""
    for c in selected_cols:
        # If sample data is available, mimic local widgets (binary selectboxes, numeric text inputs, categorical selects)
        if sample_df is not None and c in sample_df.columns and pd.api.types.is_numeric_dtype(sample_df[c]):
            vals = pd.unique(sample_df[c].dropna())
            vals_set = set([str(v) for v in vals.tolist()]) if hasattr(vals, 'tolist') else set([str(vals)])
            vmin = float(sample_df[c].min())
            vmax = float(sample_df[c].max())
            placeholder = f"min: {vmin}, max: {vmax}"
            key = f"inp_{c}"
            if vals_set.issubset({"0", "1", "0.0", "1.0", "True", "False"}) or set(vals.tolist()).issubset({0, 1}):
                val = st.selectbox(f"{feature_label(c)}", options=["", "Yes", "No"], key=key)
                if val == "":
                    inputs[c] = {"value": None, "min": 0, "max": 1, "type": "number", "widget": "binary"}
                else:
                    inputs[c] = {"value": (1 if val == "Yes" else 0), "min": 0, "max": 1, "type": "number", "widget": "binary"}
            else:
                val = st.text_input(f"{feature_label(c)}", placeholder=placeholder, key=key)
                inputs[c] = {"value": val, "min": vmin, "max": vmax, "type": "number", "widget": "text"}
        elif sample_df is not None and c in sample_df.columns:
            opts = pd.unique(sample_df[c].dropna())
            vals = opts.tolist() if hasattr(opts, 'tolist') else [opts]
            vals_set = set([str(v) for v in vals])
            key = f"inp_{c}"
            if vals_set.issubset({"0", "1", "True", "False"}) or set(vals).issubset({0, 1, True, False}):
                val = st.selectbox(f"{feature_label(c)}", options=["", "Yes", "No"], key=key)
                if val == "":
                    inputs[c] = {"value": None, "type": "number", "widget": "binary"}
                else:
                    inputs[c] = {"value": (1 if val == "Yes" else 0), "type": "number", "widget": "binary"}
            else:
                opt_list = [str(o) for o in vals]
                val = st.selectbox(f"{feature_label(c)}", options=[""] + opt_list, key=key)
                inputs[c] = {"value": (val if val != "" else None), "type": "text", "widget": "select"}
        else:
            val = st.text_input(f"{feature_label(c)}", key=f"inp_{c}")
            inputs[c] = {"value": val, "type": "text"}

    if st.button("Predict", key="predict_api"):
        payload = {"features": {}}
        for k, meta in inputs.items():
            sval = meta.get("value")
            if sval is None or sval == "":
                continue
            try:
                payload["features"][k] = float(sval)
            except Exception:
                payload["features"][k] = sval
        if model_choice:
            payload["model"] = model_choice
        try:
            r = requests.post(f"{api_url.rstrip('/')}/predict", json=payload, timeout=5)
            r.raise_for_status()
            data = r.json()
            labels = data.get("labels")
            preds = data.get("predictions")
            if labels:
                st.success(f"Prediction: {labels}")
            elif preds is not None:
                st.success(f"Prediction: {preds}")
            else:
                st.success("Prediction returned (no labels)")
        except Exception as exc:
            st.error(f"API request failed: {exc}")

    # When API mode is configured, do not render local model UI
    return

    models = list_models(cfg)
    if not models:
        st.warning("No model files found in models/*.joblib. Train a model first.")
    model_choice = st.selectbox("Select model file", options=models or [""], format_func=lambda x: Path(x).name, key="select_model_local")

    model = None
    if model_choice:
        try:
            # cache loaded models in session to avoid reloading on each rerun
            if "model_cache" not in st.session_state:
                st.session_state.model_cache = {}
            if model_choice in st.session_state.model_cache:
                model = st.session_state.model_cache[model_choice]
            else:
                model = joblib.load(model_choice)
                st.session_state.model_cache[model_choice] = model
            st.success(f"Loaded model: {Path(model_choice).name}")
        except Exception as exc:
            st.error(f"Failed to load model: {exc}")

    test_path = (cfg.get("data") or {}).get("test") or "data/test.csv"
    sample_df = get_sample_df_cached(test_path)

    st.markdown("---")
    st.write("Select features to use in prediction and enter values below.")

    # Determine available columns (prefer model features, then sample data, then DESCRIPTIONS)
    if model is not None and hasattr(model, "feature_names_in_"):
        try:
            cols = list(model.feature_names_in_)
        except Exception:
            cols = available_features(sample_df)
    else:
        cols = available_features(sample_df)
    # filter out 'id' if present
    cols = [c for c in cols if c != "id"]
    display_options = [feature_display(c) for c in cols]
    selection = st.multiselect("Choose features", options=display_options, key="choose_features_local")

    selected_cols = []
    for disp in selection:
        if disp.endswith(")") and "(" in disp:
            col = disp.split("(")[-1].rstrip(")")
        else:
            col = disp
        if col in cols:
            selected_cols.append(col)

    st.header("Enter feature values")
    if selected_cols:
        st.write("Selected features:")
        for c in selected_cols:
            st.write(f"- {feature_label(c)}")
    else:
        st.write("No features selected")

    inputs = {}
    if st.button("Reset fields", key="reset_local"):
        for c in selected_cols:
            key = f"inp_{c}"
            if key in st.session_state:
                st.session_state[key] = ""
    for c in selected_cols:
        if sample_df is not None and c in sample_df.columns and pd.api.types.is_numeric_dtype(sample_df[c]):
            vals = pd.unique(sample_df[c].dropna())
            vals_set = set([str(v) for v in vals.tolist()]) if hasattr(vals, 'tolist') else set([str(vals)])
            vmin = float(sample_df[c].min())
            vmax = float(sample_df[c].max())
            placeholder = f"min: {vmin}, max: {vmax}"
            key = f"inp_{c}"
            if vals_set.issubset({"0", "1", "0.0", "1.0", "True", "False"}) or set(vals.tolist()).issubset({0, 1}):
                val = st.selectbox(f"{feature_label(c)}", options=["", "Yes", "No"], key=key)
                if val == "":
                    inputs[c] = {"value": None, "min": 0, "max": 1, "type": "number", "widget": "binary"}
                else:
                    mapped = 1 if val == "Yes" else 0
                    inputs[c] = {"value": mapped, "min": 0, "max": 1, "type": "number", "widget": "binary"}
            else:
                val = st.text_input(f"{feature_label(c)}", placeholder=placeholder, key=key)
                inputs[c] = {"value": val, "min": vmin, "max": vmax, "type": "number", "widget": "text"}
        elif sample_df is not None and c in sample_df.columns:
            opts = pd.unique(sample_df[c].dropna())
            vals = opts.tolist() if hasattr(opts, 'tolist') else [opts]
            vals_set = set([str(v) for v in vals])
            key = f"inp_{c}"
            if vals_set.issubset({"0", "1", "True", "False"}) or set(vals).issubset({0, 1, True, False}):
                val = st.selectbox(f"{feature_label(c)}", options=["", "Yes", "No"], key=key)
                if val == "":
                    inputs[c] = {"value": None, "type": "number", "widget": "binary"}
                else:
                    inputs[c] = {"value": (1 if val == "Yes" else 0), "type": "number", "widget": "binary"}
            else:
                opt_list = [str(o) for o in vals]
                val = st.selectbox(f"{feature_label(c)}", options=[""] + opt_list, key=key)
                inputs[c] = {"value": (val if val != "" else None), "type": "text", "widget": "select"}
        else:
            st.write(f"{feature_display(c)} — no sample data available; enter a value")
            val = st.text_input(f"{c}", placeholder="enter value", key=f"inp_{c}")
            inputs[c] = {"value": val, "type": "number", "min": None, "max": None}

    if st.button("Preview aligned features", key="preview_local") and model is not None:
        try:
            row = {}
            for k, meta in inputs.items():
                if meta.get("type") == "number":
                    sval = meta.get("value")
                    if sval is None or sval == "":
                        row[k] = 0
                    else:
                        row[k] = float(sval)
                else:
                    row[k] = meta.get("value")

            df_row = pd.DataFrame([row])

            expected = None
            if hasattr(model, "feature_names_in_"):
                expected = list(model.feature_names_in_)

            if expected is not None:
                missing = [c for c in expected if c not in df_row.columns]
                for m in missing:
                    df_row[m] = 0
                unexpected = [c for c in df_row.columns if c not in expected]
                if unexpected:
                    df_row = df_row.drop(columns=unexpected)
                df_row = df_row[expected]

            display_df = df_row.copy()
            binary_cols = []
            if sample_df is not None:
                for col in display_df.columns:
                    if col in sample_df.columns:
                        vals = pd.unique(sample_df[col].dropna())
                        vals_set = set([str(v) for v in vals.tolist()]) if hasattr(vals, 'tolist') else set([str(vals)])
                        if vals_set.issubset({"0", "1", "True", "False"}):
                            binary_cols.append(col)
            for b in binary_cols:
                display_df[b] = display_df[b].apply(lambda x: "Yes" if str(x) in {"1", "True"} else "No")

            st.subheader("Aligned feature row")
            st.dataframe(display_df)
            st.info("If this looks correct you can press Predict to run the model.")
        except Exception as exc:
            st.error(f"Failed to prepare preview: {exc}")

    if st.button("Predict", key="predict_local"):
        if model is None:
            st.error("No model loaded")
        else:
            try:
                row = {}
                for k, meta in inputs.items():
                    if meta.get("type") == "number":
                        sval = meta.get("value")
                        if sval is None or sval == "":
                            st.error(f"Numeric value required for {k}")
                            raise ValueError("missing numeric")
                        val = float(sval)
                        if meta.get("min") is not None and meta.get("max") is not None:
                            if not (meta.get("min") <= val <= meta.get("max")):
                                st.warning(f"Value for {k} is outside sample min/max ({meta.get('min')} - {meta.get('max')}).")
                        row[k] = val
                    else:
                        row[k] = meta.get("value")

                df_row = pd.DataFrame([row])

                expected = None
                if hasattr(model, "feature_names_in_"):
                    expected = list(model.feature_names_in_)

                if expected is not None:
                    missing = [c for c in expected if c not in df_row.columns]
                    for m in missing:
                        df_row[m] = 0
                    unexpected = [c for c in df_row.columns if c not in expected]
                    if unexpected:
                        st.warning(f"Dropped unexpected columns before predict: {unexpected}")
                        df_row = df_row.drop(columns=unexpected)
                    df_row = df_row[expected]

                preds = model.predict(df_row.select_dtypes(include=["number"]).fillna(0))
                mapped = [PRICE_LABELS.get(int(p), str(p)) for p in preds.tolist()]
                st.success(f"Prediction: {mapped}")
            except ValueError:
                pass
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
