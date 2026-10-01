import requests
from pathlib import Path
import streamlit as st
import pandas as pd
import re

API_URL = st.secrets.get("api_url") if "api_url" in st.secrets else "http://localhost:8000"

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

def feature_display(col: str) -> str:
    desc = DESCRIPTIONS.get(col, "")
    if desc:
        desc_clean = re.sub(r"\(\s*0\s*/?\s*1\s*\)", "", desc)
        desc_clean = desc_clean.strip()
        return f"{desc_clean} ({col})"
    return col


def get_models():
    try:
        r = requests.get(f"{API_URL}/models", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def main():
    st.set_page_config(page_title="Mobile Price Predictor (API)", layout="wide")
    st.title("Mobile Price Predictor — API-backed UI")

    models = get_models()
    if not models:
        st.warning("No models available from API. Start the backend at http://localhost:8000 and reload.")
    model_choice = st.selectbox("Select model", options=[""] + models)

    st.markdown("---")
    st.write("Enter features below and press Predict to call the API.")

    # minimal feature inputs: use DESCRIPTIONS keys
    cols = [c for c in DESCRIPTIONS.keys() if c not in ("id", "price_range")]
    inputs = {}
    for c in cols:
        val = st.text_input(feature_display(c), key=f"inp_{c}")
        inputs[c] = val

    if st.button("Predict"):
        # build feature dict: cast numeric where possible
        features = {}
        for k, v in inputs.items():
            if v is None or v == "":
                continue
            try:
                features[k] = float(v)
            except Exception:
                features[k] = v

        payload = {"features": features}
        if model_choice:
            payload["model"] = model_choice

        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
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


if __name__ == "__main__":
    main()
