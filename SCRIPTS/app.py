from __future__ import annotations

import json
import os
from datetime import date
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from SCRIPTS.utils import parse_flight_number, ConfidenceBand, confidence_badge_level


ARTIFACT_DIR = "OUTPUTS"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")
META_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")
LOOKUP_PATH = os.path.join(ARTIFACT_DIR, "flight_lookup.csv")


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_metadata():
    if not os.path.exists(META_PATH):
        return None
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_lookup():
    if not os.path.exists(LOOKUP_PATH):
        return None
    try:
        df = pd.read_csv(LOOKUP_PATH)
        return df
    except Exception:
        return None


def resolve_defaults(flight_input: str, lookup_df: Optional[pd.DataFrame]):
    parsed = parse_flight_number(flight_input)
    if parsed and lookup_df is not None:
        carrier, number = parsed
        key = f"{carrier}{number}"
        row = lookup_df.loc[lookup_df["flight_key"] == key]
        if not row.empty:
            r = row.iloc[0]
            return r["ORIGIN"], r["DEST"], int(r["dep_hour"])  # defaults
    return None, None, None


def main():
    st.set_page_config(page_title="Flight Delay Prediction (MVP)", page_icon="✈️", layout="centered")
    st.title("Flight Delay Prediction (MVP)")
    st.caption("Historical-patterns baseline (zero-cost).")

    model = load_model()
    meta = load_metadata()
    lookup = load_lookup()

    with st.sidebar:
        st.subheader("Model Status")
        if meta is None or model is None:
            st.error("Model artifacts not found. Run training first (see README).")
        else:
            st.success(f"Loaded model: {meta.get('best_model','?')} (MAE ~ {meta.get('selected_mae','?'):.1f} min)")
            st.caption(f"Train: {meta.get('n_train',0):,}  Test: {meta.get('n_test',0):,}")

    st.markdown("---")
    st.subheader("Enter Flight")
    c1, c2 = st.columns([2, 1])
    with c1:
        flight_input = st.text_input("Flight number (e.g., AA1234)", value="UA1234")
    with c2:
        flight_date = st.date_input("Date", value=date.today())

    # Defaults from lookup
    d_origin, d_dest, d_hour = resolve_defaults(flight_input, lookup)

    with st.expander("Details (fallback if route not found)"):
        col1, col2, col3 = st.columns(3)
        origin = col1.text_input("Origin (IATA)", value=d_origin or "DEN")
        dest = col2.text_input("Destination (IATA)", value=d_dest or "LAX")
        dep_hour = col3.number_input("Departure hour (0-23)", min_value=0, max_value=23, value=int(d_hour or 13))

    predict_btn = st.button("Predict Delay")

    if predict_btn:
        if model is None:
            st.error("Model not available. Please run training.")
            return

        parsed = parse_flight_number(flight_input)
        if not parsed:
            st.warning("Invalid flight format. Expect like 'AA1234'.")
            return
        carrier, number = parsed

        # Build single-row feature frame matching training features
        route = f"{origin.upper()}-{dest.upper()}"
        dow = pd.Timestamp(flight_date).weekday()
        month = pd.Timestamp(flight_date).month
        is_weekend = 1 if dow in (5, 6) else 0

        # For features requiring historical averages, we approximate using lookup aggregates if available
        # Load route/origin/airline averages from metadata if present (not stored separately here), fallback to neutral zeros
        route_avg_delay = 0.0
        origin_avg_delay = 0.0
        airline_avg_delay = 0.0

        X = pd.DataFrame([
            {
                "dep_hour": int(dep_hour),
                "dow": int(dow),
                "month": int(month),
                "is_weekend": int(is_weekend),
                "DISTANCE": float(500),  # fallback; true distance is optional for MVP
                "route": route,
                "ORIGIN": origin.upper(),
                "DEST": dest.upper(),
                "OP_CARRIER": carrier,
                "route_avg_delay": float(route_avg_delay),
                "origin_avg_delay": float(origin_avg_delay),
                "airline_avg_delay": float(airline_avg_delay),
            }
        ])

        try:
            y_hat = float(model.predict(X)[0])
        except Exception as exc:
            st.error(f"Unable to generate prediction: {exc}")
            return

        mae = float(meta.get("selected_mae", 20.0)) if meta else 20.0
        band = ConfidenceBand(point_minutes=y_hat, half_width_minutes=mae)
        badge = confidence_badge_level(mae)

        st.markdown("---")
        st.subheader(f"Flight {flight_input.upper()} on {flight_date.isoformat()} — {origin.upper()} → {dest.upper()}")
        st.metric(label=f"Predicted Delay (± MAE)", value=f"{y_hat:.0f} min", delta=f"± {mae:.0f} min")
        st.caption(f"Confidence: {badge}")

        # Feature contributions (simple proxy): show input features in a table
        with st.expander("Inputs used"):
            st.dataframe(X.T, use_container_width=True)

        st.info("This MVP uses historical patterns only. Add real-time weather later for improved accuracy.")


if __name__ == "__main__":
    main()

