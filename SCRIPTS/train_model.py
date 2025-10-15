from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from dateutil import parser as date_parser
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import MAJOR_HUBS, normalize_flight_key


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    data = df.copy()
    data["FL_DATE"] = pd.to_datetime(data["FL_DATE"], errors="coerce")
    data = data.dropna(subset=["FL_DATE"]).copy()

    data["dep_hour"] = data["DEP_HOUR"].astype(int)
    data["dow"] = data["FL_DATE"].dt.weekday
    data["month"] = data["FL_DATE"].dt.month
    data["is_weekend"] = data["dow"].isin([5, 6]).astype(int)

    # Route and keys
    data["route"] = data["ORIGIN"] + "-" + data["DEST"]
    data["flight_key"] = data.apply(lambda r: normalize_flight_key(r["OP_CARRIER"], r["FL_NUM"]), axis=1)

    # Aggregates for historical patterns
    route_avg = data.groupby("route")["DEP_DELAY"].mean().rename("route_avg_delay")
    origin_avg = data.groupby("ORIGIN")["DEP_DELAY"].mean().rename("origin_avg_delay")
    airline_avg = data.groupby("OP_CARRIER")["DEP_DELAY"].mean().rename("airline_avg_delay")

    data = data.merge(route_avg, on="route", how="left")
    data = data.merge(origin_avg, on="ORIGIN", how="left")
    data = data.merge(airline_avg, on="OP_CARRIER", how="left")

    # Target
    y = data["DEP_DELAY"].astype(float)

    # Feature matrix
    X = data[[
        "dep_hour", "dow", "month", "is_weekend", "DISTANCE",
        "route", "ORIGIN", "DEST", "OP_CARRIER",
        "route_avg_delay", "origin_avg_delay", "airline_avg_delay",
    ]].copy()

    return X, y


def temporal_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series, test_size: float = 0.2) -> Tuple:
    # Use a simple chronological split by date
    order = np.argsort(dates.values)
    split_idx = int(len(order) * (1 - test_size))
    train_idx, test_idx = order[:split_idx], order[split_idx:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def train_models(X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, dict]:
    numeric_features = ["dep_hour", "dow", "month", "is_weekend", "DISTANCE",
                        "route_avg_delay", "origin_avg_delay", "airline_avg_delay"]
    categorical_features = ["route", "ORIGIN", "DEST", "OP_CARRIER"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
        ]
    )

    lin_reg = Pipeline([
        ("pre", preprocessor),
        ("model", LinearRegression(n_jobs=None)),
    ])

    rf = Pipeline([
        ("pre", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )),
    ])

    models = {"linear": lin_reg, "random_forest": rf}

    return models, preprocessor


def evaluate(model: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)
    r2 = r2_score(y_test, pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "pred": pred}


def build_flight_lookup(df: pd.DataFrame, out_path: str):
    # Map flight_key to most common route and typical hour
    tmp = (
        df.assign(flight_key=lambda d: d.apply(lambda r: normalize_flight_key(r["OP_CARRIER"], r["FL_NUM"]), axis=1))
          .groupby(["flight_key", "ORIGIN", "DEST", "DEP_HOUR"])
          .size()
          .reset_index(name="count")
    )
    # For each flight_key, pick the (origin, dest, hour) with max count
    idx = tmp.groupby("flight_key")["count"].idxmax()
    top = tmp.loc[idx, ["flight_key", "ORIGIN", "DEST", "DEP_HOUR", "count"]].reset_index(drop=True)
    top.rename(columns={"DEP_HOUR": "dep_hour"}, inplace=True)
    top.to_csv(out_path, index=False)


def run_pca_report(preprocessor: ColumnTransformer, X_train: pd.DataFrame, out_path: str):
    # Fit preprocessor to get dense feature matrix, then PCA for analysis only
    Xt = preprocessor.fit_transform(X_train)
    pca = PCA(n_components=min(10, Xt.shape[1]))
    pca.fit(Xt)
    report = {
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "n_components": int(pca.n_components_),
        "total_explained": float(pca.explained_variance_ratio_.sum()),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Processed CSV from data_acquisition")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    # Build features
    X, y = build_features(df)

    # Temporal split
    dates = pd.to_datetime(df.loc[X.index, "FL_DATE"], errors="coerce")
    X_train, X_test, y_train, y_test = temporal_split(X, y, dates, test_size=0.2)

    models, pre = train_models(X, y)

    metrics = {}
    best_name, best_model, best_mae = None, None, float("inf")
    for name, model in models.items():
        m = evaluate(model, X_train, y_train, X_test, y_test)
        metrics[name] = {k: v for k, v in m.items() if k != "pred"}
        if m["mae"] < best_mae:
            best_name, best_model, best_mae = name, model, m["mae"]

    # Persist artifacts
    joblib.dump(best_model, os.path.join(args.out_dir, "model.joblib"))
    # Preprocessor is embedded in pipeline; still export a standalone fitted preprocessor for analysis
    joblib.dump(best_model.named_steps["pre"], os.path.join(args.out_dir, "preprocessor.joblib"))

    # PCA analysis report (for documentation)
    run_pca_report(best_model.named_steps["pre"], X_train, os.path.join(args.out_dir, "pca_report.json"))

    # Flight lookup for app routing defaults
    build_flight_lookup(df, os.path.join(args.out_dir, "flight_lookup.csv"))

    summary = {
        "best_model": best_name,
        "metrics": metrics,
        "selected_mae": float(best_mae),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "major_hubs": sorted(list(MAJOR_HUBS)),
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

