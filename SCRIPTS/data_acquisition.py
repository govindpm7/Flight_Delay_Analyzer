import argparse
import os
import sys
from typing import List

import pandas as pd

from utils import MAJOR_HUBS


REQUIRED_COLUMNS = [
    "OP_CARRIER",  # Airline code
    "FL_NUM",      # Flight number (numeric)
    "ORIGIN",      # IATA code
    "DEST",        # IATA code
    "CRS_DEP_TIME",# Scheduled departure (HHMM int/string)
    "DEP_DELAY",   # Departure delay minutes (target)
    "FL_DATE",     # YYYY-MM-DD
    "DISTANCE",    # Miles
]


def read_all_csvs(input_dir: str) -> pd.DataFrame:
    files: List[str] = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError(f"No CSVs found in {input_dir}")
    frames = []
    for path in files:
        try:
            frames.append(pd.read_csv(path, low_memory=False))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
    if not frames:
        raise RuntimeError("No readable CSVs found.")
    df = pd.concat(frames, ignore_index=True)
    return df


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def filter_domestic_hubs(df: pd.DataFrame) -> pd.DataFrame:
    # Domestic-only assumption: BTS datasets for US are domestic; if international rows exist, drop them by simple heuristic (3-letter IATA both sides).
    df = df[df["ORIGIN"].str.len() == 3]
    df = df[df["DEST"].str.len() == 3]
    # Focus on routes touching major hubs
    mask = df["ORIGIN"].isin(MAJOR_HUBS) | df["DEST"].isin(MAJOR_HUBS)
    return df[mask].copy()


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    # CRS_DEP_TIME often HHMM like 1345; convert to hour 0-23
    def _to_hour(v):
        try:
            iv = int(v)
            return max(0, min(23, iv // 100))
        except Exception:
            return pd.NA
    df["DEP_HOUR"] = df["CRS_DEP_TIME"].apply(_to_hour)
    df = df.dropna(subset=["DEP_HOUR"])  # drop rows without hour
    df["DEP_HOUR"] = df["DEP_HOUR"].astype(int)
    # Normalize identifiers
    df["OP_CARRIER"] = df["OP_CARRIER"].astype(str).str.upper().str.strip()
    df["FL_NUM"] = df["FL_NUM"].astype(str).str.strip()
    df["ORIGIN"] = df["ORIGIN"].astype(str).str.upper().str.strip()
    df["DEST"] = df["DEST"].astype(str).str.upper().str.strip()
    # Target
    df = df[pd.to_numeric(df["DEP_DELAY"], errors="coerce").notna()].copy()
    df["DEP_DELAY"] = df["DEP_DELAY"].astype(float)
    # Drop extreme outliers > 300 min
    df = df[df["DEP_DELAY"].between(-30, 300)]  # allow early departures as negative small values
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = read_all_csvs(args.input_dir)
    df = ensure_required_columns(df)
    df = filter_domestic_hubs(df)
    df = coerce_types(df)

    df.to_csv(args.output, index=False)
    print(f"Wrote processed data: {args.output}  rows={len(df):,}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

