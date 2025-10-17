import argparse
import os
import sys
from typing import List, Dict, Optional

import pandas as pd

from utils import MAJOR_HUBS


# Traditional flight data columns
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

# BTS delay cause data columns
BTS_COLUMNS = [
    "year", "month", "carrier", "carrier_name", "airport", "airport_name",
    "arr_flights", "arr_del15", "carrier_ct", "weather_ct", "nas_ct", 
    "security_ct", "late_aircraft_ct", "arr_cancelled", "arr_diverted",
    "arr_delay", "carrier_delay", "weather_delay", "nas_delay", 
    "security_delay", "late_aircraft_delay"
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


def read_bts_data(input_dir: str) -> Optional[pd.DataFrame]:
    """Read and process BTS delay cause data"""
    bts_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower() == "airline_delay_cause.csv":
                bts_files.append(os.path.join(root, file))
    
    if not bts_files:
        print("No BTS delay cause data found")
        return None
    
    frames = []
    for path in bts_files:
        try:
            df = pd.read_csv(path, low_memory=False)
            # Check if this looks like BTS data
            if any(col in df.columns for col in ["carrier", "airport", "carrier_delay"]):
                frames.append(df)
                print(f"Loaded BTS data from {path}: {len(df)} rows")
        except Exception as exc:
            print(f"Skipping BTS file {path}: {exc}")
    
    if not frames:
        return None
    
    return pd.concat(frames, ignore_index=True)


def process_bts_data(bts_df: pd.DataFrame) -> pd.DataFrame:
    """Process BTS data into features for model training"""
    df = bts_df.copy()
    
    # Convert year/month to date
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    
    # Calculate delay rates and percentages
    df['total_delay_rate'] = df['arr_del15'] / df['arr_flights'].replace(0, 1)
    df['carrier_delay_rate'] = df['carrier_ct'] / df['arr_flights'].replace(0, 1)
    df['weather_delay_rate'] = df['weather_ct'] / df['arr_flights'].replace(0, 1)
    df['nas_delay_rate'] = df['nas_ct'] / df['arr_flights'].replace(0, 1)
    df['security_delay_rate'] = df['security_ct'] / df['arr_flights'].replace(0, 1)
    df['late_aircraft_delay_rate'] = df['late_aircraft_ct'] / df['arr_flights'].replace(0, 1)
    
    # Calculate average delay minutes per delayed flight
    df['avg_delay_minutes'] = df['arr_delay'] / df['arr_del15'].replace(0, 1)
    df['avg_carrier_delay'] = df['carrier_delay'] / df['carrier_ct'].replace(0, 1)
    df['avg_weather_delay'] = df['weather_delay'] / df['weather_ct'].replace(0, 1)
    df['avg_nas_delay'] = df['nas_delay'] / df['nas_ct'].replace(0, 1)
    df['avg_security_delay'] = df['security_delay'] / df['security_ct'].replace(0, 1)
    df['avg_late_aircraft_delay'] = df['late_aircraft_delay'] / df['late_aircraft_ct'].replace(0, 1)
    
    # Clean up carrier and airport codes
    df['carrier'] = df['carrier'].astype(str).str.upper().str.strip()
    df['airport'] = df['airport'].astype(str).str.upper().str.strip()
    
    # Filter to major hubs if specified
    if MAJOR_HUBS:
        df = df[df['airport'].isin(MAJOR_HUBS)]
    
    return df


def merge_flight_bts_data(flight_df: pd.DataFrame, bts_df: pd.DataFrame) -> pd.DataFrame:
    """Merge traditional flight data with BTS delay cause data"""
    # Create lookup from BTS data for each carrier-airport-month combination
    bts_lookup = bts_df.groupby(['carrier', 'airport', 'year', 'month']).agg({
        'total_delay_rate': 'mean',
        'carrier_delay_rate': 'mean',
        'weather_delay_rate': 'mean',
        'nas_delay_rate': 'mean',
        'security_delay_rate': 'mean',
        'late_aircraft_delay_rate': 'mean',
        'avg_delay_minutes': 'mean',
        'avg_carrier_delay': 'mean',
        'avg_weather_delay': 'mean',
        'avg_nas_delay': 'mean',
        'avg_security_delay': 'mean',
        'avg_late_aircraft_delay': 'mean',
        'arr_flights': 'sum'
    }).reset_index()
    
    # Add date columns to flight data for merging
    flight_df['year'] = pd.to_datetime(flight_df['FL_DATE']).dt.year
    flight_df['month'] = pd.to_datetime(flight_df['FL_DATE']).dt.month
    
    # Merge BTS data for origin airport
    merged = flight_df.merge(
        bts_lookup,
        left_on=['OP_CARRIER', 'ORIGIN', 'year', 'month'],
        right_on=['carrier', 'airport', 'year', 'month'],
        how='left',
        suffixes=('', '_origin')
    )
    
    # Merge BTS data for destination airport
    merged = merged.merge(
        bts_lookup,
        left_on=['OP_CARRIER', 'DEST', 'year', 'month'],
        right_on=['carrier', 'airport', 'year', 'month'],
        how='left',
        suffixes=('_origin', '_dest')
    )
    
    # Fill missing values with 0
    bts_cols = [col for col in merged.columns if any(x in col for x in ['delay_rate', 'avg_delay', 'arr_flights'])]
    merged[bts_cols] = merged[bts_cols].fillna(0)
    
    return merged


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
    parser.add_argument("--bts_output", help="Output path for processed BTS data")
    parser.add_argument("--merge_bts", action="store_true", help="Merge BTS data with flight data")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Process traditional flight data
    flight_df = None
    try:
        flight_df = read_all_csvs(args.input_dir)
        flight_df = ensure_required_columns(flight_df)
        flight_df = filter_domestic_hubs(flight_df)
        flight_df = coerce_types(flight_df)
        print(f"Processed flight data: {len(flight_df):,} rows")
    except Exception as e:
        print(f"Warning: Could not process traditional flight data: {e}")
        flight_df = None

    # Process BTS data
    bts_df = None
    try:
        bts_raw = read_bts_data(args.input_dir)
        if bts_raw is not None:
            bts_df = process_bts_data(bts_raw)
            print(f"Processed BTS data: {len(bts_df):,} rows")
            
            # Save BTS data separately if requested
            if args.bts_output:
                os.makedirs(os.path.dirname(args.bts_output), exist_ok=True)
                bts_df.to_csv(args.bts_output, index=False)
                print(f"Wrote BTS data: {args.bts_output}")
    except Exception as e:
        print(f"Warning: Could not process BTS data: {e}")
        bts_df = None

    # Merge data if requested and both datasets available
    if args.merge_bts and flight_df is not None and bts_df is not None:
        try:
            merged_df = merge_flight_bts_data(flight_df, bts_df)
            merged_df.to_csv(args.output, index=False)
            print(f"Wrote merged data: {args.output}  rows={len(merged_df):,}")
        except Exception as e:
            print(f"Error merging data: {e}")
            if flight_df is not None:
                flight_df.to_csv(args.output, index=False)
                print(f"Wrote flight data only: {args.output}  rows={len(flight_df):,}")
    elif flight_df is not None:
        flight_df.to_csv(args.output, index=False)
        print(f"Wrote flight data: {args.output}  rows={len(flight_df):,}")
    else:
        print("Error: No data to output")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

