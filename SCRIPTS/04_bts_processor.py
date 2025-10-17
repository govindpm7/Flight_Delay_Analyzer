"""
BTS Data Processing Pipeline
Processes Bureau of Transportation Statistics delay cause data for flight delay analysis
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from utils import MAJOR_HUBS


def load_bts_data(input_path: str) -> pd.DataFrame:
    """Load BTS delay cause data from CSV file"""
    try:
        df = pd.read_csv(input_path, low_memory=False)
        print(f"Loaded BTS data: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading BTS data: {e}")
        sys.exit(1)


def validate_bts_columns(df: pd.DataFrame) -> bool:
    """Validate that required BTS columns are present"""
    required_cols = [
        'year', 'month', 'carrier', 'airport', 'arr_flights', 'arr_del15',
        'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct',
        'arr_delay', 'carrier_delay', 'weather_delay', 'nas_delay', 
        'security_delay', 'late_aircraft_delay'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return False
    
    return True


def clean_bts_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize BTS data"""
    df = df.copy()
    
    # Clean carrier and airport codes
    df['carrier'] = df['carrier'].astype(str).str.upper().str.strip()
    df['airport'] = df['airport'].astype(str).str.upper().str.strip()
    
    # Convert numeric columns, handling any string values
    numeric_cols = [
        'year', 'month', 'arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct',
        'nas_ct', 'security_ct', 'late_aircraft_ct', 'arr_cancelled', 'arr_diverted',
        'arr_delay', 'carrier_delay', 'weather_delay', 'nas_delay', 
        'security_delay', 'late_aircraft_delay'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with invalid data
    df = df.dropna(subset=['year', 'month', 'carrier', 'airport'])
    df = df[df['arr_flights'] > 0]  # Only include airports with flights
    
    return df


def calculate_delay_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive delay metrics from BTS data"""
    df = df.copy()
    
    # Basic delay rates
    df['total_delay_rate'] = df['arr_del15'] / df['arr_flights']
    df['cancellation_rate'] = df['arr_cancelled'] / df['arr_flights']
    df['diversion_rate'] = df['arr_diverted'] / df['arr_flights']
    
    # Delay cause rates
    df['carrier_delay_rate'] = df['carrier_ct'] / df['arr_flights']
    df['weather_delay_rate'] = df['weather_ct'] / df['arr_flights']
    df['nas_delay_rate'] = df['nas_ct'] / df['arr_flights']
    df['security_delay_rate'] = df['security_ct'] / df['arr_flights']
    df['late_aircraft_delay_rate'] = df['late_aircraft_ct'] / df['arr_flights']
    
    # Average delay minutes per delayed flight
    df['avg_delay_minutes'] = np.where(
        df['arr_del15'] > 0, 
        df['arr_delay'] / df['arr_del15'], 
        0
    )
    
    df['avg_carrier_delay'] = np.where(
        df['carrier_ct'] > 0,
        df['carrier_delay'] / df['carrier_ct'],
        0
    )
    
    df['avg_weather_delay'] = np.where(
        df['weather_ct'] > 0,
        df['weather_delay'] / df['weather_ct'],
        0
    )
    
    df['avg_nas_delay'] = np.where(
        df['nas_ct'] > 0,
        df['nas_delay'] / df['nas_ct'],
        0
    )
    
    df['avg_security_delay'] = np.where(
        df['security_ct'] > 0,
        df['security_delay'] / df['security_ct'],
        0
    )
    
    df['avg_late_aircraft_delay'] = np.where(
        df['late_aircraft_ct'] > 0,
        df['late_aircraft_delay'] / df['late_aircraft_ct'],
        0
    )
    
    # Delay cause percentages (what % of delays are due to each cause)
    total_delays = df['carrier_ct'] + df['weather_ct'] + df['nas_ct'] + df['security_ct'] + df['late_aircraft_ct']
    total_delays = np.where(total_delays > 0, total_delays, 1)  # Avoid division by zero
    
    df['carrier_delay_pct'] = df['carrier_ct'] / total_delays * 100
    df['weather_delay_pct'] = df['weather_ct'] / total_delays * 100
    df['nas_delay_pct'] = df['nas_ct'] / total_delays * 100
    df['security_delay_pct'] = df['security_ct'] / total_delays * 100
    df['late_aircraft_delay_pct'] = df['late_aircraft_ct'] / total_delays * 100
    
    return df


def create_airport_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create airport-level summary statistics"""
    airport_summary = df.groupby(['airport', 'year', 'month']).agg({
        'arr_flights': 'sum',
        'arr_del15': 'sum',
        'arr_cancelled': 'sum',
        'arr_diverted': 'sum',
        'arr_delay': 'sum',
        'carrier_ct': 'sum',
        'weather_ct': 'sum',
        'nas_ct': 'sum',
        'security_ct': 'sum',
        'late_aircraft_ct': 'sum',
        'carrier_delay': 'sum',
        'weather_delay': 'sum',
        'nas_delay': 'sum',
        'security_delay': 'sum',
        'late_aircraft_delay': 'sum',
        'total_delay_rate': 'mean',
        'cancellation_rate': 'mean',
        'diversion_rate': 'mean',
        'avg_delay_minutes': 'mean',
        'carrier_delay_pct': 'mean',
        'weather_delay_pct': 'mean',
        'nas_delay_pct': 'mean',
        'security_delay_pct': 'mean',
        'late_aircraft_delay_pct': 'mean'
    }).reset_index()
    
    return airport_summary


def create_carrier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create carrier-level summary statistics"""
    carrier_summary = df.groupby(['carrier', 'year', 'month']).agg({
        'arr_flights': 'sum',
        'arr_del15': 'sum',
        'arr_cancelled': 'sum',
        'arr_diverted': 'sum',
        'arr_delay': 'sum',
        'carrier_ct': 'sum',
        'weather_ct': 'sum',
        'nas_ct': 'sum',
        'security_ct': 'sum',
        'late_aircraft_ct': 'sum',
        'carrier_delay': 'sum',
        'weather_delay': 'sum',
        'weather_delay': 'sum',
        'nas_delay': 'sum',
        'security_delay': 'sum',
        'late_aircraft_delay': 'sum',
        'total_delay_rate': 'mean',
        'cancellation_rate': 'mean',
        'diversion_rate': 'mean',
        'avg_delay_minutes': 'mean',
        'carrier_delay_pct': 'mean',
        'weather_delay_pct': 'mean',
        'nas_delay_pct': 'mean',
        'security_delay_pct': 'mean',
        'late_aircraft_delay_pct': 'mean'
    }).reset_index()
    
    return carrier_summary


def filter_major_hubs(df: pd.DataFrame) -> pd.DataFrame:
    """Filter data to focus on major hub airports"""
    if MAJOR_HUBS:
        df = df[df['airport'].isin(MAJOR_HUBS)]
        print(f"Filtered to major hubs {MAJOR_HUBS}: {len(df)} rows")
    return df


def create_delay_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Create time series trends for delay patterns"""
    # Monthly trends
    monthly_trends = df.groupby(['year', 'month']).agg({
        'arr_flights': 'sum',
        'arr_del15': 'sum',
        'arr_delay': 'sum',
        'total_delay_rate': 'mean',
        'avg_delay_minutes': 'mean',
        'carrier_delay_pct': 'mean',
        'weather_delay_pct': 'mean',
        'nas_delay_pct': 'mean'
    }).reset_index()
    
    # Add date column for easier plotting
    monthly_trends['date'] = pd.to_datetime(monthly_trends[['year', 'month']].assign(day=1))
    
    return monthly_trends


def main():
    parser = argparse.ArgumentParser(description="Process BTS delay cause data")
    parser.add_argument("--input", required=True, help="Path to BTS CSV file")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed data")
    parser.add_argument("--filter_hubs", action="store_true", help="Filter to major hub airports only")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and validate data
    df = load_bts_data(args.input)
    
    if not validate_bts_columns(df):
        print("Invalid BTS data format")
        sys.exit(1)
    
    # Clean data
    df = clean_bts_data(df)
    print(f"After cleaning: {len(df)} rows")
    
    # Calculate delay metrics
    df = calculate_delay_metrics(df)
    
    # Filter to major hubs if requested
    if args.filter_hubs:
        df = filter_major_hubs(df)
    
    # Create summary tables
    airport_summary = create_airport_summary(df)
    carrier_summary = create_carrier_summary(df)
    monthly_trends = create_delay_trends(df)
    
    # Save processed data
    df.to_csv(os.path.join(args.output_dir, "bts_processed.csv"), index=False)
    airport_summary.to_csv(os.path.join(args.output_dir, "airport_summary.csv"), index=False)
    carrier_summary.to_csv(os.path.join(args.output_dir, "carrier_summary.csv"), index=False)
    monthly_trends.to_csv(os.path.join(args.output_dir, "monthly_trends.csv"), index=False)
    
    # Print summary statistics
    print("\n=== BTS Data Processing Summary ===")
    print(f"Total records processed: {len(df):,}")
    print(f"Airports: {df['airport'].nunique()}")
    print(f"Carriers: {df['carrier'].nunique()}")
    print(f"Date range: {df['year'].min()}-{df['month'].min():02d} to {df['year'].max()}-{df['month'].max():02d}")
    print(f"Total flights: {df['arr_flights'].sum():,}")
    print(f"Total delayed flights: {df['arr_del15'].sum():,}")
    print(f"Overall delay rate: {df['arr_del15'].sum() / df['arr_flights'].sum() * 100:.1f}%")
    print(f"Average delay minutes: {df['avg_delay_minutes'].mean():.1f}")
    
    print("\n=== Top Delay Causes ===")
    cause_totals = {
        'Carrier': df['carrier_ct'].sum(),
        'Weather': df['weather_ct'].sum(),
        'NAS': df['nas_ct'].sum(),
        'Security': df['security_ct'].sum(),
        'Late Aircraft': df['late_aircraft_ct'].sum()
    }
    for cause, count in sorted(cause_totals.items(), key=lambda x: x[1], reverse=True):
        pct = count / sum(cause_totals.values()) * 100
        print(f"{cause}: {count:,} delays ({pct:.1f}%)")
    
    print(f"\nProcessed data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
