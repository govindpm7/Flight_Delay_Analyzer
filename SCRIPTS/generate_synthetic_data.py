"""
Generate synthetic flight delay data based on BTS patterns
This creates realistic flight data with DEP_DELAY column for testing the improved model
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_flight_data(n_samples=75000, output_path="OUTPUTS/synthetic_flight_data.csv"):
    """Generate synthetic flight data with realistic delay patterns"""
    np.random.seed(42)
    
    # Airports and carriers from BTS data
    airports = ['ATL', 'DEN', 'IAD', 'LAX', 'JFK', 'ORD', 'DFW', 'SFO', 'SEA', 'BOS']
    carriers = ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'HA', 'F9', 'NK', 'G4']
    
    # Generate base data
    data = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(n_samples):
        # Random selection
        origin = np.random.choice(airports)
        dest = np.random.choice([a for a in airports if a != origin])
        carrier = np.random.choice(carriers)
        
        # Generate features
        dep_hour = np.random.randint(0, 24)
        dow = np.random.randint(0, 7)
        month = np.random.randint(1, 13)
        is_weekend = 1 if dow in [5, 6] else 0
        distance = np.random.uniform(200, 2000)
        
        # Generate flight date
        days_offset = np.random.randint(0, 365)
        flight_date = start_date + timedelta(days=days_offset)
        
        # Generate flight number
        flight_num = np.random.randint(100, 9999)
        
        # Generate scheduled departure time
        crs_dep_time = dep_hour * 100 + np.random.randint(0, 60)
        
        # Generate delay based on realistic patterns
        base_delay = 0
        
        # Time-based patterns
        if dep_hour in [6, 7, 8, 17, 18, 19]:  # Rush hours
            base_delay += np.random.exponential(5)
        elif dep_hour in [22, 23, 0, 1, 2, 3]:  # Late night/early morning
            base_delay += np.random.exponential(3)
        
        # Day-of-week patterns
        if is_weekend:
            base_delay += np.random.exponential(2)
        
        # Month patterns (winter delays)
        if month in [12, 1, 2]:
            base_delay += np.random.exponential(8)
        elif month in [6, 7, 8]:  # Summer
            base_delay += np.random.exponential(3)
        
        # Carrier-specific patterns
        carrier_delays = {
            'AA': 0.5, 'DL': 0.3, 'UA': 0.4, 'WN': 0.2, 'B6': 0.6,
            'AS': 0.1, 'HA': 0.8, 'F9': 0.7, 'NK': 0.9, 'G4': 0.6
        }
        base_delay += np.random.exponential(carrier_delays.get(carrier, 0.4) * 5)
        
        # Airport-specific patterns
        airport_delays = {
            'ATL': 0.2, 'DEN': 0.4, 'IAD': 0.3, 'LAX': 0.5, 'JFK': 0.6,
            'ORD': 0.4, 'DFW': 0.3, 'SFO': 0.5, 'SEA': 0.3, 'BOS': 0.4
        }
        base_delay += np.random.exponential(airport_delays.get(origin, 0.3) * 3)
        
        # Distance-based delays (longer flights more likely to be delayed)
        if distance > 1000:
            base_delay += np.random.exponential(2)
        
        # Add random noise
        delay = max(0, base_delay + np.random.normal(0, 5))
        
        # Ensure some flights are on-time or early
        if np.random.random() < 0.3:  # 30% chance of being on-time or early
            delay = max(-15, np.random.normal(-2, 3))
        
        data.append({
            'OP_CARRIER': carrier,
            'FL_NUM': str(flight_num),
            'ORIGIN': origin,
            'DEST': dest,
            'CRS_DEP_TIME': crs_dep_time,
            'DEP_HOUR': dep_hour,
            'DEP_DELAY': delay,
            'FL_DATE': flight_date.strftime('%Y-%m-%d'),
            'DISTANCE': distance
        })
    
    df = pd.DataFrame(data)
    
    # Add some BTS-like features for consistency
    df['year'] = pd.to_datetime(df['FL_DATE']).dt.year
    df['month'] = pd.to_datetime(df['FL_DATE']).dt.month
    
    # Save the data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated synthetic flight data: {len(df):,} rows")
    print(f"Delay distribution:")
    print(df['DEP_DELAY'].describe())
    print(f"Delay > 0: {(df['DEP_DELAY'] > 0).mean() * 100:.1f}%")
    print(f"Delay > 15: {(df['DEP_DELAY'] > 15).mean() * 100:.1f}%")
    print(f"Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    generate_synthetic_flight_data()
