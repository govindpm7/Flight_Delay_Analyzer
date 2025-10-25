"""
Simplified PCA data preparation that works with available BTS data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def create_training_dataset():
    """Create training dataset from available BTS data"""
    print("Creating training dataset from BTS data...")
    
    # Load BTS processed data
    bts_df = pd.read_csv("../../OUTPUTS/processed_bts_data.csv")
    
    # Create training samples by simulating flight scenarios
    training_samples = []
    
    # Get unique combinations for sampling
    unique_combinations = bts_df[['carrier', 'airport']].drop_duplicates()
    
    # Sample combinations for training
    sample_size = min(1000, len(unique_combinations))
    sampled_combinations = unique_combinations.sample(n=sample_size, random_state=42)
    
    print(f"   Creating {sample_size} training samples...")
    
    for idx, row in sampled_combinations.iterrows():
        if idx % 100 == 0:
            print(f"   Processed {idx}/{sample_size} samples...")
        
        carrier = row['carrier']
        airport = row['airport']
        
        # Get carrier data
        carrier_data = bts_df[(bts_df['carrier'] == carrier) & (bts_df['airport'] == airport)]
        if carrier_data.empty:
            continue
            
        # Take the first record for this carrier-airport combination
        record = carrier_data.iloc[0]
        
        # Create feature vector
        features = {
            'carrier': carrier,
            'airport': airport,
            'total_delay_rate': record['total_delay_rate'],
            'carrier_delay_rate': record['carrier_delay_rate'],
            'weather_delay_rate': record['weather_delay_rate'],
            'nas_delay_rate': record['nas_delay_rate'],
            'security_delay_rate': record['security_delay_rate'],
            'late_aircraft_delay_rate': record['late_aircraft_delay_rate'],
            'avg_delay_minutes': record['avg_delay_minutes'],
            'avg_carrier_delay': record['avg_carrier_delay'],
            'avg_weather_delay': record['avg_weather_delay'],
            'avg_nas_delay': record['avg_nas_delay'],
            'avg_security_delay': record['avg_security_delay'],
            'avg_late_aircraft_delay': record['avg_late_aircraft_delay'],
            'arr_flights': record['arr_flights'],
            'arr_del15': record['arr_del15'],
            'month': record['month']
        }
        
        training_samples.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(training_samples)
    
    # Create output directory
    os.makedirs("../../OUTPUTS/pca_analysis", exist_ok=True)
    
    # Save training data
    df.to_csv("../../OUTPUTS/pca_analysis/training_data.csv", index=False)
    print(f"   Saved {len(df)} training samples to OUTPUTS/pca_analysis/training_data.csv")
    
    # Prepare features for PCA
    feature_columns = [col for col in df.columns if col not in ['carrier', 'airport']]
    X = df[feature_columns].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaled data
    np.save("../../OUTPUTS/pca_analysis/X_scaled.npy", X_scaled)
    np.save("../../OUTPUTS/pca_analysis/feature_names.npy", np.array(feature_columns))
    
    print(f"   Saved scaled features with shape: {X_scaled.shape}")
    print(f"   Feature columns: {feature_columns}")
    
    return df, X_scaled, feature_columns

if __name__ == "__main__":
    df, X_scaled, feature_columns = create_training_dataset()
    print("Data preparation complete!")

