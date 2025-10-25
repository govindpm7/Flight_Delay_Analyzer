"""
Prepare data for PCA analysis
Extract features from training data and standardize
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import json
import os
from datetime import date, timedelta

def create_training_dataset_from_bts():
    """
    Create training dataset from BTS processed data
    This simulates the feature engineering process used in the actual model
    """
    print("Creating training dataset from BTS data...")
    
    # Load BTS processed data
    bts_df = pd.read_csv("OUTPUTS/processed_bts_data.csv")
    
    # Load lookup tables
    airport_lookup = pd.read_csv("OUTPUTS/bts_lookup_airport.csv")
    carrier_lookup = pd.read_csv("OUTPUTS/bts_lookup_carrier.csv")
    flight_lookup = pd.read_csv("OUTPUTS/flight_lookup.csv")
    
    # Create training samples by simulating flight scenarios
    training_samples = []
    
    # Get unique combinations for sampling
    unique_combinations = bts_df[['carrier', 'airport']].drop_duplicates()
    
    # Sample 10,000 combinations for training
    sample_size = min(10000, len(unique_combinations))
    sampled_combinations = unique_combinations.sample(n=sample_size, random_state=42)
    
    print(f"   Creating {sample_size} training samples...")
    
    for idx, row in sampled_combinations.iterrows():
        if idx % 1000 == 0:
            print(f"   Processed {idx}/{sample_size} samples...")
        
        carrier = row['carrier']
        airport = row['airport']
        
        # Get carrier data
        carrier_data = bts_df[(bts_df['carrier'] == carrier) & (bts_df['airport'] == airport)]
        if carrier_data.empty:
            continue
            
        # Sample a random date and hour
        flight_date = date(2025, 7, np.random.randint(1, 32))
        dep_hour = np.random.randint(0, 24)
        
        # Get origin airport stats
        origin_stats = airport_lookup[airport_lookup['ORIGIN'] == airport]
        if origin_stats.empty:
            continue
            
        # Get destination (sample from available airports)
        available_dests = airport_lookup['ORIGIN'].unique()
        dest = np.random.choice(available_dests)
        dest_stats = airport_lookup[airport_lookup['ORIGIN'] == dest]
        if dest_stats.empty:
            continue
            
        # Get carrier stats
        carrier_stats = carrier_lookup[carrier_lookup['OP_CARRIER'] == carrier]
        if carrier_stats.empty:
            continue
            
        # Get route info - create route key and use defaults since flight_lookup doesn't have route info
        route_key = f"{airport}-{dest}"
        # Use default values since flight_lookup doesn't have distance or delay info
        distance = 500.0
        route_avg_delay = 10.0
        
        # Calculate temporal features
        dow = flight_date.weekday()
        month = flight_date.month
        is_weekend = 1 if dow in (5, 6) else 0
        
        # Create feature vector (matching the exact structure from FeatureEngine)
        features = {
            'dep_hour': int(dep_hour),
            'dow': int(dow),
            'month': int(month),
            'is_weekend': int(is_weekend),
            'DISTANCE': float(distance),
            'route': route_key,
            'ORIGIN': airport,
            'DEST': dest,
            'OP_CARRIER': carrier,
            'route_avg_delay': float(route_avg_delay),
            'origin_avg_delay': float(origin_stats.iloc[0]['avg_delay_minutes_origin']),
            'airline_avg_delay': float(carrier_stats.iloc[0]['avg_delay_minutes_origin']),
            'total_delay_rate_origin': float(origin_stats.iloc[0]['total_delay_rate_origin']),
            'carrier_delay_rate_origin': float(origin_stats.iloc[0]['carrier_delay_rate_origin']),
            'weather_delay_rate_origin': float(origin_stats.iloc[0]['weather_delay_rate_origin']),
            'nas_delay_rate_origin': float(origin_stats.iloc[0]['nas_delay_rate_origin']),
            'security_delay_rate_origin': float(origin_stats.iloc[0]['security_delay_rate_origin']),
            'late_aircraft_delay_rate_origin': float(origin_stats.iloc[0]['late_aircraft_delay_rate_origin']),
            'avg_delay_minutes_origin': float(origin_stats.iloc[0]['avg_delay_minutes_origin']),
            'avg_carrier_delay_origin': float(origin_stats.iloc[0]['avg_carrier_delay_origin']),
            'avg_weather_delay_origin': float(origin_stats.iloc[0]['avg_weather_delay_origin']),
            'avg_nas_delay_origin': float(origin_stats.iloc[0]['avg_nas_delay_origin']),
            'avg_security_delay_origin': float(origin_stats.iloc[0]['avg_security_delay_origin']),
            'avg_late_aircraft_delay_origin': float(origin_stats.iloc[0]['avg_late_aircraft_delay_origin']),
            'total_delay_rate_dest': float(dest_stats.iloc[0]['total_delay_rate_origin']),
            'carrier_delay_rate_dest': float(dest_stats.iloc[0]['carrier_delay_rate_origin']),
            'weather_delay_rate_dest': float(dest_stats.iloc[0]['weather_delay_rate_origin']),
            'nas_delay_rate_dest': float(dest_stats.iloc[0]['nas_delay_rate_origin']),
            'security_delay_rate_dest': float(dest_stats.iloc[0]['security_delay_rate_origin']),
            'late_aircraft_delay_rate_dest': float(dest_stats.iloc[0]['late_aircraft_delay_rate_origin']),
            'avg_delay_minutes_dest': float(dest_stats.iloc[0]['avg_delay_minutes_origin']),
            'avg_carrier_delay_dest': float(dest_stats.iloc[0]['avg_carrier_delay_origin']),
            'avg_weather_delay_dest': float(dest_stats.iloc[0]['avg_weather_delay_origin']),
            'avg_nas_delay_dest': float(dest_stats.iloc[0]['avg_nas_delay_origin']),
            'avg_security_delay_dest': float(dest_stats.iloc[0]['avg_security_delay_origin']),
            'avg_late_aircraft_delay_dest': float(dest_stats.iloc[0]['avg_late_aircraft_delay_origin']),
        }
        
        # Add a synthetic target (delay in minutes) based on the features
        # This simulates the actual delay prediction target
        base_delay = (features['origin_avg_delay'] + features['airline_avg_delay']) / 2
        time_factor = 1.2 if features['is_weekend'] else 1.0
        hour_factor = 1.3 if features['dep_hour'] in [6, 7, 8, 16, 17, 18, 19] else 1.0
        distance_factor = 1 + (features['DISTANCE'] / 2000) * 0.1
        
        features['DEP_DELAY'] = base_delay * time_factor * hour_factor * distance_factor + np.random.normal(0, 5)
        features['DEP_DELAY'] = max(0, features['DEP_DELAY'])  # Ensure non-negative
        
        training_samples.append(features)
    
    # Convert to DataFrame
    training_df = pd.DataFrame(training_samples)
    
    print(f"Created training dataset with {len(training_df)} samples")
    return training_df

def prepare_features_for_pca(df: pd.DataFrame, metadata_path: str = "OUTPUTS/metadata.json"):
    """
    Extract and prepare features for PCA
    Separate numerical and categorical features
    """
    # Load feature list from metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    feature_columns = metadata.get('feature_columns', [])
    
    # Separate feature types
    numerical_features = []
    categorical_features = []
    
    for col in feature_columns:
        if col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                numerical_features.append(col)
            else:
                categorical_features.append(col)
    
    return df, numerical_features, categorical_features

def encode_categorical_features(df: pd.DataFrame, categorical_features: list):
    """
    Encode categorical features for PCA
    """
    df_encoded = df.copy()
    encoders = {}
    
    for col in categorical_features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    return df_encoded, encoders

def standardize_features(df: pd.DataFrame, feature_columns: list):
    """
    Standardize features for PCA (mean=0, std=1)
    Critical for PCA since it's sensitive to scale
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_columns])
    
    return X_scaled, scaler

if __name__ == "__main__":
    # Create output directory
    os.makedirs("OUTPUTS/pca_analysis", exist_ok=True)
    
    # Create training dataset
    df = create_training_dataset_from_bts()
    
    # Save training dataset
    df.to_csv("OUTPUTS/pca_analysis/training_data.csv", index=False)
    print(f"Saved training data to OUTPUTS/pca_analysis/training_data.csv")
    
    # Prepare features
    df, numerical_features, categorical_features = prepare_features_for_pca(df)
    
    print(f"\nFeature Analysis:")
    print(f"   Numerical features: {len(numerical_features)}")
    print(f"   Categorical features: {len(categorical_features)}")
    print(f"   Total samples: {len(df)}")
    
    # Encode categorical
    df_encoded, encoders = encode_categorical_features(df, categorical_features)
    
    # Get all features
    all_features = numerical_features + categorical_features
    
    # Standardize
    X_scaled, scaler = standardize_features(df_encoded, all_features)
    
    # Save preprocessed data
    np.save("OUTPUTS/pca_analysis/X_scaled.npy", X_scaled)
    
    with open("OUTPUTS/pca_analysis/feature_names.pkl", 'wb') as f:
        pickle.dump(all_features, f)
    
    with open("OUTPUTS/pca_analysis/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    with open("OUTPUTS/pca_analysis/encoders.pkl", 'wb') as f:
        pickle.dump(encoders, f)
    
    # Save target variable
    np.save("OUTPUTS/pca_analysis/y_target.npy", df['DEP_DELAY'].values)
    
    print(f"\nData prepared for PCA analysis")
    print(f"   Saved to: OUTPUTS/pca_analysis/")
    print(f"   Scaled data shape: {X_scaled.shape}")
    print(f"   Target shape: {df['DEP_DELAY'].shape}")
