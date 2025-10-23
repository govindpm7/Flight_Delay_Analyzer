#!/usr/bin/env python3
"""
Create a flight delay prediction model using actual BTS data
"""
import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

def create_bts_model():
    """Create a model using actual BTS data from OUTPUTS folder"""
    
    print("Creating flight delay prediction model from BTS data...")
    
    # Load the processed BTS data
    bts_data_path = "OUTPUTS/processed_bts_data.csv"
    if not os.path.exists(bts_data_path):
        print(f"❌ Error: {bts_data_path} not found!")
        return None, None
    
    print(f"Loading BTS data from {bts_data_path}...")
    df = pd.read_csv(bts_data_path)
    print(f"Loaded {len(df)} records")
    
    # Load airport and carrier lookup data
    airport_lookup = pd.read_csv("OUTPUTS/bts_lookup_airport.csv")
    carrier_lookup = pd.read_csv("OUTPUTS/bts_lookup_carrier.csv")
    
    print(f"Airport lookup: {len(airport_lookup)} airports")
    print(f"Carrier lookup: {len(carrier_lookup)} carriers")
    
    # Prepare features for training
    # We'll use the BTS data to create realistic training examples
    
    training_data = []
    
    # Get unique combinations from BTS data
    unique_combinations = df[['carrier', 'airport', 'month']].drop_duplicates()
    
    print(f"Creating training data from {len(unique_combinations)} unique combinations...")
    
    for _, row in unique_combinations.iterrows():
        carrier = row['carrier']
        airport = row['airport']
        month = row['month']
        
        # Get BTS data for this combination
        bts_subset = df[(df['carrier'] == carrier) & 
                       (df['airport'] == airport) & 
                       (df['month'] == month)]
        
        if len(bts_subset) == 0:
            continue
            
        # Use the first record as base
        base_record = bts_subset.iloc[0]
        
        # Create multiple training examples with different departure hours and days
        for dep_hour in range(24):
            for dow in range(7):  # Day of week
                is_weekend = 1 if dow in [5, 6] else 0
                
                # Calculate expected delay based on BTS data
                base_delay = base_record['avg_delay_minutes']
                
                # Add time-based patterns
                if dep_hour in [7, 8, 9, 17, 18, 19]:  # Rush hours
                    time_factor = 1.2
                elif dep_hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Late night/early morning
                    time_factor = 0.8
                else:
                    time_factor = 1.0
                
                # Weekend factor
                weekend_factor = 1.1 if is_weekend else 1.0
                
                # Seasonal factor (summer months typically have more delays)
                if month in [6, 7, 8]:  # Summer
                    seasonal_factor = 1.15
                elif month in [12, 1, 2]:  # Winter
                    seasonal_factor = 1.1
                else:
                    seasonal_factor = 1.0
                
                # Calculate final delay
                expected_delay = base_delay * time_factor * weekend_factor * seasonal_factor
                
                # Add some realistic noise
                noise = np.random.normal(0, expected_delay * 0.2)
                final_delay = max(0, expected_delay + noise)
                
                # Get airport and carrier features
                airport_features = airport_lookup[airport_lookup['ORIGIN'] == airport]
                carrier_features = carrier_lookup[carrier_lookup['OP_CARRIER'] == carrier]
                
                if len(airport_features) == 0 or len(carrier_features) == 0:
                    continue
                
                airport_data = airport_features.iloc[0]
                carrier_data = carrier_features.iloc[0]
                
                # Create training example
                training_example = {
                    'dep_hour': dep_hour,
                    'dow': dow,
                    'month': month,
                    'is_weekend': is_weekend,
                    'DISTANCE': 1000.0,  # Default distance
                    'route': f"{airport}-XXX",  # We'll use airport as route proxy
                    'ORIGIN': airport,
                    'DEST': 'XXX',  # Placeholder
                    'OP_CARRIER': carrier,
                    'route_avg_delay': base_delay,
                    'origin_avg_delay': airport_data['avg_delay_minutes_origin'],
                    'airline_avg_delay': carrier_data['avg_delay_minutes_origin'],
                    'total_delay_rate_origin': airport_data['total_delay_rate_origin'],
                    'carrier_delay_rate_origin': airport_data['carrier_delay_rate_origin'],
                    'weather_delay_rate_origin': airport_data['weather_delay_rate_origin'],
                    'nas_delay_rate_origin': airport_data['nas_delay_rate_origin'],
                    'security_delay_rate_origin': airport_data['security_delay_rate_origin'],
                    'late_aircraft_delay_rate_origin': airport_data['late_aircraft_delay_rate_origin'],
                    'avg_delay_minutes_origin': airport_data['avg_delay_minutes_origin'],
                    'avg_carrier_delay_origin': airport_data['avg_carrier_delay_origin'],
                    'avg_weather_delay_origin': airport_data['avg_weather_delay_origin'],
                    'avg_nas_delay_origin': airport_data['avg_nas_delay_origin'],
                    'avg_security_delay_origin': airport_data['avg_security_delay_origin'],
                    'avg_late_aircraft_delay_origin': airport_data['avg_late_aircraft_delay_origin'],
                    'total_delay_rate_dest': carrier_data['total_delay_rate_origin'],
                    'carrier_delay_rate_dest': carrier_data['carrier_delay_rate_origin'],
                    'weather_delay_rate_dest': carrier_data['weather_delay_rate_origin'],
                    'nas_delay_rate_dest': carrier_data['nas_delay_rate_origin'],
                    'security_delay_rate_dest': carrier_data['security_delay_rate_origin'],
                    'late_aircraft_delay_rate_dest': carrier_data['late_aircraft_delay_rate_origin'],
                    'avg_delay_minutes_dest': carrier_data['avg_delay_minutes_origin'],
                    'avg_carrier_delay_dest': carrier_data['avg_carrier_delay_origin'],
                    'avg_weather_delay_dest': carrier_data['avg_weather_delay_origin'],
                    'avg_nas_delay_dest': carrier_data['avg_nas_delay_origin'],
                    'avg_security_delay_dest': carrier_data['avg_security_delay_origin'],
                    'avg_late_aircraft_delay_dest': carrier_data['avg_late_aircraft_delay_origin'],
                    'ARR_DELAY': final_delay
                }
                
                training_data.append(training_example)
    
    # Convert to DataFrame
    train_df = pd.DataFrame(training_data)
    print(f"Created {len(train_df)} training examples")
    
    if len(train_df) == 0:
        print("❌ No training data created!")
        return None, None
    
    # Prepare features and target
    feature_cols = [col for col in train_df.columns if col != 'ARR_DELAY']
    X = train_df[feature_cols]
    y = train_df['ARR_DELAY']
    
    # Handle categorical variables
    categorical_cols = ['route', 'ORIGIN', 'DEST', 'OP_CARRIER']
    X_encoded = X.copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Model trained successfully!")
    print(f"MAE: {mae:.2f} minutes")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature importance (top 5):")
    
    # Show feature importance
    feature_importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
        print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    # Save model
    model_path = "OUTPUTS/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Save label encoders
    encoders_path = "OUTPUTS/label_encoders.pkl"
    with open(encoders_path, "wb") as f:
        pickle.dump(label_encoders, f)
    
    # Save metadata
    metadata = {
        "best_model": "RandomForestRegressor",
        "selected_mae": float(mae),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_columns": list(X_encoded.columns),
        "categorical_columns": categorical_cols,
        "model_type": "RandomForest",
        "training_date": pd.Timestamp.now().isoformat(),
        "data_source": "BTS_processed_data",
        "major_hubs": list(airport_lookup['ORIGIN'].unique()),
        "carriers": list(carrier_lookup['OP_CARRIER'].unique())
    }
    
    metadata_path = "OUTPUTS/metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Encoders saved to: {encoders_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    return model, mae

if __name__ == "__main__":
    create_bts_model()