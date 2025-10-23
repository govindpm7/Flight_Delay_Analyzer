"""
Train a simple model from BTS data for the app
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_training_data():
    """Create training data from BTS data"""
    # Read processed BTS data
    bts_data = pd.read_csv("OUTPUTS/bts_processed.csv")
    
    # Create synthetic training data based on BTS patterns
    np.random.seed(42)
    n_samples = 10000
    
    # Generate synthetic flight data
    airports = ['ATL', 'DEN', 'IAD', 'LAX']
    carriers = bts_data['carrier'].unique()[:10]  # Top 10 carriers
    
    data = []
    for _ in range(n_samples):
        # Random selection
        origin = np.random.choice(airports)
        dest = np.random.choice([a for a in airports if a != origin])
        carrier = np.random.choice(carriers)
        
        # Get BTS statistics for this combination
        origin_stats = bts_data[bts_data['airport'] == origin].iloc[0] if len(bts_data[bts_data['airport'] == origin]) > 0 else None
        carrier_stats = bts_data[bts_data['carrier'] == carrier].iloc[0] if len(bts_data[bts_data['carrier'] == carrier]) > 0 else None
        
        # Generate features
        dep_hour = np.random.randint(0, 24)
        dow = np.random.randint(0, 7)
        month = np.random.randint(1, 13)
        is_weekend = 1 if dow in [5, 6] else 0
        distance = np.random.uniform(200, 2000)  # Random distance
        
        # Calculate delay based on BTS patterns
        base_delay = 0
        if origin_stats is not None:
            base_delay += origin_stats.get('avg_delay_minutes', 0) * 0.3
        if carrier_stats is not None:
            base_delay += carrier_stats.get('avg_delay_minutes', 0) * 0.3
        
        # Add time-based patterns
        if dep_hour in [6, 7, 8, 17, 18, 19]:  # Rush hours
            base_delay += np.random.uniform(5, 15)
        if is_weekend:
            base_delay += np.random.uniform(0, 10)
        if month in [12, 1, 2]:  # Winter months
            base_delay += np.random.uniform(5, 20)
        
        # Add noise
        delay = max(0, base_delay + np.random.normal(0, 10))
        
        data.append({
            'dep_hour': dep_hour,
            'dow': dow,
            'month': month,
            'is_weekend': is_weekend,
            'DISTANCE': distance,
            'route': f"{origin}-{dest}",
            'ORIGIN': origin,
            'DEST': dest,
            'OP_CARRIER': carrier,
            'route_avg_delay': base_delay * 0.4,
            'origin_avg_delay': origin_stats.get('avg_delay_minutes', 0) * 0.3 if origin_stats is not None else 0,
            'airline_avg_delay': carrier_stats.get('avg_delay_minutes', 0) * 0.3 if carrier_stats is not None else 0,
            'DEP_DELAY': delay
        })
    
    return pd.DataFrame(data)

def train_model():
    """Train a simple model"""
    print("Creating training data from BTS patterns...")
    df = create_training_data()
    
    # Prepare features
    feature_cols = [
        'dep_hour', 'dow', 'month', 'is_weekend', 'DISTANCE',
        'route_avg_delay', 'origin_avg_delay', 'airline_avg_delay'
    ]
    
    # Encode categorical variables
    df['route_encoded'] = pd.Categorical(df['route']).codes
    df['origin_encoded'] = pd.Categorical(df['ORIGIN']).codes
    df['dest_encoded'] = pd.Categorical(df['DEST']).codes
    df['carrier_encoded'] = pd.Categorical(df['OP_CARRIER']).codes
    
    feature_cols.extend(['route_encoded', 'origin_encoded', 'dest_encoded', 'carrier_encoded'])
    
    X = df[feature_cols]
    y = df['DEP_DELAY']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"MAE: {mae:.2f} minutes")
    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.3f}")
    
    # Save model
    os.makedirs("OUTPUTS", exist_ok=True)
    with open("OUTPUTS/model.joblib", "wb") as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        "best_model": "RandomForestRegressor",
        "selected_mae": float(mae),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_columns": feature_cols,
        "r2_score": float(r2)
    }
    
    with open("OUTPUTS/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to OUTPUTS/model.joblib")
    print(f"Metadata saved to OUTPUTS/metadata.json")
    
    return model, metadata

if __name__ == "__main__":
    train_model()
