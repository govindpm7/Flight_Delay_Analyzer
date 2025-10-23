"""
Improved Flight Delay Model Training with XGBoost and Stratified Sampling
Addresses the zero-prediction issue by:
1. Reducing sample size with stratified sampling
2. Using XGBoost with sample weights for imbalanced data
3. Enhanced evaluation and visualization
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import List, Tuple
import matplotlib.pyplot as plt

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
import xgboost as xgb

from utils import MAJOR_HUBS, normalize_flight_key


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build features for the model"""
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

    # BTS delay cause features (if available)
    bts_features = []
    bts_feature_names = [
        'total_delay_rate_origin', 'carrier_delay_rate_origin', 'weather_delay_rate_origin',
        'nas_delay_rate_origin', 'security_delay_rate_origin', 'late_aircraft_delay_rate_origin',
        'avg_delay_minutes_origin', 'avg_carrier_delay_origin', 'avg_weather_delay_origin',
        'avg_nas_delay_origin', 'avg_security_delay_origin', 'avg_late_aircraft_delay_origin',
        'total_delay_rate_dest', 'carrier_delay_rate_dest', 'weather_delay_rate_dest',
        'nas_delay_rate_dest', 'security_delay_rate_dest', 'late_aircraft_delay_rate_dest',
        'avg_delay_minutes_dest', 'avg_carrier_delay_dest', 'avg_weather_delay_dest',
        'avg_nas_delay_dest', 'avg_security_delay_dest', 'avg_late_aircraft_delay_dest'
    ]
    
    for feature in bts_feature_names:
        if feature in data.columns:
            bts_features.append(feature)
            # Fill missing values with 0 for BTS features
            data[feature] = data[feature].fillna(0)
        else:
            # Add missing BTS features as zeros
            data[feature] = 0.0
            bts_features.append(feature)

    # Target
    y = data["DEP_DELAY"].astype(float)

    # Feature matrix - include BTS features
    feature_columns = [
        "dep_hour", "dow", "month", "is_weekend", "DISTANCE",
        "route", "ORIGIN", "DEST", "OP_CARRIER",
        "route_avg_delay", "origin_avg_delay", "airline_avg_delay",
    ] + bts_features

    X = data[feature_columns].copy()

    return X, y


def stratified_sampling(df: pd.DataFrame, sample_size: int = 15000) -> pd.DataFrame:
    """
    Perform stratified sampling to maintain representation of delayed flights
    """
    print(f"Original dataset size: {len(df)}")
    
    # Create delay categories for stratification
    df['delay_category'] = pd.cut(df['DEP_DELAY'], 
                                   bins=[-np.inf, 0, 15, 30, 60, np.inf],
                                   labels=['early', 'on_time', 'short_delay', 'medium_delay', 'long_delay'])
    
    # Check category distribution
    print("Delay category distribution:")
    print(df['delay_category'].value_counts())
    
    # Sample proportionally from each category
    df_sampled = df.groupby('delay_category', group_keys=False).apply(
        lambda x: x.sample(frac=min(1.0, sample_size / len(df)), random_state=42)
    )
    
    # If we still have too many samples, randomly sample down
    if len(df_sampled) > sample_size:
        df_sampled = df_sampled.sample(n=sample_size, random_state=42)
    
    # Drop the temporary category column
    df_sampled = df_sampled.drop('delay_category', axis=1)
    
    print(f"Reduced dataset size: {len(df_sampled)}")
    print("New delay distribution:")
    print(df_sampled['DEP_DELAY'].describe())
    print(f"Delay > 0: {(df_sampled['DEP_DELAY'] > 0).mean() * 100:.1f}%")
    print(f"Delay > 15: {(df_sampled['DEP_DELAY'] > 15).mean() * 100:.1f}%")
    
    return df_sampled


def temporal_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series, test_size: float = 0.2) -> Tuple:
    """Use a simple chronological split by date"""
    order = np.argsort(dates.values)
    split_idx = int(len(order) * (1 - test_size))
    train_idx, test_idx = order[:split_idx], order[split_idx:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def train_models(X: pd.DataFrame, y: pd.Series) -> Tuple[dict, ColumnTransformer]:
    """Train both Random Forest and XGBoost models for comparison"""
    # Identify BTS features dynamically
    bts_features = [col for col in X.columns if any(x in col for x in ['delay_rate', 'avg_delay'])]
    
    numeric_features = ["dep_hour", "dow", "month", "is_weekend", "DISTANCE",
                        "route_avg_delay", "origin_avg_delay", "airline_avg_delay"] + bts_features
    categorical_features = ["route", "ORIGIN", "DEST", "OP_CARRIER"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    # Random Forest (original)
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

    # XGBoost with sample weights
    xgb_model = Pipeline([
        ("pre", preprocessor),
        ("model", xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )),
    ])

    models = {"random_forest": rf, "xgboost": xgb_model}

    return models, preprocessor


def calculate_sample_weights(y_train: pd.Series) -> np.ndarray:
    """Calculate sample weights to handle class imbalance"""
    # Give more weight to delayed flights
    weights = np.where(y_train > 15, 2.0, 1.0)  # 2x weight for delays > 15 min
    
    # More aggressive weighting for very delayed flights
    weights = np.where(y_train > 30, 3.0, weights)
    weights = np.where(y_train > 60, 4.0, weights)
    
    print(f"Weight distribution: min={weights.min()}, max={weights.max()}, mean={weights.mean():.2f}")
    return weights


def evaluate_model(model: Pipeline, X_train, y_train, X_test, y_test, 
                  sample_weights: np.ndarray = None, model_name: str = "Model") -> dict:
    """Enhanced evaluation with detailed metrics and visualization"""
    print(f"\n=== Evaluating {model_name} ===")
    
    # Train model
    if sample_weights is not None and hasattr(model.named_steps['model'], 'fit'):
        # XGBoost with sample weights
        X_train_transformed = model.named_steps['pre'].fit_transform(X_train)
        model.named_steps['model'].fit(
            X_train_transformed, 
            y_train,
            sample_weight=sample_weights,
            eval_set=[(model.named_steps['pre'].transform(X_test), y_test)],
            verbose=False
        )
    else:
        # Standard fit
        model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"MAE: {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")
    print(f"R² Score: {r2:.4f}")
    
    # Check prediction distribution
    print(f"\nPrediction Statistics:")
    print(f"Mean prediction: {y_pred.mean():.2f} minutes")
    print(f"Std prediction: {y_pred.std():.2f} minutes")
    print(f"Min prediction: {y_pred.min():.2f} minutes")
    print(f"Max prediction: {y_pred.max():.2f} minutes")
    
    # Compare with actual distribution
    print(f"\nActual Statistics:")
    print(f"Mean actual: {y_test.mean():.2f} minutes")
    print(f"Std actual: {y_test.std():.2f} minutes")
    
    # Check for zero predictions
    zero_predictions = (y_pred == 0).sum()
    print(f"Zero predictions: {zero_predictions} ({zero_predictions/len(y_pred)*100:.1f}%)")
    
    return {
        "mae": float(mae), 
        "rmse": float(rmse), 
        "r2": float(r2), 
        "pred": y_pred,
        "zero_predictions": int(zero_predictions),
        "zero_pct": float(zero_predictions/len(y_pred)*100)
    }


def plot_evaluation(y_test, y_pred, model_name: str, out_dir: str):
    """Create evaluation plots"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram of predictions vs actuals
    axes[0].hist(y_test, bins=50, alpha=0.5, label='Actual', color='blue', density=True)
    axes[0].hist(y_pred, bins=50, alpha=0.5, label='Predicted', color='red', density=True)
    axes[0].set_xlabel('Delay (minutes)')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'{model_name}: Distribution of Predictions vs Actuals')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Scatter plot
    axes[1].scatter(y_test, y_pred, alpha=0.3, s=1)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Delay (minutes)')
    axes[1].set_ylabel('Predicted Delay (minutes)')
    axes[1].set_title(f'{model_name}: Predicted vs Actual Delays')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{model_name.lower()}_evaluation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Evaluation plot saved as '{model_name.lower()}_evaluation.png'")


def plot_feature_importance(model, X_train, out_dir: str, model_name: str):
    """Plot feature importance"""
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        # Get feature names after preprocessing
        preprocessor = model.named_steps['pre']
        X_transformed = preprocessor.fit_transform(X_train)
        
        # Get feature names
        feature_names = []
        for name, transformer, features in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                # For one-hot encoded features
                if hasattr(transformer, 'get_feature_names_out'):
                    cat_features = transformer.get_feature_names_out(features)
                    feature_names.extend(cat_features)
                else:
                    feature_names.extend([f"{name}_{i}" for i in range(len(features))])
        
        # Get importance scores
        importances = model.named_steps['model'].feature_importances_
        
        # Create DataFrame and sort
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features for {model_name}:")
        print(feature_importance.head(10))
        
        # Plot
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'{model_name}: Top 20 Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{model_name.lower()}_feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved as '{model_name.lower()}_feature_importance.png'")


def build_flight_lookup(df: pd.DataFrame, out_path: str):
    """Build flight lookup for app routing defaults"""
    tmp = (
        df.assign(flight_key=lambda d: d.apply(lambda r: normalize_flight_key(r["OP_CARRIER"], r["FL_NUM"]), axis=1))
          .groupby(["flight_key", "ORIGIN", "DEST", "DEP_HOUR"])
          .size()
          .reset_index(name="count")
    )
    idx = tmp.groupby("flight_key")["count"].idxmax()
    top = tmp.loc[idx, ["flight_key", "ORIGIN", "DEST", "DEP_HOUR", "count"]].reset_index(drop=True)
    top.rename(columns={"DEP_HOUR": "dep_hour"}, inplace=True)
    top.to_csv(out_path, index=False)


def build_bts_lookup(df: pd.DataFrame, out_path: str):
    """Build BTS delay cause lookup for airports and carriers"""
    if not any(col in df.columns for col in ['total_delay_rate_origin', 'carrier_delay_rate_origin']):
        print("No BTS features found, skipping BTS lookup creation")
        return
    
    # Create airport-level BTS lookup
    airport_bts = df.groupby(['ORIGIN']).agg({
        'total_delay_rate_origin': 'mean',
        'carrier_delay_rate_origin': 'mean',
        'weather_delay_rate_origin': 'mean',
        'nas_delay_rate_origin': 'mean',
        'security_delay_rate_origin': 'mean',
        'late_aircraft_delay_rate_origin': 'mean',
        'avg_delay_minutes_origin': 'mean',
        'avg_carrier_delay_origin': 'mean',
        'avg_weather_delay_origin': 'mean',
        'avg_nas_delay_origin': 'mean',
        'avg_security_delay_origin': 'mean',
        'avg_late_aircraft_delay_origin': 'mean'
    }).reset_index()
    
    # Create carrier-level BTS lookup
    carrier_bts = df.groupby(['OP_CARRIER']).agg({
        'total_delay_rate_origin': 'mean',
        'carrier_delay_rate_origin': 'mean',
        'weather_delay_rate_origin': 'mean',
        'nas_delay_rate_origin': 'mean',
        'security_delay_rate_origin': 'mean',
        'late_aircraft_delay_rate_origin': 'mean',
        'avg_delay_minutes_origin': 'mean',
        'avg_carrier_delay_origin': 'mean',
        'avg_weather_delay_origin': 'mean',
        'avg_nas_delay_origin': 'mean',
        'avg_security_delay_origin': 'mean',
        'avg_late_aircraft_delay_origin': 'mean'
    }).reset_index()
    
    # Save both lookups
    airport_bts.to_csv(out_path.replace('.csv', '_airport.csv'), index=False)
    carrier_bts.to_csv(out_path.replace('.csv', '_carrier.csv'), index=False)
    print(f"Created BTS lookups: {out_path.replace('.csv', '_airport.csv')} and {out_path.replace('.csv', '_carrier.csv')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Processed CSV from data_acquisition")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_size", type=int, default=15000, help="Sample size for training")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(args.input)
    print(f"Original dataset shape: {df.shape}")
    
    # Apply stratified sampling
    print("\nApplying stratified sampling...")
    df_sampled = stratified_sampling(df, args.sample_size)
    
    # Build features
    print("\nBuilding features...")
    X, y = build_features(df_sampled)

    # Temporal split
    print("\nSplitting data...")
    dates = pd.to_datetime(df_sampled["FL_DATE"], errors="coerce")
    X_train, X_test, y_train, y_test = temporal_split(X, y, dates, test_size=0.2)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Calculate sample weights for XGBoost
    print("\nCalculating sample weights...")
    sample_weights = calculate_sample_weights(y_train)

    # Train models
    print("\nTraining models...")
    models, preprocessor = train_models(X, y)

    # Evaluate models
    metrics = {}
    best_name, best_model, best_mae = None, None, float("inf")
    
    for name, model in models.items():
        weights = sample_weights if name == "xgboost" else None
        m = evaluate_model(model, X_train, y_train, X_test, y_test, weights, name.title())
        metrics[name] = {k: v for k, v in m.items() if k not in ["pred"]}
        
        # Create evaluation plots
        plot_evaluation(y_test, m["pred"], name.title(), args.out_dir)
        
        # Create feature importance plots
        plot_feature_importance(model, X_train, args.out_dir, name.title())
        
        if m["mae"] < best_mae:
            best_name, best_model, best_mae = name, model, m["mae"]

    # Persist artifacts
    print(f"\nSaving best model: {best_name}")
    with open(os.path.join(args.out_dir, "model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    
    with open(os.path.join(args.out_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(best_model.named_steps["pre"], f)

    # Flight lookup for app routing defaults
    build_flight_lookup(df_sampled, os.path.join(args.out_dir, "flight_lookup.csv"))
    
    # BTS lookup for delay cause analysis
    build_bts_lookup(df_sampled, os.path.join(args.out_dir, "bts_lookup.csv"))

    # Save comprehensive metadata
    summary = {
        "best_model": best_name,
        "metrics": metrics,
        "selected_mae": float(best_mae),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "original_dataset_size": int(len(df)),
        "sampled_dataset_size": int(len(df_sampled)),
        "sample_size_used": args.sample_size,
        "major_hubs": sorted(list(MAJOR_HUBS)),
        "delay_distribution": {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "pct_delayed": float((y > 0).mean() * 100),
            "pct_highly_delayed": float((y > 15).mean() * 100)
        }
    }
    
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(json.dumps(summary, indent=2))
    
    # Print comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    for name, metric in metrics.items():
        print(f"{name.upper()}:")
        print(f"  MAE: {metric['mae']:.2f} min")
        print(f"  R²: {metric['r2']:.4f}")
        print(f"  Zero predictions: {metric['zero_pct']:.1f}%")
        print()


if __name__ == "__main__":
    main()
