#!/usr/bin/env python3
"""
Train model on BTS processed data for airport-level delay prediction
Target: avg_delay_minutes (69.8 min average)
Features: airport, carrier, month, delay rates, etc.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_bts_data():
    """Load and prepare BTS processed data"""
    print("Loading BTS processed data...")
    df = pd.read_csv('OUTPUTS/processing/processed_bts_data.csv')
    
    print(f"Loaded {len(df)} records")
    print(f"Airports: {df['airport'].unique()}")
    print(f"Average delay: {df['avg_delay_minutes'].mean():.1f} min")
    
    return df

def prepare_features(df):
    """Prepare features for training"""
    print("\nPreparing features...")
    
    # Create a copy for feature engineering
    X = df.copy()
    
    # Target variable
    y = X['avg_delay_minutes'].astype(float)
    
    # Remove target and non-feature columns
    feature_cols = [
        'airport', 'carrier', 'year', 'month',
        'arr_flights', 'arr_del15', 'arr_cancelled', 'arr_diverted',
        'total_delay_rate', 'carrier_delay_rate', 'weather_delay_rate',
        'nas_delay_rate', 'security_delay_rate', 'late_aircraft_delay_rate',
        'avg_carrier_delay', 'avg_weather_delay', 'avg_nas_delay',
        'avg_security_delay', 'avg_late_aircraft_delay'
    ]
    
    # Select only available columns
    available_cols = [col for col in feature_cols if col in X.columns]
    X = X[available_cols]
    
    print(f"Selected features: {available_cols}")
    
    # Handle categorical variables
    categorical_cols = ['airport', 'carrier']
    numerical_cols = [col for col in available_cols if col not in categorical_cols]
    
    print(f"Categorical features: {categorical_cols}")
    print(f"Numerical features: {numerical_cols}")
    
    return X, y, categorical_cols, numerical_cols

def create_model_pipeline(categorical_cols, numerical_cols):
    """Create preprocessing and model pipeline"""
    
    # Create preprocessing steps
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    # Preprocessing for categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ]
    )
    
    # Create models
    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    # Create pipelines
    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
    
    return pipelines

def evaluate_models(pipelines, X, y):
    """Evaluate all models"""
    print("\nEvaluating models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X['airport']
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    results = {}
    
    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        results[name] = {
            'pipeline': pipeline,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_mae': cv_mae,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"MAE: {mae:.2f} min")
        print(f"RMSE: {rmse:.2f} min")
        print(f"R²: {r2:.3f}")
        print(f"CV MAE: {cv_mae:.2f} min")
        
        # Prediction statistics
        print(f"Mean prediction: {y_pred.mean():.1f} min")
        print(f"Mean actual: {y_test.mean():.1f} min")
        print(f"Min prediction: {y_pred.min():.1f} min")
        print(f"Max prediction: {y_pred.max():.1f} min")
    
    return results, X_train, X_test, y_train, y_test

def plot_results(results, output_dir):
    """Create evaluation plots"""
    print("\nCreating evaluation plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for name, result in results.items():
        y_test = result['y_test']
        y_pred = result['y_pred']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{name.title()} Model Evaluation', fontsize=16)
        
        # 1. Distribution comparison
        axes[0, 0].hist(y_test, bins=20, alpha=0.7, label='Actual', color='blue', density=True)
        axes[0, 0].hist(y_pred, bins=20, alpha=0.7, label='Predicted', color='red', density=True)
        axes[0, 0].set_xlabel('Delay (minutes)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Distribution Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Scatter plot
        axes[0, 1].scatter(y_test, y_pred, alpha=0.6, s=20)
        axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Delay (minutes)')
        axes[0, 1].set_ylabel('Predicted Delay (minutes)')
        axes[0, 1].set_title(f'Predictions vs Actual (R² = {result["r2"]:.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals
        residuals = y_test - y_pred
        axes[1, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Delay (minutes)')
        axes[1, 0].set_ylabel('Residuals (minutes)')
        axes[1, 0].set_title('Residual Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature importance (if available)
        if hasattr(result['pipeline'].named_steps['model'], 'feature_importances_'):
            feature_names = result['pipeline'].named_steps['preprocessor'].get_feature_names_out()
            importances = result['pipeline'].named_steps['model'].feature_importances_
            
            # Get top 10 features
            top_indices = np.argsort(importances)[-10:]
            top_features = [feature_names[i] for i in top_indices]
            top_importances = importances[top_indices]
            
            axes[1, 1].barh(range(len(top_features)), top_importances)
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features)
            axes[1, 1].set_xlabel('Feature Importance')
            axes[1, 1].set_title('Top 10 Feature Importances')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{name}_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {name} evaluation plot")

def save_model_and_metadata(best_model, best_name, results, X_train, y_train, output_dir):
    """Save the best model and metadata"""
    print(f"\nSaving best model: {best_name}")
    
    # Save model
    model_path = f'{output_dir}/model.pkl'
    joblib.dump(best_model, model_path)
    
    # Save preprocessor separately for easier access
    preprocessor_path = f'{output_dir}/preprocessor.pkl'
    joblib.dump(best_model.named_steps['preprocessor'], preprocessor_path)
    
    # Create metadata
    metadata = {
        'best_model': best_name,
        'training_date': datetime.now().isoformat(),
        'n_train': len(X_train),
        'n_test': len(X_train) // 4,  # Approximate test size
        'target_variable': 'avg_delay_minutes',
        'target_mean': float(y_train.mean()),
        'target_std': float(y_train.std()),
        'airports': sorted(X_train['airport'].unique().tolist()),
        'carriers': sorted(X_train['carrier'].unique().tolist()),
        'metrics': {}
    }
    
    # Add metrics for all models
    for name, result in results.items():
        metadata['metrics'][name] = {
            'mae': float(result['mae']),
            'rmse': float(result['rmse']),
            'r2': float(result['r2']),
            'cv_mae': float(result['cv_mae'])
        }
    
    # Add best model metrics
    best_result = results[best_name]
    metadata['selected_mae'] = float(best_result['mae'])
    metadata['selected_rmse'] = float(best_result['rmse'])
    metadata['selected_r2'] = float(best_result['r2'])
    
    # Save metadata
    metadata_path = f'{output_dir}/metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    return metadata

def main():
    """Main training pipeline"""
    print("=== BTS Airport-Level Delay Prediction Model Training ===")
    print("Target: avg_delay_minutes (airport-level delays)")
    print("Expected average: ~69.8 minutes")
    print()
    
    # Load data
    df = load_bts_data()
    
    # Prepare features
    X, y, categorical_cols, numerical_cols = prepare_features(df)
    
    # Create models
    pipelines = create_model_pipeline(categorical_cols, numerical_cols)
    
    # Evaluate models
    results, X_train, X_test, y_train, y_test = evaluate_models(pipelines, X, y)
    
    # Select best model (lowest MAE)
    best_name = min(results.keys(), key=lambda k: results[k]['mae'])
    best_model = results[best_name]['pipeline']
    
    print(f"\n=== BEST MODEL: {best_name.upper()} ===")
    print(f"MAE: {results[best_name]['mae']:.2f} minutes")
    print(f"RMSE: {results[best_name]['rmse']:.2f} minutes")
    print(f"R²: {results[best_name]['r2']:.3f}")
    
    # Create output directory
    output_dir = 'OUTPUTS/airport_model'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots
    plot_results(results, output_dir)
    
    # Save model and metadata
    metadata = save_model_and_metadata(best_model, best_name, results, X_train, y_train, output_dir)
    
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Best model: {best_name}")
    print(f"MAE: {results[best_name]['mae']:.2f} minutes")
    print(f"Expected predictions: 60-80+ minutes for hub airports")
    print(f"Model saved to: {output_dir}/")
    
    return best_model, metadata

if __name__ == "__main__":
    model, metadata = main()
