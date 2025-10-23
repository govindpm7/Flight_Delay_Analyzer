"""
Retrain model with PCA-derived features
Compare performance before and after
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import json
from datetime import datetime
import os
import sys

# Add the modeling directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modeling'))

# Import feature engineering
from feature_engineering_advanced import AdvancedFeatureEngineer

def load_baseline_results(path="OUTPUTS/metadata.json"):
    """Load baseline model performance"""
    with open(path, 'r') as f:
        metadata = json.load(f)
    return metadata

def prepare_training_data_with_new_features(data_path="OUTPUTS/pca_analysis/training_data.csv"):
    """
    Load training data and apply advanced feature engineering
    """
    # Load original training data
    df = pd.read_csv(data_path)
    
    # Initialize advanced feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Apply feature engineering
    df_engineered = engineer.engineer_all_features(df, priority='high')
    
    print(f"\nData prepared:")
    print(f"   Samples: {len(df_engineered)}")
    print(f"   Original features: {df.shape[1]}")
    print(f"   Total features: {df_engineered.shape[1]}")
    print(f"   New features: {len(engineer.get_feature_list())}")
    
    return df_engineered, engineer

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type='rf'):
    """
    Train model and evaluate
    """
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gbm':
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        )
    
    print(f"\nTraining {model_type.upper()} model...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    return model, metrics

def compare_models(baseline_metrics, new_metrics):
    """
    Compare baseline vs new model performance
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    print(f"\nTest MAE:")
    print(f"  Baseline: {baseline_metrics.get('selected_mae', 'N/A')} minutes")
    print(f"  New Model: {new_metrics['test_mae']:.2f} minutes")
    
    if 'selected_mae' in baseline_metrics:
        improvement = baseline_metrics['selected_mae'] - new_metrics['test_mae']
        pct_improvement = (improvement / baseline_metrics['selected_mae']) * 100
        print(f"  Improvement: {improvement:.2f} minutes ({pct_improvement:.1f}%)")
    
    print(f"\nTest R²:")
    print(f"  New Model: {new_metrics['test_r2']:.4f}")
    
    print(f"\nTest RMSE:")
    print(f"  New Model: {new_metrics['test_rmse']:.2f} minutes")

def analyze_feature_importance(model, feature_names, top_n=20):
    """
    Analyze feature importance from trained model
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Feature Importances:")
        print(feature_importance_df.head(top_n).to_string(index=False))
        
        # Identify new features in top performers
        # Assuming new features have specific naming patterns
        new_feature_indicators = ['_X_', '_DIV_', '_squared', 'log_', 'composite_', 'temporal_', 'route_congestion', 'delay_type_diversity', '_group']
        
        top_features = feature_importance_df.head(top_n)
        new_features_in_top = []
        
        for _, row in top_features.iterrows():
            if any(indicator in row['feature'] for indicator in new_feature_indicators):
                new_features_in_top.append(row['feature'])
        
        if new_features_in_top:
            print(f"\nNew PCA-derived features in top {top_n}:")
            for feat in new_features_in_top:
                importance = feature_importance_df[feature_importance_df['feature'] == feat]['importance'].values[0]
                print(f"   - {feat}: {importance:.4f}")
        
        return feature_importance_df
    
    return None

def analyze_new_feature_impact(engineer, df_engineered):
    """
    Analyze the impact of new features
    """
    print(f"\nNew Feature Analysis:")
    print("=" * 80)
    
    # Get feature importance ranking
    importance_df = engineer.get_feature_importance_ranking(df_engineered)
    
    if not importance_df.empty:
        print(f"\nTop 10 New Features by Importance:")
        print(importance_df.head(10).to_string(index=False))
        
        # Analyze feature types
        feature_types = {
            'interaction': [f for f in engineer.get_feature_list() if '_X_' in f or '_DIV_' in f],
            'polynomial': [f for f in engineer.get_feature_list() if '_squared' in f or 'log_' in f],
            'composite': [f for f in engineer.get_feature_list() if any(x in f for x in ['composite_', 'temporal_', 'route_congestion', 'delay_type_diversity'])],
            'aggregation': [f for f in engineer.get_feature_list() if '_group' in f]
        }
        
        print(f"\nFeature Type Breakdown:")
        for ftype, features in feature_types.items():
            if features:
                print(f"   {ftype.title()}: {len(features)} features")
                # Show top feature of each type
                type_importance = importance_df[importance_df['feature'].isin(features)]
                if not type_importance.empty:
                    top_feature = type_importance.iloc[0]
                    print(f"      Top: {top_feature['feature']} (importance: {top_feature['importance_score']:.4f})")
    
    return importance_df

if __name__ == "__main__":
    print("Retraining Model with PCA-Derived Features")
    print("=" * 80)
    
    # Load baseline performance
    baseline_metrics = load_baseline_results()
    print(f"\nBaseline Model Performance:")
    print(f"   Model: {baseline_metrics.get('best_model', 'N/A')}")
    print(f"   Test MAE: {baseline_metrics.get('selected_mae', 'N/A')} minutes")
    
    # Prepare data with new features
    df_engineered, engineer = prepare_training_data_with_new_features()
    
    # Split features and target
    target_col = 'DEP_DELAY'
    
    # Separate numerical and categorical features
    numerical_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_engineered.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target from numerical features
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    # For now, only use numerical features for training
    feature_cols = numerical_cols
    
    print(f"   Numerical features: {len(numerical_cols)}")
    print(f"   Categorical features: {len(categorical_cols)}")
    print(f"   Using {len(feature_cols)} features for training")
    
    X = df_engineered[feature_cols]
    y = df_engineered[target_col]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining split:")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X.shape[1]}")
    
    # Train model
    model, new_metrics = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, 
        model_type='rf'
    )
    
    # Compare models
    compare_models(baseline_metrics, new_metrics)
    
    # Analyze feature importance
    feature_importance_df = analyze_feature_importance(model, feature_cols, top_n=25)
    
    # Analyze new feature impact
    new_feature_importance = analyze_new_feature_impact(engineer, df_engineered)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"OUTPUTS/pca_retraining_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    with open(f"{output_dir}/model_pca_features.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'baseline_mae': baseline_metrics.get('selected_mae'),
        'new_mae': new_metrics['test_mae'],
        'new_rmse': new_metrics['test_rmse'],
        'new_r2': new_metrics['test_r2'],
        'improvement_minutes': baseline_metrics.get('selected_mae', 0) - new_metrics['test_mae'],
        'improvement_percentage': ((baseline_metrics.get('selected_mae', 0) - new_metrics['test_mae']) / baseline_metrics.get('selected_mae', 1)) * 100,
        'n_original_features': baseline_metrics.get('n_features', len(baseline_metrics.get('feature_columns', []))),
        'n_total_features': len(feature_cols),
        'n_new_features': len(engineer.get_feature_list()),
        'new_features': engineer.get_feature_list(),
        'model_type': 'RandomForestRegressor'
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save feature importance
    if feature_importance_df is not None:
        feature_importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    
    # Save new feature analysis
    if not new_feature_importance.empty:
        new_feature_importance.to_csv(f"{output_dir}/new_feature_importance.csv", index=False)
    
    # Save training data with new features
    df_engineered.to_csv(f"{output_dir}/training_data_with_new_features.csv", index=False)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"\nRetraining complete!")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"   Original features: {metadata['n_original_features']}")
    print(f"   New features added: {metadata['n_new_features']}")
    print(f"   Total features: {metadata['n_total_features']}")
    print(f"   MAE improvement: {metadata['improvement_minutes']:.2f} minutes ({metadata['improvement_percentage']:.1f}%)")
    print(f"   New R²: {metadata['new_r2']:.4f}")
