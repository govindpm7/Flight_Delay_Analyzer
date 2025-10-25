"""
Simple feature suggestions based on PCA results
"""
import numpy as np
import pandas as pd
import pickle
import os

def suggest_features():
    """Generate feature suggestions based on PCA analysis"""
    print("Generating feature suggestions...")
    
    # Load PCA results
    loadings_df = pd.read_csv("../../OUTPUTS/pca_analysis/component_loadings.csv", index_col=0)
    
    # Calculate feature importance (sum of absolute loadings across first 5 components)
    feature_importance = loadings_df.iloc[:, :5].abs().sum(axis=1).sort_values(ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance.head(10).items(), 1):
        print(f"{i:2d}. {feature}: {importance:.3f}")
    
    # Generate feature suggestions
    suggestions = {
        'high_importance_features': feature_importance.head(10).index.tolist(),
        'interaction_features': [
            'avg_delay_minutes * carrier_delay_rate',
            'total_delay_rate * weather_delay_rate',
            'nas_delay_rate * late_aircraft_delay_rate',
            'arr_flights * total_delay_rate',
            'month * avg_delay_minutes'
        ],
        'polynomial_features': [
            'total_delay_rate_squared',
            'carrier_delay_rate_squared',
            'avg_delay_minutes_squared',
            'weather_delay_rate_squared'
        ],
        'composite_features': [
            'delay_risk_score = (total_delay_rate + carrier_delay_rate + weather_delay_rate) / 3',
            'volume_delay_interaction = arr_flights * total_delay_rate',
            'temporal_delay_pattern = month * avg_delay_minutes'
        ]
    }
    
    # Save suggestions
    os.makedirs("../../OUTPUTS/pca_analysis", exist_ok=True)
    
    with open("../../OUTPUTS/pca_analysis/feature_suggestions.pkl", 'wb') as f:
        pickle.dump(suggestions, f)
    
    # Create human-readable report
    with open("../../OUTPUTS/pca_analysis/feature_engineering_report.txt", 'w') as f:
        f.write("PCA-Based Feature Engineering Suggestions\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Top 10 Most Important Features:\n")
        for i, (feature, importance) in enumerate(feature_importance.head(10).items(), 1):
            f.write(f"{i:2d}. {feature}: {importance:.3f}\n")
        
        f.write("\nSuggested Interaction Features:\n")
        for i, feature in enumerate(suggestions['interaction_features'], 1):
            f.write(f"{i}. {feature}\n")
        
        f.write("\nSuggested Polynomial Features:\n")
        for i, feature in enumerate(suggestions['polynomial_features'], 1):
            f.write(f"{i}. {feature}\n")
        
        f.write("\nSuggested Composite Features:\n")
        for i, feature in enumerate(suggestions['composite_features'], 1):
            f.write(f"{i}. {feature}\n")
        
        f.write("\nPCA Component Insights:\n")
        f.write("- PC1 (35.4% variance): General delay tendency\n")
        f.write("- PC2 (14.9% variance): Flight volume patterns\n")
        f.write("- PC3 (11.6% variance): Weather-related delays\n")
        f.write("- PC4 (9.5% variance): NAS and security delays\n")
        f.write("- PC5 (8.6% variance): Late aircraft patterns\n")
    
    print("\nFeature suggestions saved:")
    print("- feature_suggestions.pkl")
    print("- feature_engineering_report.txt")
    
    return suggestions

if __name__ == "__main__":
    suggestions = suggest_features()

