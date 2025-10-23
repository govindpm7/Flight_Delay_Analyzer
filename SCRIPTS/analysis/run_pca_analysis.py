"""
Run comprehensive PCA analysis
Identify high-influence variables and component structure
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_pca_analysis(X_scaled, n_components=None):
    """
    Run PCA and return results
    If n_components=None, use all components
    """
    if n_components is None:
        n_components = min(X_scaled.shape)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return pca, X_pca

def analyze_explained_variance(pca):
    """
    Analyze explained variance by components
    """
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    results = {
        'explained_variance_ratio': explained_var,
        'cumulative_variance': cumulative_var,
        'n_components_90pct': np.argmax(cumulative_var >= 0.90) + 1,
        'n_components_95pct': np.argmax(cumulative_var >= 0.95) + 1,
        'n_components_99pct': np.argmax(cumulative_var >= 0.99) + 1,
    }
    
    return results

def identify_high_influence_features(pca, feature_names, n_top_components=5):
    """
    Identify features with highest influence on top principal components
    
    Returns:
        Dictionary mapping component -> top features with their loadings
    """
    loadings = pca.components_  # Shape: (n_components, n_features)
    
    high_influence_features = {}
    
    for i in range(min(n_top_components, loadings.shape[0])):
        component_loadings = loadings[i]
        
        # Get absolute loadings and sort
        abs_loadings = np.abs(component_loadings)
        top_indices = np.argsort(abs_loadings)[::-1][:10]  # Top 10 features
        
        feature_importance = [
            {
                'feature': feature_names[idx],
                'loading': component_loadings[idx],
                'abs_loading': abs_loadings[idx]
            }
            for idx in top_indices
        ]
        
        high_influence_features[f'PC{i+1}'] = feature_importance
    
    return high_influence_features

def create_feature_importance_matrix(pca, feature_names):
    """
    Create a matrix showing feature contributions to each component
    """
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=feature_names
    )
    
    # Add absolute sum across top 5 PCs as overall importance
    loadings_df['Overall_Importance'] = np.abs(
        loadings_df.iloc[:, :5]
    ).sum(axis=1)
    
    return loadings_df.sort_values('Overall_Importance', ascending=False)

def analyze_feature_correlations_in_pc_space(pca, feature_names):
    """
    Identify which features cluster together in PC space
    Useful for understanding feature relationships
    """
    # Correlation of features in PC space
    loadings = pca.components_[:5].T  # Top 5 PCs
    correlation_matrix = np.corrcoef(loadings)
    
    correlation_df = pd.DataFrame(
        correlation_matrix,
        index=feature_names,
        columns=feature_names
    )
    
    return correlation_df

def analyze_feature_relationships(high_influence_features):
    """
    Analyze relationships between features based on PCA loadings
    """
    relationships = {
        'delay_clusters': [],
        'temporal_clusters': [],
        'airport_clusters': [],
        'carrier_clusters': []
    }
    
    for pc_name, features in high_influence_features.items():
        pc_features = [f['feature'] for f in features[:5]]  # Top 5 features
        
        # Identify delay-related features
        delay_features = [f for f in pc_features if 'delay' in f.lower()]
        if len(delay_features) >= 2:
            relationships['delay_clusters'].append({
                'pc': pc_name,
                'features': delay_features,
                'loadings': [f['loading'] for f in features[:5] if f['feature'] in delay_features]
            })
        
        # Identify temporal features
        temporal_features = [f for f in pc_features if f in ['dep_hour', 'dow', 'month', 'is_weekend']]
        if len(temporal_features) >= 2:
            relationships['temporal_clusters'].append({
                'pc': pc_name,
                'features': temporal_features,
                'loadings': [f['loading'] for f in features[:5] if f['feature'] in temporal_features]
            })
        
        # Identify airport-related features
        airport_features = [f for f in pc_features if 'origin' in f.lower() or 'dest' in f.lower()]
        if len(airport_features) >= 2:
            relationships['airport_clusters'].append({
                'pc': pc_name,
                'features': airport_features,
                'loadings': [f['loading'] for f in features[:5] if f['feature'] in airport_features]
            })
        
        # Identify carrier-related features
        carrier_features = [f for f in pc_features if 'carrier' in f.lower() or 'airline' in f.lower()]
        if len(carrier_features) >= 2:
            relationships['carrier_clusters'].append({
                'pc': pc_name,
                'features': carrier_features,
                'loadings': [f['loading'] for f in features[:5] if f['feature'] in carrier_features]
            })
    
    return relationships

if __name__ == "__main__":
    # Create output directory
    os.makedirs("OUTPUTS/pca_analysis", exist_ok=True)
    
    # Load prepared data
    X_scaled = np.load("OUTPUTS/pca_analysis/X_scaled.npy")
    
    with open("OUTPUTS/pca_analysis/feature_names.pkl", 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"Running PCA Analysis")
    print(f"   Samples: {X_scaled.shape[0]}")
    print(f"   Features: {X_scaled.shape[1]}")
    print("=" * 80)
    
    # Run PCA
    pca, X_pca = run_pca_analysis(X_scaled)
    
    # Analyze explained variance
    variance_results = analyze_explained_variance(pca)
    print(f"\nVariance Analysis:")
    print(f"   Components for 90% variance: {variance_results['n_components_90pct']}")
    print(f"   Components for 95% variance: {variance_results['n_components_95pct']}")
    print(f"   Components for 99% variance: {variance_results['n_components_99pct']}")
    
    # Identify high influence features
    print(f"\nHigh Influence Features:")
    high_influence = identify_high_influence_features(pca, feature_names, n_top_components=5)
    
    for component, features in high_influence.items():
        pc_num = int(component[2:]) - 1
        variance_explained = pca.explained_variance_ratio_[pc_num] * 100
        print(f"\n{component} (Explains {variance_explained:.2f}% variance):")
        for feat in features[:5]:  # Top 5
            print(f"   - {feat['feature']}: {feat['loading']:.4f}")
    
    # Create feature importance matrix
    importance_matrix = create_feature_importance_matrix(pca, feature_names)
    print(f"\nOverall Feature Importance (Top 10):")
    print(importance_matrix.head(10)[['Overall_Importance']])
    
    # Analyze correlations
    correlation_df = analyze_feature_correlations_in_pc_space(pca, feature_names)
    
    # Analyze feature relationships
    relationships = analyze_feature_relationships(high_influence)
    
    print(f"\nFeature Relationship Analysis:")
    for category, clusters in relationships.items():
        if clusters:
            print(f"\n{category.replace('_', ' ').title()}:")
            for cluster in clusters:
                print(f"   {cluster['pc']}: {', '.join(cluster['features'])}")
    
    # Save results
    with open("OUTPUTS/pca_analysis/pca_model.pkl", 'wb') as f:
        pickle.dump(pca, f)
    
    np.save("OUTPUTS/pca_analysis/X_pca.npy", X_pca)
    importance_matrix.to_csv("OUTPUTS/pca_analysis/feature_importance_matrix.csv")
    correlation_df.to_csv("OUTPUTS/pca_analysis/feature_correlations_pc_space.csv")
    
    with open("OUTPUTS/pca_analysis/high_influence_features.pkl", 'wb') as f:
        pickle.dump(high_influence, f)
    
    with open("OUTPUTS/pca_analysis/variance_results.pkl", 'wb') as f:
        pickle.dump(variance_results, f)
    
    with open("OUTPUTS/pca_analysis/feature_relationships.pkl", 'wb') as f:
        pickle.dump(relationships, f)
    
    print(f"\nPCA analysis complete. Results saved to OUTPUTS/pca_analysis/")
    print(f"   - PCA model: pca_model.pkl")
    print(f"   - Transformed data: X_pca.npy")
    print(f"   - Feature importance: feature_importance_matrix.csv")
    print(f"   - Feature correlations: feature_correlations_pc_space.csv")
    print(f"   - High influence features: high_influence_features.pkl")
    print(f"   - Variance results: variance_results.pkl")
    print(f"   - Feature relationships: feature_relationships.pkl")
