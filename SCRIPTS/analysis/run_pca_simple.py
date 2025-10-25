"""
Simplified PCA analysis
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_pca_analysis():
    """Run PCA analysis on the prepared data"""
    print("Running PCA analysis...")
    
    # Load the scaled data
    X_scaled = np.load("../../OUTPUTS/pca_analysis/X_scaled.npy")
    feature_names = np.load("../../OUTPUTS/pca_analysis/feature_names.npy")
    
    print(f"Data shape: {X_scaled.shape}")
    print(f"Features: {list(feature_names)}")
    
    # Run PCA
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)
    
    # Print results
    print("\nPCA Results:")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Create component loadings DataFrame
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=feature_names
    )
    
    print("\nComponent Loadings (Top 5 components):")
    print(loadings_df.iloc[:, :5].round(3))
    
    # Save results
    os.makedirs("../../OUTPUTS/pca_analysis", exist_ok=True)
    
    # Save loadings
    loadings_df.to_csv("../../OUTPUTS/pca_analysis/component_loadings.csv")
    
    # Save PCA results
    np.save("../../OUTPUTS/pca_analysis/pca_result.npy", pca_result)
    
    # Create scree plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             pca.explained_variance_ratio_, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Scree Plot')
    plt.grid(True)
    plt.savefig("../../OUTPUTS/pca_analysis/scree_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create loadings heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(loadings_df.iloc[:, :5], annot=True, cmap='RdBu_r', center=0)
    plt.title('PCA Component Loadings (First 5 Components)')
    plt.tight_layout()
    plt.savefig("../../OUTPUTS/pca_analysis/loadings_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nPCA analysis complete!")
    print("Files saved:")
    print("- component_loadings.csv")
    print("- pca_result.npy")
    print("- scree_plot.png")
    print("- loadings_heatmap.png")
    
    return pca, loadings_df

if __name__ == "__main__":
    pca, loadings_df = run_pca_analysis()
