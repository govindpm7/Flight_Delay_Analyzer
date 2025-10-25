"""
Simple PCA visualization
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_visualizations():
    """Create PCA visualizations"""
    print("Creating PCA visualizations...")
    
    # Load the data
    loadings_df = pd.read_csv("../../OUTPUTS/pca_analysis/component_loadings.csv", index_col=0)
    pca_result = np.load("../../OUTPUTS/pca_analysis/pca_result.npy")
    
    # Create output directory for plots
    os.makedirs("../../OUTPUTS/pca_analysis/plots", exist_ok=True)
    
    # 1. Biplot (PC1 vs PC2)
    plt.figure(figsize=(12, 8))
    
    # Plot data points
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=50)
    
    # Plot feature vectors
    feature_names = loadings_df.index
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, loadings_df.iloc[i, 0]*3, loadings_df.iloc[i, 1]*3, 
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
        plt.text(loadings_df.iloc[i, 0]*3.2, loadings_df.iloc[i, 1]*3.2, 
                feature, fontsize=8, ha='center', va='center')
    
    plt.xlabel(f'PC1 ({loadings_df.columns[0]})')
    plt.ylabel(f'PC2 ({loadings_df.columns[1]})')
    plt.title('PCA Biplot: PC1 vs PC2')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig("../../OUTPUTS/pca_analysis/plots/biplot_pc1_pc2.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature importance heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(loadings_df.iloc[:, :5], annot=True, cmap='RdBu_r', center=0, 
                fmt='.3f', cbar_kws={'label': 'Loading'})
    plt.title('PCA Component Loadings (First 5 Components)')
    plt.xlabel('Principal Components')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig("../../OUTPUTS/pca_analysis/plots/feature_importance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Explained variance plot
    explained_var = np.load("../../OUTPUTS/pca_analysis/X_scaled.npy")
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(explained_var)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance by Component')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../../OUTPUTS/pca_analysis/plots/explained_variance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations created:")
    print("- biplot_pc1_pc2.png")
    print("- feature_importance_heatmap.png") 
    print("- explained_variance.png")

if __name__ == "__main__":
    create_visualizations()

