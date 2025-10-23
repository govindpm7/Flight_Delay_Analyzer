"""
Create visualizations for PCA analysis results
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def plot_scree_plot(pca, save_path="OUTPUTS/pca_analysis/plots/"):
    """
    Scree plot showing explained variance by component
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    n_components = len(pca.explained_variance_ratio_)
    x = range(1, min(21, n_components + 1))  # First 20 components
    
    # Individual variance
    ax1.bar(x, pca.explained_variance_ratio_[:20], alpha=0.7, color='steelblue')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Scree Plot - Individual Variance Explained', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Cumulative variance
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(x, cumulative_var[:20], marker='o', linewidth=2, color='darkred')
    ax2.axhline(y=0.90, color='green', linestyle='--', label='90% Variance', linewidth=2)
    ax2.axhline(y=0.95, color='orange', linestyle='--', label='95% Variance', linewidth=2)
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}scree_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Scree plot saved")

def plot_component_loadings(pca, feature_names, n_components=3, save_path="OUTPUTS/pca_analysis/plots/"):
    """
    Visualize feature loadings for top components
    """
    fig, axes = plt.subplots(1, n_components, figsize=(20, 6))
    
    for i in range(n_components):
        loadings = pca.components_[i]
        sorted_idx = np.argsort(np.abs(loadings))[::-1][:15]  # Top 15
        
        sorted_features = [feature_names[idx] for idx in sorted_idx]
        sorted_loadings = loadings[sorted_idx]
        
        colors = ['red' if x < 0 else 'green' for x in sorted_loadings]
        
        axes[i].barh(range(len(sorted_features)), sorted_loadings, color=colors, alpha=0.7)
        axes[i].set_yticks(range(len(sorted_features)))
        axes[i].set_yticklabels(sorted_features, fontsize=9)
        axes[i].set_xlabel('Loading', fontsize=11)
        axes[i].set_title(
            f'PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}% var)',
            fontsize=12,
            fontweight='bold'
        )
        axes[i].axvline(x=0, color='black', linewidth=0.8)
        axes[i].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}component_loadings.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Component loadings plot saved")

def plot_biplot(pca, X_pca, feature_names, pc_x=0, pc_y=1, save_path="OUTPUTS/pca_analysis/plots/"):
    """
    Biplot showing both samples and feature vectors
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Sample scatter (subsample for clarity)
    sample_size = min(5000, X_pca.shape[0])
    indices = np.random.choice(X_pca.shape[0], sample_size, replace=False)
    
    scatter = ax.scatter(
        X_pca[indices, pc_x],
        X_pca[indices, pc_y],
        alpha=0.3,
        s=5,
        c='lightblue',
        edgecolors='none'
    )
    
    # Feature vectors (top 10 by loading magnitude)
    loadings = pca.components_[[pc_x, pc_y]].T
    loading_magnitude = np.sqrt((loadings ** 2).sum(axis=1))
    top_features_idx = np.argsort(loading_magnitude)[::-1][:10]
    
    scale_factor = 3  # Scale arrows for visibility
    
    for idx in top_features_idx:
        ax.arrow(
            0, 0,
            loadings[idx, 0] * scale_factor,
            loadings[idx, 1] * scale_factor,
            head_width=0.1,
            head_length=0.1,
            fc='red',
            ec='darkred',
            alpha=0.7,
            linewidth=2
        )
        ax.text(
            loadings[idx, 0] * scale_factor * 1.15,
            loadings[idx, 1] * scale_factor * 1.15,
            feature_names[idx],
            fontsize=9,
            fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )
    
    ax.set_xlabel(
        f'PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]*100:.2f}% variance)',
        fontsize=12
    )
    ax.set_ylabel(
        f'PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]*100:.2f}% variance)',
        fontsize=12
    )
    ax.set_title('PCA Biplot - Samples and Feature Vectors', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}biplot_pc{pc_x+1}_pc{pc_y+1}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Biplot (PC{pc_x+1} vs PC{pc_y+1}) saved")

def plot_feature_importance_heatmap(importance_matrix, save_path="OUTPUTS/pca_analysis/plots/"):
    """
    Heatmap of feature loadings across components
    """
    fig, ax = plt.subplots(figsize=(12, 16))
    
    # Top 30 features by overall importance
    top_features = importance_matrix.head(30)
    
    # Select first 10 PCs
    heatmap_data = top_features.iloc[:, :10]
    
    sns.heatmap(
        heatmap_data,
        cmap='RdBu_r',
        center=0,
        annot=False,
        fmt='.2f',
        cbar_kws={'label': 'Loading'},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title('Feature Loadings Across Top 10 Principal Components', fontsize=14, fontweight='bold')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}feature_importance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Feature importance heatmap saved")

def plot_correlation_heatmap(correlation_df, n_features=20, save_path="OUTPUTS/pca_analysis/plots/"):
    """
    Heatmap showing feature correlations in PC space
    """
    # Get top features by variance
    feature_variance = correlation_df.var()
    top_features = feature_variance.nlargest(n_features).index
    
    corr_subset = correlation_df.loc[top_features, top_features]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(
        corr_subset,
        cmap='coolwarm',
        center=0,
        annot=False,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )
    
    ax.set_title('Feature Correlations in PC Space (Top 20 Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_path}feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Correlation heatmap saved")

def plot_feature_relationship_analysis(relationships, save_path="OUTPUTS/pca_analysis/plots/"):
    """
    Create visualizations for feature relationship analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    categories = ['delay_clusters', 'temporal_clusters', 'airport_clusters', 'carrier_clusters']
    titles = ['Delay Feature Clusters', 'Temporal Feature Clusters', 'Airport Feature Clusters', 'Carrier Feature Clusters']
    
    for i, (category, title) in enumerate(zip(categories, titles)):
        ax = axes[i]
        
        if relationships[category]:
            # Count features by PC
            pc_counts = {}
            for cluster in relationships[category]:
                pc = cluster['pc']
                if pc not in pc_counts:
                    pc_counts[pc] = 0
                pc_counts[pc] += len(cluster['features'])
            
            # Create bar plot
            pcs = list(pc_counts.keys())
            counts = list(pc_counts.values())
            
            bars = ax.bar(pcs, counts, alpha=0.7, color=['steelblue', 'orange', 'green', 'red', 'purple'][:len(pcs)])
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Number of Related Features')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No clusters found', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, style='italic')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f"{save_path}feature_relationship_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Feature relationship analysis plot saved")

if __name__ == "__main__":
    # Create plots directory
    os.makedirs("OUTPUTS/pca_analysis/plots", exist_ok=True)
    
    # Load results
    with open("OUTPUTS/pca_analysis/pca_model.pkl", 'rb') as f:
        pca = pickle.load(f)
    
    with open("OUTPUTS/pca_analysis/feature_names.pkl", 'rb') as f:
        feature_names = pickle.load(f)
    
    X_pca = np.load("OUTPUTS/pca_analysis/X_pca.npy")
    importance_matrix = pd.read_csv("OUTPUTS/pca_analysis/feature_importance_matrix.csv", index_col=0)
    correlation_df = pd.read_csv("OUTPUTS/pca_analysis/feature_correlations_pc_space.csv", index_col=0)
    
    with open("OUTPUTS/pca_analysis/feature_relationships.pkl", 'rb') as f:
        relationships = pickle.load(f)
    
    print("Creating visualizations...")
    print("=" * 80)
    
    # Generate all plots
    plot_scree_plot(pca)
    plot_component_loadings(pca, feature_names, n_components=3)
    plot_biplot(pca, X_pca, feature_names, pc_x=0, pc_y=1)
    plot_biplot(pca, X_pca, feature_names, pc_x=0, pc_y=2)
    plot_feature_importance_heatmap(importance_matrix)
    plot_correlation_heatmap(correlation_df, n_features=20)
    plot_feature_relationship_analysis(relationships)
    
    print("=" * 80)
    print(f"All visualizations saved to OUTPUTS/pca_analysis/plots/")
    print(f"   - scree_plot.png")
    print(f"   - component_loadings.png")
    print(f"   - biplot_pc1_pc2.png")
    print(f"   - biplot_pc1_pc3.png")
    print(f"   - feature_importance_heatmap.png")
    print(f"   - feature_correlation_heatmap.png")
    print(f"   - feature_relationship_analysis.png")
