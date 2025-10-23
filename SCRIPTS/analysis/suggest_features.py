"""
Analyze PCA results and automatically suggest new features
"""
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Tuple

class FeatureSuggestionEngine:
    """
    Analyzes PCA results to suggest new engineered features
    """
    
    def __init__(self, pca_results_dir="OUTPUTS/pca_analysis/"):
        self.pca_results_dir = pca_results_dir
        self.load_results()
    
    def load_results(self):
        """Load PCA analysis results"""
        with open(f"{self.pca_results_dir}/pca_model.pkl", 'rb') as f:
            self.pca = pickle.load(f)
        
        with open(f"{self.pca_results_dir}/feature_names.pkl", 'rb') as f:
            self.feature_names = pickle.load(f)
        
        with open(f"{self.pca_results_dir}/high_influence_features.pkl", 'rb') as f:
            self.high_influence = pickle.load(f)
        
        self.importance_matrix = pd.read_csv(
            f"{self.pca_results_dir}/feature_importance_matrix.csv",
            index_col=0
        )
        
        self.correlation_df = pd.read_csv(
            f"{self.pca_results_dir}/feature_correlations_pc_space.csv",
            index_col=0
        )
        
        with open(f"{self.pca_results_dir}/feature_relationships.pkl", 'rb') as f:
            self.relationships = pickle.load(f)
    
    def suggest_interaction_features(self, top_n=15) -> List[Dict]:
        """
        Suggest interaction features based on feature clustering in PC space
        Features that load similarly suggest multiplicative interactions
        """
        suggestions = []
        
        # Look at top 3 PCs
        for pc_idx in range(min(3, self.pca.n_components_)):
            loadings = self.pca.components_[pc_idx]
            
            # Get features with high positive loadings
            high_positive_idx = np.where(loadings > 0.2)[0]
            high_negative_idx = np.where(loadings < -0.2)[0]
            
            # Suggest interactions within same loading direction
            for i in range(len(high_positive_idx)):
                for j in range(i+1, min(i+5, len(high_positive_idx))):
                    feat1 = self.feature_names[high_positive_idx[i]]
                    feat2 = self.feature_names[high_positive_idx[j]]
                    
                    suggestions.append({
                        'type': 'interaction',
                        'operation': 'multiply',
                        'features': [feat1, feat2],
                        'new_feature_name': f"{feat1}_X_{feat2}",
                        'rationale': f"Both load positively on PC{pc_idx+1} (loadings: {loadings[high_positive_idx[i]]:.3f}, {loadings[high_positive_idx[j]]:.3f})",
                        'priority': 'high' if pc_idx == 0 else 'medium'
                    })
            
            # Suggest ratios for opposite loading features
            for i in range(min(5, len(high_positive_idx))):
                for j in range(min(5, len(high_negative_idx))):
                    feat1 = self.feature_names[high_positive_idx[i]]
                    feat2 = self.feature_names[high_negative_idx[j]]
                    
                    suggestions.append({
                        'type': 'interaction',
                        'operation': 'divide',
                        'features': [feat1, feat2],
                        'new_feature_name': f"{feat1}_DIV_{feat2}",
                        'rationale': f"Opposite loadings on PC{pc_idx+1} suggest ratio relationship",
                        'priority': 'medium'
                    })
        
        return suggestions[:top_n]
    
    def suggest_polynomial_features(self, top_n=10) -> List[Dict]:
        """
        Suggest polynomial transformations for high-importance features
        """
        suggestions = []
        
        # Get top features by overall importance
        top_features = self.importance_matrix.head(top_n).index.tolist()
        
        for feature in top_features:
            importance = self.importance_matrix.loc[feature, 'Overall_Importance']
            
            # Square transformation
            suggestions.append({
                'type': 'polynomial',
                'operation': 'square',
                'features': [feature],
                'new_feature_name': f"{feature}_squared",
                'rationale': f"High overall importance ({importance:.3f}) suggests non-linear effects",
                'priority': 'high' if importance > self.importance_matrix['Overall_Importance'].quantile(0.9) else 'medium'
            })
            
            # Log transformation (for positive features)
            suggestions.append({
                'type': 'polynomial',
                'operation': 'log',
                'features': [feature],
                'new_feature_name': f"log_{feature}",
                'rationale': f"May have multiplicative effects (importance: {importance:.3f})",
                'priority': 'medium'
            })
        
        return suggestions
    
    def suggest_aggregation_features(self) -> List[Dict]:
        """
        Suggest aggregate features based on correlated feature groups
        """
        suggestions = []
        
        # Find strongly correlated feature groups
        corr_matrix = self.correlation_df.abs()
        
        # Identify clusters of correlated features
        processed_features = set()
        
        for feature in corr_matrix.index:
            if feature in processed_features:
                continue
            
            # Find features highly correlated with this one
            correlated = corr_matrix[feature][corr_matrix[feature] > 0.7].index.tolist()
            
            if len(correlated) >= 3:  # At least 3 correlated features
                suggestions.append({
                    'type': 'aggregation',
                    'operation': 'mean',
                    'features': correlated,
                    'new_feature_name': f"avg_{'_'.join(correlated[:2])}_group",
                    'rationale': f"These {len(correlated)} features are highly correlated (>0.7), suggesting a common underlying factor",
                    'priority': 'medium'
                })
                
                suggestions.append({
                    'type': 'aggregation',
                    'operation': 'sum',
                    'features': correlated,
                    'new_feature_name': f"sum_{'_'.join(correlated[:2])}_group",
                    'rationale': f"Additive effect of correlated features",
                    'priority': 'low'
                })
                
                processed_features.update(correlated)
        
        return suggestions
    
    def suggest_domain_specific_features(self) -> List[Dict]:
        """
        Suggest domain-specific features for flight delays
        Based on high-influence features identified by PCA
        """
        suggestions = []
        
        # Analyze what types of features are important
        top_features = self.importance_matrix.head(20).index.tolist()
        
        # Check for delay-related features
        delay_features = [f for f in top_features if 'delay' in f.lower()]
        rate_features = [f for f in top_features if 'rate' in f.lower()]
        time_features = [f for f in top_features if any(x in f.lower() for x in ['hour', 'dow', 'month'])]
        
        # Suggest composite delay risk score
        if len(delay_features) >= 2:
            suggestions.append({
                'type': 'domain_composite',
                'operation': 'weighted_average',
                'features': delay_features,
                'new_feature_name': 'composite_delay_risk_score',
                'rationale': f"Multiple delay features ({len(delay_features)}) are important - create weighted composite",
                'priority': 'high',
                'implementation': 'Weighted average of normalized delay features'
            })
        
        # Suggest time-based risk score
        if len(time_features) >= 2:
            suggestions.append({
                'type': 'domain_composite',
                'operation': 'categorical_encoding',
                'features': time_features,
                'new_feature_name': 'temporal_risk_category',
                'rationale': f"Multiple temporal features ({len(time_features)}) - bin into high/medium/low risk periods",
                'priority': 'high',
                'implementation': 'Create risk categories based on time patterns'
            })
        
        # Suggest origin-dest interaction if both are important
        if any('origin' in f.lower() for f in top_features) and any('dest' in f.lower() for f in top_features):
            suggestions.append({
                'type': 'domain_composite',
                'operation': 'route_risk',
                'features': ['origin delay features', 'dest delay features'],
                'new_feature_name': 'route_congestion_score',
                'rationale': "Both origin and destination delays are important - create route-level risk score",
                'priority': 'high',
                'implementation': 'Combine origin and dest delay probabilities'
            })
        
        # Suggest delay type interaction
        delay_type_features = [f for f in top_features if any(dt in f.lower() for dt in ['carrier', 'weather', 'nas', 'security', 'late_aircraft'])]
        if len(delay_type_features) >= 3:
            suggestions.append({
                'type': 'domain_composite',
                'operation': 'delay_type_balance',
                'features': delay_type_features,
                'new_feature_name': 'delay_type_diversity_score',
                'rationale': f"Multiple delay types ({len(delay_type_features)}) - measure diversity/balance",
                'priority': 'medium',
                'implementation': 'Calculate entropy or variance of delay type contributions'
            })
        
        return suggestions
    
    def suggest_pca_derived_features(self) -> List[Dict]:
        """
        Suggest features derived directly from PCA components
        """
        suggestions = []
        
        # Use top 5 principal components as features
        for i in range(min(5, self.pca.n_components_)):
            variance_explained = self.pca.explained_variance_ratio_[i] * 100
            
            suggestions.append({
                'type': 'pca_derived',
                'operation': 'principal_component',
                'features': [f'PC{i+1}'],
                'new_feature_name': f'principal_component_{i+1}',
                'rationale': f"PC{i+1} explains {variance_explained:.1f}% of variance - captures key patterns",
                'priority': 'high' if variance_explained > 15 else 'medium',
                'implementation': f'Use PC{i+1} scores directly as feature'
            })
        
        return suggestions
    
    def generate_feature_engineering_report(self, output_path="OUTPUTS/pca_analysis/"):
        """
        Generate comprehensive feature engineering report
        """
        report = {
            'interaction_features': self.suggest_interaction_features(top_n=15),
            'polynomial_features': self.suggest_polynomial_features(top_n=10),
            'aggregation_features': self.suggest_aggregation_features(),
            'domain_specific_features': self.suggest_domain_specific_features(),
            'pca_derived_features': self.suggest_pca_derived_features()
        }
        
        # Save as pickle for programmatic access
        with open(f"{output_path}/feature_suggestions.pkl", 'wb') as f:
            pickle.dump(report, f)
        
        # Save as human-readable text
        with open(f"{output_path}/feature_engineering_report.txt", 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PCA-DRIVEN FEATURE ENGINEERING RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary statistics
            total_suggestions = sum(len(suggestions) for suggestions in report.values())
            f.write(f"SUMMARY:\n")
            f.write(f"- Total feature suggestions: {total_suggestions}\n")
            f.write(f"- Interaction features: {len(report['interaction_features'])}\n")
            f.write(f"- Polynomial features: {len(report['polynomial_features'])}\n")
            f.write(f"- Aggregation features: {len(report['aggregation_features'])}\n")
            f.write(f"- Domain-specific features: {len(report['domain_specific_features'])}\n")
            f.write(f"- PCA-derived features: {len(report['pca_derived_features'])}\n\n")
            
            for category, suggestions in report.items():
                f.write(f"\n{category.upper().replace('_', ' ')}\n")
                f.write("-" * 80 + "\n")
                
                for i, suggestion in enumerate(suggestions, 1):
                    f.write(f"\n{i}. {suggestion['new_feature_name']}\n")
                    f.write(f"   Type: {suggestion['type']}\n")
                    f.write(f"   Operation: {suggestion['operation']}\n")
                    f.write(f"   Input Features: {', '.join(map(str, suggestion['features']))}\n")
                    f.write(f"   Rationale: {suggestion['rationale']}\n")
                    f.write(f"   Priority: {suggestion['priority']}\n")
                    if 'implementation' in suggestion:
                        f.write(f"   Implementation: {suggestion['implementation']}\n")
                    f.write("\n")
        
        return report

if __name__ == "__main__":
    engine = FeatureSuggestionEngine()
    
    print("Analyzing PCA results...")
    print("=" * 80)
    
    report = engine.generate_feature_engineering_report()
    
    print(f"\nFeature engineering report generated!")
    print(f"\nSuggested {len(report['interaction_features'])} interaction features")
    print(f"Suggested {len(report['polynomial_features'])} polynomial features")
    print(f"Suggested {len(report['aggregation_features'])} aggregation features")
    print(f"Suggested {len(report['domain_specific_features'])} domain-specific features")
    print(f"Suggested {len(report['pca_derived_features'])} PCA-derived features")
    
    print(f"\nFull report saved to: OUTPUTS/pca_analysis/feature_engineering_report.txt")
    print(f"Programmatic access: OUTPUTS/pca_analysis/feature_suggestions.pkl")
