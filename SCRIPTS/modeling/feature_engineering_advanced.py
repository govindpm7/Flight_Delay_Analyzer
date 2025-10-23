"""
Advanced feature engineering based on PCA insights
Implements suggested features from PCA analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List
import pickle
import os

class AdvancedFeatureEngineer:
    """
    Implements advanced features identified through PCA analysis
    """
    
    def __init__(self, feature_suggestions_path="OUTPUTS/pca_analysis/feature_suggestions.pkl"):
        """Load feature suggestions"""
        try:
            with open(feature_suggestions_path, 'rb') as f:
                self.suggestions = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not load feature suggestions from {feature_suggestions_path}")
            self.suggestions = {
                'interaction_features': [],
                'polynomial_features': [],
                'aggregation_features': [],
                'domain_specific_features': [],
                'pca_derived_features': []
            }
        
        self.implemented_features = []
    
    def add_interaction_features(self, df: pd.DataFrame, priority_filter='high') -> pd.DataFrame:
        """
        Add interaction features from suggestions
        """
        df_new = df.copy()
        
        for suggestion in self.suggestions['interaction_features']:
            if suggestion['priority'] != priority_filter and priority_filter != 'all':
                continue
            
            features = suggestion['features']
            new_feature_name = suggestion['new_feature_name']
            operation = suggestion['operation']
            
            # Check if features exist
            if not all(f in df.columns for f in features):
                continue
            
            try:
                if operation == 'multiply':
                    df_new[new_feature_name] = df[features[0]] * df[features[1]]
                elif operation == 'divide':
                    # Avoid division by zero
                    df_new[new_feature_name] = df[features[0]] / (df[features[1]] + 1e-8)
                
                self.implemented_features.append(new_feature_name)
                print(f"Added interaction feature: {new_feature_name}")
                
            except Exception as e:
                print(f"Failed to create {new_feature_name}: {e}")
        
        return df_new
    
    def add_polynomial_features(self, df: pd.DataFrame, priority_filter='high') -> pd.DataFrame:
        """
        Add polynomial transformations
        """
        df_new = df.copy()
        
        for suggestion in self.suggestions['polynomial_features']:
            if suggestion['priority'] != priority_filter and priority_filter != 'all':
                continue
            
            feature = suggestion['features'][0]
            new_feature_name = suggestion['new_feature_name']
            operation = suggestion['operation']
            
            if feature not in df.columns:
                continue
            
            try:
                if operation == 'square':
                    df_new[new_feature_name] = df[feature] ** 2
                elif operation == 'log':
                    # Only apply log to positive values
                    df_new[new_feature_name] = np.log1p(np.maximum(df[feature], 0))
                elif operation == 'sqrt':
                    df_new[new_feature_name] = np.sqrt(np.maximum(df[feature], 0))
                
                self.implemented_features.append(new_feature_name)
                print(f"Added polynomial feature: {new_feature_name}")
                
            except Exception as e:
                print(f"Failed to create {new_feature_name}: {e}")
        
        return df_new
    
    def add_domain_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add domain-specific composite features
        """
        df_new = df.copy()
        
        for suggestion in self.suggestions['domain_specific_features']:
            new_feature_name = suggestion['new_feature_name']
            
            try:
                if 'composite_delay_risk_score' in new_feature_name:
                    # Create weighted delay risk score
                    delay_cols = [col for col in df.columns if 'delay' in col.lower() and 'avg' in col.lower()]
                    if delay_cols:
                        # Normalize and average
                        normalized = df[delay_cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))
                        df_new[new_feature_name] = normalized.mean(axis=1)
                        self.implemented_features.append(new_feature_name)
                        print(f"Added composite feature: {new_feature_name}")
                
                elif 'temporal_risk_category' in new_feature_name:
                    # Create time-based risk categories
                    if 'dep_hour' in df.columns and 'dow' in df.columns:
                        # Rush hour + weekend combinations
                        is_rush = df['dep_hour'].isin([6, 7, 8, 16, 17, 18, 19])
                        is_weekend = df['dow'].isin([5, 6])
                        
                        df_new[new_feature_name] = 0  # Low risk
                        df_new.loc[is_rush & ~is_weekend, new_feature_name] = 2  # High risk
                        df_new.loc[is_rush & is_weekend, new_feature_name] = 1  # Medium risk
                        df_new.loc[~is_rush & is_weekend, new_feature_name] = 1  # Medium risk
                        
                        self.implemented_features.append(new_feature_name)
                        print(f"Added temporal feature: {new_feature_name}")
                
                elif 'route_congestion_score' in new_feature_name:
                    # Create route-level congestion score
                    origin_delay_cols = [col for col in df.columns if 'origin' in col.lower() and 'delay' in col.lower()]
                    dest_delay_cols = [col for col in df.columns if 'dest' in col.lower() and 'delay' in col.lower()]
                    
                    if origin_delay_cols and dest_delay_cols:
                        origin_risk = df[origin_delay_cols].mean(axis=1)
                        dest_risk = df[dest_delay_cols].mean(axis=1)
                        
                        # Combined risk (not just additive - use max to capture bottleneck)
                        df_new[new_feature_name] = np.maximum(origin_risk, dest_risk) + 0.5 * np.minimum(origin_risk, dest_risk)
                        
                        self.implemented_features.append(new_feature_name)
                        print(f"Added route feature: {new_feature_name}")
                
                elif 'delay_type_diversity_score' in new_feature_name:
                    # Create delay type diversity score
                    delay_type_cols = [col for col in df.columns if any(dt in col.lower() for dt in ['carrier', 'weather', 'nas', 'security', 'late_aircraft']) and 'rate' in col.lower()]
                    
                    if len(delay_type_cols) >= 3:
                        # Calculate entropy of delay type distribution
                        delay_rates = df[delay_type_cols].values
                        delay_rates = np.maximum(delay_rates, 1e-8)  # Avoid log(0)
                        delay_rates = delay_rates / delay_rates.sum(axis=1, keepdims=True)  # Normalize
                        
                        entropy = -np.sum(delay_rates * np.log(delay_rates), axis=1)
                        df_new[new_feature_name] = entropy
                        
                        self.implemented_features.append(new_feature_name)
                        print(f"Added diversity feature: {new_feature_name}")
                
            except Exception as e:
                print(f"Failed to create {new_feature_name}: {e}")
        
        return df_new
    
    def add_aggregation_features(self, df: pd.DataFrame, priority_filter='medium') -> pd.DataFrame:
        """
        Add aggregation features based on correlated feature groups
        """
        df_new = df.copy()
        
        for suggestion in self.suggestions['aggregation_features']:
            if suggestion['priority'] != priority_filter and priority_filter != 'all':
                continue
            
            features = suggestion['features']
            new_feature_name = suggestion['new_feature_name']
            operation = suggestion['operation']
            
            # Check if features exist
            available_features = [f for f in features if f in df.columns]
            if len(available_features) < 2:
                continue
            
            try:
                if operation == 'mean':
                    df_new[new_feature_name] = df[available_features].mean(axis=1)
                elif operation == 'sum':
                    df_new[new_feature_name] = df[available_features].sum(axis=1)
                elif operation == 'max':
                    df_new[new_feature_name] = df[available_features].max(axis=1)
                elif operation == 'min':
                    df_new[new_feature_name] = df[available_features].min(axis=1)
                
                self.implemented_features.append(new_feature_name)
                print(f"Added aggregation feature: {new_feature_name}")
                
            except Exception as e:
                print(f"Failed to create {new_feature_name}: {e}")
        
        return df_new
    
    def engineer_all_features(self, df: pd.DataFrame, priority='high') -> pd.DataFrame:
        """
        Apply all feature engineering steps
        """
        print("Starting advanced feature engineering...")
        print("=" * 80)
        
        df_engineered = df.copy()
        
        # Add features by category
        df_engineered = self.add_interaction_features(df_engineered, priority_filter=priority)
        df_engineered = self.add_polynomial_features(df_engineered, priority_filter=priority)
        df_engineered = self.add_domain_specific_features(df_engineered)
        df_engineered = self.add_aggregation_features(df_engineered, priority_filter='medium')
        
        print("=" * 80)
        print(f"Feature engineering complete!")
        print(f"   Original features: {df.shape[1]}")
        print(f"   New features added: {len(self.implemented_features)}")
        print(f"   Total features: {df_engineered.shape[1]}")
        
        return df_engineered
    
    def get_feature_list(self) -> List[str]:
        """Return list of newly created features"""
        return self.implemented_features
    
    def get_feature_importance_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank the new features by their potential importance
        Based on variance and correlation with target
        """
        if 'DEP_DELAY' not in df.columns:
            print("Warning: No target variable found for importance ranking")
            return pd.DataFrame()
        
        importance_scores = []
        
        for feature in self.implemented_features:
            if feature in df.columns:
                # Calculate variance (higher = more informative)
                variance = df[feature].var()
                
                # Calculate correlation with target
                correlation = abs(df[feature].corr(df['DEP_DELAY']))
                
                # Combined importance score
                importance_score = variance * correlation if not pd.isna(correlation) else variance
                
                importance_scores.append({
                    'feature': feature,
                    'variance': variance,
                    'correlation_with_target': correlation,
                    'importance_score': importance_score
                })
        
        importance_df = pd.DataFrame(importance_scores)
        importance_df = importance_df.sort_values('importance_score', ascending=False)
        
        return importance_df

# Integration with existing FeatureEngine
def extend_feature_engine(original_features: Dict, advanced_engineer: AdvancedFeatureEngineer) -> Dict:
    """
    Extend the original feature dictionary with advanced features
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame([original_features])
    
    # Apply advanced engineering
    df_extended = advanced_engineer.engineer_all_features(df, priority='high')
    
    # Convert back to dictionary
    extended_features = df_extended.iloc[0].to_dict()
    
    return extended_features

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    
    # This would be called during model training
    print("Advanced Feature Engineering Module")
    print("This module extends the base FeatureEngine with PCA-derived features")
    
    # Test with sample data
    sample_data = {
        'dep_hour': 14,
        'dow': 1,
        'month': 7,
        'is_weekend': 0,
        'DISTANCE': 500.0,
        'origin_avg_delay': 15.0,
        'airline_avg_delay': 12.0,
        'total_delay_rate_origin': 0.2,
        'carrier_delay_rate_origin': 0.1,
        'weather_delay_rate_origin': 0.05,
        'nas_delay_rate_origin': 0.08,
        'security_delay_rate_origin': 0.02,
        'late_aircraft_delay_rate_origin': 0.15,
        'avg_delay_minutes_origin': 15.0,
        'avg_carrier_delay_origin': 10.0,
        'avg_weather_delay_origin': 8.0,
        'avg_nas_delay_origin': 12.0,
        'avg_security_delay_origin': 5.0,
        'avg_late_aircraft_delay_origin': 18.0,
        'total_delay_rate_dest': 0.18,
        'carrier_delay_rate_dest': 0.09,
        'weather_delay_rate_dest': 0.04,
        'nas_delay_rate_dest': 0.07,
        'security_delay_rate_dest': 0.01,
        'late_aircraft_delay_rate_dest': 0.13,
        'avg_delay_minutes_dest': 14.0,
        'avg_carrier_delay_dest': 9.0,
        'avg_weather_delay_dest': 7.0,
        'avg_nas_delay_dest': 11.0,
        'avg_security_delay_dest': 4.0,
        'avg_late_aircraft_delay_dest': 16.0,
        'DEP_DELAY': 20.0
    }
    
    # Test the advanced feature engineer
    engineer = AdvancedFeatureEngineer()
    df = pd.DataFrame([sample_data])
    
    print("\nTesting with sample data...")
    df_engineered = engineer.engineer_all_features(df, priority='high')
    
    print(f"\nNew features created: {len(engineer.get_feature_list())}")
    print("Feature list:", engineer.get_feature_list())
