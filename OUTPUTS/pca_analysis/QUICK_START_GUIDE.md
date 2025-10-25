# PCA Feature Engineering - Quick Start Guide

## 🚀 Quick Start

### 1. Run Complete PCA Workflow
```bash
# Navigate to analysis directory
cd SCRIPTS/analysis

# Step 1: Prepare data for PCA
python prepare_pca_data.py

# Step 2: Run PCA analysis
python run_pca_analysis.py

# Step 3: Generate visualizations
python visualize_pca.py

# Step 4: Generate feature suggestions
python suggest_features.py
```

### 2. Retrain Model with New Features
```bash
# Navigate to training directory
cd ../training

# Retrain model with PCA-derived features
python retrain_with_pca_features.py
```

### 3. Test Integration
```bash
# Test the enhanced FeatureEngine
cd ..
python test_advanced_features.py
```

## 📊 Key Results Summary

- **Performance**: 35.2% improvement in MAE (12.42 → 8.05 minutes)
- **Features**: 25 new PCA-derived features added
- **R² Score**: 0.5687
- **Integration**: Seamless with existing codebase

## 🔧 Using Enhanced FeatureEngine

### Basic Usage (Advanced Features Enabled by Default)
```python
from SCRIPTS.modeling.feature_engine import FeatureEngine
import pandas as pd
from datetime import date

# Load lookup tables
airport_lookup = pd.read_csv("OUTPUTS/bts_lookup_airport.csv")
carrier_lookup = pd.read_csv("OUTPUTS/bts_lookup_carrier.csv")
flight_lookup = pd.read_csv("OUTPUTS/flight_lookup.csv")

# Initialize with advanced features (default)
feature_engine = FeatureEngine(
    bts_airport_df=airport_lookup,
    bts_carrier_df=carrier_lookup,
    lookup_df=flight_lookup,
    use_advanced_features=True  # This is the default
)

# Build features - advanced features are automatically applied
features = feature_engine.build_features(
    carrier="AA",
    origin="LAX",
    dest="JFK",
    flight_date=date(2025, 7, 15),
    dep_hour=14
)

print(f"Total features: {len(features)}")  # Should be ~61 features
```

### Disable Advanced Features (Fallback to Original)
```python
# Initialize without advanced features
feature_engine = FeatureEngine(
    bts_airport_df=airport_lookup,
    bts_carrier_df=carrier_lookup,
    lookup_df=flight_lookup,
    use_advanced_features=False  # Disable advanced features
)

# Build features - only original features
features = feature_engine.build_features(
    carrier="AA",
    origin="LAX", 
    dest="JFK",
    flight_date=date(2025, 7, 15),
    dep_hour=14
)

print(f"Total features: {len(features)}")  # Should be ~36 features
```

## 📈 Top New Features Created

### 1. Composite Features
- **composite_delay_risk_score**: Weighted composite of delay features
- **route_congestion_score**: Combined origin/destination risk
- **delay_type_diversity_score**: Entropy-based delay type diversity
- **temporal_risk_category**: Time-based risk classification

### 2. Interaction Features (Top 5)
- `origin_avg_delay_X_late_aircraft_delay_rate_origin`
- `avg_carrier_delay_origin_X_avg_weather_delay_origin`
- `avg_delay_minutes_origin_X_avg_nas_delay_origin`
- `late_aircraft_delay_rate_origin_X_avg_carrier_delay_origin`
- `avg_weather_delay_origin_X_avg_nas_delay_origin`

### 3. Polynomial Features
- `nas_delay_rate_dest_squared`
- `carrier_delay_rate_dest_squared`
- `avg_late_aircraft_delay_dest_squared`
- `total_delay_rate_dest_squared`

## 📁 Output Files

### Analysis Results
- `OUTPUTS/pca_analysis/training_data.csv` - Generated training dataset
- `OUTPUTS/pca_analysis/feature_engineering_report.txt` - Human-readable feature suggestions
- `OUTPUTS/pca_analysis/feature_suggestions.pkl` - Programmatic feature suggestions

### Visualizations
- `OUTPUTS/pca_analysis/plots/scree_plot.png` - Variance explained by components
- `OUTPUTS/pca_analysis/plots/component_loadings.png` - Feature loadings
- `OUTPUTS/pca_analysis/plots/biplot_pc1_pc2.png` - Biplot visualization
- `OUTPUTS/pca_analysis/plots/feature_importance_heatmap.png` - Feature importance heatmap

### Model Results
- `OUTPUTS/pca_retraining_*/model_pca_features.pkl` - Retrained model
- `OUTPUTS/pca_retraining_*/feature_importance.csv` - Feature importance rankings
- `OUTPUTS/pca_retraining_*/metadata.json` - Performance metrics

## 🔍 Understanding the Results

### PCA Analysis
- **PC1**: General delay tendency (29.96% variance)
- **PC2**: Destination-specific patterns (22.75% variance)
- **PC3**: Origin delay patterns (12.72% variance)
- **PC4**: Weather/late aircraft interactions (9.65% variance)
- **PC5**: Temporal patterns (6.51% variance)

### Feature Importance
1. **airline_avg_delay** (0.4365) - Most important original feature
2. **composite_delay_risk_score** (0.3743) - Most important new feature
3. **dow** (0.0595) - Day of week
4. **delay_type_diversity_score** (0.0580) - Delay diversity measure

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Make sure you're in the project root directory
   import sys
   sys.path.append('SCRIPTS/modeling')
   ```

2. **Advanced Features Not Loading**
   ```python
   # Check if feature suggestions file exists
   import os
   if not os.path.exists("OUTPUTS/pca_analysis/feature_suggestions.pkl"):
       print("Run the PCA analysis first!")
   ```

3. **Memory Issues**
   ```python
   # For large datasets, reduce the number of features
   engineer = AdvancedFeatureEngineer()
   df_engineered = engineer.engineer_all_features(df, priority='high')  # Only high priority
   ```

### Performance Tips

1. **Feature Selection**: Use only high-priority features for faster processing
2. **Caching**: The FeatureEngine caches lookups for better performance
3. **Batch Processing**: Process multiple flights together for efficiency

## 📞 Support

For questions or issues:
1. Check the full documentation in `PCA_FEATURE_ENGINEERING_SUMMARY.md`
2. Review the generated reports in `OUTPUTS/pca_analysis/`
3. Test with the provided test scripts
4. Ensure all dependencies are installed

## 🎯 Next Steps

1. **Monitor Performance**: Track model performance with new features
2. **Feature Selection**: Experiment with different feature combinations
3. **Continuous Improvement**: Re-run PCA analysis with new data
4. **Extend Features**: Add more domain-specific features based on insights

