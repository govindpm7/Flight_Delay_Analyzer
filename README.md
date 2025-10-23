Flight Delay Prediction Engine
==============================

Overview
--------
This is a comprehensive flight delay prediction system that predicts US domestic flight departure delays using historical data. It provides a Streamlit frontend for entering a flight number and date, then returns a delay estimate with a confidence band and contributing factors.

Key Features
------------
- **Advanced Machine Learning Models**: Random Forest, XGBoost, and Linear Regression with automatic model selection
- **PCA Analysis & Feature Engineering**: Comprehensive Principal Component Analysis for feature discovery and advanced feature engineering based on PCA insights
- **Route Search Integration**: SERP API integration for automatic flight route discovery with fallback to manual input
- **Synthetic Data Training**: BTS-based synthetic data generation for model training when real data is limited
- **BTS Data Integration**: Bureau of Transportation Statistics delay cause analysis and airport/carrier performance metrics
- **Real-time Predictions**: Streamlit web interface with confidence intervals and contributing factor analysis

Project structure
-----------------
- `DATA/` — place raw CSV(s) here (BTS or Kaggle). The training script will read from this folder.
- `OUTPUTS/` — trained model artifacts, PCA analysis results, BTS lookup tables, and synthetic data.
- `SCRIPTS/` — comprehensive data processing, training, and analysis code:
  - `modeling/` — advanced feature engineering, model training, and prediction modules
  - `analysis/` — PCA analysis, feature relationship analysis, and visualization tools
  - `acquisition/` — SERP API integration for route search and flight data acquisition
  - `training/` — model retraining with PCA-derived features
- `SCRIPTS/03_app.py` — Streamlit web interface with route search and prediction capabilities.

Advanced Features
------------------
- **PCA Analysis**: Comprehensive Principal Component Analysis with feature importance analysis, biplots, and variance explanation
- **Feature Engineering**: Advanced feature creation based on PCA insights, including temporal patterns, route characteristics, and delay cause analysis
- **Route Search**: SERP API integration for automatic flight route discovery with Google Search and Google Flights data
- **Synthetic Data Generation**: BTS-based synthetic flight data generation for training when real data is limited
- **Multiple Model Support**: Random Forest, XGBoost, and Linear Regression with automatic best model selection
- **BTS Integration**: Bureau of Transportation Statistics delay cause analysis (carrier, weather, NAS, security, late aircraft delays)

Setup
-----
1) Create and activate a Python 3.10+ environment.

2) Install dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```

3) Acquire data (historical, zero-cost):
   - Download BTS On-Time Performance (recommended) or Kaggle “Flight Delays and Cancellations”.
   - Ensure your CSV(s) include (or can be derived to) at least: `OP_CARRIER`, `FL_NUM`, `ORIGIN`, `DEST`, `CRS_DEP_TIME`, `DEP_DELAY`, `FL_DATE`, `DISTANCE`.
   - Place the raw files inside `DATA/`.

4) Optional quick clean/merge step:
   
   ```bash
   python SCRIPTS/data_acquisition.py --input_dir DATA --output DATA/processed.csv
   ```

5) Train the model:
   
   ```bash
   # Basic training
   python SCRIPTS/02_train_model.py --input DATA/processed.csv --out_dir OUTPUTS
   
   # Advanced training with XGBoost and improved features
   python SCRIPTS/02_train_model_improved.py --input DATA/processed.csv --out_dir OUTPUTS/improved_model
   
   # Run PCA analysis
   python SCRIPTS/analysis/prepare_pca_data.py
   python SCRIPTS/analysis/run_pca_analysis.py
   
   # Generate synthetic data (optional)
   python SCRIPTS/generate_synthetic_data.py
   ```

6) Run the app:
   
   ```bash
   streamlit run SCRIPTS/03_app.py
   ```

Usage notes
-----------
- Enter a flight number like `AA1234` and a date. The app will try to resolve route/time using SERP API search. If not found, manually enter the origin and destination airports.
- Confidence interval uses validation MAE as ± bound for a simple, interpretable estimate.
- The system uses BTS historical data for all predictions, ensuring consistent accuracy regardless of route discovery method.

System Capabilities
-------------------
- **Route Discovery**: Automatic flight route lookup via SERP API with manual fallback
- **Delay Prediction**: Multi-model approach with Random Forest, XGBoost, and Linear Regression
- **Feature Analysis**: PCA-driven feature engineering with comprehensive analysis reports
- **Data Sources**: BTS On-Time Performance data with delay cause breakdown
- **Synthetic Training**: BTS-based synthetic data generation for enhanced model training
- **Real-time Interface**: Streamlit web app with confidence intervals and factor analysis

Deployment
----------
- Local: `streamlit run SCRIPTS/03_app.py`
- Streamlit Community Cloud: push this repo public to GitHub and configure the app to run `SCRIPTS/03_app.py`.

Troubleshooting
---------------
- If the app says model artifacts are missing, run the training step.
- If columns don’t match, ensure the processed file contains required fields (see `SCRIPTS/data_acquisition.py`).

