Flight Delay Prediction Engine
==============================

Overview
--------
This is a comprehensive flight delay prediction system that predicts US domestic flight departure delays using historical BTS (Bureau of Transportation Statistics) data. It features a **dynamic prediction system** that only uses carriers and routes with available historical data, ensuring accurate predictions based on real performance metrics.

Key Features
------------
- **Dynamic Prediction System**: Only predicts for carriers and routes with BTS historical data - no fallback predictions
- **BTS Data Integration**: Uses actual Bureau of Transportation Statistics data for 20 carriers and 4 major hub airports
- **SERP API Integration**: Real-time flight search with automatic filtering to match BTS data availability
- **Advanced Machine Learning**: Random Forest, XGBoost, and Gradient Boosting models with automatic selection
- **PCA Analysis & Feature Engineering**: Comprehensive Principal Component Analysis for feature discovery
- **Transparent Predictions**: Shows exactly which data components (carrier, origin, destination) were available for the prediction
- **Real-time Interface**: Streamlit web app with confidence intervals and data availability indicators

Project Structure
-----------------
- `DATA/` — Raw CSV files (BTS or Kaggle datasets)
- `OUTPUTS/` — Model artifacts, PCA analysis, BTS lookup tables:
  - `airport_model/` — Main model with BTS data integration
  - `improved_model/` — Enhanced model with XGBoost and advanced features
  - `pca_analysis/` — PCA analysis results and feature engineering reports
  - `processing/` — Processed BTS data and lookup tables
- `SCRIPTS/` — Core application code:
  - `03_app.py` — Main Streamlit application with dynamic prediction system
  - `airport_model_adapter.py` — Dynamic prediction adapter for BTS data filtering
  - `train_bts_airport_model.py` — BTS-based model training
  - `modeling/` — Feature engineering and prediction modules
  - `analysis/` — PCA analysis and visualization tools
  - `acquisition/` — SERP API integration for flight search

Advanced Features
------------------
- **Dynamic Prediction System**: Only uses carriers and routes with BTS historical data
- **BTS Data Filtering**: 20 carriers (AA, DL, UA, WN, etc.) and 4 hub airports (ATL, DEN, IAD, LAX)
- **SERP API Integration**: Real-time flight search with automatic filtering to BTS carriers only
- **Transparent Data Usage**: Shows which data components (carrier, origin, destination) were available
- **PCA Analysis**: Comprehensive Principal Component Analysis with feature importance analysis
- **Feature Engineering**: Advanced feature creation based on PCA insights and BTS delay causes
- **Multiple Model Support**: Random Forest, XGBoost, and Gradient Boosting with automatic selection
- **Confidence Intervals**: MAE-based confidence bands for prediction uncertainty

Setup
-----
1) **Create and activate a Python 3.10+ environment:**
   ```bash
   conda create -n flight_delay python=3.10
   conda activate flight_delay
   ```

2) **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3) **Quick Start (Recommended):**
   ```bash
   # Run the complete pipeline
   .\run_pipeline.bat
   
   # Or use PowerShell
   .\run_pipeline.ps1
   ```

4) **Manual Setup (if needed):**
   ```bash
   # Process BTS data
   python SCRIPTS/01_data_acquisition.py --input_dir DATA --output DATA/processed.csv
   
   # Train the model
   python SCRIPTS/train_bts_airport_model.py
   
   # Run PCA analysis (optional)
   python SCRIPTS/analysis/run_pca_analysis.py
   ```

5) **Launch the Application:**
   ```bash
   # Simple launcher
   .\simple_start.bat
   
   # Or manually
   cd SCRIPTS
   python -m streamlit run 03_app.py --server.port 8501
   ```

Usage
-----
### **How the Dynamic Prediction System Works:**

1. **Route Selection**: Choose from 4 major hub airports (ATL, DEN, IAD, LAX)
2. **Carrier Selection**: 
   - **Primary**: "📊 Show Airlines with BTS Data" - Shows 20 carriers with historical data
   - **Secondary**: "🔍 Search Additional Flight Info (SERP API)" - Searches real-time flights but filters to BTS carriers only
3. **Prediction**: Only generates predictions for carriers/routes with BTS historical data
4. **Transparency**: Shows exactly which data components were available (carrier ✓, origin ✓, destination ✓)

### **Available Carriers (BTS Data):**
- **Major Airlines**: AA (American), DL (Delta), UA (United), WN (Southwest)
- **Regional Carriers**: 9E (Endeavor), AS (Alaska), B6 (JetBlue), F9 (Frontier), NK (Spirit)
- **And 11 more carriers** with complete BTS historical data

### **Prediction Features:**
- **Dynamic Data Usage**: Only uses available BTS data components
- **Confidence Intervals**: MAE-based uncertainty bounds
- **Data Availability Display**: Shows which components contributed to the prediction
- **No Fallback Predictions**: If no BTS data available, shows "Cannot generate prediction"

System Capabilities
-------------------
- **Dynamic Prediction**: Only predicts for carriers/routes with BTS historical data
- **BTS Data Integration**: 20 carriers × 4 hub airports with complete historical performance
- **SERP API Integration**: Real-time flight search with automatic BTS carrier filtering
- **Multi-Model Architecture**: Random Forest, XGBoost, and Gradient Boosting with automatic selection
- **PCA Analysis**: Comprehensive feature engineering and analysis reports
- **Transparent Predictions**: Shows data availability and contributing factors
- **Real-time Interface**: Streamlit web app with confidence intervals and data indicators

Deployment
----------
### **Local Deployment:**
```bash
# Quick start
.\simple_start.bat

# Or manual
cd SCRIPTS
python -m streamlit run 03_app.py --server.port 8501
```

### **Cloud Deployment:**
- **Streamlit Community Cloud**: Push to GitHub and configure to run `SCRIPTS/03_app.py`
- **Docker**: Use the provided Dockerfile for containerized deployment

Troubleshooting
---------------
### **Common Issues:**

1. **"Model artifacts are missing"**
   - **Solution**: Run `.\run_pipeline.bat` to train the model

2. **"UnboundLocalError: available_airports"**
   - **Solution**: This has been fixed in the latest version

3. **"File does not exist: 03_app.py"**
   - **Solution**: Make sure you're in the `SCRIPTS` directory or use `.\simple_start.bat`

4. **"Python is not recognized"**
   - **Solution**: Install Python 3.10+ and add to PATH, or use conda environment

5. **"Streamlit is not installed"**
   - **Solution**: Run `pip install streamlit`

### **Launcher Scripts:**
- `simple_start.bat` - Simple launcher with error checking
- `run_pipeline.bat` - Complete pipeline (data processing + training + app)
- `start_app_powershell.ps1` - PowerShell launcher with detailed error messages
- `troubleshoot_start.bat` - Comprehensive troubleshooting launcher

### **VS Code Integration:**
- Use the "Flight Delay Analyzer - Streamlit App" debug configuration
- Set Python interpreter to your conda environment

