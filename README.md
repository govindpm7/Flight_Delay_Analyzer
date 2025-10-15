Flight Delay Prediction Engine (MVP)
===================================

Overview
--------
This is a zero-cost MVP that predicts US domestic flight departure delays using historical data. It provides a Streamlit frontend for entering a flight number and date, then returns a delay estimate with a confidence band and contributing factors.

Project structure
-----------------
- `DATA/` — place raw CSV(s) here (BTS or Kaggle). The training script will read from this folder.
- `OUTPUTS/` — trained model artifacts (`model.joblib`, `preprocessor.joblib`), metadata, and lookup tables.
- `SCRIPTS/` — data acquisition/processing and training code.
- `app.py` — Streamlit UI.

Zero-cost, historical-only baseline
-----------------------------------
The MVP uses historical patterns only (no paid APIs). You can optionally extend with free weather APIs later.

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
   python SCRIPTS/train_model.py --input DATA/processed.csv --out_dir OUTPUTS
   ```

6) Run the app:
   
   ```bash
   streamlit run app.py
   ```

Usage notes
-----------
- Enter a flight number like `AA1234` and a date. The app will try to resolve route/time from a lookup built during training. If not found, it will fall back to route/airline averages and ask for minimal clarification in the UI.
- Confidence interval uses validation MAE as ± bound for a simple, interpretable estimate.

MVP scope alignment
-------------------
- Domestic-only focus with major hubs: DEN, IAD, ATL, LAX (plus connected routes).
- Feature engineering includes temporal, route/airline, airport averages; PCA report is computed for analysis, while the production model uses tree-based features for interpretability.
- Trained model artifacts and PCA summary are saved to `OUTPUTS/`.

Deployment
----------
- Local: `streamlit run app.py`
- Streamlit Community Cloud: push this repo public to GitHub and configure the app to run `app.py`.

Troubleshooting
---------------
- If the app says model artifacts are missing, run the training step.
- If columns don’t match, ensure the processed file contains required fields (see `SCRIPTS/data_acquisition.py`).

