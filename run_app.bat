@echo off
echo Starting Flight Delay Analyzer App...
echo.

REM Change to the project directory
cd /d "%~dp0"

REM Check if OUTPUTS directory exists and has recent data
if not exist "OUTPUTS" (
    echo Processing flight data first...
    goto :process_data
) else (
    echo Checking if data needs updating...
    REM Check if BTS lookup files exist and are recent (less than 7 days old)
    forfiles /p OUTPUTS /m bts_lookup_*.csv /c "cmd /c echo @path @fdate" 2>nul | findstr /i "bts_lookup" >nul
    if errorlevel 1 (
        echo Data files missing or outdated, reprocessing...
        goto :process_data
    ) else (
        echo Data files are up to date.
        goto :start_app
    )
)

:process_data
echo.
echo ========================================
echo PROCESSING FLIGHT DATA
echo ========================================

echo Step 1: Processing BTS delay cause data...
"C:\Users\Govind Molakalapalli\Documents\GitHub\.conda\Scripts\python.exe" SCRIPTS\04_bts_processor.py --input "DATA\ot_delaycause1_DL\Airline_Delay_Cause.csv" --output_dir OUTPUTS --filter_hubs

echo.
echo Step 2: Creating lookup files for the app...
"C:\Users\Govind Molakalapalli\Documents\GitHub\.conda\Scripts\python.exe" SCRIPTS\05_create_bts_lookups.py

echo.
echo Step 3: Training model from BTS data...
"C:\Users\Govind Molakalapalli\Documents\GitHub\.conda\Scripts\python.exe" SCRIPTS\create_bts_model.py

echo.
echo Data processing complete!
echo.

:start_app
echo ========================================
echo STARTING FLIGHT DELAY ANALYZER APP
echo ========================================
cd SCRIPTS

REM Run the Streamlit app using the conda environment
"C:\Users\Govind Molakalapalli\Documents\GitHub\.conda\Scripts\streamlit.exe" run 03_app.py

echo.
echo App has been stopped.
pause
