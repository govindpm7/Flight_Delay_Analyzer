@echo off
echo Processing BTS Flight Data...
echo.

REM Create OUTPUTS directory if it doesn't exist
if not exist "OUTPUTS" mkdir OUTPUTS

REM Process BTS data
echo Step 1: Processing BTS delay cause data...
python SCRIPTS\04_bts_processor.py --input "DATA\ot_delaycause1_DL\Airline_Delay_Cause.csv" --output_dir OUTPUTS --filter_hubs

echo.
echo Step 2: Creating lookup files for the app...
python SCRIPTS\create_bts_lookups.py

echo.
echo Step 3: Training model from BTS data...
python SCRIPTS\train_model_from_bts.py

echo.
echo Data processing complete!
echo The app now has access to real flight delay data and a trained model.
echo.
pause
