# Process BTS Flight Data for the App
Write-Host "Processing BTS Flight Data..." -ForegroundColor Green
Write-Host ""

# Create OUTPUTS directory if it doesn't exist
if (!(Test-Path "OUTPUTS")) {
    New-Item -ItemType Directory -Path "OUTPUTS"
}

# Process BTS data
Write-Host "Step 1: Processing BTS delay cause data..." -ForegroundColor Yellow
& "C:\Users\Govind Molakalapalli\Documents\GitHub\.conda\Scripts\python.exe" SCRIPTS\04_bts_processor.py --input "DATA\ot_delaycause1_DL\Airline_Delay_Cause.csv" --output_dir OUTPUTS --filter_hubs

Write-Host ""
Write-Host "Step 2: Creating lookup files for the app..." -ForegroundColor Yellow
& "C:\Users\Govind Molakalapalli\Documents\GitHub\.conda\python.exe" SCRIPTS\create_bts_lookups.py

Write-Host ""
Write-Host "Step 3: Training model from BTS data..." -ForegroundColor Yellow
& "C:\Users\Govind Molakalapalli\Documents\GitHub\.conda\python.exe" SCRIPTS\train_model_from_bts.py

Write-Host ""
Write-Host "Data processing complete!" -ForegroundColor Green
Write-Host "The app now has access to real flight delay data and a trained model." -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"
