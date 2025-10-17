# Flight Delay Analyzer App Launcher
Write-Host "Starting Flight Delay Analyzer App..." -ForegroundColor Green
Write-Host ""

# Change to the project directory
Set-Location "$PSScriptRoot"

# Check if OUTPUTS directory exists and has recent data
if (!(Test-Path "OUTPUTS")) {
    Write-Host "Processing flight data first..." -ForegroundColor Yellow
    $processData = $true
} else {
    Write-Host "Checking if data needs updating..." -ForegroundColor Yellow
    
    # Check if BTS lookup files exist and are recent (less than 7 days old)
    $btsFiles = Get-ChildItem -Path "OUTPUTS" -Filter "bts_lookup_*.csv" -ErrorAction SilentlyContinue
    $needsUpdate = $false
    
    if ($btsFiles.Count -eq 0) {
        $needsUpdate = $true
    } else {
        $oldestFile = $btsFiles | Sort-Object LastWriteTime | Select-Object -First 1
        $daysOld = (Get-Date) - $oldestFile.LastWriteTime
        if ($daysOld.Days -gt 7) {
            $needsUpdate = $true
        }
    }
    
    if ($needsUpdate) {
        Write-Host "Data files missing or outdated, reprocessing..." -ForegroundColor Yellow
        $processData = $true
    } else {
        Write-Host "Data files are up to date." -ForegroundColor Green
        $processData = $false
    }
}

if ($processData) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "PROCESSING FLIGHT DATA" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    Write-Host "Step 1: Processing BTS delay cause data..." -ForegroundColor Yellow
    & "C:\Users\Govind Molakalapalli\Documents\GitHub\.conda\Scripts\python.exe" SCRIPTS\04_bts_processor.py --input "DATA\ot_delaycause1_DL\Airline_Delay_Cause.csv" --output_dir OUTPUTS --filter_hubs
    
    Write-Host ""
    Write-Host "Step 2: Creating lookup files for the app..." -ForegroundColor Yellow
    & "C:\Users\Govind Molakalapalli\Documents\GitHub\.conda\Scripts\python.exe" SCRIPTS\create_bts_lookups.py
    
    Write-Host ""
    Write-Host "Step 3: Training model from BTS data..." -ForegroundColor Yellow
    & "C:\Users\Govind Molakalapalli\Documents\GitHub\.conda\Scripts\python.exe" SCRIPTS\train_model_from_bts.py
    
    Write-Host ""
    Write-Host "Data processing complete!" -ForegroundColor Green
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "STARTING FLIGHT DELAY ANALYZER APP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Set-Location "$PSScriptRoot\SCRIPTS"

# Run the Streamlit app using the conda environment
& "C:\Users\Govind Molakalapalli\Documents\GitHub\.conda\python.exe" -m streamlit run 03_app.py

Write-Host ""
Write-Host "App has been stopped." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
