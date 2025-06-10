# Smart Shopping VLM Application Startup Script
Write-Host "Starting Smart Shopping VLM Application..." -ForegroundColor Green

# Set environment variables to handle OpenMP conflicts
Write-Host "Setting environment variables..." -ForegroundColor Yellow
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"

# Check if conda environment is activated
Write-Host "Checking conda environment..." -ForegroundColor Yellow
$condaEnv = $env:CONDA_DEFAULT_ENV
if ($condaEnv) {
    Write-Host "Conda environment active: $condaEnv" -ForegroundColor Green
} else {
    Write-Host "Warning: No conda environment detected. Consider activating 'smart_shopping_vlm'" -ForegroundColor Red
}

# Check CUDA availability
Write-Host "Checking CUDA availability..." -ForegroundColor Yellow
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU\"}') if torch.cuda.is_available() else print('Using CPU')"

Write-Host "Starting application..." -ForegroundColor Green
python app.py
