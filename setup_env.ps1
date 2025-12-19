# PowerShell script for setting up DeepIsles environment on Windows
# Usage: .\setup_env.ps1 [CUDA_VERSION]
# Example: .\setup_env.ps1 12

$ErrorActionPreference = "Stop"

$ENV_NAME = "deepisles"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# Parse command-line arguments
$CUDA_VERSION = if ($args.Count -gt 0) { $args[0] } else { "11.8" }

# Validate CUDA version
if ($CUDA_VERSION -notmatch "^(11\.8|12|13)$") {
    Write-Host "Error: CUDA version must be 11.8, 12, or 13" -ForegroundColor Red
    Write-Host "Usage: .\setup_env.ps1 [CUDA_VERSION]"
    Write-Host "Example: .\setup_env.ps1 12"
    exit 1
}

# Map CUDA version to PyTorch index
switch ($CUDA_VERSION) {
    "11.8" { $PYTORCH_CUDA = "cu118" }
    "12"   { $PYTORCH_CUDA = "cu121" }  # PyTorch uses cu121 for CUDA 12.x
    "13"   { $PYTORCH_CUDA = "cu131" }  # PyTorch uses cu131 for CUDA 13.x (or latest)
}

Write-Host "=== CUDA Version: $CUDA_VERSION (PyTorch index: $PYTORCH_CUDA) ===" -ForegroundColor Cyan

Write-Host "=== Creating conda environment: $ENV_NAME ===" -ForegroundColor Cyan
conda create -n $ENV_NAME python=3.12 pip -y

Write-Host "=== Activating environment ===" -ForegroundColor Cyan
# Note: If conda activate doesn't work, you may need to run 'conda init powershell' 
# and restart your PowerShell session first
conda activate $ENV_NAME

Write-Host "=== Installing uv ===" -ForegroundColor Cyan
pip install uv

Write-Host "=== Installing PyTorch with CUDA $CUDA_VERSION via uv (PyTorch index: $PYTORCH_CUDA) ===" -ForegroundColor Cyan
Set-Location $SCRIPT_DIR
uv pip install --extra-index-url "https://download.pytorch.org/whl/$PYTORCH_CUDA" `
    torch>=2.1.2 torchvision torchaudio

Write-Host "=== Installing all other dependencies via uv ===" -ForegroundColor Cyan
uv pip install -e .

Write-Host "=== Installing test dependencies (pytest, pytest-cov, pytest-mock) ===" -ForegroundColor Cyan
uv pip install -e ".[dev]"

Write-Host "=== Resolving SimpleITK package conflict ===" -ForegroundColor Cyan
# Check if both SimpleITK and SimpleITK-SimpleElastix are installed
$simpleitkInstalled = uv pip list | Select-String "^simpleitk "
$simpleElastixInstalled = uv pip list | Select-String "^SimpleITK-SimpleElastix "
if ($simpleitkInstalled -and $simpleElastixInstalled) {
    Write-Host "WARNING: Both SimpleITK and SimpleITK-SimpleElastix are installed. Removing conflicting SimpleITK package..." -ForegroundColor Yellow
    uv pip uninstall simpleitk -y
    Write-Host "Reinstalling SimpleITK-SimpleElastix to ensure it's properly installed..." -ForegroundColor Cyan
    uv pip install SimpleITK-SimpleElastix --force-reinstall
    Write-Host "SimpleITK conflict resolved." -ForegroundColor Green
} else {
    Write-Host "No SimpleITK conflict detected." -ForegroundColor Green
}

Write-Host "=== Setting up PYTHONPATH ===" -ForegroundColor Cyan
# Add to conda activate script for persistence
$ACTIVATE_SCRIPT = "$env:CONDA_PREFIX\etc\conda\activate.d\env_vars.ps1"
$ACTIVATE_DIR = Split-Path -Parent $ACTIVATE_SCRIPT
if (-not (Test-Path $ACTIVATE_DIR)) {
    New-Item -ItemType Directory -Path $ACTIVATE_DIR -Force | Out-Null
}

# Convert paths to Windows format and use semicolon separator
$SEALS_PATH = "$SCRIPT_DIR\src\SEALS"
$FACTORIZER_PATH = "$SCRIPT_DIR\src\FACTORIZER\model\factorizer"
$HD_BET_PATH = "$SCRIPT_DIR\src\HD-BET"

# Build PYTHONPATH: append new paths to existing PYTHONPATH if it exists
$PYTHONPATH_CONTENT = @"
# PowerShell activation script for PYTHONPATH
if (`$env:PYTHONPATH) {
    `$env:PYTHONPATH = "`$env:PYTHONPATH;$SEALS_PATH;$FACTORIZER_PATH;$HD_BET_PATH"
} else {
    `$env:PYTHONPATH = "$SEALS_PATH;$FACTORIZER_PATH;$HD_BET_PATH"
}
"@

$PYTHONPATH_CONTENT | Out-File -FilePath $ACTIVATE_SCRIPT -Encoding UTF8

Write-Host ""
Write-Host "=== Done! ===" -ForegroundColor Green
Write-Host "Activate the environment with: conda activate $ENV_NAME"
Write-Host "The PYTHONPATH is automatically set when you activate the environment."

