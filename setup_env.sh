#!/bin/bash
set -e

ENV_NAME="deepisles"
# More portable script directory detection (works with both bash and zsh)
if [ -n "$BASH_SOURCE" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

# Parse command-line arguments
CUDA_VERSION="${1:-11.8}"  # Default to 11.8 if not provided

# Validate CUDA version
if [[ ! "$CUDA_VERSION" =~ ^(11\.8|12|13)$ ]]; then
    echo "Error: CUDA version must be 11.8, 12, or 13"
    echo "Usage: $0 [CUDA_VERSION]"
    echo "Example: $0 12"
    exit 1
fi

# Map CUDA version to PyTorch index
case "$CUDA_VERSION" in
    11.8)
        PYTORCH_CUDA="cu118"
        ;;
    12)
        PYTORCH_CUDA="cu121"  # PyTorch uses cu121 for CUDA 12.x
        ;;
    13)
        PYTORCH_CUDA="cu131"  # PyTorch uses cu131 for CUDA 13.x (or latest)
        ;;
esac

echo "=== CUDA Version: $CUDA_VERSION (PyTorch index: $PYTORCH_CUDA) ==="

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n $ENV_NAME python=3.12 pip -y

echo "=== Activating environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "=== Installing uv ==="
pip install uv

echo "=== Installing PyTorch with CUDA $CUDA_VERSION via uv (PyTorch index: $PYTORCH_CUDA) ==="
cd "$SCRIPT_DIR"
uv pip install --extra-index-url https://download.pytorch.org/whl/$PYTORCH_CUDA \
    "torch>=2.1.2" torchvision torchaudio

echo "=== Installing all other dependencies via uv ==="
uv pip install -e .

echo "=== Installing test dependencies (pytest, pytest-cov, pytest-mock) ==="
uv pip install -e ".[dev]"

echo "=== Resolving SimpleITK package conflict ==="
# Check if both SimpleITK and SimpleITK-SimpleElastix are installed
if uv pip list | grep -q "^simpleitk " && uv pip list | grep -q "^SimpleITK-SimpleElastix "; then
    echo "WARNING: Both SimpleITK and SimpleITK-SimpleElastix are installed. Removing conflicting SimpleITK package..."
    uv pip uninstall simpleitk -y
    echo "Reinstalling SimpleITK-SimpleElastix to ensure it's properly installed..."
    uv pip install SimpleITK-SimpleElastix --force-reinstall
    echo "SimpleITK conflict resolved."
else
    echo "No SimpleITK conflict detected."
fi

echo "=== Setting up PYTHONPATH ==="
# Add to conda activate script for persistence
ACTIVATE_SCRIPT="$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
mkdir -p "$(dirname "$ACTIVATE_SCRIPT")"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
export PYTHONPATH="\${PYTHONPATH}:${SCRIPT_DIR}/src/SEALS:${SCRIPT_DIR}/src/FACTORIZER/model/factorizer:${SCRIPT_DIR}/src/HD-BET"
EOF
chmod +x "$ACTIVATE_SCRIPT"

echo ""
echo "=== Done! ==="
echo "Activate the environment with: conda activate $ENV_NAME"
echo "The PYTHONPATH is automatically set when you activate the environment."

