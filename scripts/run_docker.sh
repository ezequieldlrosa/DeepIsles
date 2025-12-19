#!/bin/bash
# Script to run isleschallenge/deepisles:latest Docker container on sample data
# and export the output to example_test/ with a clear naming convention

set -e  # Exit on error

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Input data filenames
DWI_FILE="sub-strokecase0001_ses-0001_dwi.nii.gz"
ADC_FILE="sub-strokecase0001_ses-0001_adc.nii.gz"
FLAIR_FILE="sub-strokecase0001_ses-0001_flair.nii.gz"

# Paths
DATA_DIR="$PROJECT_ROOT/data"
OUTPUT_DIR="$PROJECT_ROOT/example_test"
DOCKER_OUTPUT="$OUTPUT_DIR/lesion_msk_docker.nii.gz"

# Check if data files exist
if [ ! -f "$DATA_DIR/$DWI_FILE" ]; then
    echo "Error: DWI file not found: $DATA_DIR/$DWI_FILE"
    exit 1
fi

if [ ! -f "$DATA_DIR/$ADC_FILE" ]; then
    echo "Error: ADC file not found: $DATA_DIR/$ADC_FILE"
    exit 1
fi

if [ ! -f "$DATA_DIR/$FLAIR_FILE" ]; then
    echo "Error: FLAIR file not found: $DATA_DIR/$FLAIR_FILE"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Running Docker container: isleschallenge/deepisles:latest"
echo "Input data directory: $DATA_DIR"
echo "Output will be saved to: $DOCKER_OUTPUT"

# Run Docker container
docker run --gpus all \
    -v "$DATA_DIR:/app/data" \
    isleschallenge/deepisles:latest \
    --dwi_file_name "$DWI_FILE" \
    --adc_file_name "$ADC_FILE" \
    --flair_file_name "$FLAIR_FILE"

# Check if Docker output was created
DOCKER_RESULTS="$DATA_DIR/results/lesion_msk.nii.gz"
if [ ! -f "$DOCKER_RESULTS" ]; then
    echo "Error: Docker output not found at $DOCKER_RESULTS"
    exit 1
fi

# Copy output to example_test with docker naming
echo "Copying Docker output to $DOCKER_OUTPUT"
cp "$DOCKER_RESULTS" "$DOCKER_OUTPUT"

# Clean up temporary results directory (optional - comment out if you want to keep it)
if [ -d "$DATA_DIR/results" ]; then
    echo "Cleaning up temporary results directory..."
    rm -rf "$DATA_DIR/results"
fi

echo "Success! Docker output saved to: $DOCKER_OUTPUT"

