#!/bin/bash
# Launcher script for professional dashboard

# Navigate to dashboard directory
cd "$(dirname "$0")"

# Activate conda environment
eval "$(/data/virtualenvs/miniforge3/condabin/conda shell.bash hook)"
conda activate pytorch_ML_312

# Run dashboard
echo "Starting Professional RL Trading Dashboard..."
echo "Open http://localhost:5051 in your browser"
echo ""
python professional_dashboard.py
