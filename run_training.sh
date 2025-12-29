#!/bin/bash
#SBATCH --job-name=hedging_td3
#SBATCH --output=logs/hedging_td3_%A.out
#SBATCH --error=logs/hedging_td3_%A.err
#SBATCH --partition=general            # Default partition (all users have access)
#SBATCH --time=12:00:00                # Time limit (12 hours for inference)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=32G                      # Memory per task
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx3090
#SBATCH --mail-type=BEGIN,END,FAIL     # Send email on job begin, end, and fail
#SBATCH --mail-user=aitor.diez@opendeusto.es

# Create logs and output directories if they don't exist
mkdir -p logs
mkdir -p output
mkdir -p results

# Activate virtual environment
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/aitordiez/conda-env/rl_env

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Print useful information
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Starting time: $(date)"
echo "GPU assigned: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "========================================================================"
echo ""
echo "Training TD3 Agent for Options Hedging"
echo "Configuration:"
echo "  - Training years: 2005-2010"
echo "  - Validation year: 2011"
echo "  - Test year: 2012"
echo "  - Epochs: 10"
echo ""
echo "========================================================================"
echo ""

# Run the training script
python src/run_training.py

# Print completion information
echo ""
echo "========================================================================"
echo "Job completed"
echo "Ending time: $(date)"
echo "========================================================================"

# List generated files
echo ""
echo "Generated files:"
echo "Results directory:"
ls -lh results/
echo ""
echo "Output directory:"
ls -lh output/
