#!/bin/bash
#SBATCH --job-name=optuna_td3_hedging
#SBATCH --output=logs/optuna_td3_%A.out
#SBATCH --error=logs/optuna_td3_%A.err
#SBATCH --time=23:00:00                # Time limit (23 hours)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=16G                      # Memory per task
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx3090
#SBATCH --mail-type=BEGIN,END,FAIL     # Send email on job begin, end, and fail
#SBATCH --mail-user=aitor.diez@opendeusto.es  # Replace with your email address

# Create logs directory if it doesn't exist
mkdir -p logs

# Create optuna_studies directory if it doesn't exist
mkdir -p optuna_studies

# Activate virtual environment
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/aitordiez/conda-env/rl_env

# Print some useful information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Starting time: $(date)"
echo "GPU assigned: $CUDA_VISIBLE_DEVICES"
echo "Running: optuna_td3.py"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Change to src directory and run the script
cd src
python optuna_td3.py

echo "Ending time: $(date)"
