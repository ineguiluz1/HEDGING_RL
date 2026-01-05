#!/bin/bash
# Script to launch multiple parallel Optuna workers for SAC optimization
# Usage: ./launch_parallel_optuna_sac.sh [n_workers]
# Example: ./launch_parallel_optuna_sac.sh 4  # Launch 4 parallel workers

N_WORKERS=${1:-4}  # Default to 4 workers if not specified
TOTAL_TRIALS=${2:-100}  # Default to 100 total trials if not specified
TRIALS_PER_WORKER=$((TOTAL_TRIALS / N_WORKERS))

echo "Launching $N_WORKERS parallel Optuna SAC workers..."
echo "Total trials: $TOTAL_TRIALS"
echo "Trials per worker: ~$TRIALS_PER_WORKER"
echo "Each worker will execute trials independently"
echo "All workers share the same SQLite database: optuna_studies/sac_optimization.db"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Launch multiple SBATCH jobs
for i in $(seq 1 $N_WORKERS); do
    JOB_ID=$(sbatch --parsable --export=OPTUNA_N_TRIALS=$TRIALS_PER_WORKER run_optuna_sac.sh)
    echo "Worker $i launched with Job ID: $JOB_ID (will run ~$TRIALS_PER_WORKER trials)"
    sleep 1  # Small delay to avoid overwhelming the scheduler
done

echo ""
echo "All workers launched!"
echo "Monitor progress with: squeue -u $USER"
echo "View logs in: logs/optuna_sac_*.out"
