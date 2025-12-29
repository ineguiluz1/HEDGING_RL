"# HEDGING_RL

A Reinforcement Learning approach to Options Hedging using TD3 (Twin Delayed Deep Deterministic Policy Gradient).

## Overview

This project implements a TD3 agent that learns optimal hedging strategies for options, comparing its performance against traditional Delta Hedging (Black-Scholes). The agent learns to minimize hedging costs while managing risk through interaction with a simulated trading environment.

**Implementation**: Now using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for a robust, tested TD3 implementation. The original custom implementation is preserved in `td3_agent_custom.py` and `trainer_custom.py` for reference.

## Project Structure

```
HEDGING_RL/
├── src/
│   ├── config.py              # Configuration parameters
│   ├── hedging_env.py         # Gymnasium environment for hedging
│   ├── td3_agent.py           # TD3 Agent (Stable-Baselines3 wrapper)
│   ├── td3_agent_custom.py    # Original custom TD3 implementation
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── trainer.py             # Training utilities (SB3)
│   ├── trainer_custom.py      # Original custom trainer
│   ├── benchmark.py           # Delta Hedging benchmark
│   ├── run_training.py        # Main training script
│   └── volatility_utils.py    # Volatility calculation utilities
├── data/
│   └── historical_hedging_data.csv  # Historical options data
├── output/                    # Plots and visualizations
├── results/                   # Training results and saved models
└── requirements.txt           # Python dependencies
```

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd HEDGING_RL

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Quick Test with Visualization

For a quick test (1 epoch, 1 year) with immediate visual feedback:

```bash
python src/quick_test_visual.py
```

This will:
- Train on year 2005 (1 epoch)
- Validate on year 2006
- Test on year 2007
- **Show an interactive comparison plot** (similar to academic papers)

### Run Full Training Pipeline

```bash
# Train with default configuration (2005-2010 train, 2011 validation, 2012 test)
python src/run_training.py

# Or use the visual training script
python src/train_full_visual.py

# Train with custom parameters
python src/run_training.py --epochs 20 --train-years 2005 2006 2007 2008 2009 2010

# Test a pre-trained model
python src/run_training.py --test-only --model-path results/td3_model_best.pth
```

### Visualization

The comparison plot shows:
- **Main Plot**: Cumulative Return (%) over time - RL Agent vs Delta Hedging
- **Hedge Ratio Comparison**: Actions taken by both strategies
- **Step P&L**: Smoothed P&L at each trading step
- **Distribution**: Histogram of hedge ratios used
- **Statistics Table**: Comprehensive performance metrics

Colors:
- 🔵 Blue = RL Agent
- 🔴 Red = Delta Hedging Benchmark

**Output Files**: All comparison plots are automatically saved in the `results/run_TIMESTAMP/` directory:
- `comparison_year_YYYY.png` - Combined plot with all metrics
- `comparison_year_YYYY_cumulative_pnl.png` - Cumulative P&L only
- `comparison_year_YYYY_hedge_ratio.png` - Hedge ratio comparison
- `comparison_year_YYYY_step_pnl.png` - Step-by-step P&L
- `comparison_year_YYYY_distribution.png` - Distribution of hedge ratios
- `training_curves.png` - Training progress metrics

This allows for easy inclusion in papers, presentations, and reports. Each training run gets its own timestamped directory.

### Configuration

All parameters are configurable in `src/config.py`:

```python
# Data splits
"train_years": [2005, 2006, 2007, 2008, 2009, 2010],
"validation_year": 2011,
"test_year": 2012,

# Training parameters
"num_epochs": 10,          # Epochs for multi-year training
"batch_size": 256,         # Batch size for experience replay
"replay_buffer_size": 100000,

# TD3 Algorithm
"tau": 0.001,              # Soft update rate

# GPU
"use_gpu": True,           # Enable GPU training (faster)
```

### Training Progress

The training scripts now include **progress bars** powered by `tqdm`:

- **Epoch progress**: Shows overall epoch completion
- **Year progress**: Shows which year is being trained within each epoch
- **Metrics displayed**: Real-time reward, P&L, and noise levels

Example output during training:
```
Training Epochs:  50%|████████████▌            | 5/10 [01:23<01:23, avg_reward: -45.23, avg_pnl: -44.12]
└─ Epoch 5/10 - Year 2008: 100%|██████████| 6/6 [00:16<00:00, reward: -47.32, pnl: -45.89, noise: 0.125]
```
"gamma": 0.99,             # Discount factor
"policy_noise": 0.1,       # Target policy smoothing noise
"policy_freq": 2,          # Delayed policy updates

# Exploration
"initial_noise": 0.3,
"final_noise": 0.02,
"noise_decay_steps": 500000,
```

## Results

After training, results are saved to:

**Model & Metrics**: `results/run_<timestamp>/`
- `td3_model.pth` - Trained model weights
- `td3_model_best.pth` - Best model (based on validation)
- `training_metrics.json` - Training metrics in JSON format
- `results.json` - Summary of results
- `rl_agent_steps.csv` - Detailed RL agent step data
- `benchmark_steps.csv` - Detailed benchmark step data

**Plots & Visualizations**: `output/`
- `training_curves.png` - Training metrics over time
- `comparison_year_<year>.png` - Complete comparison dashboard (all subplots)
- `comparison_year_<year>_cumulative_return.png` - Cumulative return plot only
- `comparison_year_<year>_hedge_ratio.png` - Hedge ratio comparison
- `comparison_year_<year>_step_pnl.png` - Step-by-step P&L analysis
- `comparison_year_<year>_distribution.png` - Hedge ratio distribution
- `quick_test_comparison_<year>.png` - Quick test complete dashboard
- `quick_test_comparison_<year>_*.png` - Individual quick test plots
- `evaluation_year_<year>.png` - Evaluation complete dashboard
- `evaluation_year_<year>_*.png` - Individual evaluation plots

### Example Output

```
Out-of-Sample Cumulative Return (%): RL Agent vs Delta Hedging

Final Returns:
  RL Agent: +45.23%
  Delta Hedging: -38.45%
  Improvement: +83.68%
```

The visualization includes:
1. **Main cumulative return plot** showing performance over time
2. **Hedge ratio comparison** showing strategy decisions
3. **Step P&L analysis** showing profit/loss at each step
4. **Distribution analysis** comparing hedge ratio distributions
5. **Statistics table** with comprehensive metrics

### Evaluate Pre-trained Model

```bash
# Evaluate a specific model with visualization
python src/evaluate_model.py
```

Or modify the script to test on different years:
```python
MODEL_PATH = "results/td3_model_best.pth"
TEST_YEAR = 2012  # Change this to test on different years
```

## Architecture

### TD3 Agent
- **Actor Network**: 3 hidden layers (256, 256, 128 neurons) with LayerNorm and LeakyReLU
- **Critic Networks**: Twin Q-networks with same architecture
- **Exploration**: Ornstein-Uhlenbeck noise with Gaussian noise combination
- **Target Policy Smoothing**: Added noise to target actions for smoother value estimates

### Hedging Environment
- **State Space (6D)**: Option price, stock price, TTM, moneyness, realized volatility, current position
- **Action Space (1D)**: Hedge ratio (delta) in [-1, 1]
- **Reward**: Risk-adjusted P&L with penalty for deviation

## Benchmark Comparison

The agent is compared against traditional Delta Hedging:
- Uses Black-Scholes delta calculation
- Same transaction costs and risk parameters
- Evaluated on the same test period

## References

- Fujimoto et al. (2018) "Addressing Function Approximation Error in Actor-Critic Methods" (TD3)
- Deep Hedging literature
" 
