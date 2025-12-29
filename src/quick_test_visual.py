#!/usr/bin/env python3
"""
Quick Test with Visualization
Shows comparison plot after training
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend

# Configure for quick test
CONFIG['verbose_evaluation'] = False
CONFIG['num_epochs'] = 1
CONFIG['training_report_interval'] = 1
CONFIG['save_model'] = False
CONFIG['save_plots'] = True  # Save to output folder

from trainer import train_multi_year, evaluate_agent, plot_comparison
from data_loader import load_hedging_data, create_environment
from benchmark import run_benchmark_simple

print('=' * 70)
print('QUICK TRAINING TEST WITH VISUALIZATION')
print('=' * 70)

# Quick training with 1 epoch, 1 training year
print('\nTraining agent (1 epoch, 1 year - 2005)...')
agent, metrics, norm_stats = train_multi_year(
    data_path='./data/historical_hedging_data.csv',
    train_years=[2005],
    validation_year=2006,
    num_epochs=1,
    shuffle_years=False,
    verbose=True
)

# Evaluate on test year
test_year = 2007
print(f'\nEvaluating on test year ({test_year})...')
df = load_hedging_data('./data/historical_hedging_data.csv')
datetime_col = 'timestamp'
df[datetime_col] = pd.to_datetime(df[datetime_col])
test_df = df[df[datetime_col].dt.year == test_year].copy().reset_index(drop=True)

test_env = create_environment(
    test_df,
    normalization_stats=norm_stats,
    normalize=True,
    verbose=False
)

rl_stats = evaluate_agent(agent, test_env, verbose=True)

# Run benchmark
print('\nRunning benchmark...')
benchmark_df = run_benchmark_simple(df, year=test_year, verbose=False)
benchmark_pnl = benchmark_df['Cumulative PnL'].iloc[-1]

# Show comparison
# Convert to same units for fair comparison
# RL's cumulative_pnl is in dollars, benchmark's is normalized
# Convert RL to normalized for comparison
notional = CONFIG.get('notional', 1000)
rl_cumulative_pnl_normalized = rl_stats["cumulative_pnl"] / notional

print('\n' + '=' * 70)
print('QUICK TEST RESULTS')
print('=' * 70)
print(f'RL Cumulative P&L (from env): ${rl_stats["cumulative_pnl"]:.2f}')
print(f'RL Normalized: {rl_cumulative_pnl_normalized:.4f}')
print(f'Benchmark Normalized: {benchmark_pnl:.4f}')
print(f'Benchmark in $: ${benchmark_pnl * notional:.2f}')
print(f'Difference (normalized): {rl_cumulative_pnl_normalized - benchmark_pnl:+.4f}')
print(f'Difference ($): ${(rl_cumulative_pnl_normalized - benchmark_pnl) * notional:+.2f}')
print(f'RL is better by: ${(benchmark_pnl - rl_cumulative_pnl_normalized) * notional:.2f}')
print('=' * 70)

# Save and show plot
print('\nGenerating comparison plot...')
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join('results', f'quick_test_{timestamp}')
os.makedirs(results_dir, exist_ok=True)
plot_path = os.path.join(results_dir, f'quick_test_comparison_{test_year}.png')
plot_comparison(rl_stats, benchmark_df, test_env, save_path=plot_path, test_year=test_year, output_dir=results_dir)

print(f'\nPlot saved to: {plot_path}')
print('\nQuick test completed!')
