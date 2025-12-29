"""
Example: Using the trained agent for evaluation with visualization
"""

import sys
import os
sys.path.insert(0, 'src')

from config import CONFIG
from td3_agent import TD3Agent
from data_loader import load_hedging_data, create_environment
from trainer import evaluate_agent, plot_comparison
from benchmark import run_benchmark_simple
import pandas as pd

# Configuration
MODEL_PATH = "results/td3_model_best.pth"  # Path to your trained model
TEST_YEAR = 2012  # Year to test on
DATA_PATH = "./data/historical_hedging_data.csv"

def evaluate_trained_model(model_path, test_year, data_path=DATA_PATH):
    """
    Evaluate a trained TD3 model and compare with benchmark
    
    Args:
        model_path: Path to saved model (.pth file)
        test_year: Year to test on
        data_path: Path to data file
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING TRAINED TD3 MODEL")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Test Year: {test_year}")
    print(f"{'='*70}\n")
    
    # Load data
    df = load_hedging_data(data_path)
    datetime_col = 'timestamp'
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    test_df = df[df[datetime_col].dt.year == test_year].copy().reset_index(drop=True)
    
    if len(test_df) == 0:
        raise ValueError(f"No data found for year {test_year}")
    
    print(f"Test data: {len(test_df)} rows\n")
    
    # Create test environment
    test_env = create_environment(
        test_df,
        normalize=CONFIG.get("normalize_data", True),
        verbose=False
    )
    
    # Load trained agent
    print("Loading trained agent...")
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.shape[0]
    
    agent = TD3Agent(state_dim, action_dim)
    agent.load(model_path)
    print("Model loaded successfully!\n")
    
    # Evaluate RL agent
    print("Evaluating RL Agent...")
    rl_stats = evaluate_agent(agent, test_env, verbose=True)
    
    # Run benchmark
    print("\nRunning Delta Hedging Benchmark...")
    benchmark_df = run_benchmark_simple(df, year=test_year, verbose=True)
    
    # Calculate improvements
    benchmark_pnl = benchmark_df['Cumulative PnL'].iloc[-1]
    benchmark_sharpe = benchmark_df['PnL'].mean() / (benchmark_df['PnL'].std() + 1e-8) * 252**0.5
    
    pnl_improvement = rl_stats['cumulative_pnl'] - benchmark_pnl
    sharpe_improvement = rl_stats['sharpe_ratio'] - benchmark_sharpe
    
    # Print comparison
    print(f"\n{'='*70}")
    print(f"PERFORMANCE COMPARISON - Year {test_year}")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'RL Agent':<20} {'Delta Hedge':<20}")
    print(f"{'-'*70}")
    print(f"{'Cumulative P&L':<30} {rl_stats['cumulative_pnl']:<20.4f} {benchmark_pnl:<20.4f}")
    print(f"{'Sharpe Ratio':<30} {rl_stats['sharpe_ratio']:<20.4f} {benchmark_sharpe:<20.4f}")
    print(f"{'Mean Hedge Ratio':<30} {rl_stats['mean_action']:<20.4f} {benchmark_df['Delta'].mean():<20.4f}")
    print(f"{'='*70}")
    print(f"\nIMPROVEMENTS:")
    print(f"  P&L: {pnl_improvement:+.4f} ({pnl_improvement/abs(benchmark_pnl + 1e-8)*100:+.2f}%)")
    print(f"  Sharpe: {sharpe_improvement:+.4f}")
    
    if pnl_improvement > 0:
        print(f"\n✅ RL Agent OUTPERFORMS Delta Hedging!")
    else:
        print(f"\n❌ Delta Hedging outperforms RL Agent")
    
    # Show and save visualization
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATION...")
    print(f"{'='*70}\n")
    
    # Save to results directory (use model path to determine location)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', f'evaluation_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, f'evaluation_year_{test_year}.png')
    plot_comparison(rl_stats, benchmark_df, test_env, save_path=plot_path, test_year=test_year, output_dir=results_dir)
    print(f"Plot saved to: {plot_path}")
    
    return {
        'rl_stats': rl_stats,
        'benchmark_df': benchmark_df,
        'improvements': {
            'pnl': pnl_improvement,
            'sharpe': sharpe_improvement
        }
    }


if __name__ == "__main__":
    import os
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n⚠️  Model not found: {MODEL_PATH}")
        print("\nPlease train a model first:")
        print("  python src/run_training.py")
        print("\nOr run a quick test:")
        print("  python src/quick_test_visual.py")
    else:
        # Evaluate the model
        results = evaluate_trained_model(MODEL_PATH, TEST_YEAR)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETED!")
        print("="*70)
        print("\nThe comparison plot is now displayed.")
        print("Close the plot window to exit.")
        print("="*70 + "\n")
