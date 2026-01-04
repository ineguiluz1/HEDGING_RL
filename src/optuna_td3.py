#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for TD3 Hedging Agent
=========================================================

This script performs hyperparameter optimization using Optuna for the TD3 algorithm.
It optimizes both algorithm-specific parameters and environment/reward configuration.

Parameters optimized:
- TD3 Algorithm: learning rates, tau, gamma, policy noise, hidden dimensions, etc.
- Reward function: weights for different reward components
- Exploration: noise parameters, warmup steps

Parameters NOT optimized (problem-specific constants):
- transaction_cost: This is a market reality, not a tunable parameter
- risk_free_rate: Market constant
- notional: Scale factor
- episode_length: Determined by option expiry (30 days)

Usage:
    python optuna_td3.py
    
Results:
    - SQLite database: optuna_studies/td3_optimization.db
    - Best parameters saved to: optuna_studies/td3_best_params.json
"""

import os
import sys
import json
import random
import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from td3_agent import TD3Agent, device
from data_loader import create_environments_for_training
from trainer import TrainingMetrics
from metrics import evaluate_agent_with_metrics, evaluate_benchmark_with_metrics


# Import training function from run_training
from run_training import train_multi_env


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_with_params(params: dict, train_envs: list, test_envs: list, seed: int = 1234, writer: SummaryWriter = None):
    """
    Train a TD3 agent with specific hyperparameters using the same training flow as run_training.py.
    
    Args:
        params: Dictionary of hyperparameters
        train_envs: Pre-created training environments (shared across trials)
        test_envs: Pre-created test environments (shared across trials)
        seed: Random seed
        writer: TensorBoard SummaryWriter for logging
    
    Returns:
        dict: Evaluation metrics
    """
    set_all_seeds(seed)
    
    # Override CONFIG with trial parameters
    original_config = CONFIG.copy()
    
    # Update CONFIG with trial parameters
    for key, value in params.items():
        CONFIG[key] = value
    
    # Force TD3
    CONFIG['model_type'] = 'TD3'
    
    try:
        # Use the same training function as run_training.py
        # This includes: curriculum learning, early stopping, all the training logic
        agent, metrics = train_multi_env(train_envs=train_envs, verbose=False)
        
        # Log training metrics to TensorBoard
        if writer is not None:
            for i, reward in enumerate(metrics.episode_rewards):
                writer.add_scalar('training/episode_reward', reward, i)
            
            cumulative_reward = np.cumsum(metrics.episode_rewards)
            for i, cum_reward in enumerate(cumulative_reward):
                writer.add_scalar('training/cumulative_reward', cum_reward, i)
        
        # Evaluate on test set
        agent_metrics, _ = evaluate_agent_with_metrics(agent, test_envs, verbose=False)
        benchmark_metrics, _ = evaluate_benchmark_with_metrics(test_envs, verbose=False)
        
        # Calculate improvement metrics
        pnl_improvement = agent_metrics.total_pnl - benchmark_metrics.total_pnl
        sharpe_improvement = agent_metrics.sharpe_ratio - benchmark_metrics.sharpe_ratio
        
        results = {
            'total_pnl': agent_metrics.total_pnl,
            'sharpe_ratio': agent_metrics.sharpe_ratio,
            'pnl_improvement': pnl_improvement,
            'sharpe_improvement': sharpe_improvement,
            'mean_episode_pnl': agent_metrics.mean_episode_pnl,
            'pnl_variance': agent_metrics.pnl_variance,
            'max_drawdown': agent_metrics.max_drawdown,
            'training_reward': np.mean(metrics.episode_rewards) if metrics.episode_rewards else 0.0,
            'episodes_trained': len(metrics.episode_rewards)
        }
        
        # Log final metrics to TensorBoard using add_hparams
        # This creates visual comparisons in the HPARAMS tab (table, parallel coordinates, scatter plots)
        if writer is not None:
            # Convert params to format suitable for add_hparams (only numeric values)
            hparam_dict = {k: v for k, v in params.items() if isinstance(v, (int, float))}
            metric_dict = {
                'metric/sharpe_improvement': results['sharpe_improvement'],
                'metric/pnl_improvement': results['pnl_improvement'],
                'metric/total_pnl': results['total_pnl'],
                'metric/sharpe_ratio': results['sharpe_ratio'],
                'metric/max_drawdown': results['max_drawdown'],
                'metric/pnl_variance': results['pnl_variance']
            }
            # run_name='.' prevents creating subdirectories with timestamps
            writer.add_hparams(hparam_dict, metric_dict, run_name='.')
            writer.flush()
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        results = {
            'total_pnl': -999999,
            'sharpe_ratio': -999,
            'pnl_improvement': -999999,
            'sharpe_improvement': -999,
            'mean_episode_pnl': -999999,
            'pnl_variance': 999999,
            'max_drawdown': 999999,
            'training_reward': -999999,
            'episodes_trained': 0
        }
    
    finally:
        # Restore original config
        for key, value in original_config.items():
            CONFIG[key] = value
    
    return results


def create_td3_objective(train_envs: list, test_envs: list):
    """Create the Optuna objective function for TD3 optimization."""
    
    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function for TD3 hyperparameter optimization."""
        
        # =================================================================
        # TD3 ALGORITHM PARAMETERS
        # =================================================================
        
        # Learning rates (log scale for better sampling)
        actor_lr = trial.suggest_float('actor_lr', 1e-5, 1e-3, log=True)
        critic_lr = trial.suggest_float('critic_lr', 1e-5, 1e-3, log=True)
        
        # Discount factor (typically 0.95-0.999 for continuous control)
        gamma = trial.suggest_float('gamma', 0.95, 0.999)
        
        # Soft update rate (tau)
        tau = trial.suggest_float('tau', 0.001, 0.02)
        
        # TD3 specific: policy smoothing noise
        policy_noise = trial.suggest_float('policy_noise', 0.01, 0.3)
        noise_clip = trial.suggest_float('noise_clip', 0.1, 0.5)
        
        # Policy update frequency (TD3 specific)
        policy_freq = trial.suggest_int('policy_freq', 1, 4)
        
        # Network architecture
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
        
        # Batch size and buffer
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024])
        replay_buffer_size = trial.suggest_categorical('replay_buffer_size', [50000, 100000, 200000])
        
        # =================================================================
        # EXPLORATION PARAMETERS (OU Process for TD3)
        # =================================================================
        
        initial_noise = trial.suggest_float('initial_noise', 0.3, 1.0)
        final_noise = trial.suggest_float('final_noise', 0.01, 0.2)
        ou_theta = trial.suggest_float('ou_theta', 0.05, 0.3)
        ou_sigma = trial.suggest_float('ou_sigma', 0.1, 0.6)
        
        # Warmup steps (exploration before learning)
        warmup_steps = trial.suggest_int('warmup_steps', 1000, 10000, step=1000)
        
        # =================================================================
        # REWARD FUNCTION PARAMETERS
        # =================================================================
        
        # Reward weights (these affect learning, not the problem definition)
        delta_tracking_weight = trial.suggest_float('delta_tracking_weight', 0.01, 1.0, log=True)
        pnl_variance_weight = trial.suggest_float('pnl_variance_weight', 0.5, 5.0)
        transaction_cost_weight = trial.suggest_float('transaction_cost_weight', 0.1, 3.0)
        reward_scale = trial.suggest_float('reward_scale', 10.0, 500.0)
        
        # Risk aversion (for profit_seeking reward type)
        risk_aversion = trial.suggest_float('risk_aversion', 0.001, 0.1, log=True)
        
        # =================================================================
        # ACTION SPACE CONFIGURATION
        # =================================================================
        
        # Maximum action adjustment from delta
        max_action = trial.suggest_float('max_action', 0.1, 0.5)
        
        # =================================================================
        # BUILD PARAMETER DICTIONARY
        # =================================================================
        
        params = {
            # TD3 Algorithm
            'actor_lr': actor_lr,
            'critic_lr': critic_lr,
            'gamma': gamma,
            'tau': tau,
            'policy_noise': policy_noise,
            'noise_clip': noise_clip,
            'policy_freq': policy_freq,
            'hidden_dim': hidden_dim,
            'batch_size': batch_size,
            'replay_buffer_size': replay_buffer_size,
            
            # Exploration
            'initial_noise': initial_noise,
            'final_noise': final_noise,
            'ou_theta': ou_theta,
            'ou_sigma': ou_sigma,
            'warmup_steps': warmup_steps,
            
            # Reward function
            'delta_tracking_weight': delta_tracking_weight,
            'pnl_variance_weight': pnl_variance_weight,
            'transaction_cost_weight': transaction_cost_weight,
            'reward_scale': reward_scale,
            'risk_aversion': risk_aversion,
            
            # Action space
            'max_action': max_action,
            'min_action': -max_action,
        }
        
        # Create TensorBoard writer for this trial
        log_dir = f"optuna_studies/tensorboard/td3/trial_{trial.number}"
        writer = SummaryWriter(log_dir=log_dir)
        
        # Log hyperparameters as text (add_hparams will handle metrics visualization)
        writer.add_text('hyperparameters', str(params))
        
        # Train and evaluate
        seed = 1234 + trial.number  # Different seed for each trial
        results = train_with_params(params, train_envs=train_envs, test_envs=test_envs, 
                                   seed=seed, writer=writer)
        
        # Log final objective value
        objective_value = results['sharpe_improvement'] + 0.01 * results['pnl_improvement']
        writer.add_scalar('objective/value', objective_value, 0)
        writer.close()
        
        # Store additional metrics as user attributes
        trial.set_user_attr('total_pnl', results['total_pnl'])
        trial.set_user_attr('sharpe_ratio', results['sharpe_ratio'])
        trial.set_user_attr('pnl_improvement', results['pnl_improvement'])
        trial.set_user_attr('sharpe_improvement', results['sharpe_improvement'])
        trial.set_user_attr('max_drawdown', results['max_drawdown'])
        trial.set_user_attr('training_reward', results['training_reward'])
        trial.set_user_attr('episodes_trained', results['episodes_trained'])
        
        return objective_value
    
    return objective


def get_current_td3_params():
    """Get the current TD3 parameters from CONFIG for baseline trial."""
    return {
        'actor_lr': CONFIG.get('actor_lr', 1e-4),
        'critic_lr': CONFIG.get('critic_lr', 1e-4),
        'gamma': CONFIG.get('gamma', 0.99),
        'tau': CONFIG.get('tau', 0.001),
        'policy_noise': CONFIG.get('policy_noise', 0.05),
        'noise_clip': CONFIG.get('noise_clip', 0.2),
        'policy_freq': CONFIG.get('policy_freq', 2),
        'hidden_dim': CONFIG.get('hidden_dim', 256),
        'batch_size': CONFIG.get('batch_size', 512),
        'replay_buffer_size': CONFIG.get('replay_buffer_size', 200000),
        'initial_noise': CONFIG.get('initial_noise', 0.8),
        'final_noise': CONFIG.get('final_noise', 0.05),
        'ou_theta': CONFIG.get('ou_theta', 0.15),
        'ou_sigma': CONFIG.get('ou_sigma', 0.4),
        'warmup_steps': CONFIG.get('warmup_steps', 5000),
        'delta_tracking_weight': CONFIG.get('delta_tracking_weight', 0.1),
        'pnl_variance_weight': CONFIG.get('pnl_variance_weight', 2.0),
        'transaction_cost_weight': CONFIG.get('transaction_cost_weight', 1.0),
        'reward_scale': CONFIG.get('reward_scale', 100.0),
        'risk_aversion': CONFIG.get('risk_aversion', 0.01),
        'max_action': CONFIG.get('max_action', 0.3),
        'min_action': CONFIG.get('min_action', -0.3),
    }


def run_baseline_trial(study: optuna.Study, train_envs: list, test_envs: list):
    """Run a trial with current CONFIG parameters as baseline.
    
    Note: This creates a 'baseline' directory in TensorBoard, separate from Optuna trials.
    Optuna trials will be numbered starting from 0 (trial_0, trial_1, etc.)
    """
    print("\n" + "="*60)
    print("BASELINE TRIAL: Using current CONFIG parameters")
    print("="*60)
    
    current_params = get_current_td3_params()
    
    # Create TensorBoard writer for baseline (separate from Optuna trials)
    log_dir = "optuna_studies/tensorboard/td3/baseline"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Log baseline hyperparameters as text (add_hparams will handle metrics visualization)
    writer.add_text('hyperparameters', str(current_params))
    
    # Train with current parameters
    results = train_with_params(current_params, train_envs=train_envs, test_envs=test_envs, 
                               seed=1234, writer=writer)
    
    # Log baseline results
    writer.add_scalar('objective/value', results['sharpe_improvement'] + 0.01 * results['pnl_improvement'], 0)
    writer.close()
    
    print(f"\nBaseline Results:")
    print(f"  Total P&L: {results['total_pnl']:.4f}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"  P&L Improvement: {results['pnl_improvement']:.4f}")
    print(f"  Sharpe Improvement: {results['sharpe_improvement']:.4f}")
    print(f"  Episodes trained: {results['episodes_trained']}")
    
    # Store baseline results for comparison (not as an Optuna trial to avoid confusion)
    # The baseline serves as a reference point, Optuna trials start fresh
    
    return results


def main():
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    N_TRIALS = 3              # Total trials (1 baseline + 99 Optuna)
    STUDY_NAME = "td3_optimization"
    
    # Create output directory
    output_dir = "optuna_studies"
    os.makedirs(output_dir, exist_ok=True)
    
    # Database path
    db_path = os.path.join(output_dir, f"{STUDY_NAME}.db")
    storage = f"sqlite:///{db_path}"
    
    print("\n" + "="*70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION - TD3")
    print("="*70)
    print(f"Total trials: {N_TRIALS}")
    print(f"Training: Using same flow as run_training.py (with early stopping)")
    print(f"Database: {db_path}")
    print(f"Device: {device}")
    print("="*70)
    
    # =========================================================================
    # CREATE ENVIRONMENTS ONCE (shared across all trials)
    # =========================================================================
    print("\n" + "="*70)
    print("CREATING ENVIRONMENTS (shared across all trials)")
    print("="*70)
    
    # Set seed for environment creation
    seed = CONFIG.get("seed", 101)
    set_all_seeds(seed)
    
    envs = create_environments_for_training(verbose=True)
    train_envs = envs['train_envs']
    test_envs = envs.get('test_envs', [])
    norm_stats = envs['normalization_stats']
    
    if len(test_envs) == 0:
        raise ValueError("No test environments created")
    
    print(f"\n  ✓ Training environments: {len(train_envs)}")
    print(f"  ✓ Test environments: {len(test_envs)}")
    print(f"  ✓ These will be reused for all {N_TRIALS} trials")
    print("="*70)
    
    # Create or load study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
        direction="maximize",  # Maximize improvement over benchmark
        sampler=sampler
    )
    
    # Check how many trials already completed
    n_existing = len(study.trials)
    n_remaining = max(0, N_TRIALS - n_existing)
    
    if n_existing > 0:
        print(f"\nResuming study with {n_existing} existing trials")
        print(f"Running {n_remaining} more trials to reach {N_TRIALS}")
    else:
        # Run baseline trial first (separate from Optuna optimization)
        print("\nStarting fresh study...")
        baseline_results = run_baseline_trial(study, train_envs=train_envs, test_envs=test_envs)
        # Baseline doesn't count against N_TRIALS - all N_TRIALS will be Optuna trials
        # This gives you: 1 baseline + N_TRIALS optimized = N_TRIALS + 1 total runs
    
    # Run optimization
    if n_remaining > 0:
        print(f"\nRunning {n_remaining} optimization trials...")
        
        objective = create_td3_objective(train_envs=train_envs, test_envs=test_envs)
        
        study.optimize(
            objective,
            n_trials=n_remaining,
            show_progress_bar=True,
            catch=(Exception,)
        )
    
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best objective value: {study.best_value:.4f}")
    
    print("\n" + "-"*40)
    print("BEST HYPERPARAMETERS:")
    print("-"*40)
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print("\n" + "-"*40)
    print("BEST TRIAL METRICS:")
    print("-"*40)
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key}: {value:.4f}")
    
    # Save best parameters
    best_params_path = os.path.join(output_dir, f"{STUDY_NAME}_best_params.json")
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial_number': study.best_trial.number,
            'best_trial_metrics': study.best_trial.user_attrs,
            'n_trials': len(study.trials),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"\nBest parameters saved to: {best_params_path}")
    
    # Print comparison with baseline
    if len(study.trials) > 1:
        baseline_trial = study.trials[0]
        print("\n" + "-"*40)
        print("COMPARISON: Best vs Baseline")
        print("-"*40)
        print(f"  Baseline Sharpe Improvement: {baseline_trial.user_attrs.get('sharpe_improvement', 'N/A')}")
        print(f"  Best Sharpe Improvement: {study.best_trial.user_attrs.get('sharpe_improvement', 'N/A')}")
        print(f"  Baseline P&L Improvement: {baseline_trial.user_attrs.get('pnl_improvement', 'N/A')}")
        print(f"  Best P&L Improvement: {study.best_trial.user_attrs.get('pnl_improvement', 'N/A')}")
    
    print(f"\n✓ Study database saved to: {db_path}")
    print("  Use optuna-dashboard or optuna.visualization to explore results")
    print(f"\n✓ TensorBoard logs saved to: optuna_studies/tensorboard/td3/")
    print("  Run: tensorboard --logdir=optuna_studies/tensorboard/td3")
    
    return study


if __name__ == "__main__":
    study = main()
