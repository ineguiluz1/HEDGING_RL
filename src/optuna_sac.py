#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for SAC Hedging Agent
=========================================================

This script performs hyperparameter optimization using Optuna for the SAC algorithm.
It optimizes both algorithm-specific parameters and environment/reward configuration.

Parameters optimized:
- SAC Algorithm: learning rate, tau, gamma, entropy coefficient, etc.
- Reward function: weights for different reward components
- Network architecture: hidden dimensions, batch size

Parameters NOT optimized (problem-specific constants):
- transaction_cost: This is a market reality, not a tunable parameter
- risk_free_rate: Market constant
- notional: Scale factor
- episode_length: Determined by option expiry (30 days)

Usage:
    python optuna_sac.py
    
Results:
    - SQLite database: optuna_studies/sac_optimization.db
    - Best parameters saved to: optuna_studies/sac_best_params.json
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
from sac_agent import SACAgent, device
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
    Train a SAC agent with specific hyperparameters using the same training flow as run_training.py.
    
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
    
    # Force SAC
    CONFIG['model_type'] = 'SAC'
    
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


def create_sac_objective(train_envs: list, test_envs: list):
    """Create the Optuna objective function for SAC optimization."""
    
    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function for SAC hyperparameter optimization."""
        
        # =================================================================
        # SAC ALGORITHM PARAMETERS
        # =================================================================
        
        # Learning rate (SAC typically uses single learning rate)
        sac_learning_rate = trial.suggest_float('sac_learning_rate', 1e-5, 1e-3, log=True)
        
        # Discount factor
        gamma = trial.suggest_float('gamma', 0.95, 0.999)
        
        # Soft update rate
        tau = trial.suggest_float('tau', 0.001, 0.02)
        
        # Entropy coefficient (SAC specific)
        # "auto" means automatic entropy tuning; otherwise fixed value
        use_auto_entropy = trial.suggest_categorical('use_auto_entropy', [True, False])
        if use_auto_entropy:
            sac_ent_coef = "auto"
            # Target entropy (only used when auto)
            # None means automatic, otherwise specify
            use_auto_target = trial.suggest_categorical('use_auto_target_entropy', [True, False])
            if use_auto_target:
                sac_target_entropy = "auto"
            else:
                sac_target_entropy = trial.suggest_float('sac_target_entropy', -5.0, -0.1)
        else:
            sac_ent_coef = trial.suggest_float('sac_ent_coef_fixed', 0.01, 0.5, log=True)
            sac_target_entropy = "auto"  # Not used when ent_coef is fixed
        
        # State-Dependent Exploration (SDE)
        sac_use_sde = trial.suggest_categorical('sac_use_sde', [True, False])
        if sac_use_sde:
            sac_sde_sample_freq = trial.suggest_int('sac_sde_sample_freq', 1, 16)
        else:
            sac_sde_sample_freq = -1  # Disabled
        
        # Network architecture
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
        
        # Batch size and buffer
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024])
        replay_buffer_size = trial.suggest_categorical('replay_buffer_size', [50000, 100000, 200000])
        
        # Gradient steps per update
        sac_gradient_steps = trial.suggest_int('sac_gradient_steps', 1, 4)
        
        # Train frequency
        sac_train_freq = trial.suggest_int('sac_train_freq', 1, 8)
        
        # =================================================================
        # REWARD FUNCTION PARAMETERS
        # =================================================================
        
        # Reward weights
        delta_tracking_weight = trial.suggest_float('delta_tracking_weight', 0.01, 1.0, log=True)
        pnl_variance_weight = trial.suggest_float('pnl_variance_weight', 0.5, 5.0)
        transaction_cost_weight = trial.suggest_float('transaction_cost_weight', 0.1, 3.0)
        reward_scale = trial.suggest_float('reward_scale', 10.0, 500.0)
        
        # Risk aversion
        risk_aversion = trial.suggest_float('risk_aversion', 0.001, 0.1, log=True)
        
        # =================================================================
        # ACTION SPACE CONFIGURATION
        # =================================================================
        
        max_action = trial.suggest_float('max_action', 0.1, 0.5)
        
        # =================================================================
        # WARMUP CONFIGURATION
        # =================================================================
        
        warmup_steps = trial.suggest_int('warmup_steps', 100, 5000, step=100)
        
        # =================================================================
        # BUILD PARAMETER DICTIONARY
        # =================================================================
        
        params = {
            # SAC Algorithm
            'sac_learning_rate': sac_learning_rate,
            'gamma': gamma,
            'tau': tau,
            'sac_ent_coef': sac_ent_coef,
            'sac_target_entropy': sac_target_entropy,
            'sac_use_sde': sac_use_sde,
            'sac_sde_sample_freq': sac_sde_sample_freq,
            'hidden_dim': hidden_dim,
            'batch_size': batch_size,
            'replay_buffer_size': replay_buffer_size,
            'sac_gradient_steps': sac_gradient_steps,
            'sac_train_freq': sac_train_freq,
            
            # Warmup
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
        log_dir = f"optuna_studies/tensorboard/sac/trial_{trial.number}"
        writer = SummaryWriter(log_dir=log_dir)
        
        # Log hyperparameters as text (add_hparams will handle metrics visualization)
        writer.add_text('hyperparameters', str(params))
        
        # Train and evaluate
        seed = 1234 + trial.number  # Different seed per trial
        results = train_with_params(params, train_envs=train_envs, test_envs=test_envs, 
                                   seed=seed, writer=writer)
        
        # Log final objective value
        objective_value = results['sharpe_improvement'] + 0.01 * results['pnl_improvement']
        writer.add_scalar('objective/value', objective_value, 0)
        writer.close()
        
        # Store additional metrics
        trial.set_user_attr('total_pnl', results['total_pnl'])
        trial.set_user_attr('sharpe_ratio', results['sharpe_ratio'])
        trial.set_user_attr('pnl_improvement', results['pnl_improvement'])
        trial.set_user_attr('sharpe_improvement', results['sharpe_improvement'])
        trial.set_user_attr('max_drawdown', results['max_drawdown'])
        trial.set_user_attr('training_reward', results['training_reward'])
        trial.set_user_attr('episodes_trained', results['episodes_trained'])
        
        return objective_value
    
    return objective


def get_current_sac_params():
    """Get the current SAC parameters from CONFIG for baseline trial."""
    return {
        'sac_learning_rate': CONFIG.get('sac_learning_rate', 3e-4),
        'gamma': CONFIG.get('gamma', 0.99),
        'tau': CONFIG.get('tau', 0.005),
        'sac_ent_coef': CONFIG.get('sac_ent_coef', 'auto'),
        'sac_target_entropy': CONFIG.get('sac_target_entropy', 'auto'),
        'sac_use_sde': CONFIG.get('sac_use_sde', False),
        'sac_sde_sample_freq': CONFIG.get('sac_sde_sample_freq', -1),
        'hidden_dim': CONFIG.get('hidden_dim', 256),
        'batch_size': CONFIG.get('batch_size', 512),
        'replay_buffer_size': CONFIG.get('replay_buffer_size', 200000),
        'sac_gradient_steps': CONFIG.get('sac_gradient_steps', 1),
        'sac_train_freq': CONFIG.get('sac_train_freq', 1),
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
    
    current_params = get_current_sac_params()
    
    # Create TensorBoard writer for baseline (separate from Optuna trials)
    log_dir = "optuna_studies/tensorboard/sac/baseline"
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
    N_TRIALS = 100              # Total trials (1 baseline + 99 Optuna)
    STUDY_NAME = "sac_optimization"
    
    # Create output directory
    output_dir = "optuna_studies"
    os.makedirs(output_dir, exist_ok=True)
    
    # Database path
    db_path = os.path.join(output_dir, f"{STUDY_NAME}.db")
    storage = f"sqlite:///{db_path}"
    
    print("\n" + "="*70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION - SAC")
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
    
    # Check existing trials
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
        
        objective = create_sac_objective(train_envs=train_envs, test_envs=test_envs)
        
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
    
    # Convert best params to config format
    best_config = study.best_params.copy()
    # Handle entropy coefficient conversion
    if best_config.get('use_auto_entropy', True):
        best_config['sac_ent_coef'] = 'auto'
    elif 'sac_ent_coef_fixed' in best_config:
        best_config['sac_ent_coef'] = best_config.pop('sac_ent_coef_fixed')
    
    if best_config.get('use_auto_target_entropy', True):
        best_config['sac_target_entropy'] = 'auto'
    
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_params': best_config,
            'best_value': study.best_value,
            'best_trial_number': study.best_trial.number,
            'best_trial_metrics': study.best_trial.user_attrs,
            'n_trials': len(study.trials),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)
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
    print(f"\n✓ TensorBoard logs saved to: optuna_studies/tensorboard/sac/")
    print("  Run: tensorboard --logdir=optuna_studies/tensorboard/sac")
    
    return study


if __name__ == "__main__":
    study = main()
