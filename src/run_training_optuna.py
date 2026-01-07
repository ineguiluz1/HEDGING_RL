#!/usr/bin/env python3
"""
Training Script with Optuna-Optimized Hyperparameters
======================================================

This script trains a TD3 or SAC agent using:
- Optimized hyperparameters from Optuna (if available)
- Default parameters from config.py (for non-optimized params)

Usage:
    python run_training_optuna.py                    # Use model_type from config.py
    python run_training_optuna.py --model SAC        # Force SAC model
    python run_training_optuna.py --model TD3        # Force TD3 model
    
The script automatically:
1. Loads best parameters from optuna_studies/{model}_optimization_best_params.json
2. Merges with default CONFIG from config.py
3. Trains the model using run_training.py infrastructure

Author: Generated for HEDGING_RL project
"""

import os
import sys
import argparse
import json
import copy
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import config first (before modifying it)
from config import CONFIG

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTUNA_STUDIES_DIR = os.path.join(PROJECT_ROOT, "optuna_studies")


def load_optuna_params(model_type: str) -> dict:
    """
    Load best parameters from Optuna optimization.
    
    Args:
        model_type: "SAC" or "TD3"
    
    Returns:
        dict: Best parameters from Optuna, or empty dict if not found
    """
    filename = f"{model_type.lower()}_optimization_best_params.json"
    filepath = os.path.join(OPTUNA_STUDIES_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"⚠ Optuna parameters file not found: {filepath}")
        print(f"  Will use default parameters from config.py")
        return {}
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        best_params = data.get("best_params", {})
        best_value = data.get("best_value", None)
        best_trial = data.get("best_trial_number", None)
        n_trials = data.get("n_trials", None)
        timestamp = data.get("timestamp", None)
        
        print(f"\n{'='*70}")
        print(f"LOADED OPTUNA OPTIMIZED PARAMETERS")
        print(f"{'='*70}")
        print(f"  File: {filepath}")
        print(f"  Best trial: #{best_trial} of {n_trials} trials")
        print(f"  Best value: {best_value:.6f}" if best_value else "  Best value: N/A")
        print(f"  Optimized on: {timestamp}" if timestamp else "")
        print(f"  Parameters loaded: {len(best_params)}")
        print(f"{'='*70}\n")
        
        return best_params
        
    except Exception as e:
        print(f"⚠ Error loading Optuna parameters: {e}")
        return {}


def merge_configs(base_config: dict, optuna_params: dict, model_type: str) -> dict:
    """
    Merge Optuna parameters with base config.
    Optuna parameters take precedence over base config.
    
    Args:
        base_config: Default configuration from config.py
        optuna_params: Parameters from Optuna optimization
        model_type: "SAC" or "TD3"
    
    Returns:
        dict: Merged configuration
    """
    # Create a deep copy of base config
    merged = copy.deepcopy(base_config)
    
    # Set model type
    merged["model_type"] = model_type
    
    # Track which parameters are from Optuna vs config
    from_optuna = []
    from_config = []
    
    # Define mapping from Optuna param names to config param names
    # Some Optuna params may have different names than config
    param_mapping = {
        # Common params (both SAC and TD3)
        "gamma": "gamma",
        "tau": "tau",
        "hidden_dim": "hidden_dim",
        "batch_size": "batch_size",
        "replay_buffer_size": "replay_buffer_size",
        "warmup_steps": "warmup_steps",
        "delta_tracking_weight": "delta_tracking_weight",
        "pnl_variance_weight": "pnl_variance_weight",
        "transaction_cost_weight": "transaction_cost_weight",
        "reward_scale": "reward_scale",
        "risk_aversion": "risk_aversion",
        "max_action": "max_action",
        
        # TD3 specific
        "actor_lr": "actor_lr",
        "critic_lr": "critic_lr",
        "policy_noise": "policy_noise",
        "noise_clip": "noise_clip",
        "policy_freq": "policy_freq",
        "initial_noise": "initial_noise",
        "final_noise": "final_noise",
        "ou_theta": "ou_theta",
        "ou_sigma": "ou_sigma",
        
        # SAC specific
        "sac_learning_rate": "sac_learning_rate",
        "sac_ent_coef": "sac_ent_coef",
        "sac_target_entropy": "sac_target_entropy",
        "sac_use_sde": "sac_use_sde",
        "sac_sde_sample_freq": "sac_sde_sample_freq",
        "sac_gradient_steps": "sac_gradient_steps",
        "sac_train_freq": "sac_train_freq",
    }
    
    # Apply Optuna parameters
    for optuna_key, optuna_value in optuna_params.items():
        config_key = param_mapping.get(optuna_key, optuna_key)
        
        if config_key in merged:
            old_value = merged[config_key]
            merged[config_key] = optuna_value
            from_optuna.append((config_key, optuna_value, old_value))
        else:
            # Parameter not in base config, add it anyway
            merged[config_key] = optuna_value
            from_optuna.append((config_key, optuna_value, "NEW"))
    
    # Handle special cases
    # If Optuna specifies use_auto_entropy=False for SAC, we need to use fixed entropy
    if "use_auto_entropy" in optuna_params:
        if not optuna_params["use_auto_entropy"]:
            # Use fixed entropy coefficient from Optuna
            if "sac_ent_coef" in optuna_params:
                merged["sac_ent_coef"] = optuna_params["sac_ent_coef"]
        else:
            merged["sac_ent_coef"] = "auto"
    
    # Update min_action to match max_action (symmetric bounds)
    if "max_action" in optuna_params:
        merged["min_action"] = -optuna_params["max_action"]
    
    # Print parameter comparison
    print(f"\n{'='*70}")
    print(f"CONFIGURATION SUMMARY FOR {model_type}")
    print(f"{'='*70}")
    print(f"\n📊 Parameters FROM OPTUNA ({len(from_optuna)} params):")
    print(f"{'-'*70}")
    for key, new_val, old_val in sorted(from_optuna):
        if isinstance(new_val, float):
            new_str = f"{new_val:.6g}"
        else:
            new_str = str(new_val)
        if isinstance(old_val, float):
            old_str = f"{old_val:.6g}"
        else:
            old_str = str(old_val)
        print(f"  {key:<30} = {new_str:<20} (was: {old_str})")
    
    # List important params still from config
    important_config_params = [
        "mc_train_trajectories", "mc_episode_length", "seed",
        "use_curriculum_learning", "use_volatility_curriculum",
        "use_early_stopping", "reward_type", "action_mode",
        "transaction_cost", "notional", "normalize_data"
    ]
    
    print(f"\n📋 Key parameters FROM CONFIG.PY:")
    print(f"{'-'*70}")
    for key in important_config_params:
        if key in merged and key not in [p[0] for p in from_optuna]:
            val = merged[key]
            if isinstance(val, float):
                val_str = f"{val:.6g}"
            else:
                val_str = str(val)
            print(f"  {key:<30} = {val_str}")
    
    print(f"{'='*70}\n")
    
    return merged


def setup_results_dir(model_type: str):
    """Create results directory with timestamp and model type"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        CONFIG.get("results_dir", "results"), 
        f"run_optuna_{model_type.lower()}_{timestamp}"
    )
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_merged_config(results_dir: str, merged_config: dict, optuna_params: dict):
    """Save merged configuration and optuna params to results directory"""
    # Save full merged config
    config_path = os.path.join(results_dir, "config.json")
    config_save = {}
    for k, v in merged_config.items():
        if isinstance(v, (int, float, str, bool, list, dict, type(None))):
            config_save[k] = v
        else:
            config_save[k] = str(v)
    
    with open(config_path, 'w') as f:
        json.dump(config_save, f, indent=2)
    
    # Save which params came from Optuna
    optuna_path = os.path.join(results_dir, "optuna_params_used.json")
    with open(optuna_path, 'w') as f:
        json.dump(optuna_params, f, indent=2)
    
    print(f"Configuration saved to {config_path}")
    print(f"Optuna params saved to {optuna_path}")


def run_training_with_optuna_params(model_type: str, verbose: bool = True):
    """
    Run training with Optuna-optimized parameters.
    
    Args:
        model_type: "SAC" or "TD3"
        verbose: Print progress
    
    Returns:
        dict: Training results
    """
    # Load Optuna parameters
    optuna_params = load_optuna_params(model_type)
    
    # Merge with base config
    merged_config = merge_configs(CONFIG, optuna_params, model_type)
    
    # Update the global CONFIG in config module
    # This is necessary because run_training.py imports CONFIG
    import config
    for key, value in merged_config.items():
        config.CONFIG[key] = value
    
    # Setup results directory
    results_dir = setup_results_dir(model_type)
    
    # Save configuration
    save_merged_config(results_dir, merged_config, optuna_params)
    
    # Now import and run the training pipeline
    # We need to reload modules after updating CONFIG
    import importlib
    
    # Reload config to ensure changes propagate
    importlib.reload(config)
    
    # Import training function
    # Note: We need to handle the agent import based on model type
    print(f"\n{'='*70}")
    print(f"STARTING {model_type} TRAINING WITH OPTUNA PARAMETERS")
    print(f"{'='*70}")
    print(f"Results will be saved to: {results_dir}")
    print(f"{'='*70}\n")
    
    # Import necessary modules
    import random
    import numpy as np
    import torch
    
    # Set seeds
    seed = merged_config.get("seed", 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"All random seeds set to {seed}")
    
    # Import agent based on model type
    if model_type == "SAC":
        from sac_agent import SACAgent as Agent, device
    elif model_type == "RTD3":
        from rtd3_agent import RTD3Agent as Agent, device
    else:
        from td3_agent import TD3Agent as Agent, device
    
    print(f"Using {model_type} Agent on device: {device}")
    
    # Import data loader and trainer
    from data_loader import create_environments_for_training
    from trainer import TrainingMetrics
    from metrics import (
        evaluate_agent_with_metrics,
        evaluate_benchmark_with_metrics,
        compare_metrics,
        print_metrics_comparison,
        plot_efficient_frontier
    )
    import pandas as pd
    
    # Store results_dir in config for data_loader
    config.CONFIG['current_results_dir'] = results_dir
    
    # =========================================================================
    # CREATE ENVIRONMENTS
    # =========================================================================
    print("\nStep 1: Creating environments...")
    
    envs = create_environments_for_training(verbose=verbose)
    
    train_envs = envs['train_envs']
    test_env = envs.get('test_env')
    test_envs = envs.get('test_envs', [])
    norm_stats = envs['normalization_stats']
    use_windowed_test = merged_config.get('use_windowed_test', True) and len(test_envs) > 1
    
    print(f"\n  Training environments: {len(train_envs)}")
    if use_windowed_test:
        print(f"  Test environments: {len(test_envs)} windows")
    else:
        print(f"  Test environment: {'Ready' if test_env else 'Not available'}")
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    print(f"\nStep 2: Training {model_type} Agent...")
    
    model_filename = f"{model_type.lower()}.pth"
    model_save_path = os.path.join(results_dir, model_filename)
    
    # Import training function from run_training
    from run_training import train_multi_env
    
    if model_type == "RTD3":
        from run_training import train_rtd3
        agent, metrics = train_rtd3(
            train_envs=train_envs,
            verbose=verbose,
            save_path=model_save_path
        )
    else:
        agent, metrics = train_multi_env(
            train_envs=train_envs,
            verbose=verbose
        )
    
    # Save model
    agent.save(model_save_path)
    print(f"\n{model_type} Model saved to: {model_save_path}")
    
    # Save normalization statistics
    norm_stats_path = os.path.join(results_dir, "normalization_stats.json")
    if norm_stats is not None:
        norm_stats_save = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                          for k, v in norm_stats.items()}
        with open(norm_stats_path, 'w') as f:
            json.dump(norm_stats_save, f, indent=2)
        print(f"Normalization stats saved to: {norm_stats_path}")
    
    # Save training metrics
    metrics_path = os.path.join(results_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2, 
                  default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    
    # =========================================================================
    # TEST EVALUATION
    # =========================================================================
    if test_env is None and len(test_envs) == 0:
        print("\n⚠ No test environment available - skipping evaluation")
        return {'agent': agent, 'metrics': metrics, 'results_dir': results_dir}
    
    print(f"\nStep 3: Evaluating on Test Data...")
    
    if use_windowed_test and len(test_envs) > 1:
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE EVALUATION: {len(test_envs)} episodes")
        print(f"{'='*70}")
        
        # Evaluate agent
        print(f"\nStep 3a: Evaluating RL Agent...")
        agent_metrics, agent_data = evaluate_agent_with_metrics(agent, test_envs, verbose=True)
        
        # Evaluate benchmark
        print(f"\nStep 3b: Evaluating Delta Hedging Benchmark...")
        benchmark_metrics, bench_data = evaluate_benchmark_with_metrics(test_envs, verbose=True)
        
        # Add comparison metrics
        agent_metrics = compare_metrics(agent_metrics, benchmark_metrics)
        
        # Print comparison
        print_metrics_comparison(agent_metrics, benchmark_metrics, 
                                title="FINAL COMPARISON: RL Agent vs Delta Hedging")
        
        # Plot efficient frontier
        if merged_config.get("save_plots", True):
            frontier_path = os.path.join(results_dir, "efficient_frontier.png")
            plot_efficient_frontier(agent_metrics, benchmark_metrics, 
                                   save_path=frontier_path, show=False)
        
        # Results
        rl_cumulative_pnl = agent_metrics.total_pnl
        rl_sharpe = agent_metrics.sharpe_ratio
        benchmark_pnl = benchmark_metrics.total_pnl
        benchmark_sharpe = benchmark_metrics.sharpe_ratio
        pnl_improvement = agent_metrics.pnl_improvement
        
        rl_stats = {
            'mean_episode_pnl': agent_metrics.mean_episode_pnl,
            'std_episode_pnl': agent_metrics.std_episode_pnl,
            'total_cumulative_pnl': agent_metrics.total_pnl,
            'mean_sharpe': agent_metrics.sharpe_ratio,
            'mean_action': agent_metrics.mean_hedge_ratio,
            'std_action': agent_metrics.std_hedge_ratio,
        }
        benchmark_results = {
            'mean_episode_pnl': benchmark_metrics.mean_episode_pnl,
            'total_cumulative_pnl': benchmark_metrics.total_pnl,
            'mean_sharpe': benchmark_metrics.sharpe_ratio,
            'mean_delta': benchmark_metrics.mean_hedge_ratio,
        }
        benchmark_df = bench_data if bench_data is not None else pd.DataFrame()
    else:
        # Single episode mode (fallback)
        from trainer import evaluate_agent
        from run_training import run_benchmark_on_env
        
        print("\nEvaluating RL Agent on single test episode...")
        rl_stats = evaluate_agent(agent, test_env, verbose=True)
        rl_cumulative_pnl = np.sum(rl_stats['pnls'])
        rl_sharpe = rl_stats['sharpe_ratio']
        
        print(f"\nRunning Delta Hedging Benchmark...")
        benchmark_results = run_benchmark_on_env(test_env, verbose=True)
        benchmark_pnl = benchmark_results['cumulative_pnl']
        benchmark_sharpe = benchmark_results['sharpe_ratio']
        benchmark_df = benchmark_results['df']
        
        pnl_improvement = rl_cumulative_pnl - benchmark_pnl
        
        agent_metrics = None
        benchmark_metrics = None
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        'mode': envs['mode'],
        'model_type': model_type,
        'used_optuna_params': len(optuna_params) > 0,
        'optuna_params_count': len(optuna_params),
        'num_train_trajectories': len(train_envs),
        'episode_length_train': merged_config.get('mc_episode_length', 30),
        'episode_length_test': merged_config.get('test_episode_length', 30),
        'use_windowed_test': use_windowed_test,
        'num_test_episodes': len(test_envs) if use_windowed_test else 1,
        'rl_agent': {
            'total_pnl': float(rl_cumulative_pnl),
            'sharpe_ratio': float(rl_sharpe),
        },
        'benchmark': {
            'total_pnl': float(benchmark_pnl),
            'sharpe_ratio': float(benchmark_sharpe),
        },
        'improvements': {
            'pnl': float(pnl_improvement),
        },
        'model_path': model_save_path,
        'results_dir': results_dir
    }
    
    # Add comprehensive metrics if available
    if agent_metrics is not None:
        results['rl_agent_metrics'] = agent_metrics.to_dict()
        results['benchmark_metrics'] = benchmark_metrics.to_dict()
        results['improvements']['tc_savings'] = float(agent_metrics.tc_savings)
        results['improvements']['tc_savings_pct'] = float(agent_metrics.tc_savings_pct)
        results['improvements']['information_ratio'] = float(agent_metrics.information_ratio)
    
    if use_windowed_test:
        results['rl_agent']['mean_episode_pnl'] = float(rl_stats['mean_episode_pnl'])
        results['rl_agent']['std_episode_pnl'] = float(rl_stats['std_episode_pnl'])
        results['benchmark']['mean_episode_pnl'] = float(benchmark_results['mean_episode_pnl'])
    
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_dir}")
    print(f"Model saved to: {model_save_path}")
    print(f"{'='*70}\n")
    
    # Save model metadata
    metadata = {
        'model_type': model_type,
        'used_optuna_optimization': len(optuna_params) > 0,
        'optuna_params': optuna_params,
        'merged_config_keys': list(merged_config.keys()),
        'training_complete': True,
        'timestamp': datetime.now().isoformat()
    }
    metadata_path = os.path.join(results_dir, f"{model_type.lower()}_model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train hedging agent with Optuna-optimized parameters"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["SAC", "TD3", "sac", "td3"],
        default=None,
        help="Model type to train (SAC or TD3). If not specified, uses model_type from config.py"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Determine model type
    if args.model:
        model_type = args.model.upper()
    else:
        model_type = CONFIG.get("model_type", "TD3").upper()
    
    if model_type not in ["SAC", "TD3", "RTD3"]:
        print(f"Error: Invalid model type '{model_type}'. Must be SAC or TD3.")
        sys.exit(1)
    
    print(f"\n{'#'*70}")
    print(f"# HEDGING RL - TRAINING WITH OPTUNA PARAMETERS")
    print(f"# Model: {model_type}")
    print(f"{'#'*70}\n")
    
    # Run training
    results = run_training_with_optuna_params(
        model_type=model_type,
        verbose=not args.quiet
    )
    
    print(f"\n✅ Training completed successfully!")
    print(f"   Results directory: {results['results_dir']}")
    
    return results


if __name__ == "__main__":
    main()
