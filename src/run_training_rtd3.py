#!/usr/bin/env python3
"""
Main Training Script for Recurrent TD3 (RTD3) Hedging Agent
===========================================================

This script trains a Recurrent TD3 agent for options hedging using:
- Monte Carlo simulated trajectories for training
- Real S&P 500 daily data for testing
- SAME training flow as SAC/TD3 (curriculum learning + early stopping)

Usage:
    python src/run_training_rtd3.py

Configuration:
    Training: mc_train_trajectories (e.g., 50) synthetic 1-year paths
    Testing: Real S&P 500 data (2004-2025)
    Flow: Streaming single-pass with curriculum phases (no epochs)

Author: Generated for HEDGING_RL project
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG, get_environment_config
from rtd3_agent import RTD3Agent, device
from data_loader import (
    load_hedging_data,
    split_data_by_years,
    create_environment,
    create_environments_for_training,
    get_year_ranges_from_data
)
from trainer import (
    plot_comparison,
    plot_multi_episode_results,
    TrainingMetrics
)
from benchmark import run_benchmark_simple, delta_hedging_simple


def setup_results_dir():
    """Create results directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(CONFIG.get("results_dir", "results"), f"run_rtd3_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_config(results_dir):
    """Save configuration to results directory"""
    config_path = os.path.join(results_dir, "config.json")
    
    # Convert config to JSON-serializable format
    config_save = {}
    for k, v in CONFIG.items():
        if isinstance(v, (int, float, str, bool, list, dict, type(None))):
            config_save[k] = v
        else:
            config_save[k] = str(v)
    
    with open(config_path, 'w') as f:
        json.dump(config_save, f, indent=2)
    
    print(f"Configuration saved to {config_path}")


def run_full_training_pipeline(
    results_dir=None,
    verbose=True
):
    """
    Run the full training pipeline with Monte Carlo trajectories for RTD3.
    """
    if results_dir is None:
        results_dir = setup_results_dir()
    
    print(f"\n{'='*70}")
    print(f"RECURRENT TD3 HEDGING AGENT - FULL TRAINING PIPELINE")
    print(f"{'='*70}")
    print(f"Results directory: {results_dir}")
    print(f"Device: {device}")
    print(f"Training episodes: {CONFIG.get('mc_train_trajectories', 50)} x {CONFIG.get('mc_episode_length', 30)} days")
    print(f"Test data: Real S&P 500 ({CONFIG.get('test_start_year', 2004)}-{CONFIG.get('test_end_year', 2025)})")
    print(f"Test mode: {'Windowed episodes' if CONFIG.get('use_windowed_test', True) else 'Single long episode'}")
    print(f"Training mode: Streaming (same as SAC/TD3 - curriculum + early stopping)")
    print(f"{'='*70}\n")
    
    # Save configuration
    save_config(results_dir)
    
    # Store results_dir in CONFIG so data_loader can access it for plotting
    CONFIG['current_results_dir'] = results_dir
    
    # =========================================================================
    # CREATE ENVIRONMENTS
    # =========================================================================
    print("Step 1: Creating environments...")
    
    envs = create_environments_for_training(verbose=verbose)
    
    train_envs = envs['train_envs']
    test_env = envs.get('test_env')
    test_envs = envs.get('test_envs', [])  # List of 30-day test windows
    norm_stats = envs['normalization_stats']
    use_windowed_test = CONFIG.get('use_windowed_test', True) and len(test_envs) > 1
    
    print(f"\n  Training environments: {len(train_envs)} ({CONFIG.get('mc_episode_length', 30)} days each)")
    if use_windowed_test:
        print(f"  Test environments: {len(test_envs)} ({CONFIG.get('test_episode_length', 30)}-day windows)")
    else:
        print(f"  Test environment: {'Ready' if test_env else 'Not available'}")
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    print(f"\nStep 2: Training RTD3 Agent on {len(train_envs)} trajectories...")
    
    model_save_path = os.path.join(results_dir, "rtd3_model.pth")
    CONFIG["model_save_path"] = model_save_path
    
    # Train using multi-environment approach
    agent, metrics = train_multi_env(
        train_envs=train_envs,
        verbose=verbose
    )
    
    # Save model
    agent.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
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
        json.dump(metrics.to_dict(), f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    
    # =========================================================================
    # TEST EVALUATION
    # =========================================================================
    if test_env is None and len(test_envs) == 0:
        print("\n⚠ No test environment available - skipping evaluation")
        return {'agent': agent, 'metrics': metrics, 'results_dir': results_dir}
    
    print(f"\nStep 3: Evaluating on Test Data...")
    
    if use_windowed_test and len(test_envs) > 1:
        # Multi-episode evaluation (30-day windows)
        print(f"\nEvaluating RTD3 Agent on {len(test_envs)} test episodes ({CONFIG.get('test_episode_length', 30)} days each)...")
        rl_stats = evaluate_agent_multi_episode(agent, test_envs, verbose=True)
        
        rl_cumulative_pnl = rl_stats['total_cumulative_pnl']
        rl_sharpe = rl_stats['mean_sharpe']
        
        # Run benchmark on all test windows
        print(f"\nStep 4: Running Delta Hedging Benchmark on {len(test_envs)} test episodes...")
        benchmark_results = run_benchmark_multi_episode(test_envs, verbose=True)
        
        benchmark_pnl = benchmark_results['total_cumulative_pnl']
        benchmark_reward = benchmark_results['total_cumulative_reward']
        benchmark_sharpe = benchmark_results['mean_sharpe']
        benchmark_df = benchmark_results['aggregated_df']
        
    else:
        # Single long episode evaluation
        print("\nEvaluating RTD3 Agent on single test episode...")
        rl_stats = evaluate_agent(agent, test_env, verbose=True)
        rl_cumulative_pnl = np.sum(rl_stats['pnls'])
        rl_sharpe = rl_stats['sharpe_ratio']
        
        print(f"\nStep 4: Running Delta Hedging Benchmark...")
        benchmark_results = run_benchmark_on_env(test_env, verbose=True)
        
        benchmark_pnl = benchmark_results['cumulative_pnl']
        benchmark_reward = benchmark_results['cumulative_reward']
        benchmark_sharpe = benchmark_results['sharpe_ratio']
        benchmark_df = benchmark_results['df']
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON: RTD3 Agent vs Delta Hedging")
    if use_windowed_test:
        print(f"  (Multi-Episode Evaluation: {len(test_envs)} x {CONFIG.get('test_episode_length', 30)} days)")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'RTD3 Agent':<20} {'Delta Hedge':<20}")
    print(f"{'-'*70}")
    
    if use_windowed_test:
        print(f"{'Mean Episode P&L':<30} {rl_stats['mean_episode_pnl']:<20.4f} {benchmark_results['mean_episode_pnl']:<20.4f}")
        print(f"{'Total P&L (all episodes)':<30} {rl_cumulative_pnl:<20.4f} {benchmark_pnl:<20.4f}")
        print(f"{'Mean Sharpe Ratio':<30} {rl_sharpe:<20.4f} {benchmark_sharpe:<20.4f}")
        print(f"{'Mean Hedge Ratio':<30} {rl_stats['mean_action']:<20.4f} {benchmark_results['mean_delta']:<20.4f}")
    else:
        print(f"{'Total Reward':<30} {rl_stats['total_reward']:<20.4f} {benchmark_reward:<20.4f}")
        print(f"{'Cumulative P&L':<30} {rl_cumulative_pnl:<20.4f} {benchmark_pnl:<20.4f}")
        print(f"{'Sharpe Ratio':<30} {rl_sharpe:<20.4f} {benchmark_sharpe:<20.4f}")
        print(f"{'Mean Action':<30} {rl_stats['mean_action']:<20.4f} {benchmark_df['Delta'].mean():<20.4f}")
    
    print(f"{'='*70}")
    
    # Calculate improvements
    pnl_improvement = rl_cumulative_pnl - benchmark_pnl
    sharpe_improvement = rl_sharpe - benchmark_sharpe
    
    print(f"\nIMPROVEMENTS:")
    print(f"  P&L Improvement: {pnl_improvement:+.4f}")
    print(f"  Sharpe Improvement: {sharpe_improvement:+.4f}")
    
    # Plot results
    if CONFIG.get("save_plots", True):
        if use_windowed_test:
            multi_ep_path = os.path.join(results_dir, "multi_episode_results.png")
            plot_multi_episode_results(rl_stats, benchmark_results, save_path=multi_ep_path)
        else:
            comparison_path = os.path.join(results_dir, "comparison_test.png")
            plot_comparison(rl_stats, benchmark_df, test_env, save_path=comparison_path, 
                           test_year="Test", output_dir=results_dir)
    
    return {'agent': agent, 'metrics': metrics, 'results_dir': results_dir}


def train_multi_env(train_envs, verbose=True):
    """
    Train RTD3 agent on multiple environments (trajectories).
    NOW USES THE SAME FLOW AS SAC/TD3:
    - Curriculum learning (volatility/drift based)
    - Early stopping per phase
    - No epochs (single pass streaming)
    """
    if len(train_envs) == 0:
        raise ValueError("No training environments provided")
    
    # Get dimensions from first environment
    state_dim = train_envs[0].observation_space.shape[0]
    action_dim = train_envs[0].action_space.shape[0]
    
    # Create RTD3 agent
    agent = RTD3Agent(state_dim, action_dim)
    
    # Metrics tracking
    metrics = TrainingMetrics()
    
    total_steps = 0
    n_trajectories = len(train_envs)
    
    # =========================================================================
    # CURRICULUM LEARNING SETUP (same as SAC/TD3)
    # =========================================================================
    use_curriculum = CONFIG.get('use_curriculum_learning', False)
    use_vol_curriculum = CONFIG.get('use_volatility_curriculum', False)
    
    # Define curriculum phases based on how trajectories were generated
    curriculum_phases = []
    
    if use_curriculum or use_vol_curriculum:
        # Calculate phase boundaries
        neutral_ratio = CONFIG.get('curriculum_neutral_ratio', 0.4)
        vol_phases = CONFIG.get('vol_curriculum_phases', {})
        
        if use_vol_curriculum and vol_phases:
            # Volatility curriculum has priority
            low_ratio = vol_phases.get('low', {}).get('ratio', 0.3)
            medium_ratio = vol_phases.get('medium', {}).get('ratio', 0.4)
            high_ratio = vol_phases.get('high', {}).get('ratio', 0.3)
            
            low_end = int(n_trajectories * low_ratio)
            medium_end = int(n_trajectories * (low_ratio + medium_ratio))
            
            curriculum_phases = [
                {'name': 'Low Volatility', 'start': 0, 'end': low_end},
                {'name': 'Medium Volatility', 'start': low_end, 'end': medium_end},
                {'name': 'High Volatility', 'start': medium_end, 'end': n_trajectories}
            ]
        elif use_curriculum:
            # Drift curriculum
            neutral_end = int(n_trajectories * neutral_ratio)
            curriculum_phases = [
                {'name': 'Neutral Drift', 'start': 0, 'end': neutral_end},
                {'name': 'Mixed Drift', 'start': neutral_end, 'end': n_trajectories}
            ]
    
    if not curriculum_phases:
        # No curriculum - single phase
        curriculum_phases = [{'name': 'Full Training', 'start': 0, 'end': n_trajectories}]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING ON {n_trajectories} TRAJECTORIES (Streaming - No Epochs)")
        print(f"{'='*60}")
        print(f"  Curriculum phases: {len(curriculum_phases)}")
        for phase in curriculum_phases:
            print(f"    - {phase['name']}: trajectories {phase['start']}-{phase['end']} ({phase['end']-phase['start']} total)")
    
    # =========================================================================
    # EARLY STOPPING SETUP (same as SAC/TD3)
    # =========================================================================
    use_early_stopping = CONFIG.get('use_early_stopping', False)
    es_patience = CONFIG.get('early_stopping_patience', 500)
    es_min_delta = CONFIG.get('early_stopping_min_delta', 0.001)
    es_window = CONFIG.get('early_stopping_window', 100)
    es_per_phase = CONFIG.get('early_stopping_per_curriculum_phase', True)
    es_min_episodes = CONFIG.get('early_stopping_min_episodes_per_phase', 200)
    
    if use_early_stopping and verbose:
        print(f"\n  Early Stopping enabled:")
        print(f"    Patience: {es_patience} episodes")
        print(f"    Min improvement: {es_min_delta}")
        print(f"    Moving average window: {es_window}")
        print(f"    Per curriculum phase: {es_per_phase}")
    
    # Shuffle environments with dedicated RNG for reproducibility
    shuffle_seed = CONFIG.get("seed", 101)
    shuffle_rng = np.random.default_rng(shuffle_seed)
    
    all_rewards = []
    all_losses = []
    total_episodes_trained = 0
    early_stopped_phases = []
    
    # =========================================================================
    # TRAINING LOOP WITH CURRICULUM PHASES (same as SAC/TD3)
    # =========================================================================
    for phase_idx, phase in enumerate(curriculum_phases):
        phase_name = phase['name']
        phase_start = phase['start']
        phase_end = phase['end']
        phase_size = phase_end - phase_start
        
        if verbose:
            print(f"\n{'-'*60}")
            print(f"PHASE {phase_idx + 1}/{len(curriculum_phases)}: {phase_name}")
            print(f"  Trajectories: {phase_start} to {phase_end} ({phase_size} total)")
            print(f"{'-'*60}")
        
        # Get indices for this phase and shuffle them
        phase_indices = list(range(phase_start, phase_end))
        shuffle_rng.shuffle(phase_indices)
        
        # Early stopping state for this phase
        phase_rewards = []
        best_avg_reward = float('-inf')
        episodes_without_improvement = 0
        phase_early_stopped = False
        
        for i, env_idx in enumerate(phase_indices):
            env = train_envs[env_idx]
            
            # Train one episode on this environment
            episode_reward, episode_steps, episode_losses = train_single_episode(
                agent, env, total_steps
            )
            
            total_steps += episode_steps
            total_episodes_trained += 1
            all_rewards.append(episode_reward)
            all_losses.extend(episode_losses)
            phase_rewards.append(episode_reward)
            
            # Record metrics
            loss = episode_losses[-1] if episode_losses else 0.0
            metrics.add_episode(
                reward=episode_reward,
                pnl=0,
                length=episode_steps,
                actor_loss=0,
                critic_loss=loss,
                noise=agent.current_noise
            )
            
            # Progress reporting
            if verbose and (i + 1) % 10 == 0:
                progress = (i + 1) / phase_size * 100
                recent_rewards = phase_rewards[-10:]
                avg_recent = np.mean(recent_rewards)
                print(f"  [{progress:5.1f}%] Episode {i+1}/{phase_size}: Reward={episode_reward:.2f}, Avg(last 10)={avg_recent:.2f}")
            
            # =========================================================================
            # EARLY STOPPING CHECK (same as SAC/TD3)
            # =========================================================================
            if use_early_stopping and es_per_phase and len(phase_rewards) >= es_min_episodes:
                # Compute moving average over recent episodes
                if len(phase_rewards) >= es_window:
                    current_avg = np.mean(phase_rewards[-es_window:])
                    
                    if current_avg > best_avg_reward + es_min_delta:
                        # Improvement detected
                        best_avg_reward = current_avg
                        episodes_without_improvement = 0
                    else:
                        episodes_without_improvement += 1
                    
                    # Check if patience exceeded
                    if episodes_without_improvement >= es_patience:
                        phase_early_stopped = True
                        early_stopped_phases.append(phase_name)
                        if verbose:
                            print(f"\n  Early stopping triggered for phase '{phase_name}'")
                            print(f"    No improvement for {es_patience} episodes")
                            print(f"    Best avg reward: {best_avg_reward:.4f}")
                            print(f"    Episodes trained in phase: {len(phase_rewards)}/{phase_size}")
                        break
        
        # Phase summary
        if verbose:
            phase_avg = np.mean(phase_rewards) if phase_rewards else 0.0
            print(f"\n  Phase '{phase_name}' completed:")
            print(f"    Episodes trained: {len(phase_rewards)}/{phase_size}")
            print(f"    Average reward: {phase_avg:.4f}")
            if phase_early_stopped:
                print(f"    Status: Early stopped")
            else:
                print(f"    Status: Completed fully")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"  Total episodes trained: {total_episodes_trained}/{n_trajectories}")
        print(f"  Total steps: {total_steps}")
        print(f"  Overall avg reward: {np.mean(all_rewards):.4f}")
        if early_stopped_phases:
            print(f"  Early stopped phases: {', '.join(early_stopped_phases)}")
    
    return agent, metrics


def train_single_episode(agent, env, global_step=0):
    """
    Train agent for a single episode (one trajectory).
    Handles hidden state reset for RTD3.
    """
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        state = reset_result[0]
    else:
        state = reset_result
        
    # Reset hidden state for new episode
    agent.reset_hidden_state()
    
    episode_reward = 0.0
    losses = []
    steps = 0
    done = False
    
    warmup_steps = CONFIG.get("warmup_steps", 1000)
    
    while not done:
        # Select action
        if global_step + steps < warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, add_noise=True)
        
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_result
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, done)
        
        # Perform training step
        if global_step + steps >= warmup_steps:
            loss_info = agent.train_step()
            if loss_info and loss_info[1] is not None:
                losses.append(loss_info[1])
        
        episode_reward += reward
        state = next_state
        steps += 1
    
    return episode_reward, steps, losses


def evaluate_agent(agent, env, verbose=True):
    """
    Evaluate RTD3 agent on a single environment.
    """
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        state = reset_result[0]
    else:
        state = reset_result
        
    # Reset hidden state
    agent.reset_hidden_state()
    
    episode_reward = 0
    episode_pnl = 0
    steps = 0
    actions = []
    rewards = []
    pnls = []
    
    done = False
    while not done:
        # Select action without noise
        action = agent.select_action(state, add_noise=False)
        
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_result
            
        state = next_state
        episode_reward += reward
        episode_pnl += info.get('step_pnl', 0)
        steps += 1
        
        actions.append(action[0] if hasattr(action, '__len__') else action)
        rewards.append(reward)
        pnls.append(info.get('step_pnl', 0))
        
    actions_arr = np.array(actions)
    pnls_arr = np.array(pnls)
    
    stats = {
        'total_reward': episode_reward,
        'total_pnl': episode_pnl,
        'steps': steps,
        'mean_action': np.mean(actions_arr),
        'std_action': np.std(actions_arr),
        'sharpe_ratio': np.mean(pnls_arr) / (np.std(pnls_arr) + 1e-8) * np.sqrt(252),
        'pnls': pnls_arr,
        'actions': actions_arr  # Added for compatibility with multi_episode evaluation
    }
    
    if verbose:
        print(f"Eval Result: Reward={episode_reward:.4f}, P&L={episode_pnl:.4f}, Sharpe={stats['sharpe_ratio']:.4f}")
        
    return stats


def evaluate_agent_multi_episode(agent, test_envs, verbose=True):
    """
    Evaluate RTD3 agent on multiple test episodes.
    Returns same format as trainer.py version for compatibility.
    """
    all_rewards = []
    all_pnls = []
    all_cumulative_pnls = []
    all_sharpes = []
    all_actions = []
    all_tracking_errors = []
    episode_stats_list = []
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MULTI-EPISODE EVALUATION ({len(test_envs)} episodes)")
        print(f"{'='*70}")
        
    for i, env in enumerate(test_envs):
        stats = evaluate_agent(agent, env, verbose=False)
        
        all_rewards.append(stats['total_reward'])
        episode_pnl = np.sum(stats['pnls'])
        all_pnls.append(episode_pnl)
        all_cumulative_pnls.append(episode_pnl)
        all_sharpes.append(stats['sharpe_ratio'])
        all_actions.extend(stats['actions'].tolist())
        episode_stats_list.append(stats)
        
        # Calculate tracking error (deviation from BS delta)
        if hasattr(env, 'bs_delta_raw') and len(env.bs_delta_raw) > 0:
            actions = stats['actions']
            n_steps = min(len(actions), len(env.bs_delta_raw) - 1)
            bs_deltas = env.bs_delta_raw[1:n_steps+1]
            tracking_error = np.mean((actions[:n_steps] - bs_deltas) ** 2)
            all_tracking_errors.append(tracking_error)
        
        if verbose and (i+1) % 50 == 0:
            print(f"  Evaluated {i+1}/{len(test_envs)} episodes...")
    
    # Calculate tracking statistics
    mean_tracking_error = np.mean(all_tracking_errors) if all_tracking_errors else 0.0
    std_tracking_error = np.std(all_tracking_errors) if all_tracking_errors else 0.0
    rmse_tracking = np.sqrt(mean_tracking_error)
            
    # Return same format as trainer.py for compatibility with plot functions
    return {
        # Per-episode means
        'mean_episode_reward': np.mean(all_rewards),
        'std_episode_reward': np.std(all_rewards),
        'mean_episode_pnl': np.mean(all_pnls),
        'std_episode_pnl': np.std(all_pnls),
        'mean_cumulative_pnl': np.mean(all_cumulative_pnls),
        'std_cumulative_pnl': np.std(all_cumulative_pnls),
        'mean_sharpe': np.mean(all_sharpes),
        'std_sharpe': np.std(all_sharpes),
        
        # Aggregated totals
        'total_reward': sum(all_rewards),
        'total_pnl': sum(all_pnls),
        'total_cumulative_pnl': sum(all_pnls),
        
        # Action statistics
        'mean_action': np.mean(all_actions),
        'std_action': np.std(all_actions),
        'min_action': np.min(all_actions),
        'max_action': np.max(all_actions),
        
        # Tracking error metrics
        'mean_tracking_error': mean_tracking_error,
        'std_tracking_error': std_tracking_error,
        'rmse_tracking': rmse_tracking,
        
        # Number of episodes
        'n_episodes': len(test_envs),
        
        # All individual episode data (needed for plotting)
        'episode_stats': episode_stats_list,
        'all_rewards': all_rewards,
        'all_pnls': all_pnls,
        'all_cumulative_pnls': all_cumulative_pnls,
        'all_sharpes': all_sharpes,
        'all_actions': np.array(all_actions),
        'all_tracking_errors': all_tracking_errors
    }


def run_benchmark_multi_episode(test_envs, verbose=True):
    """
    Run delta hedging benchmark on multiple test episodes.
    Returns same format as run_training.py version for compatibility.
    """
    all_pnls = []
    all_rewards = []
    all_sharpes = []
    all_deltas = []
    all_episode_results = []
    
    if verbose:
        print(f"\nRunning benchmark on {len(test_envs)} episodes...")
    
    for i, env in enumerate(test_envs):
        results = run_benchmark_on_env(env, verbose=False)
        
        all_pnls.append(results['cumulative_pnl'])
        all_rewards.append(results['cumulative_reward'])
        all_sharpes.append(results['sharpe_ratio'])
        all_deltas.extend(results['df']['Delta'].tolist())
        all_episode_results.append(results)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Benchmark evaluated {i + 1}/{len(test_envs)} episodes...")
    
    # Aggregate - return same format as run_training.py
    aggregated = {
        'mean_episode_pnl': np.mean(all_pnls),
        'std_episode_pnl': np.std(all_pnls),
        'total_cumulative_pnl': sum(all_pnls),
        'total_cumulative_reward': sum(all_rewards),
        'mean_sharpe': np.mean(all_sharpes),
        'std_sharpe': np.std(all_sharpes),
        'mean_delta': np.mean(all_deltas),
        'std_delta': np.std(all_deltas),
        'n_episodes': len(test_envs),
        'all_pnls': all_pnls,
        'all_rewards': all_rewards,
        'all_sharpes': all_sharpes,
        'episode_results': all_episode_results,
        'aggregated_df': pd.concat([r['df'] for r in all_episode_results], ignore_index=True)
    }
    
    if verbose:
        print(f"\nBenchmark Multi-Episode Results:")
        print(f"  Episodes: {len(test_envs)}")
        print(f"  Mean Episode P&L: {aggregated['mean_episode_pnl']:.4f} ± {aggregated['std_episode_pnl']:.4f}")
        print(f"  Total P&L: {aggregated['total_cumulative_pnl']:.4f}")
        print(f"  Mean Sharpe: {aggregated['mean_sharpe']:.4f} ± {aggregated['std_sharpe']:.4f}")
        print(f"  Mean Delta: {aggregated['mean_delta']:.4f}")
    
    return aggregated


def run_benchmark_on_env(env, verbose=True):
    """
    Run delta hedging benchmark on an environment.
    """
    from scipy.stats import norm
    
    # Get RAW (unnormalized) data from environment
    option_prices = env.option_prices_raw
    stock_prices = env.stock_prices_raw
    moneyness = env.moneyness_raw
    ttm = env.ttm_raw
    
    # Get config
    r = CONFIG.get("risk_free_rate", 0.02)
    vol = CONFIG.get("mc_volatility", 0.20)
    tc = CONFIG.get("transaction_cost", 0.001)
    notional = CONFIG.get("notional", 1000)
    xi = CONFIG.get("risk_aversion", 0.01)
    
    pnl_list = []
    reward_list = []
    delta_list = []
    
    prev_position = 0.0
    
    for i in range(len(stock_prices)):
        S_now = stock_prices[i]
        O_now = option_prices[i]
        K = S_now / moneyness[i] if moneyness[i] > 0 else S_now
        T = max(ttm[i], 1e-6)
        
        # Black-Scholes delta
        d1 = (np.log(S_now/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T) + 1e-8)
        delta = norm.cdf(d1)
        delta = np.clip(delta, 0, 1)
        
        # P&L calculation
        if i > 0:
            S_prev = stock_prices[i-1]
            O_prev = option_prices[i-1]
            
            HO_t = -1
            option_component = HO_t * (O_now - O_prev) / notional
            hedge_component = (prev_position / notional) * (S_now / S_prev - 1)
            
            hedge_adjustment = (delta * notional - prev_position) / S_now
            transaction_component = tc * S_now * abs(hedge_adjustment) / notional
            
            step_pnl = option_component + hedge_component - transaction_component
            pnl_list.append(step_pnl)
            
            reward = step_pnl - xi * abs(step_pnl)
            reward_list.append(reward)
        
        delta_list.append(delta)
        prev_position = delta * notional
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Stock_Price': stock_prices,
        'Option_Price': option_prices,
        'Delta': delta_list,
        'PnL': [0] + pnl_list,
        'Cumulative PnL': np.cumsum([0] + pnl_list),
        'Reward': [0] + reward_list,
        'Cumulative Reward': np.cumsum([0] + reward_list)
    })
    
    cum_pnl = results_df['Cumulative PnL'].iloc[-1]
    cum_reward = results_df['Cumulative Reward'].iloc[-1]
    sharpe = results_df['PnL'].mean() / (results_df['PnL'].std() + 1e-8) * np.sqrt(252)
    
    if verbose:
        print(f"Benchmark: P&L={cum_pnl:.4f}, Sharpe={sharpe:.4f}")
    
    return {
        'df': results_df,
        'cumulative_pnl': cum_pnl,
        'cumulative_reward': cum_reward,
        'sharpe_ratio': sharpe
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train RTD3 agent for options hedging')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    args = parser.parse_args()
    
    run_full_training_pipeline(verbose=not args.quiet)


if __name__ == "__main__":
    main()