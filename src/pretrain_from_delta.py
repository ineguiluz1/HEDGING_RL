#!/usr/bin/env python3
"""
Pre-train RL Agent from Delta Hedging Strategy
===============================================

This module implements behavioral cloning to initialize the RL agent's
policy network with delta hedging behavior. This significantly reduces
seed dependency because the network starts from a good policy instead
of random weights.

Benefits:
- Faster convergence (starts from good policy)
- More stable learning (less exploration needed)
- Better final performance (refined from expert baseline)
- Reduced seed dependency (initialization matters less)

Usage:
    from pretrain_from_delta import pretrain_agent_from_delta
    
    agent = TD3Agent(state_dim, action_dim)
    agent = pretrain_agent_from_delta(agent, train_envs, epochs=10)
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from config import CONFIG
from hedging_env import calculate_bs_delta


def collect_expert_demonstrations(envs, max_samples=10000, verbose=True):
    """
    Collect expert demonstrations using delta hedging on training environments.
    
    Args:
        envs: List of training environments
        max_samples: Maximum number of state-action pairs to collect
        verbose: Print progress
    
    Returns:
        tuple: (states, actions) arrays for supervised learning
    """
    states = []
    actions = []
    
    # Calculate how many environments we need to sample from
    # If we have more envs than max_samples, sample from subset
    # If we have fewer envs than max_samples, sample multiple steps per env
    envs_to_sample = min(len(envs), max_samples // 30 + 1)  # At least 30 steps per env
    samples_per_env = max(1, max_samples // envs_to_sample)  # At least 1 sample per env
    
    if verbose:
        print(f"\nCollecting expert demonstrations (Delta Hedging)...")
        print(f"  Target samples: {max_samples}")
        print(f"  Environments available: {len(envs)}")
        print(f"  Environments to sample: {envs_to_sample}")
        print(f"  Samples per environment: {samples_per_env}")
    
    for env_idx, env in enumerate(envs[:envs_to_sample]):
        # Stop if we already have enough samples
        if len(states) >= max_samples:
            break
            
        # Handle both old-style (obs only) and new-style (obs, info) reset
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state, info = reset_result
        else:
            state = reset_result
        
        done = False
        episode_samples = 0
        
        while not done and episode_samples < samples_per_env:
            # Get current market state from environment arrays
            step_idx = env.current_step
            
            # Calculate BS delta as expert action
            moneyness = env.moneyness_raw[step_idx]
            ttm = env.ttm_raw[step_idx]
            vol = env.realized_vol_raw[step_idx] if hasattr(env, 'realized_vol_raw') else 0.2
            r = CONFIG.get('risk_free_rate', 0.02)
            
            expert_delta = calculate_bs_delta(moneyness, ttm, vol, r)
            
            # In adjustment mode, action is adjustment from delta
            action_mode = CONFIG.get('action_mode', 'adjustment')
            if action_mode == 'adjustment':
                # Expert says: stay at delta (no adjustment)
                expert_action = np.array([0.0])  # Zero adjustment = follow delta
            else:
                # Expert says: set hedge ratio to delta
                expert_action = np.array([expert_delta])
            
            states.append(state.copy())
            actions.append(expert_action.copy())
            
            # Step environment - handle both 4-tuple and 5-tuple returns
            step_result = env.step(expert_action)
            if len(step_result) == 5:
                state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                state, reward, done, info = step_result
            
            episode_samples += 1
        
        if verbose and (env_idx + 1) % 50 == 0:
            print(f"  Collected from {env_idx + 1}/{len(envs)} envs ({len(states)} samples)")
    
    states = np.array(states)
    actions = np.array(actions)
    
    if verbose:
        print(f"\n  ✓ Collected {len(states)} expert demonstrations")
        print(f"  State shape: {states.shape}")
        print(f"  Action shape: {actions.shape}")
        print(f"  Action stats: mean={actions.mean():.4f}, std={actions.std():.4f}")
    
    return states, actions


def pretrain_agent_from_delta(
    agent,
    train_envs,
    epochs=10,
    batch_size=256,
    learning_rate=1e-3,
    max_samples=10000,
    verbose=True
):
    """
    Pre-train agent's actor network using behavioral cloning from delta hedging.
    
    Args:
        agent: TD3Agent to pre-train
        train_envs: List of training environments
        epochs: Number of pre-training epochs
        batch_size: Batch size for supervised learning
        learning_rate: Learning rate for pre-training
        max_samples: Maximum expert demonstrations to collect
        verbose: Print progress
    
    Returns:
        agent: Pre-trained agent (modified in-place)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"PRE-TRAINING AGENT FROM DELTA HEDGING")
        print(f"{'='*70}")
        print(f"This initializes the policy network with delta hedging behavior")
        print(f"Benefits: Faster convergence, more stable, less seed-dependent")
        print(f"{'='*70}")
    
    # Step 1: Collect expert demonstrations
    states, actions = collect_expert_demonstrations(
        train_envs, 
        max_samples=max_samples,
        verbose=verbose
    )
    
    # Step 2: Create optimizer for actor network only
    if hasattr(agent, 'model'):
        # SB3 TD3 agent
        actor = agent.model.actor
    else:
        # Custom TD3 agent
        actor = agent.actor
    
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Step 3: Supervised learning
    if verbose:
        print(f"\nPre-training actor network...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
    
    device = next(actor.parameters()).device
    
    dataset_size = len(states)
    losses = []
    
    # Use seeded RNG for reproducibility
    seed = CONFIG.get("seed", 101)
    rng = np.random.default_rng(seed)
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Shuffle data with seeded RNG
        indices = rng.permutation(dataset_size)
        
        # Mini-batch training
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            
            # Get batch
            batch_states = torch.FloatTensor(states[batch_indices]).to(device)
            batch_actions = torch.FloatTensor(actions[batch_indices]).to(device)
            
            # Forward pass
            predicted_actions = actor(batch_states)
            
            # Compute loss
            loss = criterion(predicted_actions, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
    
    if verbose:
        print(f"\n  ✓ Pre-training complete")
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Loss reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
        print(f"\n  Agent is now initialized with delta hedging behavior")
        print(f"  RL training will refine this policy for better performance")
    
    return agent


def pretrain_enabled():
    """Check if pre-training is enabled in config."""
    return CONFIG.get('use_pretraining', False)


def get_pretraining_config():
    """Get pre-training configuration."""
    return {
        'epochs': CONFIG.get('pretrain_epochs', 10),
        'batch_size': CONFIG.get('pretrain_batch_size', 256),
        'learning_rate': CONFIG.get('pretrain_lr', 1e-3),
        'max_samples': CONFIG.get('pretrain_max_samples', 10000)
    }


if __name__ == '__main__':
    """Test pre-training functionality."""
    import sys
    sys.path.insert(0, '.')
    
    from td3_agent import TD3Agent
    from data_loader import create_environments_for_training
    
    print("Testing pre-training module...")
    
    # Create environments
    envs_dict = create_environments_for_training(verbose=True)
    train_envs = envs_dict['train_envs'][:100]  # Use first 100 for testing
    
    # Create agent
    state_dim = train_envs[0].observation_space.shape[0]
    action_dim = train_envs[0].action_space.shape[0]
    agent = TD3Agent(state_dim, action_dim)
    agent.set_env(train_envs[0])
    
    # Pre-train
    agent = pretrain_agent_from_delta(
        agent,
        train_envs,
        epochs=5,
        max_samples=5000,
        verbose=True
    )
    
    print("\n✓ Pre-training test successful!")
