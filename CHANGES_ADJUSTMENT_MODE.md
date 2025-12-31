# Changes: Adjustment Mode for Delta Tracking

## Problem
The TD3 agent was collapsing to a constant hedge ratio (1.0) regardless of the Black-Scholes delta.
This happened because standard actor-critic algorithms tend to find a constant action that minimizes 
average loss across all states, rather than learning a conditional policy.

## Root Cause Analysis
Through experimentation, we found that:
1. TD3, SAC, and PPO all collapse to constant actions when learning to "copy" an input
2. The reward structure r = -|action - delta| doesn't provide enough gradient signal
3. With 200k+ training steps, agents still output constant values

## Solution: Adjustment Mode
Instead of learning `action = hedge_ratio` directly, we reformulate:
- **Action**: adjustment from delta, in range [-0.3, 0.3]
- **Actual hedge**: delta + adjustment
- **Optimal action**: 0 (follow delta exactly)

This works because:
1. The optimal action (0) is a fixed target, not state-dependent
2. RL algorithms can easily learn to output 0
3. Any non-zero adjustment gets penalized

## Implementation Changes

### config.py
```python
# Action Space Configuration
"action_mode": "adjustment",        # "adjustment" or "absolute"
"max_action": 0.3,                  # Maximum adjustment from delta
"min_action": -0.3,                 # Minimum adjustment from delta
```

### hedging_env.py
```python
# In step():
if self.action_mode == "adjustment":
    adjustment = action[0]
    target_hedge_ratio = np.clip(bs_delta + adjustment, 0.0, 1.0)
else:
    target_hedge_ratio = action[0]

# In reward calculation:
if self.action_mode == "adjustment":
    absolute_adjustment = abs(action[0])
    reward = -tracking_weight * absolute_adjustment
    reward -= pnl_weight * (step_pnl ** 2)
```

### trainer.py
- Modified evaluate_agent() to convert raw actions to actual hedge ratios
- Store both raw_actions (adjustments) and actions (hedge ratios)

### td3_agent.py
- Added proper warmup_steps from config (5000 steps of random exploration)

## Results Comparison

| Metric | Before (absolute) | After (adjustment) |
|--------|-------------------|--------------------|
| Mean Hedge Ratio | 1.0000 (constant!) | 0.7218 (dynamic) |
| Std Hedge Ratio | 0.0000 | 0.2905 |
| Action Range | [0.9997, 1.0] | [0.0, 1.0] |
| RMSE Tracking | 0.4899 | **0.1665** |
| Total P&L | +0.22 | -1.29 |
| Benchmark P&L | -1.28 | -1.28 |

**Key improvements:**
- 66% reduction in tracking error RMSE (0.49 → 0.17)
- Agent now uses full action range dynamically
- P&L now matches benchmark (was artificially high before due to directional bet)

## Why This Works
The original problem was that the network had to learn:
- Input: [opt_price, stock_price, ttm, moneyness, vol, position, log_return, delta]
- Output: hedge ratio that equals delta

This requires the network to essentially "pass through" the delta input, which is not 
what neural networks are optimized for. They tend to find averages instead.

With adjustment mode:
- Input: same features including delta
- Output: adjustment (ideally 0)

The network just needs to learn to output 0, which is trivial. The delta-following 
behavior is encoded in the environment's step() function, not learned.

## Date: 2024-12-31
