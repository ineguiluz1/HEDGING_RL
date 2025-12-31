import numpy as np
CONFIG = {
    # ENVIRONMENT PARAMETERS
    
    # Financial Parameters
    "transaction_cost": 0.001,          # Transaction cost as fraction of trade value (0.1% = 10 bps)
    "risk_free_rate": 0.02,             # Annual risk-free interest rate (2% = Treasury rate)        
    "notional": 1000,                   # Notional amount for option exposure ($1000)
    
    # Reward Function Configuration
    # Options: "delta_tracking" (PRIMARY), "variance_minimization", "cara_utility", "profit_seeking"
    "reward_type": "delta_tracking",        # Track BS delta as primary objective
    "delta_tracking_weight": 1.0,           # Weight for tracking error (main component)
    "pnl_variance_weight": 0.1,             # Weight for P&L variance (secondary)
    "cara_lambda": 1.0,                     # CARA utility risk parameter (only if reward_type="cara_utility")
    "risk_aversion": 0.01,                  # Only used if reward_type="profit_seeking"
    "reward_scale": 100.0,                  # Scale rewards for stable gradients
    
    # Delta Tracking (used by all reward types except profit_seeking)
    "use_delta_tracking_reward": True,      # Enable tracking penalty for variance_min
    "include_delta_in_state": True,         # Add Black-Scholes delta to observation space
    
    # Volatility Calculation
    "vol_window": 20,                   # Rolling window size for realized volatility calculation
    "initial_vol": 0.2,                 # Default initial volatility estimate for filling (20% annual)
    "use_random_initial_vol": True,     # Whether to use random draw for first volatility value
    "initial_vol_min": 0.02,            # Minimum for random initial volatility (10% annual)
    "initial_vol_max": 0.10,            # Maximum for random initial volatility (60% annual)
    "normalize_vol": True,              # Whether to normalize volatility along with other features
    # Data Frequency Configuration  
    "trading_days_per_year": 252,       # Trading days per year (for metrics calculation)
    "annualization_factor": 252,        # Annualization factor for daily data (sqrt(252) for vol)
    
    # Action Space Configuration
    # Two modes: "absolute" (action = hedge ratio) or "adjustment" (action = delta + adjustment)
    "action_mode": "adjustment",        # "adjustment" is better for tracking delta
    "max_action": 0.3,                  # Maximum adjustment from delta (+/-30%)
    "min_action": -0.3,                 # Minimum adjustment from delta
    "use_action_bounds": True,          # Whether to enforce action bounds
    
    # Data Processing
    "normalize_data": True,             # Whether to normalize input features
    "handle_missing_data": True,        # Whether to handle missing data points
    
    # REINFORCEMENT LEARNING PARAMETERS
    
    # Training Configuration
    "episodes": 1,                      # Number of training episodes (1 = single time series)
    "num_episodes": 100,                # Number of episodes for single-env training
    "num_epochs": 10,                   # Number of epochs for multi-year training
    "update_every": 1,                  # Training updates every N steps
    "warmup_steps": 5000,               # Random actions before training (5000 steps)
    "seed": 101,                        # Random seed for reproducibility
    "training_report_interval": 1,      # Episodes between training progress reports
    
    # Network Architecture
    "hidden_dim": 256,                  # Number of neurons in hidden layers
    "leaky_relu_alpha": 0.05,           # Negative slope for LeakyReLU activation
    
    # TD3 Algorithm Parameters
    "tau": 0.001,                       # Soft update rate for target networks (0.1% per update)
    "gamma": 0.99,                      # Discount factor for future rewards
    "policy_noise": 0.1,                # Standard deviation of target policy smoothing noise
    "noise_clip": 0.2,                  # Clipping range for target policy noise
    "policy_freq": 2,                   # Frequency of policy updates (every N critic updates)
    
    # Optimization
    "actor_lr": 1e-4,                   # Learning rate for actor network (reduced from 3e-3)
    "critic_lr": 1e-4,                  # Learning rate for critic networks (reduced from 3e-3)
    "batch_size": 256,                  # Batch size for experience replay
    "replay_buffer_size": 100000,       # Maximum size of experience replay buffer
    
    # Exploration Strategy (Ornstein-Uhlenbeck Process)
    "initial_noise": 0.8,               # Starting exploration noise level (80% - VERY high for more variation)
    "final_noise": 0.05,                # Final exploration noise level (5%)
    "noise_decay_steps": 150000,        # Steps over which to decay exploration noise (~80% of training)
    "min_noise": 0.05,                  # Minimum noise level (never go below this)
    "ou_theta": 0.15,                   # OU process mean reversion rate
    "ou_sigma": 0.4,                    # OU process volatility parameter (DOUBLED for more exploration)
    
    # =============================================================================
    # DATA CONFIGURATION
    # =============================================================================
    
    # Data Paths
    "data_path": "./data/historical_hedging_data.csv",  # Path to main dataset
    "sp500_data_path": "./data/sp500_data.csv",         # Path to real S&P 500 data for testing
    "results_dir": "results",                      # Directory for saving results
    
    # =============================================================================
    # MONTE CARLO SIMULATION CONFIGURATION (for Training & Validation)
    # =============================================================================
    
    # Monte Carlo Parameters
    "use_montecarlo_training": True,               # Use MC trajectories for training
    "mc_train_trajectories": 6000,                 # Number of MC trajectories for training
    "mc_episode_length": 30,                       # Episode length in trading days (30 = option expiry simulation)
    "mc_steps_per_year": 252,                      # Trading days per year (for annualization calculations)
    "test_start_year": 2004,                       # Start year for test data (real data)
    "test_end_year": 2025,                         # End year for test data (real data)
    
    # =============================================================================
    # CURRICULUM LEARNING CONFIGURATION
    # =============================================================================
    # Train in phases: first neutral drift only, then introduce directional markets
    "use_curriculum_learning": True,               # Enable curriculum learning
    "curriculum_neutral_ratio": 0.4,               # First 40% of trajectories are neutral drift only
    
    # Mixed Market Conditions Training (used after curriculum phase)
    "mc_use_mixed_drift": True,                    # Use mixed bullish/bearish/neutral trajectories
    "mc_drift_distribution": {                     # Distribution of drift scenarios
        "bullish": 0.33,                           # 33% bullish trajectories (μ > 0)
        "neutral": 0.34,                           # 34% neutral trajectories (μ ≈ 0)
        "bearish": 0.33                            # 33% bearish trajectories (μ < 0)
    },
    "mc_drift_ranges": {                           # Drift ranges for each scenario (annual)
        "bullish": (0.05, 0.20),                   # +5% to +20% annual drift
        "neutral": (-0.05, 0.05),                  # -5% to +5% annual drift  
        "bearish": (-0.20, -0.05)                  # -20% to -5% annual drift
    },
    
    # Evaluation Episode Configuration
    "test_episode_length": 30,                      # Episode length for test (30 days = option expiry)
    "use_windowed_test": True,                     # Whether to use 30-day windows in test data
    
    # Data Columns
    "price_column": "mid",                          # Column name for option prices
    "stock_price_column": "active_underlying_price", # Column name for stock prices
    "moneyness_column": "moneyness",                # Column name for moneyness
    "ttm_column": "maturity",                       # Column name for time to maturity
    "datetime_column": "quote_datetime",            # Column name for timestamps
    
    # =============================================================================
    # HYPERPARAMETER TUNING CONFIGURATION
    # =============================================================================
    
    # Tuning Control
    "enable_tuning": False,             # Whether to perform hyperparameter search
    "n_search_iterations": 1,          # Number of random search iterations
    "validation_metric": "pnl_sum",     # Metric for hyperparameter selection
                                        # Options: "pnl_sum", "sharpe_ratio", "cumulative_reward"
    
    # Hyperparameter Search Ranges (used only if enable_tuning=True)
    "hyperparam_ranges": {
        # TD3 Algorithm
        "tau": (0.001, 0.01),
        "gamma": (0.95, 0.999),
        "policy_noise": (0.1, 0.3),
        "noise_clip": (0.2, 0.5),
        "policy_freq": (1, 5),
        
        # Learning Rates (log-uniform sampling)
        "actor_lr": (1e-5, 1e-3),
        "critic_lr": (1e-5, 1e-3),
        "batch_size": (64, 512),
        
        # Exploration
        "initial_noise": (0.2, 0.5),
        "final_noise": (0.01, 0.1),
        "noise_decay_steps": (50000, 200000),
        "ou_theta": (0.1, 0.3),
        "ou_sigma": (0.1, 0.3),
        
        # Environment
        "vol_window": (10, 50),
        "initial_vol": (0.1, 0.4),
        "initial_vol_min": (0.05, 0.15),    # Range for minimum random initial vol
        "initial_vol_max": (0.40, 0.80)    # Range for maximum random initial vol
    },
    
    # =============================================================================
    # EVALUATION AND ANALYSIS CONFIGURATION
    # =============================================================================
    
    # Evaluation Settings
    "export_positions": True,           # Whether to export detailed position data
    "log_q_values": True,               # Whether to log Q-values during evaluation
    "log_q_values_training": False,     # Whether to log Q-values during training (slower)
    "use_consistent_evaluation": True,  # Whether to use consistent evaluation format
    
    # Benchmark Comparison
    "run_benchmark": True,              # Whether to run delta hedging benchmark
    "benchmark_vol_window": 20,         # Volatility window for benchmark (can differ from RL)
    
    # Performance Metrics
    "periods_per_year": 252,            # Trading periods per year for metric calculations
    "calculate_sharpe": True,           # Whether to calculate Sharpe ratio
    "calculate_information_ratio": True, # Whether to calculate Information ratio
    "calculate_max_drawdown": True,     # Whether to calculate maximum drawdown
    
    # Output Configuration
    "save_plots": True,                 # Whether to save performance plots
    "plot_dpi": 300,                    # DPI for saved plots
    "save_detailed_results": True,      # Whether to save detailed CSV results
    "verbose_evaluation": True,         # Whether to print detailed evaluation info
    "verbose_training": True,           # Whether to print detailed training info
    
    # =============================================================================
    # ADVANCED CONFIGURATION
    # =============================================================================
    
    # Model Checkpointing
    "save_model": True,                 # Whether to save trained model
    "model_save_path": "results/trained_model.pth", # Path for saving model
    "save_best_only": True,             # Whether to save only the best model
    
    # Computational Settings
    "use_gpu": True,                    # Whether to use GPU if available
    "num_workers": 1,                   # Number of parallel workers (for future use)
    
    # Debugging
    "debug_mode": False,                # Enable detailed debugging output
    "profile_performance": False,       # Enable performance profiling
    
    # Reproducibility
    "deterministic": True,              # Whether to ensure deterministic results
    "torch_deterministic": True,       # Whether to make PyTorch operations deterministic
}

# =============================================================================
# PARAMETER VALIDATION AND DERIVED PARAMETERS
# =============================================================================

def validate_config(config):
    """Validate configuration parameters and compute derived values."""
    
    # Validate required parameters
    assert config["initial_noise"] >= config["final_noise"], \
        "initial_noise must be >= final_noise"
    assert config["max_action"] > config["min_action"], \
        "max_action must be > min_action"
    assert config["vol_window"] > 0, \
        "vol_window must be positive"
    assert config["initial_vol"] > 0, \
        "initial_vol must be positive"
    assert config["notional"] > 0, \
        "notional must be positive"
    assert 0 < config["tau"] < 1, \
        "tau must be between 0 and 1"
    assert 0 < config["gamma"] <= 1, \
        "gamma must be between 0 and 1"
    
    # Compute derived parameters
    config["action_range"] = config["max_action"] - config["min_action"]
    
    # Legacy parameter for compatibility (not used in MC mode)
    if "train_years" in config:
        config["total_train_years"] = len(config["train_years"])
    else:
        config["total_train_years"] = config.get("mc_train_trajectories", 50)
    
    # Validate hyperparameter ranges if tuning is enabled
    if config.get("enable_tuning", False):
        ranges = config["hyperparam_ranges"]
        for param, (low, high) in ranges.items():
            assert low < high, f"Invalid range for {param}: {low} >= {high}"
    
    return config

# Validate configuration on import
CONFIG = validate_config(CONFIG)

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_data_config():
    """Get data-related configuration parameters."""
    return {
        "data_path": CONFIG["data_path"],
        "sp500_data_path": CONFIG.get("sp500_data_path", "./data/sp500_data.csv"),
        "use_montecarlo_training": CONFIG.get("use_montecarlo_training", True),
        "mc_train_trajectories": CONFIG.get("mc_train_trajectories", 50),
        "mc_val_trajectories": CONFIG.get("mc_val_trajectories", 10),
        "test_start_year": CONFIG.get("test_start_year", 2004),
        "test_end_year": CONFIG.get("test_end_year", 2025),
        "price_column": CONFIG["price_column"],
        "stock_price_column": CONFIG["stock_price_column"],
        "moneyness_column": CONFIG["moneyness_column"],
        "ttm_column": CONFIG["ttm_column"],
        "datetime_column": CONFIG["datetime_column"],
    }

def get_model_config():
    """Get model-related configuration parameters."""
    return {
        "hidden_dim": CONFIG["hidden_dim"],
        "leaky_relu_alpha": CONFIG["leaky_relu_alpha"],
        "tau": CONFIG["tau"],
        "gamma": CONFIG["gamma"],
        "policy_noise": CONFIG["policy_noise"],
        "noise_clip": CONFIG["noise_clip"],
        "policy_freq": CONFIG["policy_freq"],
        "actor_lr": CONFIG["actor_lr"],
        "critic_lr": CONFIG["critic_lr"],
        "batch_size": CONFIG["batch_size"],
        "max_action": CONFIG["max_action"],
        "replay_buffer_size": CONFIG["replay_buffer_size"],
    }

def get_environment_config():
    """Get environment-related configuration parameters."""
    return {
        "transaction_cost": CONFIG["transaction_cost"],
        "risk_free_rate": CONFIG["risk_free_rate"],
        "notional": CONFIG["notional"],
        "vol_window": CONFIG["vol_window"],
        "initial_vol": CONFIG["initial_vol"],
        "use_random_initial_vol": CONFIG["use_random_initial_vol"],
        "initial_vol_min": CONFIG["initial_vol_min"],
        "initial_vol_max": CONFIG["initial_vol_max"],
        "normalize_vol": CONFIG["normalize_vol"],
        "annualization_factor": CONFIG["annualization_factor"],
        "max_action": CONFIG["max_action"],
        "min_action": CONFIG["min_action"],
        "use_action_bounds": CONFIG["use_action_bounds"],
    }

def get_exploration_config():

    return {
        "initial_noise": CONFIG["initial_noise"],
        "final_noise": CONFIG["final_noise"],
        "noise_decay_steps": CONFIG["noise_decay_steps"],
        "min_noise": CONFIG["min_noise"],
        "ou_theta": CONFIG["ou_theta"],
        "ou_sigma": CONFIG["ou_sigma"],
    }