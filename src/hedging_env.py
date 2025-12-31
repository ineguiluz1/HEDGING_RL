import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import norm
from config import CONFIG, get_environment_config
from volatility_utils import calculate_realized_volatility, fill_initial_volatility


def calculate_bs_delta(moneyness, ttm, sigma, r=0.02):
    """
    Calculate Black-Scholes delta for a call option given moneyness.
    
    Args:
        moneyness: S/K ratio
        ttm: Time to maturity in years
        sigma: Volatility
        r: Risk-free rate
    
    Returns:
        Delta value in [0, 1]
    """
    # Handle edge cases
    if ttm <= 1/365.25:  # Less than 1 day
        return 1.0 if moneyness > 1 else 0.0
    if sigma <= 0.001 or np.isnan(sigma) or np.isnan(moneyness):
        return 1.0 if moneyness > 1 else 0.5
    
    try:
        d1 = (np.log(moneyness) + (r + 0.5 * sigma**2) * ttm) / (sigma * np.sqrt(ttm))
        return float(norm.cdf(d1))
    except:
        return 0.5


def calculate_bs_greeks(S, K, ttm, sigma, r=0.02):
    """
    Calculate Black-Scholes Greeks for a call option.
    
    Args:
        S: Stock price
        K: Strike price  
        ttm: Time to maturity in years
        sigma: Volatility
        r: Risk-free rate
    
    Returns:
        dict: Dictionary with delta, gamma, vega, vanna
    """
    # Handle edge cases
    if ttm <= 1/365.25:  # Less than 1 day
        moneyness = S / K if K > 0 else 1.0
        return {
            'delta': 1.0 if moneyness > 1 else 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'vanna': 0.0
        }
    if sigma <= 0.001 or np.isnan(sigma) or K <= 0:
        return {
            'delta': 0.5,
            'gamma': 0.0,
            'vega': 0.0,
            'vanna': 0.0
        }
    
    try:
        sqrt_t = np.sqrt(ttm)
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * ttm) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        
        # Standard normal PDF at d1
        n_d1 = norm.pdf(d1)
        
        # Delta: N(d1)
        delta = float(norm.cdf(d1))
        
        # Gamma: n(d1) / (S * sigma * sqrt(T))
        # Measures how fast delta changes with stock price
        gamma = n_d1 / (S * sigma * sqrt_t)
        
        # Vega: S * n(d1) * sqrt(T) / 100
        # Sensitivity to volatility (divided by 100 for 1% vol change)
        vega = S * n_d1 * sqrt_t / 100.0
        
        # Vanna: -n(d1) * d2 / sigma
        # Sensitivity of delta to volatility (cross-Greek)
        # Also equals: d(vega)/d(S)
        vanna = -n_d1 * d2 / sigma
        
        return {
            'delta': delta,
            'gamma': float(gamma),
            'vega': float(vega),
            'vanna': float(vanna)
        }
    except:
        return {
            'delta': 0.5,
            'gamma': 0.0,
            'vega': 0.0,
            'vanna': 0.0
        }


class HedgingEnv(gym.Env): 
    def __init__(
        self,
        option_prices,
        stock_prices,
        moneyness,
        ttm,
        timestamps,
        vol_window=None,
        initial_vol=None,
        normalize=None,
        use_action_bounds=None,
        action_low=None,
        action_high=None,
        normalization_stats=None,
        notional=None,
        verbose=None
    ):
        super().__init__()

        # Get environment configuration
        env_config = get_environment_config()
        
        # Use config defaults if parameters not provided
        if vol_window is None:
            vol_window = env_config["vol_window"]
        if initial_vol is None:
            initial_vol = env_config["initial_vol"]
        if normalize is None:
            normalize = CONFIG["normalize_data"]
        if use_action_bounds is None:
            use_action_bounds = env_config["use_action_bounds"]
        if action_low is None:
            action_low = env_config["min_action"]
        if action_high is None:
            action_high = env_config["max_action"]
        if notional is None:
            notional = env_config["notional"]
        if verbose is None:
            verbose = CONFIG.get("verbose_evaluation", True)

        self.verbose = verbose

        assert len(option_prices) == len(stock_prices) == len(moneyness) == len(ttm), "Data length mismatch"
        self.timestamps = timestamps
        self.notional = notional 

        self.cumulative_reward = 0.0
        self.reward_history_detailed = []
        self.vol_window = vol_window
        self.initial_vol = initial_vol

        self._normalize_flag = normalize
        self.use_action_bounds = use_action_bounds
        self.action_low = action_low
        self.action_high = action_high
        
        # Delta tracking configuration
        self.include_delta_in_state = CONFIG.get("include_delta_in_state", True)
        self.delta_tracking_weight = CONFIG.get("delta_tracking_weight", 0.5)
        
        # Greeks in state (for better hedging decisions)
        self.include_greeks_in_state = CONFIG.get("include_greeks_in_state", True)
        
        # Action mode: "absolute" (action = hedge ratio) or "adjustment" (action = delta + adjustment)
        self.action_mode = CONFIG.get("action_mode", "adjustment")

        self.option_prices_raw = option_prices
        self.stock_prices_raw = stock_prices
        self.moneyness_raw = moneyness
        self.ttm_raw = ttm
        self.realized_vol_raw = self._calculate_realized_volatility()
        
        # Calculate Black-Scholes delta for each timestep (always raw, not normalized)
        self.bs_delta_raw = self._calculate_bs_delta_series()
        
        # Calculate Greeks series (Gamma, Vega, Vanna) for enhanced state
        if self.include_greeks_in_state:
            self._calculate_greeks_series()
        
        # Calculate implied volatility proxy (using MC vol as "implied")
        # Vol spread = realized_vol - implied_vol (positive = gamma costs higher)
        self.implied_vol = CONFIG.get("mc_volatility", 0.20)  # Use MC vol as proxy for implied

        if normalization_stats is None:
            self._compute_normalization_stats()
        else:
            self.opt_mean = normalization_stats["opt_mean"]
            self.opt_std = normalization_stats["opt_std"]
            self.stock_mean = normalization_stats["stock_mean"]
            self.stock_std = normalization_stats["stock_std"]
            self.moneyness_mean = normalization_stats["moneyness_mean"]
            self.moneyness_std = normalization_stats["moneyness_std"]
            self.ttm_mean = normalization_stats["ttm_mean"]
            self.ttm_std = normalization_stats["ttm_std"]
            self.vol_mean = normalization_stats.get("vol_mean", CONFIG["initial_vol"])
            self.vol_std = normalization_stats.get("vol_std", 0.1)

        if self._normalize_flag:
            self.option_prices = self._normalize(self.option_prices_raw, self.opt_mean, self.opt_std)
            self.stock_prices = self._normalize(self.stock_prices_raw, self.stock_mean, self.stock_std)
            self.moneyness = self._normalize(self.moneyness_raw, self.moneyness_mean, self.moneyness_std)
            self.ttm = self._normalize(self.ttm_raw, self.ttm_mean, self.ttm_std)
            self.realized_vol = self._normalize(self.realized_vol_raw, self.vol_mean, self.vol_std)
        else:
            self.option_prices = self.option_prices_raw
            self.stock_prices = self.stock_prices_raw
            self.moneyness = self.moneyness_raw
            self.ttm = self.ttm_raw
            self.realized_vol = self.realized_vol_raw


        # Use config parameters for financial calculations
        self.transaction_cost = env_config["transaction_cost"]
        self.discount_rate = env_config["risk_free_rate"]

        self.current_step = 1
        self.position = 0.0
        self.cash_account = 0.0

        self.action_history = []
        self.pnl_history = []
        self.reward_history = []
        self.timestamp_history = []

        # Observation space dimensions:
        # Base: [option_price, stock_price, ttm, log_moneyness, realized_vol, position, log_return] = 7
        # + delta: [bs_delta] = +1
        # + greeks: [gamma, vega, vanna, vol_spread] = +4
        obs_dim = 7
        if self.include_delta_in_state:
            obs_dim += 1  # bs_delta
        if self.include_greeks_in_state:
            obs_dim += 4  # gamma, vega, vanna, vol_spread
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        if self.use_action_bounds:
            self.action_space = spaces.Box(
                low=self.action_low, high=self.action_high, shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)


        if self.verbose:
            print(f"\n{'='*70}")
            print(f"HEDGING ENVIRONMENT INITIALIZED")
            print(f"{'='*70}")
            print(f"Data Length: {len(option_prices)}")
            print(f"Notional: {self.notional}")
            print(f"Transaction Cost: {self.transaction_cost}")
            print(f"Risk-Free Rate: {self.discount_rate}")
            print(f"Risk Aversion (ξ): {CONFIG.get('risk_aversion', 0.1)}")
            print(f"Action Mode: {self.action_mode}")
            print(f"Delta Tracking Weight: {self.delta_tracking_weight}")
            print(f"Include Delta in State: {self.include_delta_in_state}")
            print(f"Normalization: {self._normalize_flag}")
            print(f"Action Bounds: {self.use_action_bounds} [{self.action_low}, {self.action_high}]")
            print(f"Volatility Window: {self.vol_window}")
            print(f"Initial Volatility: {self.initial_vol}")
            print(f"Realized Vol Range: {self.realized_vol_raw.min():.4f} - {self.realized_vol_raw.max():.4f}")
            print(f"BS Delta Range: {self.bs_delta_raw.min():.4f} - {self.bs_delta_raw.max():.4f}")
            print(f"Include Greeks: {self.include_greeks_in_state}")
            if self.include_greeks_in_state:
                print(f"  Gamma Range: {self.gamma_raw.min():.6f} - {self.gamma_raw.max():.6f}")
                print(f"  Vega Range: {self.vega_raw.min():.4f} - {self.vega_raw.max():.4f}")
                print(f"  Vanna Range: {self.vanna_raw.min():.6f} - {self.vanna_raw.max():.6f}")
            print(f"Observation Dim: {obs_dim}")
            print(f"{'='*70}\n")

    def _calculate_realized_volatility(self):
        realized_vol = calculate_realized_volatility(
            prices=pd.Series(self.stock_prices_raw),
            window=self.vol_window,
            min_periods=1,
            annualization_factor=CONFIG["annualization_factor"]
        )

        filled_vol = fill_initial_volatility(realized_vol, self.initial_vol)
        
        return filled_vol.values
    
    def _calculate_bs_delta_series(self):
        """Calculate Black-Scholes delta for each timestep."""
        deltas = np.zeros(len(self.moneyness_raw))
        for i in range(len(deltas)):
            deltas[i] = calculate_bs_delta(
                moneyness=self.moneyness_raw[i],
                ttm=self.ttm_raw[i],
                sigma=self.realized_vol_raw[i],
                r=CONFIG.get("risk_free_rate", 0.02)
            )
        return deltas
    
    def _calculate_greeks_series(self):
        """
        Calculate Greeks (Gamma, Vega, Vanna) for each timestep.
        These give the agent "forward-looking" information about how delta will change.
        """
        n = len(self.moneyness_raw)
        self.gamma_raw = np.zeros(n)
        self.vega_raw = np.zeros(n)
        self.vanna_raw = np.zeros(n)
        
        r = CONFIG.get("risk_free_rate", 0.02)
        
        for i in range(n):
            S = self.stock_prices_raw[i]
            moneyness = self.moneyness_raw[i]
            K = S / moneyness if moneyness > 0 else S
            ttm = self.ttm_raw[i]
            sigma = self.realized_vol_raw[i]
            
            greeks = calculate_bs_greeks(S, K, ttm, sigma, r)
            
            self.gamma_raw[i] = greeks['gamma']
            self.vega_raw[i] = greeks['vega']
            self.vanna_raw[i] = greeks['vanna']
        
        # Normalize Greeks for neural network stability
        # Gamma: typically very small, scale up
        self.gamma_mean = np.mean(self.gamma_raw)
        self.gamma_std = np.std(self.gamma_raw) + 1e-8
        
        # Vega: can vary widely
        self.vega_mean = np.mean(self.vega_raw)
        self.vega_std = np.std(self.vega_raw) + 1e-8
        
        # Vanna: can be positive or negative
        self.vanna_mean = np.mean(self.vanna_raw)
        self.vanna_std = np.std(self.vanna_raw) + 1e-8

    def _compute_normalization_stats(self):
        self.opt_mean = np.mean(self.option_prices_raw)
        self.opt_std = np.std(self.option_prices_raw)
        self.stock_mean = np.mean(self.stock_prices_raw)
        self.stock_std = np.std(self.stock_prices_raw)
        self.moneyness_mean = np.mean(self.moneyness_raw)
        self.moneyness_std = np.std(self.moneyness_raw)
        self.ttm_mean = np.mean(self.ttm_raw)
        self.ttm_std = np.std(self.ttm_raw)
        self.vol_mean = np.mean(self.realized_vol_raw)
        self.vol_std = np.std(self.realized_vol_raw)

    def _normalize(self, data, mean, std):
        epsilon = 1e-8
        small_shift = 1e-6
        normalized = (data - mean) / (std + epsilon)
        normalized = np.where(normalized == 0.0, small_shift, normalized)
        return normalized

    def reset(self):
        self.current_step = 1
        self.position = 0.0
        self.cash_account = 0.0
        self.action_history = []
        self.pnl_history = []
        self.reward_history = []
        self.timestamp_history = []
        self.cumulative_reward = 0.0
        self.discounted_cumulative_reward = 0.0
        # Initialize initial timestamp for discounting
        self.initial_timestamp = self.timestamps[1]  # Set initial timestamp here
        self.reward_history_detailed = []
        self.reset_step_data_history()
        
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"ENVIRONMENT RESET")
            print(f"{'='*50}")
            print(f"Initial state: Step={self.current_step}, Position={self.position:.4f}, Cash={self.cash_account:.4f}")
            
        return self._get_observation()

    def _get_observation(self):
        pos = (self.position / self.notional) if self._normalize_flag else self.position
        
        # Calculate log-return (more stable for neural networks)
        if self.current_step > 0:
            S_now = self.stock_prices_raw[self.current_step]
            S_prev = self.stock_prices_raw[self.current_step - 1]
            log_return = np.log(S_now / S_prev) if S_prev > 0 else 0.0
        else:
            log_return = 0.0
        
        # Calculate log-moneyness (more representative for options)
        log_moneyness = np.log(self.moneyness_raw[self.current_step]) if self.moneyness_raw[self.current_step] > 0 else 0.0
        
        # Enhanced observation space:
        # [option_price, stock_price, ttm, log_moneyness, realized_vol, position, log_return]
        base_obs = [
            self.option_prices[self.current_step],
            self.stock_prices[self.current_step],
            self.ttm[self.current_step], 
            log_moneyness,  # Changed from linear moneyness to log-moneyness
            self.realized_vol[self.current_step],
            pos,
            log_return  # NEW: Log-return of underlying
        ]
        
        # Add Black-Scholes delta if configured (not normalized - always in [0,1])
        if self.include_delta_in_state:
            bs_delta = self.bs_delta_raw[self.current_step]
            base_obs.append(bs_delta)
        
        # Add Greeks for forward-looking hedging decisions
        if self.include_greeks_in_state:
            # Gamma: How fast delta changes (normalized)
            # High gamma = need more frequent rebalancing
            gamma_norm = (self.gamma_raw[self.current_step] - self.gamma_mean) / self.gamma_std
            base_obs.append(gamma_norm)
            
            # Vega: Sensitivity to volatility (normalized)
            # High vega = vol changes matter more
            vega_norm = (self.vega_raw[self.current_step] - self.vega_mean) / self.vega_std
            base_obs.append(vega_norm)
            
            # Vanna: Cross-sensitivity (delta to vol)
            # High vanna = delta will shift if vol changes
            vanna_norm = (self.vanna_raw[self.current_step] - self.vanna_mean) / self.vanna_std
            base_obs.append(vanna_norm)
            
            # Vol spread: realized_vol - implied_vol
            # Positive spread = gamma costs higher than priced, hedge more aggressively
            vol_spread = self.realized_vol_raw[self.current_step] - self.implied_vol
            # Normalize vol spread (typical range -0.2 to +0.2)
            vol_spread_norm = vol_spread / 0.1  # Scale so typical values are in [-2, 2]
            base_obs.append(vol_spread_norm)
        
        obs = np.array(base_obs, dtype=np.float32)
        return obs
    
    def get_current_bs_delta(self):
        """Return the current Black-Scholes delta (for reward calculation)."""
        return self.bs_delta_raw[self.current_step]

    def step(self, action, q_values=None):
        prev_position = self.position
        prev_cash = self.cash_account
        
        if self.use_action_bounds:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Get the theoretical Black-Scholes delta
        bs_delta = self.bs_delta_raw[self.current_step]
        
        # Convert action to target hedge ratio based on action mode
        if self.action_mode == "adjustment":
            # Action is adjustment from delta: hedge = delta + adjustment
            adjustment = action[0]
            target_hedge_ratio = np.clip(bs_delta + adjustment, 0.0, 1.0)
        else:
            # Action is absolute hedge ratio
            target_hedge_ratio = action[0]
        
        self.position = np.clip(target_hedge_ratio * self.notional, -self.notional, self.notional)
        
        # Prices
        S_prev = self.stock_prices_raw[self.current_step - 1]
        S_now = self.stock_prices_raw[self.current_step]
        
        O_prev = self.option_prices_raw[self.current_step - 1]
        O_now = self.option_prices_raw[self.current_step]
        
        # HO_t = -1 (short option position) - always
        HO_t = -1
        
        # Option component: HO_t * (Ct - Ct-1)
        option_component = HO_t * (O_now - O_prev) / (self.notional)  
        hedge_component = (prev_position / self.notional) * (S_now / S_prev - 1)
        hedge_adjustment = (self.position - prev_position) / S_now
        transaction_component = self.transaction_cost * S_now * abs(hedge_adjustment) / (self.notional)   

        # Step reward (what agent optimizes)
        step_pnl = option_component + hedge_component - transaction_component
        
        # ====================================================================
        # REWARD FUNCTION: Delta Tracking is PRIMARY objective
        # For adjustment mode, reward encourages adjustment ≈ 0
        # ====================================================================
        reward_type = CONFIG.get("reward_type", "delta_tracking")
        xi = CONFIG.get("risk_aversion", 0.1)  # Always define xi for logging
        
        # Calculate tracking error (how far from delta)
        tracking_error = (target_hedge_ratio - bs_delta) ** 2
        
        if reward_type == "delta_tracking":
            # DELTA TRACKING REWARD
            # In adjustment mode: reward is based on |adjustment| (want ≈ 0)
            # Also includes P&L variance penalty and transaction cost penalty
            tracking_weight = CONFIG.get("delta_tracking_weight", 1.0)
            pnl_weight = CONFIG.get("pnl_variance_weight", 0.1)
            tc_weight = CONFIG.get("transaction_cost_weight", 1.0)  # NEW: Transaction cost penalty
            
            if self.action_mode == "adjustment":
                # For adjustment mode, reward penalizes non-zero adjustments
                # The optimal action is adjustment = 0 (i.e., follow delta exactly)
                absolute_adjustment = abs(action[0])
                reward = -tracking_weight * absolute_adjustment
                # P&L variance component for financial performance
                reward -= pnl_weight * (step_pnl ** 2)
                # Transaction cost penalty: penalize trading activity
                # This teaches the agent that rebalancing has a real cost
                reward -= tc_weight * transaction_component
            else:
                # For absolute mode, reward penalizes deviation from delta
                absolute_tracking_error = abs(target_hedge_ratio - bs_delta)
                reward = -tracking_weight * absolute_tracking_error
                reward -= pnl_weight * (step_pnl ** 2)
                reward -= tc_weight * transaction_component
            
        elif reward_type == "variance_minimization":
            # VARIANCE MINIMIZATION (Deep Hedging style - Buehler et al.)
            reward = -(step_pnl ** 2)
            # Add tracking penalty
            if CONFIG.get("use_delta_tracking_reward", False):
                delta_tracking_weight = CONFIG.get("delta_tracking_weight", 0.1)
                reward -= delta_tracking_weight * tracking_error
            
        elif reward_type == "cara_utility":
            # CARA (Constant Absolute Risk Aversion) Utility
            lambda_risk = CONFIG.get("cara_lambda", 1.0)
            reward = -np.exp(-lambda_risk * step_pnl)
            
        else:  # "profit_seeking" - original behavior
            reward = step_pnl - xi * abs(step_pnl)
        
        # Scale reward to stable range
        reward_scale = CONFIG.get("reward_scale", 100.0)
        reward = reward * reward_scale
        
        # Traditional P&L calculation (for comparison and plotting)
        hedge_pnl_traditional = prev_position * (S_now - S_prev)
        option_pnl_traditional = -1 * self.notional * (O_now - O_prev)
        trade_cost_traditional = abs(hedge_adjustment) * S_now * self.transaction_cost
        combined_pnl_traditional = option_pnl_traditional + hedge_pnl_traditional
        
        # Calculate actual time difference for cash account interest
        if self.current_step > 1:
            prev_timestamp = self.timestamps[self.current_step - 1]
            current_timestamp = self.timestamps[self.current_step]
            
            # Handle different timestamp types
            if hasattr(current_timestamp, 'total_seconds'):
                # Already pandas datetime
                time_diff = current_timestamp - prev_timestamp
                time_diff_years = time_diff.total_seconds() / (365.25 * 24 * 3600)
            else:
         
                current_ts = pd.Timestamp(current_timestamp)
                prev_ts = pd.Timestamp(prev_timestamp)
                time_diff = current_ts - prev_ts
                time_diff_years = time_diff.total_seconds() / (365.25 * 24 * 3600)
                
            daily_discount_factor = (1 + self.discount_rate) ** time_diff_years
        else:
            daily_discount_factor = 1.0
            
        self.cash_account = self.cash_account * daily_discount_factor 
        self.cash_account = self.cash_account - hedge_adjustment * S_now - trade_cost_traditional  

        total_pnl_traditional = self.cash_account + hedge_pnl_traditional
        
        portfolio_value = self.cash_account + self.position * S_now
        prev_portfolio_value = prev_cash + prev_position * S_prev

        if prev_portfolio_value != 0:
            daily_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            daily_return = 0.0
        
        if not hasattr(self, 'initial_portfolio_value') or self.current_step == 1:
            self.initial_portfolio_value = S_now * self.notional
        
        if self.initial_portfolio_value != 0:
            cumulative_return = (portfolio_value / self.initial_portfolio_value) - 1.0
        else:
            cumulative_return = 0.0

        self.cumulative_reward += reward
        self.reward_history_detailed.append(reward)
 
        current_timestamp = self.timestamps[self.current_step]
        initial_timestamp = self.initial_timestamp
        
        if hasattr(current_timestamp, 'total_seconds'):
            time_diff = current_timestamp - initial_timestamp
            time_elapsed_years = time_diff.total_seconds() / (365.25 * 24 * 3600)
        else:
            
            current_ts = pd.Timestamp(current_timestamp)
            initial_ts = pd.Timestamp(initial_timestamp)
            time_diff = current_ts - initial_ts
            time_elapsed_years = time_diff.total_seconds() / (365.25 * 24 * 3600)

        annual_risk_free_rate = self.discount_rate
        financial_discount_factor = (1 + annual_risk_free_rate) ** (-time_elapsed_years)
        discounted_reward = financial_discount_factor * reward
        self.discounted_cumulative_reward += discounted_reward

        step_data = {
            'timestamp': self.timestamps[self.current_step],
            'step': self.current_step,
            'action': action[0],
            'position': self.position,
            'prev_position': prev_position,
            'hedge_adjustment': hedge_adjustment,
            'stock_price_prev': S_prev,
            'stock_price_now': S_now,
            'option_price_prev': O_prev,
            'option_price_now': O_now,
            'ttm': self.ttm_raw[self.current_step],
            'moneyness': self.moneyness_raw[self.current_step],
            'realized_vol': self.realized_vol_raw[self.current_step],
            'reward': reward,                           
            'step_pnl': step_pnl,                  
            'option_component': option_component,       # HO_t * (Ct - Ct-1)
            'hedge_component': hedge_component,         # HS_t * (St - St-1) / notional
            'transaction_component': transaction_component,  # c * |St| * |HS_t - HS_t-1| / notional
            'risk_aversion_xi': xi,                   
            'cumulative_reward': self.cumulative_reward, 
            'discounted_cumulative_reward': self.discounted_cumulative_reward, # Time-based discounted cumulative reward
            'time_elapsed_years': time_elapsed_years,   # Actual time elapsed from start (years)
            'financial_discount_factor': financial_discount_factor,   # Present value discount factor applied
            'discounted_reward': discounted_reward,     # This step's discounted reward
            # TRADITIONAL P&L (for comparison and plotting)
            'cumulative_pnl': combined_pnl_traditional, # Traditional combined P&L
            'option_pnl': option_pnl_traditional,      # Traditional option P&L
            'hedge_pnl': hedge_pnl_traditional,        # Traditional hedge P&L
            'trade_cost': trade_cost_traditional,      # Traditional transaction costs
            'total_pnl': total_pnl_traditional,        # Traditional total P&L (includes cash)
            # OTHER METRICS
            'cash_account_prev': prev_cash,
            'cash_account': self.cash_account,
            'portfolio_value': portfolio_value,
            'prev_portfolio_value': prev_portfolio_value,
            'daily_return': daily_return,
            'daily_return_pct': daily_return * 100, 
            'cumulative_return': cumulative_return,
            'cumulative_return_pct': cumulative_return * 100,
            'discount_factor': daily_discount_factor
        }

        if q_values is not None:
            step_data.update({
                'q1_value': q_values['q1'],
                'q2_value': q_values['q2'],
                'q_mean_value': q_values['q_mean'],
                'action_raw':q_values['action_raw'],
                'state': q_values['state']
            })
        else:
            step_data.update({
                'q1_value': np.nan,
                'q2_value': np.nan,
                'q_mean_value': np.nan
            })
        
        if not hasattr(self, 'step_data_history'):
            self.step_data_history = []
        self.step_data_history.append(step_data)
        
        self.action_history.append(self.position)
        self.pnl_history.append(total_pnl_traditional)
        self.reward_history.append(reward)              
        self.timestamp_history.append(self.timestamps[self.current_step])

        self.current_step += 1
        done = self.current_step >= len(self.stock_prices_raw) - 1
        next_state = self._get_observation() if not done else np.zeros_like(self.observation_space.low)

        return next_state, reward, done, {
            'reward': reward,                      
            'cumulative_reward': self.cumulative_reward, # Undiscounted cumulative reward
            'discounted_cumulative_reward': self.discounted_cumulative_reward, # Time-based discounted cumulative reward
            'step_pnl': step_pnl,                     # Step P&L before risk aversion
            'cumulative_pnl': combined_pnl_traditional, # Traditional combined P&L
            'option_pnl': option_pnl_traditional,     # Traditional option P&L
            'hedge_pnl': hedge_pnl_traditional,       # Traditional hedge P&L
            'trade_cost': trade_cost_traditional,     # Traditional transaction costs
            'transaction_component': transaction_component,  # Normalized TC for reward
            'total_pnl': total_pnl_traditional,       # Traditional total P&L
            'portfolio_value': portfolio_value,
            'daily_return': daily_return,
            'cumulative_return': cumulative_return,
            'realized_vol': self.realized_vol_raw[self.current_step - 1],
            'hedge_adjustment': hedge_adjustment,     # Position change / S
            'target_hedge_ratio': target_hedge_ratio, # The actual hedge ratio used
            'bs_delta': bs_delta,                     # The BS delta at this step
            'q_values': q_values 
        }

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Position: {self.position:.4f}")

    def get_logs(self):
        return self.action_history, self.pnl_history, self.timestamp_history

    def set_action_bounds(self, enabled=None, low=None, high=None):
        env_config = get_environment_config()
        
        if enabled is None:
            enabled = env_config["use_action_bounds"]
        if low is None:
            low = env_config["min_action"]
        if high is None:
            high = env_config["max_action"]
            
        self.use_action_bounds = enabled
        self.action_low = low
        self.action_high = high
        
        if self.use_action_bounds:
            self.action_space = spaces.Box(low=self.action_low, high=self.action_high, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def get_action_bounds_config(self):
        return {
            "use_action_bounds": self.use_action_bounds,
            "action_low": self.action_low,
            "action_high": self.action_high
        }
        
    def set_verbose(self, verbose=True):
        self.verbose = verbose

    def get_normalization_stats(self):
        return {
            "opt_mean": self.opt_mean,
            "opt_std": self.opt_std,
            "stock_mean": self.stock_mean,
            "stock_std": self.stock_std,
            "moneyness_mean": self.moneyness_mean,
            "moneyness_std": self.moneyness_std,
            "ttm_mean": self.ttm_mean,
            "ttm_std": self.ttm_std,
            "vol_mean": self.vol_mean,  
            "vol_std": self.vol_std    
        }
    
    def export_step_data(self, filepath='step_data.csv'):
        import pandas as pd
        
        if hasattr(self, 'step_data_history') and len(self.step_data_history) > 0:
            df = pd.DataFrame(self.step_data_history)
            df.to_csv(filepath, index=False)
            return True
        else:
            if self.verbose:
                print("No step data available to export")
            return False
        
    def reset_step_data_history(self):
        if hasattr(self, 'step_data_history'):
            self.step_data_history = []