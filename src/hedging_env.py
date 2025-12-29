import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from config import CONFIG, get_environment_config
from volatility_utils import calculate_realized_volatility, fill_initial_volatility


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

        self.option_prices_raw = option_prices
        self.stock_prices_raw = stock_prices
        self.moneyness_raw = moneyness
        self.ttm_raw = ttm
        self.realized_vol_raw = self._calculate_realized_volatility()

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

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

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
            print(f"Normalization: {self._normalize_flag}")
            print(f"Action Bounds: {self.use_action_bounds} [{self.action_low}, {self.action_high}]")
            print(f"Volatility Window: {self.vol_window}")
            print(f"Initial Volatility: {self.initial_vol}")
            print(f"Realized Vol Range: {self.realized_vol_raw.min():.4f} - {self.realized_vol_raw.max():.4f}")
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
        obs = np.array([
            self.option_prices[self.current_step],
            self.stock_prices[self.current_step],
            self.ttm[self.current_step], 
            self.moneyness[self.current_step],
            self.realized_vol[self.current_step],
            pos
        ], dtype=np.float32)
        
        
        return obs

    def step(self, action, q_values=None):
        prev_position = self.position
        prev_cash = self.cash_account
        
        if self.use_action_bounds:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # self.position = action[0] * self.notional
        # max_change = self.notional * 0.1  # Max 10% position change per step
        # position_change = action[0] * max_change
        # self.position = np.clip(self.position + position_change, -self.notional, self.notional)
        target_hedge_ratio = action[0]  # new delta
        self.position = np.clip(target_hedge_ratio * self.notional, -self.notional, self.notional) # delta scaled by notional, so how much we should actually buy/sell
        #prices
        S_prev = self.stock_prices_raw[self.current_step - 1]
        S_now = self.stock_prices_raw[self.current_step]
        
        O_prev = self.option_prices_raw[self.current_step - 1]
        O_now = self.option_prices_raw[self.current_step]
        
        # HO_t = -1 (short option position) - always
        HO_t = -1
        
        # Option component: HO_t * (Ct - Ct-1)
        option_component = HO_t * (O_now - O_prev) / (self.notional)  
        # hedge_component = prev_position * (S_now - S_prev) / (self.notional) 
        hedge_component = (prev_position / self.notional) * (S_now / S_prev - 1)
        hedge_adjustment = (self.position - prev_position) / S_now
        transaction_component = self.transaction_cost * S_now * abs(hedge_adjustment) / (self.notional)   

        # Transaction cost component: c * |St| * |HS_t - HS_t-1|

        
        # Step reward (what agent optimizes)
        step_pnl = option_component + hedge_component - transaction_component
        
        # Step reward with risk aversion
        xi = CONFIG.get("risk_aversion", 0.1)  # Risk aversion parameter ξ
        reward = step_pnl - xi * abs(step_pnl)
        
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
            'total_pnl': total_pnl_traditional,       # Traditional total P&L
            'portfolio_value': portfolio_value,
            'daily_return': daily_return,
            'cumulative_return': cumulative_return,
            'realized_vol': self.realized_vol_raw[self.current_step - 1],
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