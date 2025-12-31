#!/usr/bin/env python3
"""
Comprehensive Hedging Performance Metrics Module.

This module provides standardized metrics for comparing RL agent vs benchmark:

1. Operational Efficiency (Cost vs Benefit):
   - Net P&L (P&L - Transaction Costs)
   - Trading Intensity (Turnover)
   - P&L Improvement Ratio

2. Risk & Fidelity (Tracking Error):
   - RMSE Tracking Error vs BS Delta
   - P&L Variance (Hedge Error)
   - Maximum Drawdown

3. Risk-Adjusted Performance:
   - Sharpe Ratio
   - Information Ratio vs Benchmark

4. Visualization:
   - Efficient Frontier: Tracking Error vs Transaction Cost

Author: HEDGING_RL Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm


@dataclass
class HedgingMetrics:
    """Container for all hedging performance metrics."""
    
    # Basic Statistics
    n_episodes: int = 0
    n_steps: int = 0
    
    # P&L Metrics
    total_pnl: float = 0.0
    mean_episode_pnl: float = 0.0
    std_episode_pnl: float = 0.0
    net_pnl: float = 0.0  # P&L - Transaction Costs
    
    # Transaction Cost Metrics
    total_transaction_costs: float = 0.0
    mean_tc_per_episode: float = 0.0
    tc_as_pct_of_gross_pnl: float = 0.0
    
    # Trading Intensity (Turnover)
    total_turnover: float = 0.0  # Sum of |hedge_adjustments|
    mean_turnover_per_step: float = 0.0
    mean_turnover_per_episode: float = 0.0
    
    # Tracking Error (vs BS Delta)
    rmse_tracking_error: float = 0.0
    mse_tracking_error: float = 0.0
    mean_absolute_tracking_error: float = 0.0
    
    # Hedge Ratio Statistics
    mean_hedge_ratio: float = 0.0
    std_hedge_ratio: float = 0.0
    min_hedge_ratio: float = 0.0
    max_hedge_ratio: float = 0.0
    
    # Risk Metrics
    pnl_variance: float = 0.0
    pnl_std: float = 0.0
    max_drawdown: float = 0.0
    max_episode_loss: float = 0.0
    
    # Risk-Adjusted Metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Comparison Metrics (vs Benchmark)
    pnl_improvement: float = 0.0
    pnl_improvement_pct: float = 0.0
    tc_savings: float = 0.0
    tc_savings_pct: float = 0.0
    information_ratio: float = 0.0
    
    # Raw data for plotting
    all_episode_pnls: List[float] = field(default_factory=list)
    all_step_pnls: List[float] = field(default_factory=list)
    all_hedge_ratios: List[float] = field(default_factory=list)
    all_bs_deltas: List[float] = field(default_factory=list)
    all_transaction_costs: List[float] = field(default_factory=list)
    all_turnovers: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding raw data lists for JSON serialization)."""
        return {
            'n_episodes': self.n_episodes,
            'n_steps': self.n_steps,
            'total_pnl': self.total_pnl,
            'mean_episode_pnl': self.mean_episode_pnl,
            'std_episode_pnl': self.std_episode_pnl,
            'net_pnl': self.net_pnl,
            'total_transaction_costs': self.total_transaction_costs,
            'mean_tc_per_episode': self.mean_tc_per_episode,
            'tc_as_pct_of_gross_pnl': self.tc_as_pct_of_gross_pnl,
            'total_turnover': self.total_turnover,
            'mean_turnover_per_step': self.mean_turnover_per_step,
            'mean_turnover_per_episode': self.mean_turnover_per_episode,
            'rmse_tracking_error': self.rmse_tracking_error,
            'mse_tracking_error': self.mse_tracking_error,
            'mean_absolute_tracking_error': self.mean_absolute_tracking_error,
            'mean_hedge_ratio': self.mean_hedge_ratio,
            'std_hedge_ratio': self.std_hedge_ratio,
            'min_hedge_ratio': self.min_hedge_ratio,
            'max_hedge_ratio': self.max_hedge_ratio,
            'pnl_variance': self.pnl_variance,
            'pnl_std': self.pnl_std,
            'max_drawdown': self.max_drawdown,
            'max_episode_loss': self.max_episode_loss,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'pnl_improvement': self.pnl_improvement,
            'pnl_improvement_pct': self.pnl_improvement_pct,
            'tc_savings': self.tc_savings,
            'tc_savings_pct': self.tc_savings_pct,
            'information_ratio': self.information_ratio,
        }


def calculate_max_drawdown(pnl_series: np.ndarray) -> float:
    """
    Calculate maximum drawdown from a P&L series.
    
    Args:
        pnl_series: Array of cumulative P&L values
    
    Returns:
        Maximum drawdown (positive value representing loss)
    """
    if len(pnl_series) == 0:
        return 0.0
    
    cumulative = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    return float(np.max(drawdowns))


def calculate_sharpe_ratio(pnl_series: np.ndarray, annualization: float = 252) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        pnl_series: Array of step P&L values
        annualization: Annualization factor (252 for daily)
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(pnl_series) < 2 or np.std(pnl_series) == 0:
        return 0.0
    return float(np.mean(pnl_series) / np.std(pnl_series) * np.sqrt(annualization))


def calculate_sortino_ratio(pnl_series: np.ndarray, annualization: float = 252) -> float:
    """
    Calculate Sortino ratio (only considers downside volatility).
    
    Args:
        pnl_series: Array of step P&L values
        annualization: Annualization factor
    
    Returns:
        Sortino ratio
    """
    if len(pnl_series) < 2:
        return 0.0
    
    negative_returns = pnl_series[pnl_series < 0]
    if len(negative_returns) == 0:
        return float('inf') if np.mean(pnl_series) > 0 else 0.0
    
    downside_std = np.std(negative_returns)
    if downside_std == 0:
        return 0.0
    
    return float(np.mean(pnl_series) / downside_std * np.sqrt(annualization))


def calculate_information_ratio(
    agent_pnls: np.ndarray,
    benchmark_pnls: np.ndarray,
    annualization: float = 252
) -> float:
    """
    Calculate Information Ratio: excess return over tracking error.
    
    IR = (Mean Agent P&L - Mean Benchmark P&L) / Std(Agent P&L - Benchmark P&L)
    
    Args:
        agent_pnls: Array of agent episode P&Ls
        benchmark_pnls: Array of benchmark episode P&Ls
        annualization: Annualization factor
    
    Returns:
        Information Ratio
    """
    if len(agent_pnls) != len(benchmark_pnls) or len(agent_pnls) < 2:
        return 0.0
    
    excess_returns = agent_pnls - benchmark_pnls
    tracking_error_vol = np.std(excess_returns)
    
    if tracking_error_vol == 0:
        return 0.0
    
    mean_excess = np.mean(excess_returns)
    return float(mean_excess / tracking_error_vol * np.sqrt(annualization))


def compute_metrics_from_episodes(
    episode_data: List[Dict],
    is_benchmark: bool = False
) -> HedgingMetrics:
    """
    Compute comprehensive metrics from episode data.
    
    Args:
        episode_data: List of episode dictionaries containing:
            - 'step_pnls': List of step P&Ls
            - 'hedge_ratios': List of hedge ratios (actions)
            - 'bs_deltas': List of BS deltas
            - 'transaction_costs': List of transaction costs per step
            - 'hedge_adjustments': List of position changes (for turnover)
        is_benchmark: Whether this is benchmark data (affects some calculations)
    
    Returns:
        HedgingMetrics object with all computed metrics
    """
    metrics = HedgingMetrics()
    
    if not episode_data:
        return metrics
    
    metrics.n_episodes = len(episode_data)
    
    # Aggregate across episodes
    all_step_pnls = []
    all_hedge_ratios = []
    all_bs_deltas = []
    all_tcs = []
    all_turnovers = []
    episode_pnls = []
    episode_tcs = []
    episode_turnovers = []
    
    for ep in episode_data:
        step_pnls = np.array(ep.get('step_pnls', []))
        hedge_ratios = np.array(ep.get('hedge_ratios', []))
        bs_deltas = np.array(ep.get('bs_deltas', []))
        tcs = np.array(ep.get('transaction_costs', []))
        adjustments = np.array(ep.get('hedge_adjustments', []))
        
        all_step_pnls.extend(step_pnls)
        all_hedge_ratios.extend(hedge_ratios)
        all_bs_deltas.extend(bs_deltas)
        all_tcs.extend(tcs)
        all_turnovers.extend(np.abs(adjustments))
        
        episode_pnls.append(np.sum(step_pnls))
        episode_tcs.append(np.sum(tcs))
        episode_turnovers.append(np.sum(np.abs(adjustments)))
    
    # Convert to arrays
    all_step_pnls = np.array(all_step_pnls)
    all_hedge_ratios = np.array(all_hedge_ratios)
    all_bs_deltas = np.array(all_bs_deltas)
    all_tcs = np.array(all_tcs)
    all_turnovers = np.array(all_turnovers)
    episode_pnls = np.array(episode_pnls)
    episode_tcs = np.array(episode_tcs)
    episode_turnovers = np.array(episode_turnovers)
    
    metrics.n_steps = len(all_step_pnls)
    
    # Store raw data
    metrics.all_episode_pnls = episode_pnls.tolist()
    metrics.all_step_pnls = all_step_pnls.tolist()
    metrics.all_hedge_ratios = all_hedge_ratios.tolist()
    metrics.all_bs_deltas = all_bs_deltas.tolist()
    metrics.all_transaction_costs = all_tcs.tolist()
    metrics.all_turnovers = all_turnovers.tolist()
    
    # === P&L Metrics ===
    metrics.total_pnl = float(np.sum(episode_pnls))
    metrics.mean_episode_pnl = float(np.mean(episode_pnls))
    metrics.std_episode_pnl = float(np.std(episode_pnls))
    
    # === Transaction Cost Metrics ===
    metrics.total_transaction_costs = float(np.sum(all_tcs))
    metrics.mean_tc_per_episode = float(np.mean(episode_tcs))
    
    # Net P&L (after transaction costs - but TC is already subtracted in step_pnl)
    # So we report gross P&L + TC for comparison purposes
    gross_pnl = metrics.total_pnl + metrics.total_transaction_costs
    metrics.net_pnl = metrics.total_pnl  # Already net of TC
    
    if abs(gross_pnl) > 1e-8:
        metrics.tc_as_pct_of_gross_pnl = float(metrics.total_transaction_costs / abs(gross_pnl) * 100)
    
    # === Turnover (Trading Intensity) ===
    metrics.total_turnover = float(np.sum(all_turnovers))
    metrics.mean_turnover_per_step = float(np.mean(all_turnovers)) if len(all_turnovers) > 0 else 0.0
    metrics.mean_turnover_per_episode = float(np.mean(episode_turnovers))
    
    # === Tracking Error (vs BS Delta) ===
    if len(all_hedge_ratios) > 0 and len(all_bs_deltas) > 0:
        tracking_errors = all_hedge_ratios - all_bs_deltas
        metrics.mse_tracking_error = float(np.mean(tracking_errors ** 2))
        metrics.rmse_tracking_error = float(np.sqrt(metrics.mse_tracking_error))
        metrics.mean_absolute_tracking_error = float(np.mean(np.abs(tracking_errors)))
    
    # === Hedge Ratio Statistics ===
    if len(all_hedge_ratios) > 0:
        metrics.mean_hedge_ratio = float(np.mean(all_hedge_ratios))
        metrics.std_hedge_ratio = float(np.std(all_hedge_ratios))
        metrics.min_hedge_ratio = float(np.min(all_hedge_ratios))
        metrics.max_hedge_ratio = float(np.max(all_hedge_ratios))
    
    # === Risk Metrics ===
    if len(all_step_pnls) > 0:
        metrics.pnl_variance = float(np.var(all_step_pnls))
        metrics.pnl_std = float(np.std(all_step_pnls))
        metrics.max_drawdown = calculate_max_drawdown(all_step_pnls)
    
    if len(episode_pnls) > 0:
        metrics.max_episode_loss = float(np.min(episode_pnls))
    
    # === Risk-Adjusted Metrics ===
    metrics.sharpe_ratio = calculate_sharpe_ratio(all_step_pnls)
    metrics.sortino_ratio = calculate_sortino_ratio(all_step_pnls)
    
    return metrics


def compare_metrics(
    agent_metrics: HedgingMetrics,
    benchmark_metrics: HedgingMetrics
) -> HedgingMetrics:
    """
    Add comparison metrics to agent metrics.
    
    Args:
        agent_metrics: Metrics computed for the RL agent
        benchmark_metrics: Metrics computed for the benchmark
    
    Returns:
        Updated agent_metrics with comparison fields filled
    """
    # P&L Improvement
    agent_metrics.pnl_improvement = agent_metrics.total_pnl - benchmark_metrics.total_pnl
    
    if abs(benchmark_metrics.total_pnl) > 1e-8:
        agent_metrics.pnl_improvement_pct = (agent_metrics.pnl_improvement / abs(benchmark_metrics.total_pnl)) * 100
    
    # Transaction Cost Savings
    agent_metrics.tc_savings = benchmark_metrics.total_transaction_costs - agent_metrics.total_transaction_costs
    
    if benchmark_metrics.total_transaction_costs > 1e-8:
        agent_metrics.tc_savings_pct = (agent_metrics.tc_savings / benchmark_metrics.total_transaction_costs) * 100
    
    # Information Ratio
    agent_pnls = np.array(agent_metrics.all_episode_pnls)
    bench_pnls = np.array(benchmark_metrics.all_episode_pnls)
    
    if len(agent_pnls) == len(bench_pnls):
        agent_metrics.information_ratio = calculate_information_ratio(agent_pnls, bench_pnls)
    
    return agent_metrics


def print_metrics_comparison(
    agent_metrics: HedgingMetrics,
    benchmark_metrics: HedgingMetrics,
    title: str = "PERFORMANCE COMPARISON"
) -> None:
    """
    Print a formatted comparison table of metrics.
    
    Args:
        agent_metrics: Metrics for RL agent
        benchmark_metrics: Metrics for benchmark
        title: Title for the comparison
    """
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    
    # Header
    print(f"\n{'METRIC':<40} {'RL AGENT':>18} {'BENCHMARK':>18}")
    print(f"{'-'*80}")
    
    # === OPERATIONAL EFFICIENCY ===
    print(f"\n{'--- OPERATIONAL EFFICIENCY ---':<40}")
    print(f"{'Total P&L (Net)':<40} {agent_metrics.total_pnl:>18.4f} {benchmark_metrics.total_pnl:>18.4f}")
    print(f"{'Mean Episode P&L':<40} {agent_metrics.mean_episode_pnl:>18.4f} {benchmark_metrics.mean_episode_pnl:>18.4f}")
    print(f"{'Total Transaction Costs':<40} {agent_metrics.total_transaction_costs:>18.4f} {benchmark_metrics.total_transaction_costs:>18.4f}")
    print(f"{'TC as % of Gross P&L':<40} {agent_metrics.tc_as_pct_of_gross_pnl:>17.2f}% {benchmark_metrics.tc_as_pct_of_gross_pnl:>17.2f}%")
    print(f"{'Total Turnover':<40} {agent_metrics.total_turnover:>18.4f} {benchmark_metrics.total_turnover:>18.4f}")
    print(f"{'Mean Turnover/Episode':<40} {agent_metrics.mean_turnover_per_episode:>18.4f} {benchmark_metrics.mean_turnover_per_episode:>18.4f}")
    
    # === TRACKING ERROR & FIDELITY ===
    print(f"\n{'--- TRACKING ERROR & FIDELITY ---':<40}")
    print(f"{'RMSE vs BS Delta':<40} {agent_metrics.rmse_tracking_error:>18.4f} {benchmark_metrics.rmse_tracking_error:>18.4f}")
    print(f"{'Mean Absolute TE':<40} {agent_metrics.mean_absolute_tracking_error:>18.4f} {benchmark_metrics.mean_absolute_tracking_error:>18.4f}")
    print(f"{'Mean Hedge Ratio':<40} {agent_metrics.mean_hedge_ratio:>18.4f} {benchmark_metrics.mean_hedge_ratio:>18.4f}")
    print(f"{'Std Hedge Ratio':<40} {agent_metrics.std_hedge_ratio:>18.4f} {benchmark_metrics.std_hedge_ratio:>18.4f}")
    
    # === RISK METRICS ===
    print(f"\n{'--- RISK METRICS ---':<40}")
    print(f"{'P&L Variance':<40} {agent_metrics.pnl_variance:>18.6f} {benchmark_metrics.pnl_variance:>18.6f}")
    print(f"{'P&L Std Dev':<40} {agent_metrics.pnl_std:>18.6f} {benchmark_metrics.pnl_std:>18.6f}")
    print(f"{'Max Drawdown':<40} {agent_metrics.max_drawdown:>18.4f} {benchmark_metrics.max_drawdown:>18.4f}")
    print(f"{'Worst Episode P&L':<40} {agent_metrics.max_episode_loss:>18.4f} {benchmark_metrics.max_episode_loss:>18.4f}")
    
    # === RISK-ADJUSTED PERFORMANCE ===
    print(f"\n{'--- RISK-ADJUSTED PERFORMANCE ---':<40}")
    print(f"{'Sharpe Ratio':<40} {agent_metrics.sharpe_ratio:>18.4f} {benchmark_metrics.sharpe_ratio:>18.4f}")
    print(f"{'Sortino Ratio':<40} {agent_metrics.sortino_ratio:>18.4f} {benchmark_metrics.sortino_ratio:>18.4f}")
    
    # === IMPROVEMENT SUMMARY ===
    print(f"\n{'='*80}")
    print(f"{'IMPROVEMENT SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'P&L Improvement':<40} {agent_metrics.pnl_improvement:>+18.4f} ({agent_metrics.pnl_improvement_pct:>+.2f}%)")
    print(f"{'Transaction Cost Savings':<40} {agent_metrics.tc_savings:>+18.4f} ({agent_metrics.tc_savings_pct:>+.2f}%)")
    print(f"{'Information Ratio':<40} {agent_metrics.information_ratio:>18.4f}")
    
    # === VERDICT ===
    print(f"\n{'='*80}")
    
    wins = 0
    total_criteria = 4
    
    # P&L better
    if agent_metrics.total_pnl > benchmark_metrics.total_pnl:
        wins += 1
    
    # Lower transaction costs
    if agent_metrics.total_transaction_costs < benchmark_metrics.total_transaction_costs:
        wins += 1
    
    # Lower P&L variance (better hedge)
    if agent_metrics.pnl_variance < benchmark_metrics.pnl_variance:
        wins += 1
    
    # Better Sharpe
    if agent_metrics.sharpe_ratio > benchmark_metrics.sharpe_ratio:
        wins += 1
    
    if wins >= 3:
        print(f"✅ RL AGENT OUTPERFORMS BENCHMARK ({wins}/{total_criteria} criteria)")
    elif wins >= 2:
        print(f"🔶 MIXED RESULTS ({wins}/{total_criteria} criteria favor RL Agent)")
    else:
        print(f"❌ BENCHMARK OUTPERFORMS RL AGENT ({total_criteria - wins}/{total_criteria} criteria)")
    
    print(f"{'='*80}\n")


def plot_efficient_frontier(
    agent_metrics: HedgingMetrics,
    benchmark_metrics: HedgingMetrics,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot the Efficient Frontier: Tracking Error vs Transaction Costs.
    
    The ideal position is bottom-left (low tracking error, low costs).
    
    Args:
        agent_metrics: Metrics for RL agent
        benchmark_metrics: Metrics for benchmark
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # === Plot 1: Efficient Frontier (Tracking Error vs TC) ===
    ax1 = axes[0, 0]
    
    # Plot the two strategies
    ax1.scatter(
        benchmark_metrics.total_transaction_costs,
        benchmark_metrics.rmse_tracking_error,
        s=200, c='green', marker='s', label='Delta Hedging', zorder=5, edgecolors='black'
    )
    ax1.scatter(
        agent_metrics.total_transaction_costs,
        agent_metrics.rmse_tracking_error,
        s=200, c='blue', marker='o', label='RL Agent', zorder=5, edgecolors='black'
    )
    
    # Reference point: No hedging (TC=0, max tracking error)
    ax1.scatter(0, 0.5, s=150, c='red', marker='x', label='No Hedging (hypothetical)', zorder=4)
    
    # Arrow showing improvement direction
    ax1.annotate('', xy=(0, 0), xytext=(0.1, 0.15),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax1.text(0.05, 0.08, 'Better', fontsize=10, color='gray')
    
    ax1.set_xlabel('Total Transaction Costs', fontsize=12)
    ax1.set_ylabel('RMSE Tracking Error (vs BS Delta)', fontsize=12)
    ax1.set_title('Efficient Frontier: Tracking Error vs Transaction Costs', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: P&L Distribution ===
    ax2 = axes[0, 1]
    
    bins = min(30, max(10, len(agent_metrics.all_episode_pnls) // 5))
    ax2.hist(agent_metrics.all_episode_pnls, bins=bins, alpha=0.7, 
             label=f'RL Agent (μ={agent_metrics.mean_episode_pnl:.4f})', color='blue', edgecolor='black')
    ax2.hist(benchmark_metrics.all_episode_pnls, bins=bins, alpha=0.7, 
             label=f'Benchmark (μ={benchmark_metrics.mean_episode_pnl:.4f})', color='green', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', label='Break-even', linewidth=2)
    ax2.set_xlabel('Episode P&L', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Episode P&L', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === Plot 3: Cumulative P&L Over Episodes ===
    ax3 = axes[1, 0]
    
    agent_cumsum = np.cumsum(agent_metrics.all_episode_pnls)
    bench_cumsum = np.cumsum(benchmark_metrics.all_episode_pnls)
    
    ax3.plot(agent_cumsum, label='RL Agent', color='blue', linewidth=2)
    ax3.plot(bench_cumsum, label='Benchmark', color='green', linewidth=2)
    ax3.fill_between(range(len(agent_cumsum)), agent_cumsum, bench_cumsum, 
                     alpha=0.3, color='blue' if agent_cumsum[-1] > bench_cumsum[-1] else 'green')
    ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Episode Number', fontsize=12)
    ax3.set_ylabel('Cumulative P&L', fontsize=12)
    ax3.set_title('Cumulative P&L Progression', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # === Plot 4: Metrics Comparison Bar Chart ===
    ax4 = axes[1, 1]
    
    # Normalize metrics for comparison
    metrics_names = ['P&L\n(normalized)', 'Sharpe\nRatio', 'TC Savings\n(%)', 'Turnover\nReduction (%)']
    
    # Calculate normalized values
    pnl_norm_agent = agent_metrics.total_pnl / (abs(benchmark_metrics.total_pnl) + 1e-8)
    pnl_norm_bench = 1.0  # benchmark is reference
    
    tc_reduction = (1 - agent_metrics.total_transaction_costs / (benchmark_metrics.total_transaction_costs + 1e-8)) * 100
    turnover_reduction = (1 - agent_metrics.total_turnover / (benchmark_metrics.total_turnover + 1e-8)) * 100
    
    agent_values = [pnl_norm_agent, agent_metrics.sharpe_ratio, tc_reduction, turnover_reduction]
    bench_values = [pnl_norm_bench, benchmark_metrics.sharpe_ratio, 0, 0]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, agent_values, width, label='RL Agent', color='blue', alpha=0.8)
    bars2 = ax4.bar(x + width/2, bench_values, width, label='Benchmark', color='green', alpha=0.8)
    
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('Key Metrics Comparison', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_names, fontsize=10)
    ax4.legend()
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Efficient frontier plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def evaluate_agent_with_metrics(
    agent,
    test_envs: List,
    verbose: bool = True
) -> Tuple[HedgingMetrics, List[Dict]]:
    """
    Evaluate agent on test environments and return comprehensive metrics.
    
    Args:
        agent: Trained TD3Agent
        test_envs: List of test environments
        verbose: Print progress
    
    Returns:
        Tuple of (HedgingMetrics, episode_data list)
    """
    episode_data = []
    
    if verbose:
        print(f"\nEvaluating agent on {len(test_envs)} episodes...")
    
    for i, env in enumerate(test_envs):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        step_pnls = []
        hedge_ratios = []
        bs_deltas = []
        transaction_costs = []
        hedge_adjustments = []
        
        done = False
        while not done:
            action = agent.select_action(state, add_noise=False)
            
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            step_pnls.append(info.get('step_pnl', 0))
            hedge_ratios.append(info.get('target_hedge_ratio', action[0]))
            bs_deltas.append(info.get('bs_delta', 0.5))
            transaction_costs.append(info.get('transaction_component', 0))
            hedge_adjustments.append(info.get('hedge_adjustment', 0))
            
            state = next_state
        
        episode_data.append({
            'step_pnls': step_pnls,
            'hedge_ratios': hedge_ratios,
            'bs_deltas': bs_deltas,
            'transaction_costs': transaction_costs,
            'hedge_adjustments': hedge_adjustments
        })
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Evaluated {i + 1}/{len(test_envs)} episodes...")
    
    metrics = compute_metrics_from_episodes(episode_data)
    
    if verbose:
        print(f"  ✓ Evaluation complete: {metrics.n_episodes} episodes, {metrics.n_steps} total steps")
    
    return metrics, episode_data


def evaluate_benchmark_with_metrics(
    test_envs: List,
    verbose: bool = True
) -> Tuple[HedgingMetrics, List[Dict]]:
    """
    Evaluate delta hedging benchmark on test environments and return comprehensive metrics.
    
    Args:
        test_envs: List of test environments
        verbose: Print progress
    
    Returns:
        Tuple of (HedgingMetrics, episode_data list)
    """
    from config import CONFIG
    
    # Get volatility for BS delta calculation (same as run_training.py)
    vol = CONFIG.get("mc_volatility", 0.20)
    r = CONFIG.get("risk_free_rate", 0.02)
    tc = CONFIG.get("transaction_cost", 0.001)
    notional = CONFIG.get("notional", 1000)
    
    episode_data = []
    
    if verbose:
        print(f"\nRunning benchmark on {len(test_envs)} episodes...")
    
    for i, env in enumerate(test_envs):
        # Get raw data from environment
        option_prices = env.option_prices_raw
        stock_prices = env.stock_prices_raw
        moneyness = env.moneyness_raw
        ttm = env.ttm_raw
        
        step_pnls = []
        hedge_ratios = []
        bs_deltas = []
        transaction_costs = []
        hedge_adjustments = []
        
        prev_position = 0.0
        
        for j in range(len(stock_prices)):
            S_now = stock_prices[j]
            O_now = option_prices[j]
            K = S_now / moneyness[j] if moneyness[j] > 0 else S_now
            T = max(ttm[j], 1e-6)
            
            # Black-Scholes delta (same calculation as run_training.py)
            d1 = (np.log(S_now/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T) + 1e-8)
            delta = float(norm.cdf(d1))
            delta = np.clip(delta, 0, 1)
            
            bs_deltas.append(delta)
            hedge_ratios.append(delta)  # Benchmark follows delta exactly
            
            # P&L calculation (same as hedging_env.py)
            if j > 0:
                S_prev = stock_prices[j-1]
                O_prev = option_prices[j-1]
                
                # Option component
                HO_t = -1
                option_component = HO_t * (O_now - O_prev) / notional
                
                # Hedge component
                hedge_component = (prev_position / notional) * (S_now / S_prev - 1)
                
                # Transaction component
                new_position = delta * notional
                hedge_adj = (new_position - prev_position) / S_now
                transaction_component = tc * S_now * abs(hedge_adj) / notional
                
                step_pnl = option_component + hedge_component - transaction_component
                
                step_pnls.append(step_pnl)
                transaction_costs.append(transaction_component)
                hedge_adjustments.append(hedge_adj)
            
            prev_position = delta * notional
        
        episode_data.append({
            'step_pnls': step_pnls,
            'hedge_ratios': hedge_ratios[1:],  # Align with step_pnls
            'bs_deltas': bs_deltas[1:],
            'transaction_costs': transaction_costs,
            'hedge_adjustments': hedge_adjustments
        })
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Benchmark evaluated {i + 1}/{len(test_envs)} episodes...")
    
    metrics = compute_metrics_from_episodes(episode_data, is_benchmark=True)
    
    if verbose:
        print(f"  ✓ Benchmark complete: {metrics.n_episodes} episodes, {metrics.n_steps} total steps")
    
    return metrics, episode_data


# Convenience function for quick comparison
def run_full_comparison(
    agent,
    test_envs: List,
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[HedgingMetrics, HedgingMetrics]:
    """
    Run full evaluation and comparison of agent vs benchmark.
    
    Args:
        agent: Trained TD3Agent
        test_envs: List of test environments
        save_dir: Directory to save plots (optional)
        verbose: Print progress and results
    
    Returns:
        Tuple of (agent_metrics, benchmark_metrics)
    """
    # Evaluate agent
    agent_metrics, agent_data = evaluate_agent_with_metrics(agent, test_envs, verbose)
    
    # Evaluate benchmark
    benchmark_metrics, bench_data = evaluate_benchmark_with_metrics(test_envs, verbose)
    
    # Add comparison metrics
    agent_metrics = compare_metrics(agent_metrics, benchmark_metrics)
    
    # Print comparison
    if verbose:
        print_metrics_comparison(agent_metrics, benchmark_metrics)
    
    # Plot efficient frontier
    if save_dir:
        import os
        plot_path = os.path.join(save_dir, 'efficient_frontier.png')
        plot_efficient_frontier(agent_metrics, benchmark_metrics, save_path=plot_path, show=False)
    
    return agent_metrics, benchmark_metrics
