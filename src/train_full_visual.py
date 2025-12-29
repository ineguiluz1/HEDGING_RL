#!/usr/bin/env python3
"""
Full Training Pipeline with Enhanced Visualization
Trains on 2005-2010, validates on 2011, tests on 2012
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_training import run_full_training_pipeline

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FULL TD3 TRAINING PIPELINE - ENHANCED VISUALIZATION")
    print("="*70)
    print("\nConfiguration:")
    print("  Training years: 2005-2010")
    print("  Validation year: 2011")
    print("  Test year: 2012")
    print("  Epochs: 10 (configurable)")
    print("="*70 + "\n")
    
    # Run the full pipeline with visualization
    results = run_full_training_pipeline(
        train_years=[2005, 2006, 2007, 2008, 2009, 2010],
        validation_year=2011,
        test_year=2012,
        num_epochs=10,  # Can be adjusted
        verbose=True
    )
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETED!")
    print("="*70)
    print(f"\nResults saved to: {results['results_dir']}")
    print(f"Model saved to: {results['model_path']}")
    print("\nKey Metrics:")
    print(f"  RL Agent P&L: {results['rl_agent']['cumulative_pnl']:.4f}")
    print(f"  Benchmark P&L: {results['benchmark']['cumulative_pnl']:.4f}")
    print(f"  Improvement: {results['improvements']['pnl']:+.4f}")
    print("="*70 + "\n")
