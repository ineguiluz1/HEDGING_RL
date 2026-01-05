#!/usr/bin/env python3
"""
Export Optuna Study Visualizations
===================================

This script generates all visualization plots from Optuna hyperparameter 
optimization studies for both SAC and TD3 agents.

Generates:
- Optimization history (objective value over trials)
- Parameter importances
- Parallel coordinate plot
- Contour plots (2D parameter relationships)
- Slice plots (parameter vs objective)
- Empirical distribution function
- Timeline plot
- Pareto front (if multi-objective)

Usage:
    python export_optuna_plots.py [--output-dir OUTPUT_DIR]
"""

import os
import sys
import argparse
from pathlib import Path
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
    plot_slice,
    plot_edf,
    plot_timeline,
)

try:
    from optuna.visualization import plot_pareto_front
    HAS_PARETO = True
except ImportError:
    HAS_PARETO = False


def load_study(db_path: str, study_name: str) -> optuna.Study:
    """
    Load an Optuna study from SQLite database.
    
    Args:
        db_path: Path to SQLite database
        study_name: Name of the study
    
    Returns:
        Optuna study object
    """
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    return study


def export_all_plots(study: optuna.Study, output_dir: Path, prefix: str = ""):
    """
    Export all available Optuna plots for a study.
    
    Args:
        study: Optuna study object
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames (e.g., "sac_" or "td3_")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_trials = len(study.trials)
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    
    print(f"\n{'='*70}")
    print(f"Exporting plots for: {study.study_name}")
    print(f"{'='*70}")
    print(f"Total trials: {n_trials}")
    print(f"Complete trials: {n_complete}")
    print(f"Output directory: {output_dir}")
    print(f"Prefix: {prefix if prefix else '(none)'}")
    print(f"{'='*70}\n")
    
    if n_complete < 2:
        print(f"⚠️  Warning: Only {n_complete} complete trial(s). Skipping visualizations.")
        return
    
    plots_generated = []
    
    # 1. Optimization History
    try:
        print("  Generating optimization history plot...")
        fig = plot_optimization_history(study)
        filepath = output_dir / f"{prefix}optimization_history.html"
        fig.write_html(str(filepath))
        plots_generated.append(("Optimization History", filepath))
        print(f"    ✓ Saved to: {filepath}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # 2. Parameter Importances
    try:
        print("  Generating parameter importances plot...")
        fig = plot_param_importances(study)
        filepath = output_dir / f"{prefix}param_importances.html"
        fig.write_html(str(filepath))
        plots_generated.append(("Parameter Importances", filepath))
        print(f"    ✓ Saved to: {filepath}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # 3. Parallel Coordinate Plot
    try:
        print("  Generating parallel coordinate plot...")
        fig = plot_parallel_coordinate(study)
        filepath = output_dir / f"{prefix}parallel_coordinate.html"
        fig.write_html(str(filepath))
        plots_generated.append(("Parallel Coordinate", filepath))
        print(f"    ✓ Saved to: {filepath}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # 4. Contour Plots
    try:
        print("  Generating contour plot...")
        fig = plot_contour(study)
        filepath = output_dir / f"{prefix}contour.html"
        fig.write_html(str(filepath))
        plots_generated.append(("Contour Plot", filepath))
        print(f"    ✓ Saved to: {filepath}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # 5. Slice Plot
    try:
        print("  Generating slice plot...")
        fig = plot_slice(study)
        filepath = output_dir / f"{prefix}slice.html"
        fig.write_html(str(filepath))
        plots_generated.append(("Slice Plot", filepath))
        print(f"    ✓ Saved to: {filepath}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # 6. EDF (Empirical Distribution Function)
    try:
        print("  Generating EDF plot...")
        fig = plot_edf(study)
        filepath = output_dir / f"{prefix}edf.html"
        fig.write_html(str(filepath))
        plots_generated.append(("Empirical Distribution", filepath))
        print(f"    ✓ Saved to: {filepath}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # 7. Timeline
    try:
        print("  Generating timeline plot...")
        fig = plot_timeline(study)
        filepath = output_dir / f"{prefix}timeline.html"
        fig.write_html(str(filepath))
        plots_generated.append(("Timeline", filepath))
        print(f"    ✓ Saved to: {filepath}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # 8. Pareto Front (if multi-objective and available)
    if HAS_PARETO and len(study.directions) > 1:
        try:
            print("  Generating Pareto front plot...")
            fig = plot_pareto_front(study)
            filepath = output_dir / f"{prefix}pareto_front.html"
            fig.write_html(str(filepath))
            plots_generated.append(("Pareto Front", filepath))
            print(f"    ✓ Saved to: {filepath}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Summary for {study.study_name}")
    print(f"{'='*70}")
    print(f"Plots generated: {len(plots_generated)}/{8 if not HAS_PARETO or len(study.directions) == 1 else 9}")
    for name, path in plots_generated:
        print(f"  ✓ {name}: {path}")
    print(f"{'='*70}\n")


def export_study_info(study: optuna.Study, output_dir: Path, prefix: str = ""):
    """
    Export study information to a text file.
    
    Args:
        study: Optuna study object
        output_dir: Directory to save info
        prefix: Prefix for filename
    """
    info_path = output_dir / f"{prefix}study_info.txt"
    
    best_trial = study.best_trial
    n_trials = len(study.trials)
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    
    with open(info_path, 'w') as f:
        f.write(f"Optuna Study Information\n")
        f.write(f"{'='*70}\n\n")
        
        f.write(f"Study Name: {study.study_name}\n")
        f.write(f"Direction: {study.direction.name}\n")
        f.write(f"Total Trials: {n_trials}\n")
        f.write(f"  - Complete: {n_complete}\n")
        f.write(f"  - Failed: {n_failed}\n")
        f.write(f"  - Pruned: {n_pruned}\n\n")
        
        f.write(f"Best Trial:\n")
        f.write(f"  - Number: {best_trial.number}\n")
        f.write(f"  - Value: {best_trial.value:.6f}\n")
        f.write(f"  - Params:\n")
        for key, value in best_trial.params.items():
            f.write(f"      {key}: {value}\n")
        
        f.write(f"\n{'='*70}\n")
        f.write(f"All Trials Summary:\n")
        f.write(f"{'='*70}\n\n")
        
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                f.write(f"Trial {trial.number}: Value={trial.value:.6f}\n")
                for key, value in trial.params.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
    
    print(f"Study info saved to: {info_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Export all Optuna visualization plots',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='optuna_studies/plots',
        help='Directory to save plots (default: optuna_studies/plots)'
    )
    
    parser.add_argument(
        '--studies-dir',
        type=str,
        default='optuna_studies',
        help='Directory containing Optuna database files (default: optuna_studies)'
    )
    
    parser.add_argument(
        '--sac-only',
        action='store_true',
        help='Only export SAC plots'
    )
    
    parser.add_argument(
        '--td3-only',
        action='store_true',
        help='Only export TD3 plots'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    output_dir = Path(args.output_dir)
    studies_dir = Path(args.studies_dir)
    
    print(f"\n{'='*70}")
    print(f"OPTUNA VISUALIZATION EXPORT")
    print(f"{'='*70}")
    print(f"Studies directory: {studies_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # Define study configurations
    studies_config = []
    
    if not args.td3_only:
        sac_db = studies_dir / "sac_optimization.db"
        if sac_db.exists():
            studies_config.append({
                'name': 'SAC',
                'db_path': str(sac_db),
                'study_name': 'sac_optimization',
                'output_subdir': output_dir / 'sac',
                'prefix': 'sac_'
            })
        else:
            print(f"⚠️  Warning: SAC database not found at {sac_db}")
    
    if not args.sac_only:
        td3_db = studies_dir / "td3_optimization.db"
        if td3_db.exists():
            studies_config.append({
                'name': 'TD3',
                'db_path': str(td3_db),
                'study_name': 'td3_optimization',
                'output_subdir': output_dir / 'td3',
                'prefix': 'td3_'
            })
        else:
            print(f"⚠️  Warning: TD3 database not found at {td3_db}")
    
    if not studies_config:
        print("❌ Error: No Optuna databases found!")
        sys.exit(1)
    
    # Process each study
    for config in studies_config:
        print(f"\n{'#'*70}")
        print(f"# Processing {config['name']} Study")
        print(f"{'#'*70}\n")
        
        try:
            # Load study
            print(f"Loading study from {config['db_path']}...")
            study = load_study(config['db_path'], config['study_name'])
            
            # Export plots
            export_all_plots(study, config['output_subdir'], config['prefix'])
            
            # Export study info
            export_study_info(study, config['output_subdir'], config['prefix'])
            
        except Exception as e:
            print(f"❌ Error processing {config['name']} study: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"EXPORT COMPLETE")
    print(f"{'='*70}")
    print(f"All plots saved to: {output_dir}")
    print(f"\nTo view the plots, open the .html files in a web browser:")
    print(f"  - SAC plots: {output_dir}/sac/")
    print(f"  - TD3 plots: {output_dir}/td3/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
