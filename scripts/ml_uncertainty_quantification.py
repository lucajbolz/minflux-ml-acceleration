"""
Uncertainty Quantification for MINFLUX ML Models using Conformal Prediction

Uses MAPIE (Model Agnostic Prediction Interval Estimator) to provide
90% confidence intervals for distance predictions.

Usage:
    python ml_uncertainty_quantification.py --model static
    python ml_uncertainty_quantification.py --model dynamic
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from mapie.regression import SplitConformalRegressor

# Nature Physics style
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300


def engineer_features(X_raw):
    """
    Apply same feature engineering as training.

    Args:
        X_raw: (N, 12) array with [photons, positions] interleaved

    Returns:
        X: (N, 15) engineered features
    """
    photons = X_raw[:, 0::2]  # 6 photon counts
    positions = X_raw[:, 1::2]  # 6 positions

    # Photon ratios
    total_photons = photons.sum(axis=1, keepdims=True)
    photon_ratios = photons / (total_photons + 1e-8)

    # Modulation depth
    mod_x = photons[:, 0] + photons[:, 2] - 2 * photons[:, 1]
    mod_y = photons[:, 3] + photons[:, 5] - 2 * photons[:, 4]
    modulation = np.stack([mod_x, mod_y], axis=1) / (total_photons + 1e-8)

    # Log total photons
    log_total = np.log(total_photons + 1)

    # Normalize positions
    pos_mean = positions.mean(axis=0, keepdims=True)
    pos_std = positions.std(axis=0, keepdims=True) + 1e-8
    positions_norm = (positions - pos_mean) / pos_std

    # Combine features
    X = np.concatenate([
        photon_ratios,      # 6
        positions_norm,     # 6
        modulation,         # 2
        log_total          # 1
    ], axis=1).astype(np.float32)

    return X


def load_model_and_data(model_type='static'):
    """
    Load XGBoost model and corresponding data.

    Args:
        model_type: 'static' or 'dynamic'

    Returns:
        model, X, y
    """
    print(f"\n[1] Loading {model_type} model and data...")

    if model_type == 'static':
        model_path = 'models/xgboost_optimized.pkl'
        data_path = 'data/paper_data_with_pos'
    elif model_type == 'dynamic':
        model_path = 'models/xgboost_dynamic.pkl'
        data_path = 'data/dynamic_data'
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"   ✓ Loaded model: {model_path}")

    # Load data
    X_raw = np.load(f'{data_path}_X.npy')
    y = np.load(f'{data_path}_y.npy')
    print(f"   ✓ Loaded data: {len(X_raw):,} samples")

    # Engineer features
    X = engineer_features(X_raw)
    print(f"   ✓ Engineered features: {X.shape[1]} features")

    return model, X, y


def calibrate_uncertainty(model, X, y, alpha=0.1):
    """
    Calibrate uncertainty using MAPIE with conformal prediction.

    Args:
        model: Trained XGBoost model
        X: Feature matrix
        y: Target values
        alpha: Significance level (0.1 for 90% CI)

    Returns:
        mapie_estimator: Calibrated MAPIE estimator
        X_test, y_test: Test set
        y_pred, y_intervals: Predictions and intervals
        coverage: Empirical coverage
    """
    print(f"\n[2] Calibrating uncertainty quantification...")

    # Split: 70% train, 15% calibration, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42  # 0.15/0.85 ≈ 0.176
    )

    print(f"   Train set:       {len(X_train):,} samples")
    print(f"   Calibration set: {len(X_calib):,} samples")
    print(f"   Test set:        {len(X_test):,} samples")

    # Create Split Conformal Regressor with pretrained model
    mapie = SplitConformalRegressor(
        estimator=model,
        prefit=True,
        confidence_level=1 - alpha  # 90% CI -> confidence_level=0.9
    )

    print(f"\n   Calibrating MAPIE estimator...")
    # Conformalize using calibration data (model is already trained)
    mapie.conformalize(X_calib, y_calib)

    # Predict with uncertainty
    print(f"   Computing predictions and intervals...")
    y_pred, y_intervals = mapie.predict_interval(X_test)

    # Calculate coverage
    y_lower = y_intervals[:, 0, 0]
    y_upper = y_intervals[:, 1, 0]
    coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))

    print(f"\n   ✓ Calibration complete")
    print(f"   Target coverage: {(1-alpha)*100:.0f}%")
    print(f"   Empirical coverage: {coverage*100:.1f}%")

    return mapie, X_test, y_test, y_pred, y_intervals, coverage


def evaluate_uncertainty(y_test, y_pred, y_intervals, model_type):
    """
    Evaluate uncertainty quantification quality.

    Args:
        y_test: Ground truth
        y_pred: Predictions
        y_intervals: Prediction intervals (N, 2, 1)
        model_type: 'static' or 'dynamic'

    Returns:
        metrics: Dictionary of metrics
    """
    print(f"\n[3] Evaluating uncertainty quality...")

    y_lower = y_intervals[:, 0, 0]
    y_upper = y_intervals[:, 1, 0]
    interval_widths = y_upper - y_lower

    # Coverage
    coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))

    # RMSE
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))

    # Mean interval width
    mean_width = np.mean(interval_widths)
    median_width = np.median(interval_widths)

    # Sharpness (interval width / prediction)
    sharpness = np.mean(interval_widths / (y_pred + 1e-8))

    metrics = {
        'coverage': coverage,
        'rmse': rmse,
        'mean_interval_width': mean_width,
        'median_interval_width': median_width,
        'sharpness': sharpness
    }

    print(f"\n   Metrics:")
    print(f"   ├─ Coverage:          {coverage*100:.1f}%")
    print(f"   ├─ RMSE:              {rmse:.2f} nm")
    print(f"   ├─ Mean interval:     {mean_width:.2f} nm")
    print(f"   ├─ Median interval:   {median_width:.2f} nm")
    print(f"   └─ Sharpness:         {sharpness:.2%}")

    return metrics


def plot_uncertainty(y_test, y_pred, y_intervals, model_type, metrics):
    """
    Create publication-quality plots for uncertainty quantification.

    Args:
        y_test: Ground truth
        y_pred: Predictions
        y_intervals: Prediction intervals
        model_type: 'static' or 'dynamic'
        metrics: Evaluation metrics
    """
    print(f"\n[4] Creating visualization...")

    y_lower = y_intervals[:, 0, 0]
    y_upper = y_intervals[:, 1, 0]

    # Sort by ground truth for better visualization
    sort_idx = np.argsort(y_test)
    y_test_sorted = y_test[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    y_lower_sorted = y_lower[sort_idx]
    y_upper_sorted = y_upper[sort_idx]

    # Sample 200 points for cleaner plot
    n_samples = min(200, len(y_test))
    step = len(y_test) // n_samples
    idx = np.arange(0, len(y_test), step)[:n_samples]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Predictions with confidence intervals
    ax1.fill_between(
        idx,
        y_lower_sorted[idx],
        y_upper_sorted[idx],
        alpha=0.3, color='#2563eb', label='90% Confidence Interval'
    )
    ax1.scatter(idx, y_test_sorted[idx], s=20, color='#dc2626',
                alpha=0.6, label='Ground Truth', zorder=3)
    ax1.scatter(idx, y_pred_sorted[idx], s=15, color='#059669',
                alpha=0.8, label='Prediction', zorder=4, marker='x')

    ax1.set_xlabel('Sample Index (sorted)', fontweight='normal')
    ax1.set_ylabel('Distance (nm)', fontweight='normal')
    ax1.set_title(f'{model_type.capitalize()} Model - Uncertainty Quantification',
                  fontweight='bold')
    ax1.legend(loc='best', frameon=True, edgecolor='black')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.2, linestyle='--')

    # Plot 2: Residuals with error bars
    residuals = y_test_sorted[idx] - y_pred_sorted[idx]
    errors = (y_upper_sorted[idx] - y_lower_sorted[idx]) / 2

    ax2.errorbar(y_test_sorted[idx], residuals, yerr=errors,
                 fmt='o', markersize=4, alpha=0.5, color='#2563eb',
                 ecolor='#93c5fd', capsize=2, capthick=1)
    ax2.axhline(y=0, color='#64748b', linestyle='--', linewidth=1.5)

    ax2.set_xlabel('Ground Truth Distance (nm)', fontweight='normal')
    ax2.set_ylabel('Residual (nm)', fontweight='normal')
    ax2.set_title('Residuals with 90% CI', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.2, linestyle='--')

    # Add metrics text
    metrics_text = (
        f"Coverage: {metrics['coverage']*100:.1f}%\n"
        f"RMSE: {metrics['rmse']:.2f} nm\n"
        f"Mean CI width: {metrics['mean_interval_width']:.2f} nm"
    )
    ax2.text(0.98, 0.02, metrics_text,
             transform=ax2.transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

    plt.tight_layout()

    output_file = f'uncertainty_{model_type}.png'
    plt.savefig(output_file, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {output_file}")


def save_calibrated_model(mapie_estimator, model_type):
    """
    Save calibrated MAPIE estimator.

    Args:
        mapie_estimator: Calibrated MAPIE model
        model_type: 'static' or 'dynamic'
    """
    output_path = f'models/mapie_{model_type}.pkl'

    with open(output_path, 'wb') as f:
        pickle.dump(mapie_estimator, f)

    print(f"\n[5] Saved calibrated model: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Uncertainty Quantification for MINFLUX ML'
    )
    parser.add_argument('--model', type=str, default='static',
                       choices=['static', 'dynamic'],
                       help='Model to calibrate (static or dynamic)')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Significance level (0.1 for 90%% CI)')

    args = parser.parse_args()

    print("="*70)
    print("MINFLUX ML - UNCERTAINTY QUANTIFICATION")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Confidence level: {(1-args.alpha)*100:.0f}%")

    # Load model and data
    model, X, y = load_model_and_data(args.model)

    # Calibrate uncertainty
    mapie, X_test, y_test, y_pred, y_intervals, coverage = calibrate_uncertainty(
        model, X, y, alpha=args.alpha
    )

    # Evaluate
    metrics = evaluate_uncertainty(y_test, y_pred, y_intervals, args.model)

    # Visualize
    plot_uncertainty(y_test, y_pred, y_intervals, args.model, metrics)

    # Save calibrated model
    save_calibrated_model(mapie, args.model)

    print("\n" + "="*70)
    print("✓ UNCERTAINTY QUANTIFICATION COMPLETE!")
    print("="*70)
    print(f"\nKey Results:")
    print(f"  Target coverage:    {(1-args.alpha)*100:.0f}%")
    print(f"  Empirical coverage: {coverage*100:.1f}%")
    print(f"  RMSE:               {metrics['rmse']:.2f} nm")
    print(f"  Mean CI width:      {metrics['mean_interval_width']:.2f} nm")
    print(f"\nOutput files:")
    print(f"  - uncertainty_{args.model}.png")
    print(f"  - models/mapie_{args.model}.pkl")
    print("="*70)


if __name__ == '__main__':
    main()
