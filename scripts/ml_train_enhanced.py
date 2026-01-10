#!/usr/bin/env python3
"""
Train XGBoost with enhanced physics-informed features.

Adds new features based on MINFLUX physics:
- SNR (Signal-to-Noise Ratio)
- Contrast per axis
- Pattern asymmetry
- Cramér-Rao bound related terms

Usage:
    python ml_train_enhanced.py
    python ml_train_enhanced.py --compare
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


def engineer_features_baseline(X_raw: np.ndarray) -> np.ndarray:
    """Original 15 features (baseline for comparison)."""
    photons = X_raw[:, :6].copy()
    positions = X_raw[:, 6:].copy()

    photons = np.maximum(photons, 0)
    total_photons = photons.sum(axis=1, keepdims=True)
    total_photons = np.maximum(total_photons, 1e-8)

    photon_ratios = photons / total_photons

    mod_x = photons[:, 0] + photons[:, 2] - 2 * photons[:, 1]
    mod_y = photons[:, 3] + photons[:, 5] - 2 * photons[:, 4]
    modulation = np.stack([mod_x, mod_y], axis=1) / total_photons

    log_total = np.log(np.maximum(total_photons, 1))

    pos_mean = positions.mean(axis=0, keepdims=True)
    pos_std = positions.std(axis=0, keepdims=True) + 1e-8
    positions_norm = (positions - pos_mean) / pos_std

    features = np.concatenate([
        photon_ratios, positions_norm, modulation, log_total
    ], axis=1).astype(np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def engineer_features_enhanced(X_raw: np.ndarray) -> np.ndarray:
    """
    Enhanced features with physics-informed additions.

    Original (15):
        - photon_ratios (6)
        - positions_norm (6)
        - modulation_x, modulation_y (2)
        - log_total (1)

    New physics-informed (7):
        - snr: Signal-to-Noise Ratio = sqrt(total)
        - contrast_x: (max-min)/(max+min) for x-beams
        - contrast_y: (max-min)/(max+min) for y-beams
        - asymmetry: |mod_x - mod_y| / (|mod_x| + |mod_y|)
        - crb_x: mod_x² / total (Cramér-Rao related)
        - crb_y: mod_y² / total (Cramér-Rao related)
        - total_modulation: sqrt(mod_x² + mod_y²) / total

    Total: 22 features
    """
    photons = X_raw[:, :6].copy()
    positions = X_raw[:, 6:].copy()

    photons = np.maximum(photons, 0)
    total_photons = photons.sum(axis=1, keepdims=True)
    total_photons_flat = total_photons.flatten()
    total_photons = np.maximum(total_photons, 1e-8)

    # === Original features ===
    photon_ratios = photons / total_photons

    mod_x = photons[:, 0] + photons[:, 2] - 2 * photons[:, 1]
    mod_y = photons[:, 3] + photons[:, 5] - 2 * photons[:, 4]
    modulation = np.stack([mod_x, mod_y], axis=1) / total_photons

    log_total = np.log(np.maximum(total_photons, 1))

    pos_mean = positions.mean(axis=0, keepdims=True)
    pos_std = positions.std(axis=0, keepdims=True) + 1e-8
    positions_norm = (positions - pos_mean) / pos_std

    # === New physics-informed features ===

    # 1. SNR: Signal-to-Noise Ratio
    # For Poisson: σ = √N, so SNR = N/σ = √N
    snr = np.sqrt(np.maximum(total_photons_flat, 1)).reshape(-1, 1)

    # 2. Contrast per axis: (max - min) / (max + min)
    # Measures how much the signal varies (related to distance from center)
    photons_x = photons[:, :3]  # First 3 beams (x-direction)
    photons_y = photons[:, 3:]  # Last 3 beams (y-direction)

    contrast_x = (photons_x.max(axis=1) - photons_x.min(axis=1)) / \
                 (photons_x.max(axis=1) + photons_x.min(axis=1) + 1e-8)
    contrast_y = (photons_y.max(axis=1) - photons_y.min(axis=1)) / \
                 (photons_y.max(axis=1) + photons_y.min(axis=1) + 1e-8)
    contrast = np.stack([contrast_x, contrast_y], axis=1)

    # 3. Asymmetry between x and y modulation
    # If the pattern is symmetric, both modulations should be similar
    mod_x_abs = np.abs(mod_x)
    mod_y_abs = np.abs(mod_y)
    asymmetry = (np.abs(mod_x_abs - mod_y_abs) /
                 (mod_x_abs + mod_y_abs + 1e-8)).reshape(-1, 1)

    # 4. Cramér-Rao related terms
    # Fisher information ~ (derivative of intensity)² / intensity
    # For modulation-based estimation: ~ mod² / total
    crb_x = (mod_x ** 2 / (total_photons_flat + 1e-8)).reshape(-1, 1)
    crb_y = (mod_y ** 2 / (total_photons_flat + 1e-8)).reshape(-1, 1)

    # 5. Total modulation magnitude
    total_mod = (np.sqrt(mod_x**2 + mod_y**2) /
                 (total_photons_flat + 1e-8)).reshape(-1, 1)

    # Combine all features
    features = np.concatenate([
        # Original (15)
        photon_ratios,      # 6
        positions_norm,     # 6
        modulation,         # 2
        log_total,          # 1
        # New physics-informed (7)
        snr,                # 1
        contrast,           # 2
        asymmetry,          # 1
        crb_x,              # 1
        crb_y,              # 1
        total_mod,          # 1
    ], axis=1).astype(np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency sample weights."""
    unique_distances = np.unique(y)
    counts = {d: np.sum(y == d) for d in unique_distances}
    max_count = max(counts.values())

    weights = np.ones(len(y))
    for d in unique_distances:
        weights[y == d] = max_count / counts[d]

    weights = weights * len(y) / weights.sum()
    return weights


def train_and_evaluate(X: np.ndarray, y: np.ndarray,
                       sample_weights: np.ndarray,
                       feature_name: str) -> dict:
    """Train model and return metrics."""

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, sample_weights,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )

    start = time.time()
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

    # Per-distance metrics
    per_dist = {}
    for d in sorted(np.unique(y_test)):
        mask = y_test == d
        d_rmse = np.sqrt(np.mean((y_pred[mask] - y_test[mask]) ** 2))
        d_bias = np.mean(y_pred[mask] - y_test[mask])
        per_dist[d] = {'rmse': d_rmse, 'bias': d_bias}

    return {
        'feature_set': feature_name,
        'n_features': X.shape[1],
        'rmse': rmse,
        'per_distance': per_dist,
        'train_time': train_time,
        'model': model
    }


def main():
    parser = argparse.ArgumentParser(description='Train with enhanced features')
    parser.add_argument('--compare', action='store_true',
                        help='Compare baseline vs enhanced features')
    parser.add_argument('--save', action='store_true',
                        help='Save enhanced model')
    args = parser.parse_args()

    print("=" * 70)
    print("ENHANCED FEATURE TRAINING")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    X_raw = np.load('data/dynamic_data_X.npy')
    y = np.load('data/dynamic_data_y.npy')
    print(f"    Samples: {len(y):,}")

    # Compute sample weights
    sample_weights = compute_sample_weights(y)

    # Generate feature sets
    print("\n[2] Engineering features...")
    X_baseline = engineer_features_baseline(X_raw)
    X_enhanced = engineer_features_enhanced(X_raw)
    print(f"    Baseline features: {X_baseline.shape[1]}")
    print(f"    Enhanced features: {X_enhanced.shape[1]}")

    # Train both
    print("\n[3] Training models...")
    print("\n--- Baseline (15 features) ---")
    result_baseline = train_and_evaluate(X_baseline, y, sample_weights, "Baseline")

    print("\n--- Enhanced (22 features) ---")
    result_enhanced = train_and_evaluate(X_enhanced, y, sample_weights, "Enhanced")

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Baseline (15)':<18} {'Enhanced (22)':<18} {'Δ':<12}")
    print("-" * 70)

    rmse_diff = result_enhanced['rmse'] - result_baseline['rmse']
    rmse_pct = rmse_diff / result_baseline['rmse'] * 100
    print(f"{'Overall RMSE':<20} {result_baseline['rmse']:<18.3f} {result_enhanced['rmse']:<18.3f} {rmse_diff:+.3f} ({rmse_pct:+.1f}%)")

    print(f"\n{'Distance':<12} {'Baseline RMSE':<15} {'Enhanced RMSE':<15} {'Baseline Bias':<15} {'Enhanced Bias':<15}")
    print("-" * 70)

    for d in sorted(result_baseline['per_distance'].keys()):
        b = result_baseline['per_distance'][d]
        e = result_enhanced['per_distance'][d]
        print(f"{d:<12.0f} {b['rmse']:<15.3f} {e['rmse']:<15.3f} {b['bias']:+<15.3f} {e['bias']:+<15.3f}")

    # Feature importance for enhanced model
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Enhanced Model)")
    print("=" * 70)

    feature_names = [
        'ratio_0', 'ratio_1', 'ratio_2', 'ratio_3', 'ratio_4', 'ratio_5',
        'pos_0', 'pos_1', 'pos_2', 'pos_3', 'pos_4', 'pos_5',
        'mod_x', 'mod_y', 'log_total',
        'snr', 'contrast_x', 'contrast_y', 'asymmetry', 'crb_x', 'crb_y', 'total_mod'
    ]

    importances = result_enhanced['model'].feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n{'Rank':<6} {'Feature':<15} {'Importance':<12} {'New?':<6}")
    print("-" * 45)
    for i, idx in enumerate(sorted_idx[:15]):
        is_new = "✓" if idx >= 15 else ""
        print(f"{i+1:<6} {feature_names[idx]:<15} {importances[idx]:<12.4f} {is_new:<6}")

    # Save if requested
    if args.save:
        output_path = 'models/xgboost_enhanced.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(result_enhanced['model'], f)
        print(f"\n✓ Enhanced model saved: {output_path}")

    print("\n" + "=" * 70)
    if result_enhanced['rmse'] < result_baseline['rmse']:
        improvement = (result_baseline['rmse'] - result_enhanced['rmse']) / result_baseline['rmse'] * 100
        print(f"✓ Enhanced features IMPROVED RMSE by {improvement:.2f}%")
    else:
        print("✗ Enhanced features did not improve RMSE")
    print("=" * 70)


if __name__ == '__main__':
    main()
