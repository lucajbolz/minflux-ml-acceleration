#!/usr/bin/env python3
"""
Train XGBoost model with balanced sample weights to fix systematic bias.

The original dynamic model is biased toward 20nm because 72% of training
data comes from that distance. This script applies inverse-frequency
weighting to give equal importance to all distances.
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


def engineer_features(X_raw: np.ndarray) -> np.ndarray:
    """
    Apply feature engineering to raw MINFLUX data.

    Args:
        X_raw: Array of shape (n_samples, 12) with [photons(6), positions(6)]

    Returns:
        Engineered features of shape (n_samples, 15)
    """
    photons = X_raw[:, :6].copy()
    positions = X_raw[:, 6:].copy()

    # Ensure non-negative photons
    photons = np.maximum(photons, 0)

    total_photons = photons.sum(axis=1, keepdims=True)
    total_photons = np.maximum(total_photons, 1e-8)  # Avoid division by zero

    photon_ratios = photons / total_photons

    # Modulation depth
    mod_x = photons[:, 0] + photons[:, 2] - 2 * photons[:, 1]
    mod_y = photons[:, 3] + photons[:, 5] - 2 * photons[:, 4]
    modulation = np.stack([mod_x, mod_y], axis=1) / total_photons

    # Log total (ensure positive input)
    log_total = np.log(np.maximum(total_photons, 1))

    # Normalize positions
    pos_mean = positions.mean(axis=0, keepdims=True)
    pos_std = positions.std(axis=0, keepdims=True) + 1e-8
    positions_norm = (positions - pos_mean) / pos_std

    features = np.concatenate([
        photon_ratios, positions_norm, modulation, log_total
    ], axis=1).astype(np.float32)

    # Replace any remaining inf/nan with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def compute_sample_weights(y: np.ndarray, method: str = 'inverse') -> np.ndarray:
    """
    Compute sample weights for balanced training.

    Args:
        y: Target distances
        method: 'inverse' (inverse frequency) or 'sqrt_inverse' (less aggressive)

    Returns:
        Sample weights array
    """
    unique_distances = np.unique(y)
    counts = {d: np.sum(y == d) for d in unique_distances}
    total = len(y)

    print(f"\nClass distribution:")
    for d in sorted(counts.keys()):
        pct = counts[d] / total * 100
        print(f"  {d:.0f}nm: {counts[d]:,} samples ({pct:.1f}%)")

    # Compute weights
    weights = np.ones(len(y))

    if method == 'inverse':
        # Inverse frequency: rare classes get higher weight
        max_count = max(counts.values())
        for d in unique_distances:
            mask = y == d
            weights[mask] = max_count / counts[d]
    elif method == 'sqrt_inverse':
        # Square root of inverse: less aggressive balancing
        max_count = max(counts.values())
        for d in unique_distances:
            mask = y == d
            weights[mask] = np.sqrt(max_count / counts[d])
    elif method == 'equal':
        # Equal weight per class (regardless of sample count)
        n_classes = len(unique_distances)
        for d in unique_distances:
            mask = y == d
            weights[mask] = total / (n_classes * counts[d])

    # Normalize weights to sum to n_samples
    weights = weights * len(y) / weights.sum()

    print(f"\nEffective weights ({method}):")
    for d in sorted(counts.keys()):
        mask = y == d
        print(f"  {d:.0f}nm: {weights[mask].mean():.2f}x")

    return weights


def train_balanced_model(
    data_dir: str = 'data',
    output_path: str = 'models/xgboost_balanced.pkl',
    weight_method: str = 'inverse'
) -> dict:
    """
    Train XGBoost model with balanced sample weights.

    Args:
        data_dir: Directory containing extracted data
        output_path: Where to save trained model
        weight_method: 'inverse', 'sqrt_inverse', or 'equal'

    Returns:
        Dictionary with training results
    """
    print("=" * 70)
    print("BALANCED MODEL TRAINING")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    X_path = Path(data_dir) / 'dynamic_data_X.npy'
    y_path = Path(data_dir) / 'dynamic_data_y.npy'

    if not X_path.exists():
        raise FileNotFoundError(f"Data not found at {X_path}")

    X_raw = np.load(X_path)
    y = np.load(y_path)
    print(f"    Loaded {len(y):,} samples")

    # Engineer features
    print("\n[2] Engineering features...")
    X = engineer_features(X_raw)
    print(f"    Features: {X.shape[1]}")

    # Compute sample weights
    print("\n[3] Computing sample weights...")
    sample_weights = compute_sample_weights(y, method=weight_method)

    # Train/test split (stratified)
    print("\n[4] Splitting data...")
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"    Train: {len(y_train):,} samples")
    print(f"    Test:  {len(y_test):,} samples")

    # Train model
    print("\n[5] Training XGBoost with sample weights...")
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

    start_time = time.time()
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    train_time = time.time() - start_time
    print(f"    Training time: {train_time:.1f}s")
    print(f"    Best iteration: {model.best_iteration}")

    # Evaluate
    print("\n[6] Evaluating model...")
    y_pred = model.predict(X_test)

    # Overall metrics
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    mae = np.mean(np.abs(y_pred - y_test))
    print(f"\n    Overall RMSE: {rmse:.3f} nm")
    print(f"    Overall MAE:  {mae:.3f} nm")

    # Per-distance metrics
    print("\n    Per-distance performance:")
    print("    Distance | RMSE (nm) | Bias (nm) | Count")
    print("    ---------|-----------|-----------|-------")

    results_per_dist = {}
    for d in sorted(np.unique(y_test)):
        mask = y_test == d
        d_pred = y_pred[mask]
        d_true = y_test[mask]

        d_rmse = np.sqrt(np.mean((d_pred - d_true) ** 2))
        d_bias = np.mean(d_pred - d_true)
        d_count = mask.sum()

        results_per_dist[d] = {'rmse': d_rmse, 'bias': d_bias, 'count': d_count}
        print(f"    {d:7.0f} | {d_rmse:9.3f} | {d_bias:+9.3f} | {d_count:,}")

    # Save model
    print(f"\n[7] Saving model to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return {
        'rmse': rmse,
        'mae': mae,
        'per_distance': results_per_dist,
        'train_time': train_time,
        'best_iteration': model.best_iteration
    }


def compare_models():
    """Compare original vs balanced model."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # Load test data
    X_raw = np.load('data/dynamic_data_X.npy')
    y = np.load('data/dynamic_data_y.npy')
    X = engineer_features(X_raw)

    # Use same test split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Original': 'models/xgboost_dynamic.pkl',
        'Balanced': 'models/xgboost_balanced.pkl'
    }

    print("\n    Distance | Original RMSE | Balanced RMSE | Improvement")
    print("    ---------|---------------|---------------|------------")

    for dist in sorted(np.unique(y_test)):
        mask = y_test == dist
        y_true = y_test[mask]
        X_dist = X_test[mask]

        orig_rmse = None
        bal_rmse = None

        for name, path in models.items():
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                y_pred = model.predict(X_dist)
                rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

                if name == 'Original':
                    orig_rmse = rmse
                else:
                    bal_rmse = rmse
            except FileNotFoundError:
                pass

        if orig_rmse and bal_rmse:
            improvement = (orig_rmse - bal_rmse) / orig_rmse * 100
            print(f"    {dist:7.0f} | {orig_rmse:13.3f} | {bal_rmse:13.3f} | {improvement:+10.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Train balanced XGBoost model for MINFLUX distance estimation'
    )
    parser.add_argument(
        '--data_dir',
        default='data',
        help='Directory containing extracted data'
    )
    parser.add_argument(
        '--output',
        default='models/xgboost_balanced.pkl',
        help='Output path for trained model'
    )
    parser.add_argument(
        '--weight_method',
        choices=['inverse', 'sqrt_inverse', 'equal'],
        default='inverse',
        help='Sample weighting method'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare with original model after training'
    )

    args = parser.parse_args()

    train_balanced_model(args.data_dir, args.output, args.weight_method)

    if args.compare:
        compare_models()


if __name__ == '__main__':
    main()
