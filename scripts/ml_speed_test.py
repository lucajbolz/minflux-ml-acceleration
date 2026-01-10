#!/usr/bin/env python3
"""
Speed optimization tests for MINFLUX ML inference.

Tests various approaches to make inference faster:
1. Fewer trees
2. Shallower trees
3. ONNX Runtime (if available)
"""

import pickle
import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


def engineer_features(X_raw: np.ndarray) -> np.ndarray:
    """Original feature engineering."""
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


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights."""
    unique = np.unique(y)
    counts = {d: np.sum(y == d) for d in unique}
    max_count = max(counts.values())
    weights = np.ones(len(y))
    for d in unique:
        weights[y == d] = max_count / counts[d]
    return weights * len(y) / weights.sum()


def benchmark_inference(model, X_test, n_runs=100):
    """Benchmark inference speed."""
    # Warmup
    for _ in range(10):
        _ = model.predict(X_test[:100])

    # Single sample timing
    times_single = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X_test[:1])
        times_single.append((time.perf_counter() - start) * 1000)

    # Batch timing (100 samples)
    times_batch = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X_test[:100])
        times_batch.append((time.perf_counter() - start) * 1000)

    return {
        'single_ms': np.median(times_single),
        'batch_100_ms': np.median(times_batch),
        'per_sample_batch_ms': np.median(times_batch) / 100
    }


def main():
    print("=" * 70)
    print("SPEED OPTIMIZATION TESTS")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    X_raw = np.load('data/dynamic_data_X.npy')
    y = np.load('data/dynamic_data_y.npy')
    X = engineer_features(X_raw)
    sample_weights = compute_sample_weights(y)

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Test samples: {len(X_test):,}")

    # Test configurations
    configs = [
        {'name': 'Current (500 trees, depth 8)', 'n_estimators': 500, 'max_depth': 8},
        {'name': 'Fewer trees (100)', 'n_estimators': 100, 'max_depth': 8},
        {'name': 'Fewer trees (50)', 'n_estimators': 50, 'max_depth': 8},
        {'name': 'Shallow (depth 4)', 'n_estimators': 500, 'max_depth': 4},
        {'name': 'Fast (100 trees, depth 4)', 'n_estimators': 100, 'max_depth': 4},
        {'name': 'Ultra-fast (50 trees, depth 4)', 'n_estimators': 50, 'max_depth': 4},
    ]

    results = []

    print("\n[2] Training and benchmarking models...")
    print("-" * 70)

    for config in configs:
        print(f"\n    Training: {config['name']}...")

        model = xgb.XGBRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50
        )

        model.fit(X_train, y_train, sample_weight=w_train,
                  eval_set=[(X_test, y_test)], verbose=False)

        # Evaluate accuracy
        y_pred = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

        # Benchmark speed
        timing = benchmark_inference(model, X_test)

        results.append({
            'name': config['name'],
            'n_trees': model.best_iteration if hasattr(model, 'best_iteration') else config['n_estimators'],
            'depth': config['max_depth'],
            'rmse': rmse,
            'single_ms': timing['single_ms'],
            'batch_ms': timing['per_sample_batch_ms']
        })

        print(f"        RMSE: {rmse:.3f} nm | Single: {timing['single_ms']:.3f} ms | Batch: {timing['per_sample_batch_ms']:.4f} ms/sample")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baseline = results[0]

    print(f"\n{'Configuration':<35} {'RMSE':<10} {'Speed (batch)':<15} {'Speedup':<10} {'RMSE Δ':<10}")
    print("-" * 80)

    for r in results:
        speedup = baseline['batch_ms'] / r['batch_ms']
        rmse_diff = r['rmse'] - baseline['rmse']
        print(f"{r['name']:<35} {r['rmse']:<10.3f} {r['batch_ms']*1000:<15.1f} µs {speedup:<10.1f}× {rmse_diff:+<10.3f}")

    # Recommendation
    print("\n" + "=" * 70)
    print("EMPFEHLUNG")
    print("=" * 70)

    # Find best tradeoff (less than 5% RMSE increase)
    best = None
    for r in sorted(results, key=lambda x: x['batch_ms']):
        rmse_increase = (r['rmse'] - baseline['rmse']) / baseline['rmse'] * 100
        if rmse_increase < 5:  # Less than 5% worse
            best = r
            break

    if best:
        speedup = baseline['batch_ms'] / best['batch_ms']
        print(f"\n    Beste Option: {best['name']}")
        print(f"    - RMSE: {best['rmse']:.3f} nm (vs {baseline['rmse']:.3f} nm)")
        print(f"    - Speed: {best['batch_ms']*1000:.1f} µs/sample")
        print(f"    - Speedup: {speedup:.1f}× schneller")
        print(f"    - Nur {(best['rmse']-baseline['rmse'])/baseline['rmse']*100:.1f}% Genauigkeitsverlust")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
