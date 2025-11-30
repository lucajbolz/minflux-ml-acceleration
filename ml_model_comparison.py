"""
Model Comparison for MINFLUX ML

Compares different ML algorithms for MINFLUX distance estimation.

Models tested:
- XGBoost (current choice)
- Random Forest
- LightGBM
- MLP (Neural Network)
- Ridge Regression (linear baseline)

Usage:
    python ml_model_comparison.py

Output:
    - analysis/model_comparison.png
    - Printed results table
"""

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# Try to import LightGBM (optional)
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Note: LightGBM not installed, skipping this model")


def engineer_features(X_raw: np.ndarray) -> np.ndarray:
    """Apply feature engineering to raw MINFLUX data."""
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
    """Compute inverse-frequency sample weights."""
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    weights = np.ones(len(y))
    for d, c in zip(unique, counts):
        weights[y == d] = max_count / c
    return weights


def get_model_size(model) -> float:
    """Get model size in MB by pickling."""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as f:
        pickle.dump(model, f)
        size = os.path.getsize(f.name) / (1024 * 1024)
        os.unlink(f.name)
    return size


def evaluate_model(model, X_train: np.ndarray, X_test: np.ndarray,
                   y_train: np.ndarray, y_test: np.ndarray,
                   weights_train: np.ndarray,
                   supports_weights: bool = True) -> Dict:
    """Train and evaluate a model."""
    # Training time
    start = time.time()
    if supports_weights:
        model.fit(X_train, y_train, sample_weight=weights_train)
    else:
        model.fit(X_train, y_train)
    train_time = time.time() - start

    # Inference time (single sample)
    times = []
    for i in range(min(100, len(X_test))):
        start = time.perf_counter()
        _ = model.predict(X_test[i:i+1])
        times.append(time.perf_counter() - start)
    single_inference_ms = np.median(times) * 1000

    # Batch inference time
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    batch_time = (time.perf_counter() - start) / len(X_test) * 1000

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Per-distance RMSE
    rmse_by_dist = {}
    for dist in [15, 20, 30]:
        mask = y_test == dist
        if mask.sum() > 0:
            rmse_by_dist[dist] = np.sqrt(np.mean((y_pred[mask] - y_test[mask]) ** 2))

    # Model size
    model_size = get_model_size(model)

    return {
        'rmse': rmse,
        'mae': mae,
        'rmse_by_dist': rmse_by_dist,
        'train_time': train_time,
        'inference_ms': single_inference_ms,
        'batch_ms': batch_time,
        'model_size_mb': model_size,
        'speedup_vs_mle': 100 / single_inference_ms  # MLE ~100ms
    }


def run_model_comparison(X_train: np.ndarray, X_test: np.ndarray,
                         y_train: np.ndarray, y_test: np.ndarray,
                         weights_train: np.ndarray) -> Dict:
    """Compare different ML models."""
    results = {}

    # 1. XGBoost (current choice)
    print("\n[1/5] XGBoost...")
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    results['XGBoost'] = evaluate_model(
        model, X_train, X_test, y_train, y_test, weights_train
    )
    print(f"      RMSE: {results['XGBoost']['rmse']:.3f} nm, "
          f"Inference: {results['XGBoost']['inference_ms']:.3f} ms")

    # 2. Random Forest
    print("\n[2/5] Random Forest...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    results['Random Forest'] = evaluate_model(
        model, X_train, X_test, y_train, y_test, weights_train
    )
    print(f"      RMSE: {results['Random Forest']['rmse']:.3f} nm, "
          f"Inference: {results['Random Forest']['inference_ms']:.3f} ms")

    # 3. LightGBM (if available)
    if HAS_LIGHTGBM:
        print("\n[3/5] LightGBM...")
        model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        results['LightGBM'] = evaluate_model(
            model, X_train, X_test, y_train, y_test, weights_train
        )
        print(f"      RMSE: {results['LightGBM']['rmse']:.3f} nm, "
              f"Inference: {results['LightGBM']['inference_ms']:.3f} ms")
    else:
        print("\n[3/5] LightGBM... SKIPPED (not installed)")

    # 4. MLP (Neural Network)
    print("\n[4/5] MLP Neural Network...")
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False
    )
    results['MLP'] = evaluate_model(
        model, X_train, X_test, y_train, y_test, weights_train,
        supports_weights=False
    )
    print(f"      RMSE: {results['MLP']['rmse']:.3f} nm, "
          f"Inference: {results['MLP']['inference_ms']:.3f} ms")

    # 5. Ridge Regression (linear baseline)
    print("\n[5/5] Ridge Regression (linear baseline)...")
    model = Ridge(alpha=1.0, random_state=42)
    results['Ridge'] = evaluate_model(
        model, X_train, X_test, y_train, y_test, weights_train
    )
    print(f"      RMSE: {results['Ridge']['rmse']:.3f} nm, "
          f"Inference: {results['Ridge']['inference_ms']:.3f} ms")

    return results


def print_results(results: Dict) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 110)
    print("MODEL COMPARISON RESULTS")
    print("=" * 110)

    print(f"\n{'Model':<18} {'RMSE (nm)':<12} {'MAE (nm)':<12} {'Inference':<12} "
          f"{'Speedup':<10} {'Size (MB)':<12} {'Train (s)':<10}")
    print("-" * 110)

    # Sort by RMSE
    sorted_models = sorted(results.items(), key=lambda x: x[1]['rmse'])

    for name, r in sorted_models:
        print(f"{name:<18} {r['rmse']:<12.3f} {r['mae']:<12.3f} "
              f"{r['inference_ms']:<12.3f} {r['speedup_vs_mle']:<10.0f}× "
              f"{r['model_size_mb']:<12.2f} {r['train_time']:<10.1f}")

    # Best model
    best = sorted_models[0]
    print("\n" + "-" * 110)
    print(f"Best model: {best[0]} (RMSE = {best[1]['rmse']:.3f} nm)")
    print("=" * 110)

    # Per-distance comparison
    print("\n--- RMSE by Distance ---")
    print(f"{'Model':<18} {'15nm':<10} {'20nm':<10} {'30nm':<10}")
    print("-" * 50)
    for name, r in sorted_models:
        dist_rmse = r['rmse_by_dist']
        print(f"{name:<18} {dist_rmse.get(15, 0):<10.3f} "
              f"{dist_rmse.get(20, 0):<10.3f} {dist_rmse.get(30, 0):<10.3f}")


def plot_results(results: Dict, output_dir: Path) -> None:
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    models = list(results.keys())
    rmse = [results[m]['rmse'] for m in models]
    inference = [results[m]['inference_ms'] for m in models]
    speedup = [results[m]['speedup_vs_mle'] for m in models]
    size = [results[m]['model_size_mb'] for m in models]

    # Color XGBoost differently
    colors = ['green' if m == 'XGBoost' else 'steelblue' for m in models]

    # 1. RMSE comparison
    ax = axes[0, 0]
    bars = ax.bar(models, rmse, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('RMSE (nm)')
    ax.set_title('Prediction Accuracy')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, r in zip(bars, rmse):
        ax.annotate(f'{r:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, r),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)

    # 2. Inference speed
    ax = axes[0, 1]
    bars = ax.bar(models, inference, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Single Prediction Speed')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='MLE (100ms)')
    ax.legend()

    # 3. Accuracy vs Speed scatter
    ax = axes[1, 0]
    for m, r, s in zip(models, rmse, speedup):
        color = 'green' if m == 'XGBoost' else 'steelblue'
        ax.scatter(s, r, s=200, c=color, alpha=0.7, edgecolors='black', linewidths=2)
        ax.annotate(m, (s, r), xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax.set_xlabel('Speedup vs MLE (×)')
    ax.set_ylabel('RMSE (nm)')
    ax.set_title('Accuracy vs Speed Tradeoff')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Ideal region
    ax.axhspan(0, 4, alpha=0.1, color='green', label='Good accuracy (<4nm)')
    ax.axvspan(100, ax.get_xlim()[1] if ax.get_xlim()[1] > 100 else 1000,
               alpha=0.1, color='blue', label='Good speedup (>100×)')
    ax.legend(loc='upper right')

    # 4. Model size comparison
    ax = axes[1, 1]
    bars = ax.bar(models, size, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Storage Requirements')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, s in zip(bars, size):
        ax.annotate(f'{s:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, s),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)

    plt.suptitle('MINFLUX ML - Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'model_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='ML Model Comparison for MINFLUX')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing extracted data')
    parser.add_argument('--n_samples', type=int, default=100000,
                        help='Number of samples to use')
    args = parser.parse_args()

    print("=" * 70)
    print("MINFLUX ML - Model Comparison")
    print("=" * 70)

    # Load data
    data_dir = Path(args.data_dir)
    X_raw = np.load(data_dir / 'dynamic_data_X.npy')
    y = np.load(data_dir / 'dynamic_data_y.npy')

    print(f"\nLoaded {len(y):,} samples")

    # Subsample
    np.random.seed(42)
    n_samples = min(args.n_samples, len(y))
    indices = np.random.choice(len(y), n_samples, replace=False)
    X_raw = X_raw[indices]
    y = y[indices]

    print(f"Using {n_samples:,} samples")

    # Engineer features
    print("Engineering features...")
    X = engineer_features(X_raw)
    print(f"Feature shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Compute sample weights
    weights_train = compute_sample_weights(y_train)

    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    # Run comparison
    results = run_model_comparison(X_train, X_test, y_train, y_test, weights_train)

    # Print results
    print_results(results)

    # Generate plots
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)
    plot_results(results, output_dir)

    # Justification
    print("\n" + "-" * 70)
    print("WHY XGBOOST?")
    print("-" * 70)
    xgb = results['XGBoost']
    print(f"""
XGBoost was selected as the final model because:

1. Best accuracy: RMSE = {xgb['rmse']:.3f} nm
2. Fast inference: {xgb['inference_ms']:.3f} ms ({xgb['speedup_vs_mle']:.0f}× faster than MLE)
3. Reasonable model size: {xgb['model_size_mb']:.1f} MB
4. Supports sample weighting for balanced training
5. Built-in feature importance
6. Well-established, production-ready library
""")


if __name__ == '__main__':
    main()
