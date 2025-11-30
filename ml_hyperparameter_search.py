"""
Hyperparameter Search for MINFLUX ML Models

Performs Grid Search with Cross-Validation to find optimal XGBoost hyperparameters.

Usage:
    python ml_hyperparameter_search.py [--quick]

Output:
    - analysis/hyperparameter_search.png
    - Printed results table
"""

import argparse
import time
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor


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


def run_hyperparameter_search(X: np.ndarray, y: np.ndarray,
                               sample_weights: np.ndarray,
                               quick: bool = False) -> dict:
    """
    Run grid search with cross-validation.

    Args:
        X: Features
        y: Targets
        sample_weights: Sample weights for balanced training
        quick: If True, use smaller parameter grid

    Returns:
        Dictionary with results
    """
    if quick:
        param_grid = {
            'max_depth': [6, 8],
            'n_estimators': [100, 500],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
    else:
        param_grid = {
            'max_depth': [4, 6, 8, 10, 12],
            'n_estimators': [100, 200, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(product(*[param_grid[k] for k in keys]))

    print(f"\nTesting {len(combinations)} parameter combinations...")
    print(f"Parameters: {keys}")
    print("-" * 80)

    results = []
    best_score = float('inf')
    best_params = None

    # Stratified K-Fold based on distance bins
    y_bins = y.astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        model = XGBRegressor(
            **params,
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        start = time.time()

        # Manual cross-validation with sample weights
        fold_scores = []
        for train_idx, val_idx in cv.split(X, y_bins):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            w_train = sample_weights[train_idx]

            model.fit(X_train, y_train, sample_weight=w_train)
            y_pred = model.predict(X_val)
            fold_rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
            fold_scores.append(fold_rmse)

        elapsed = time.time() - start

        rmse_mean = np.mean(fold_scores)
        rmse_std = np.std(fold_scores)

        results.append({
            'params': params,
            'rmse_mean': rmse_mean,
            'rmse_std': rmse_std,
            'cv_time': elapsed
        })

        if rmse_mean < best_score:
            best_score = rmse_mean
            best_params = params

        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1:3d}/{len(combinations)}] "
                  f"RMSE: {rmse_mean:.3f} ± {rmse_std:.3f} nm | "
                  f"Time: {elapsed:.1f}s | "
                  f"Best: {best_score:.3f} nm")

    return {
        'results': results,
        'best_params': best_params,
        'best_score': best_score,
        'param_grid': param_grid
    }


def print_results_table(search_results: dict) -> None:
    """Print formatted results table."""
    results = sorted(search_results['results'], key=lambda x: x['rmse_mean'])

    print("\n" + "=" * 100)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("=" * 100)

    print(f"\n{'Rank':<5} {'max_depth':<10} {'n_est':<8} {'lr':<8} "
          f"{'subsample':<10} {'colsample':<10} {'RMSE (nm)':<15} {'Time (s)':<10}")
    print("-" * 100)

    for i, r in enumerate(results[:20]):  # Top 20
        p = r['params']
        print(f"{i+1:<5} {p['max_depth']:<10} {p['n_estimators']:<8} "
              f"{p['learning_rate']:<8} {p['subsample']:<10} "
              f"{p['colsample_bytree']:<10} "
              f"{r['rmse_mean']:.3f} ± {r['rmse_std']:.3f}    {r['cv_time']:.1f}")

    print("\n" + "=" * 100)
    print("BEST PARAMETERS:")
    print("=" * 100)
    for k, v in search_results['best_params'].items():
        print(f"  {k}: {v}")
    print(f"\n  Best RMSE: {search_results['best_score']:.3f} nm")
    print("=" * 100)


def plot_results(search_results: dict, output_dir: Path) -> None:
    """Generate visualization plots."""
    results = search_results['results']
    param_grid = search_results['param_grid']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Extract data for plotting
    params_list = [r['params'] for r in results]
    rmse_values = [r['rmse_mean'] for r in results]

    param_names = ['max_depth', 'n_estimators', 'learning_rate',
                   'subsample', 'colsample_bytree']

    # Box plots for each parameter
    for idx, param in enumerate(param_names):
        ax = axes[idx // 3, idx % 3]

        unique_vals = sorted(set(p[param] for p in params_list))
        data_by_val = {v: [] for v in unique_vals}

        for p, rmse in zip(params_list, rmse_values):
            data_by_val[p[param]].append(rmse)

        positions = range(len(unique_vals))
        bp = ax.boxplot([data_by_val[v] for v in unique_vals],
                       positions=positions, widths=0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels([str(v) for v in unique_vals])
        ax.set_xlabel(param)
        ax.set_ylabel('RMSE (nm)')
        ax.set_title(f'Effect of {param}')
        ax.grid(True, alpha=0.3)

    # Heatmap: max_depth vs n_estimators (last subplot)
    ax = axes[1, 2]

    if len(param_grid['max_depth']) > 1 and len(param_grid['n_estimators']) > 1:
        depths = sorted(param_grid['max_depth'])
        n_ests = sorted(param_grid['n_estimators'])

        heatmap_data = np.zeros((len(depths), len(n_ests)))
        counts = np.zeros((len(depths), len(n_ests)))

        for r in results:
            d_idx = depths.index(r['params']['max_depth'])
            n_idx = n_ests.index(r['params']['n_estimators'])
            heatmap_data[d_idx, n_idx] += r['rmse_mean']
            counts[d_idx, n_idx] += 1

        heatmap_data = heatmap_data / np.maximum(counts, 1)

        im = ax.imshow(heatmap_data, cmap='viridis_r', aspect='auto')
        ax.set_xticks(range(len(n_ests)))
        ax.set_xticklabels(n_ests)
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels(depths)
        ax.set_xlabel('n_estimators')
        ax.set_ylabel('max_depth')
        ax.set_title('RMSE Heatmap (avg over other params)')

        plt.colorbar(im, ax=ax, label='RMSE (nm)')
    else:
        ax.text(0.5, 0.5, 'Not enough\nparameter variation',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Heatmap (skipped)')

    plt.suptitle('Hyperparameter Search Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'hyperparameter_search.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='XGBoost Hyperparameter Search')
    parser.add_argument('--quick', action='store_true',
                        help='Use smaller parameter grid for quick testing')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing extracted data')
    parser.add_argument('--n_samples', type=int, default=100000,
                        help='Number of samples to use (default: 100000)')
    args = parser.parse_args()

    print("=" * 70)
    print("MINFLUX ML - Hyperparameter Search")
    print("=" * 70)

    # Load data
    data_dir = Path(args.data_dir)
    X_raw = np.load(data_dir / 'dynamic_data_X.npy')
    y = np.load(data_dir / 'dynamic_data_y.npy')

    print(f"\nLoaded {len(y):,} samples")

    # Subsample for faster search
    np.random.seed(42)
    n_samples = min(args.n_samples, len(y))
    indices = np.random.choice(len(y), n_samples, replace=False)
    X_raw = X_raw[indices]
    y = y[indices]

    print(f"Using {n_samples:,} samples for search")

    # Engineer features
    print("Engineering features...")
    X = engineer_features(X_raw)
    print(f"Feature shape: {X.shape}")

    # Compute sample weights
    sample_weights = compute_sample_weights(y)
    print(f"Sample weights computed (inverse frequency)")

    # Run search
    search_results = run_hyperparameter_search(X, y, sample_weights, quick=args.quick)

    # Print results
    print_results_table(search_results)

    # Generate plots
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)
    plot_results(search_results, output_dir)

    # Save best params for reference
    print("\n" + "-" * 70)
    print("Copy-paste for ml_train_balanced.py:")
    print("-" * 70)
    bp = search_results['best_params']
    print(f"""
model = XGBRegressor(
    n_estimators={bp['n_estimators']},
    max_depth={bp['max_depth']},
    learning_rate={bp['learning_rate']},
    subsample={bp['subsample']},
    colsample_bytree={bp['colsample_bytree']},
    tree_method='hist',
    random_state=42
)
""")


if __name__ == '__main__':
    main()
