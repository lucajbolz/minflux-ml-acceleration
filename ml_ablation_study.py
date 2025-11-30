"""
Ablation Study for MINFLUX ML Features

Tests the importance of each feature group by removing them and measuring impact.

Usage:
    python ml_ablation_study.py

Output:
    - analysis/ablation_study.png
    - Printed results table
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# Feature group definitions
FEATURE_GROUPS = {
    'photon_ratios': list(range(0, 6)),      # Features 0-5
    'positions_norm': list(range(6, 12)),    # Features 6-11
    'modulation': list(range(12, 14)),       # Features 12-13
    'log_total': [14]                         # Feature 14
}

FEATURE_NAMES = [
    'ratio_0', 'ratio_1', 'ratio_2', 'ratio_3', 'ratio_4', 'ratio_5',
    'pos_0', 'pos_1', 'pos_2', 'pos_3', 'pos_4', 'pos_5',
    'mod_x', 'mod_y',
    'log_total'
]


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


def train_and_evaluate(X_train: np.ndarray, X_test: np.ndarray,
                       y_train: np.ndarray, y_test: np.ndarray,
                       weights_train: np.ndarray) -> Tuple[float, float, float]:
    """Train model and return metrics."""
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

    model.fit(X_train, y_train, sample_weight=weights_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    mae = np.mean(np.abs(y_pred - y_test))
    bias = np.mean(y_pred - y_test)

    return rmse, mae, bias


def run_ablation_study(X: np.ndarray, y: np.ndarray,
                       sample_weights: np.ndarray) -> Dict:
    """
    Run ablation study by removing feature groups.

    Returns:
        Dictionary with all results
    """
    # Split data
    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42
    )

    results = {}

    # 1. Baseline with all features
    print("\n[1/7] Training baseline (all features)...")
    rmse, mae, bias = train_and_evaluate(X_train, X_test, y_train, y_test, w_train)
    results['all_features'] = {
        'rmse': rmse, 'mae': mae, 'bias': bias,
        'n_features': X.shape[1],
        'features_used': 'all'
    }
    baseline_rmse = rmse
    print(f"      RMSE: {rmse:.3f} nm")

    # 2. Remove each feature group
    print("\n[2/7] Ablation: Remove feature groups...")
    for group_name, indices in FEATURE_GROUPS.items():
        # Keep all features except this group
        keep_indices = [i for i in range(X.shape[1]) if i not in indices]
        X_train_abl = X_train[:, keep_indices]
        X_test_abl = X_test[:, keep_indices]

        rmse, mae, bias = train_and_evaluate(
            X_train_abl, X_test_abl, y_train, y_test, w_train
        )

        delta = rmse - baseline_rmse
        results[f'without_{group_name}'] = {
            'rmse': rmse, 'mae': mae, 'bias': bias,
            'n_features': len(keep_indices),
            'delta_rmse': delta,
            'features_removed': group_name
        }
        print(f"      Without {group_name:15s}: RMSE = {rmse:.3f} nm (Δ = {delta:+.3f})")

    # 3. Only single feature groups
    print("\n[3/7] Single feature groups only...")
    for group_name, indices in FEATURE_GROUPS.items():
        X_train_only = X_train[:, indices]
        X_test_only = X_test[:, indices]

        rmse, mae, bias = train_and_evaluate(
            X_train_only, X_test_only, y_train, y_test, w_train
        )

        results[f'only_{group_name}'] = {
            'rmse': rmse, 'mae': mae, 'bias': bias,
            'n_features': len(indices),
            'features_used': group_name
        }
        print(f"      Only {group_name:15s}: RMSE = {rmse:.3f} nm")

    # 4. Leave-one-out for individual features
    print("\n[4/7] Leave-one-out (individual features)...")
    loo_results = {}
    for i, feat_name in enumerate(FEATURE_NAMES):
        keep_indices = [j for j in range(X.shape[1]) if j != i]
        X_train_loo = X_train[:, keep_indices]
        X_test_loo = X_test[:, keep_indices]

        rmse, mae, bias = train_and_evaluate(
            X_train_loo, X_test_loo, y_train, y_test, w_train
        )

        delta = rmse - baseline_rmse
        loo_results[feat_name] = {
            'rmse': rmse,
            'delta_rmse': delta
        }

    results['leave_one_out'] = loo_results

    # 5. Feature importance ranking by delta RMSE
    print("\n[5/7] Computing importance ranking...")
    importance_ranking = sorted(
        loo_results.items(),
        key=lambda x: x[1]['delta_rmse'],
        reverse=True
    )

    results['importance_ranking'] = importance_ranking

    return results


def print_results(results: Dict) -> None:
    """Print formatted results."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)

    # Baseline
    baseline = results['all_features']
    print(f"\nBaseline (all 15 features): RMSE = {baseline['rmse']:.3f} nm")

    # Feature group removal
    print("\n--- Feature Group Removal ---")
    print(f"{'Configuration':<30} {'RMSE (nm)':<12} {'Δ RMSE':<12} {'#Features':<10}")
    print("-" * 70)

    for key, val in results.items():
        if key.startswith('without_'):
            group = key.replace('without_', '')
            print(f"Without {group:<22} {val['rmse']:<12.3f} {val['delta_rmse']:+.3f}        "
                  f"{val['n_features']}")

    # Single groups
    print("\n--- Single Feature Groups Only ---")
    print(f"{'Configuration':<30} {'RMSE (nm)':<12} {'#Features':<10}")
    print("-" * 70)

    for key, val in results.items():
        if key.startswith('only_'):
            group = key.replace('only_', '')
            print(f"Only {group:<25} {val['rmse']:<12.3f} {val['n_features']}")

    # Feature importance ranking
    print("\n--- Feature Importance (by Δ RMSE when removed) ---")
    print(f"{'Rank':<6} {'Feature':<15} {'Δ RMSE (nm)':<15} {'Importance':<15}")
    print("-" * 60)

    ranking = results['importance_ranking']
    max_delta = max(r[1]['delta_rmse'] for r in ranking) if ranking else 1

    for i, (feat, data) in enumerate(ranking):
        importance = data['delta_rmse'] / max_delta if max_delta > 0 else 0
        bar = '█' * int(importance * 20)
        print(f"{i+1:<6} {feat:<15} {data['delta_rmse']:+.4f}         {bar}")

    print("=" * 80)


def plot_results(results: Dict, output_dir: Path) -> None:
    """Generate visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    baseline_rmse = results['all_features']['rmse']

    # 1. Feature group removal impact
    ax = axes[0, 0]
    groups = []
    deltas = []
    for key, val in results.items():
        if key.startswith('without_'):
            groups.append(key.replace('without_', '').replace('_', '\n'))
            deltas.append(val['delta_rmse'])

    colors = ['red' if d > 0 else 'green' for d in deltas]
    bars = ax.bar(groups, deltas, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Δ RMSE (nm)')
    ax.set_title('Impact of Removing Feature Groups\n(positive = worse performance)')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        ax.annotate(f'{delta:+.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3 if height >= 0 else -10),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=10)

    # 2. Single feature group performance
    ax = axes[0, 1]
    single_groups = []
    single_rmse = []
    for key, val in results.items():
        if key.startswith('only_'):
            single_groups.append(key.replace('only_', '').replace('_', '\n'))
            single_rmse.append(val['rmse'])

    bars = ax.bar(single_groups, single_rmse, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(y=baseline_rmse, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_rmse:.2f}nm)')
    ax.set_ylabel('RMSE (nm)')
    ax.set_title('Performance Using Single Feature Groups')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, rmse in zip(bars, single_rmse):
        ax.annotate(f'{rmse:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, rmse),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=10)

    # 3. Leave-one-out feature importance
    ax = axes[1, 0]
    ranking = results['importance_ranking']
    features = [r[0] for r in ranking]
    deltas = [r[1]['delta_rmse'] for r in ranking]

    colors = ['red' if d > 0 else 'green' for d in deltas]
    y_pos = range(len(features))
    ax.barh(y_pos, deltas, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Δ RMSE (nm)')
    ax.set_title('Leave-One-Out Feature Importance\n(larger = more important)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # 4. Summary comparison
    ax = axes[1, 1]
    configs = ['All Features', 'w/o Modulation', 'w/o Log Total',
               'w/o Positions', 'w/o Ratios']
    rmse_vals = [
        results['all_features']['rmse'],
        results.get('without_modulation', {}).get('rmse', 0),
        results.get('without_log_total', {}).get('rmse', 0),
        results.get('without_positions_norm', {}).get('rmse', 0),
        results.get('without_photon_ratios', {}).get('rmse', 0)
    ]

    colors = ['green'] + ['orange' if r > baseline_rmse else 'steelblue' for r in rmse_vals[1:]]
    bars = ax.bar(configs, rmse_vals, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('RMSE (nm)')
    ax.set_title('Overall Comparison')
    ax.set_xticklabels(configs, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, rmse in zip(bars, rmse_vals):
        if rmse > 0:
            ax.annotate(f'{rmse:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, rmse),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10)

    plt.suptitle('MINFLUX ML - Ablation Study', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'ablation_study.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Ablation Study for MINFLUX ML')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing extracted data')
    parser.add_argument('--n_samples', type=int, default=100000,
                        help='Number of samples to use')
    args = parser.parse_args()

    print("=" * 70)
    print("MINFLUX ML - Ablation Study")
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

    # Compute sample weights
    sample_weights = compute_sample_weights(y)

    # Run ablation study
    results = run_ablation_study(X, y, sample_weights)

    # Print results
    print_results(results)

    # Generate plots
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)
    plot_results(results, output_dir)

    # Key findings
    print("\n" + "-" * 70)
    print("KEY FINDINGS:")
    print("-" * 70)

    # Most important feature group
    group_deltas = {k.replace('without_', ''): v['delta_rmse']
                   for k, v in results.items() if k.startswith('without_')}
    most_important = max(group_deltas.items(), key=lambda x: x[1])
    print(f"• Most important feature group: {most_important[0]} "
          f"(Δ RMSE = {most_important[1]:+.3f} nm when removed)")

    # Most important single feature
    ranking = results['importance_ranking']
    if ranking:
        top_feat = ranking[0]
        print(f"• Most important single feature: {top_feat[0]} "
              f"(Δ RMSE = {top_feat[1]['delta_rmse']:+.4f} nm when removed)")


if __name__ == '__main__':
    main()
