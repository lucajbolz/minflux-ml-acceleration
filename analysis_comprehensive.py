#!/usr/bin/env python3
"""
Comprehensive analysis for MINFLUX ML Bachelorarbeit.

Generates all analysis plots:
1. Error Analysis (residuals, scatter plots)
2. Feature Importance (XGBoost built-in)
3. UQ Calibration per distance
4. Robustness tests (photon budget, noise)
5. Speedup-Accuracy tradeoff
"""

import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set publication style
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['savefig.dpi'] = 150
mpl.rcParams['figure.facecolor'] = 'white'

# Colors
COLORS = {
    'blue': '#2563eb',
    'green': '#059669',
    'red': '#dc2626',
    'orange': '#ea580c',
    'purple': '#7c3aed',
    'gray': '#6b7280'
}


def load_data_and_model():
    """Load model and test data."""
    # Load balanced model
    with open('models/xgboost_balanced.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load data
    X_raw = np.load('data/dynamic_data_X.npy')
    y = np.load('data/dynamic_data_y.npy')

    return model, X_raw, y


def engineer_features(X_raw):
    """Feature engineering."""
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


def plot_error_analysis(model, X, y, output_dir='analysis'):
    """Generate error analysis plots."""
    print("\n[1] Error Analysis...")

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_pred = model.predict(X_test)
    residuals = y_pred - y_test

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Scatter: Prediction vs Ground Truth
    ax = axes[0, 0]
    for dist in [15, 20, 30]:
        mask = y_test == dist
        ax.scatter(y_test[mask], y_pred[mask], alpha=0.3, s=5, label=f'{dist}nm')
    ax.plot([10, 35], [10, 35], 'k--', linewidth=2, label='Perfect')
    ax.set_xlabel('Ground Truth (nm)')
    ax.set_ylabel('Prediction (nm)')
    ax.set_title('Prediction vs Ground Truth')
    ax.legend()
    ax.set_xlim(10, 35)
    ax.set_ylim(10, 35)

    # 2. Residual histogram
    ax = axes[0, 1]
    for dist, color in zip([15, 20, 30], [COLORS['blue'], COLORS['green'], COLORS['red']]):
        mask = y_test == dist
        ax.hist(residuals[mask], bins=50, alpha=0.5, label=f'{dist}nm', color=color)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (nm)')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distribution')
    ax.legend()

    # 3. Residual vs Prediction
    ax = axes[1, 0]
    ax.scatter(y_pred, residuals, alpha=0.1, s=3, c=COLORS['blue'])
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction (nm)')
    ax.set_ylabel('Residual (nm)')
    ax.set_title('Residual vs Prediction')

    # 4. Error by distance (box plot)
    ax = axes[1, 1]
    errors_by_dist = [np.abs(residuals[y_test == d]) for d in [15, 20, 30]]
    bp = ax.boxplot(errors_by_dist, labels=['15nm', '20nm', '30nm'], patch_artist=True)
    for patch, color in zip(bp['boxes'], [COLORS['blue'], COLORS['green'], COLORS['red']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel('Absolute Error (nm)')
    ax.set_title('Error Distribution by Distance')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_analysis.png', bbox_inches='tight')
    print(f"    Saved: {output_dir}/error_analysis.png")
    plt.close()


def plot_feature_importance(model, output_dir='analysis'):
    """Plot feature importance."""
    print("\n[2] Feature Importance...")

    feature_names = [
        'Photon Ratio 1', 'Photon Ratio 2', 'Photon Ratio 3',
        'Photon Ratio 4', 'Photon Ratio 5', 'Photon Ratio 6',
        'Position Norm 1', 'Position Norm 2', 'Position Norm 3',
        'Position Norm 4', 'Position Norm 5', 'Position Norm 6',
        'Modulation X', 'Modulation Y', 'Log Total Photons'
    ]

    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS['blue'] if 'Photon' in feature_names[i] else
              COLORS['green'] if 'Position' in feature_names[i] else
              COLORS['orange'] for i in indices]

    ax.barh(range(len(importance)), importance[indices], color=colors)
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_title('XGBoost Feature Importance')
    ax.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['blue'], label='Photon Ratios'),
        Patch(facecolor=COLORS['green'], label='Positions'),
        Patch(facecolor=COLORS['orange'], label='Derived Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', bbox_inches='tight')
    print(f"    Saved: {output_dir}/feature_importance.png")
    plt.close()


def plot_uq_calibration(output_dir='analysis'):
    """Plot UQ calibration per distance."""
    print("\n[3] UQ Calibration per Distance...")

    # Load MAPIE model
    with open('models/mapie_balanced.pkl', 'rb') as f:
        mapie = pickle.load(f)

    # Load data
    X_raw = np.load('data/dynamic_data_X.npy')
    y = np.load('data/dynamic_data_y.npy')
    X = engineer_features(X_raw)

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Predict with intervals
    y_pred, y_intervals = mapie.predict_interval(X_test)
    y_lower = y_intervals[:, 0, 0]
    y_upper = y_intervals[:, 1, 0]

    # Coverage per distance
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, dist in enumerate([15, 20, 30]):
        mask = y_test == dist
        coverage = np.mean((y_test[mask] >= y_lower[mask]) & (y_test[mask] <= y_upper[mask]))
        mean_width = np.mean(y_upper[mask] - y_lower[mask])

        ax = axes[idx]
        n_show = min(100, mask.sum())
        indices = np.where(mask)[0][:n_show]

        x = np.arange(n_show)
        ax.errorbar(x, y_pred[indices], yerr=[y_pred[indices] - y_lower[indices],
                                                y_upper[indices] - y_pred[indices]],
                    fmt='none', alpha=0.5, color=COLORS['blue'])
        ax.scatter(x, y_pred[indices], s=10, color=COLORS['blue'], label='Prediction')
        ax.axhline(y=dist, color=COLORS['green'], linestyle='--', linewidth=2, label=f'True ({dist}nm)')

        ax.set_xlabel('Sample')
        ax.set_ylabel('Distance (nm)')
        ax.set_title(f'{dist}nm: Coverage={coverage*100:.1f}%, Width={mean_width:.1f}nm')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 40)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/uq_calibration.png', bbox_inches='tight')
    print(f"    Saved: {output_dir}/uq_calibration.png")
    plt.close()


def plot_robustness(model, X_raw, y, output_dir='analysis'):
    """Test robustness to photon budget and noise."""
    print("\n[4] Robustness Analysis...")

    from sklearn.model_selection import train_test_split
    _, X_test_raw, _, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Photon budget scaling
    ax = axes[0]
    scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    rmse_by_scale = []

    for scale in scales:
        X_scaled = X_test_raw.copy()
        X_scaled[:, :6] = X_scaled[:, :6] * scale  # Scale photons
        X_feat = engineer_features(X_scaled)
        y_pred = model.predict(X_feat)
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        rmse_by_scale.append(rmse)

    ax.plot(scales, rmse_by_scale, 'o-', color=COLORS['blue'], linewidth=2, markersize=8)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Photon Budget Scale Factor')
    ax.set_ylabel('RMSE (nm)')
    ax.set_title('Robustness to Photon Budget')
    ax.grid(True, alpha=0.3)

    # 2. Noise injection
    ax = axes[1]
    noise_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    rmse_by_noise = []

    for noise in noise_levels:
        X_noisy = X_test_raw.copy()
        # Add Gaussian noise to photons
        photon_noise = np.random.normal(0, noise * X_noisy[:, :6].mean(), X_noisy[:, :6].shape)
        X_noisy[:, :6] = np.maximum(X_noisy[:, :6] + photon_noise, 0)
        X_feat = engineer_features(X_noisy)
        y_pred = model.predict(X_feat)
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        rmse_by_noise.append(rmse)

    ax.plot([n*100 for n in noise_levels], rmse_by_noise, 'o-', color=COLORS['red'], linewidth=2, markersize=8)
    ax.set_xlabel('Noise Level (%)')
    ax.set_ylabel('RMSE (nm)')
    ax.set_title('Robustness to Measurement Noise')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/robustness.png', bbox_inches='tight')
    print(f"    Saved: {output_dir}/robustness.png")
    plt.close()


def plot_speedup_accuracy_tradeoff(output_dir='analysis'):
    """Plot speedup vs accuracy for different model sizes."""
    print("\n[5] Speedup-Accuracy Tradeoff...")

    # Load test data
    X_raw = np.load('data/dynamic_data_X.npy')
    y = np.load('data/dynamic_data_y.npy')
    X = engineer_features(X_raw)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train models with different n_estimators
    import xgboost as xgb

    results = []
    n_estimators_list = [10, 25, 50, 100, 200, 500]

    for n_est in n_estimators_list:
        print(f"    Training model with {n_est} estimators...")
        model = xgb.XGBRegressor(
            n_estimators=n_est,
            max_depth=8,
            learning_rate=0.1,
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Measure inference time
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = model.predict(X_test[:100])
            times.append(time.perf_counter() - start)

        inference_time = np.median(times) / 100 * 1000  # ms per sample

        # Calculate RMSE
        y_pred = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

        results.append({
            'n_estimators': n_est,
            'rmse': rmse,
            'inference_ms': inference_time,
            'speedup': 100 / inference_time  # vs 100ms MLE
        })

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    rmses = [r['rmse'] for r in results]
    speedups = [r['speedup'] for r in results]
    n_ests = [r['n_estimators'] for r in results]

    scatter = ax.scatter(speedups, rmses, c=n_ests, cmap='viridis', s=200, edgecolors='black', linewidth=2)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Trees')

    for r in results:
        ax.annotate(f"{r['n_estimators']}", (r['speedup'], r['rmse']),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)

    # Add MLE reference
    ax.axhline(y=4.24, color=COLORS['green'], linestyle='--', linewidth=2, label='MLE (4.24nm)')
    ax.axvline(x=1, color=COLORS['gray'], linestyle=':', linewidth=1.5, label='MLE Speed (1×)')

    ax.set_xlabel('Speedup vs MLE (×)', fontsize=12)
    ax.set_ylabel('RMSE (nm)', fontsize=12)
    ax.set_title('Speedup-Accuracy Tradeoff', fontsize=14)
    ax.set_xscale('log')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_accuracy_tradeoff.png', bbox_inches='tight')
    print(f"    Saved: {output_dir}/speedup_accuracy_tradeoff.png")
    plt.close()


def main():
    print("=" * 70)
    print("MINFLUX ML - COMPREHENSIVE ANALYSIS")
    print("=" * 70)

    # Create output directory
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)

    # Load data
    model, X_raw, y = load_data_and_model()
    X = engineer_features(X_raw)

    # Run all analyses
    plot_error_analysis(model, X, y, str(output_dir))
    plot_feature_importance(model, str(output_dir))
    plot_uq_calibration(str(output_dir))
    plot_robustness(model, X_raw, y, str(output_dir))
    plot_speedup_accuracy_tradeoff(str(output_dir))

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll plots saved to: {output_dir}/")
    print("  - error_analysis.png")
    print("  - feature_importance.png")
    print("  - uq_calibration.png")
    print("  - robustness.png")
    print("  - speedup_accuracy_tradeoff.png")


if __name__ == '__main__':
    main()
