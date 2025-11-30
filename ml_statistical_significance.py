"""
Statistical Significance Analysis for MINFLUX ML

Computes Bootstrap Confidence Intervals for all reported metrics.

Usage:
    python ml_statistical_significance.py

Output:
    - analysis/bootstrap_confidence_intervals.png
    - Printed results with 95% CIs
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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


def bootstrap_metric(y_true: np.ndarray, y_pred: np.ndarray,
                     metric_fn, n_bootstrap: int = 1000,
                     confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metric_fn: Function(y_true, y_pred) -> float
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    n = len(y_true)
    bootstrap_values = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        value = metric_fn(y_true_boot, y_pred_boot)
        bootstrap_values.append(value)

    bootstrap_values = np.array(bootstrap_values)

    # Point estimate
    point_estimate = metric_fn(y_true, y_pred)

    # Percentile confidence interval
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_values, alpha * 100)
    upper = np.percentile(bootstrap_values, (1 - alpha) * 100)

    return point_estimate, lower, upper, bootstrap_values


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_pred - y_true))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Bias."""
    return np.mean(y_pred - y_true)


def run_bootstrap_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                           n_bootstrap: int = 1000) -> Dict:
    """
    Run complete bootstrap analysis.

    Returns:
        Dictionary with all confidence intervals
    """
    results = {}

    # Overall metrics
    print("\n[1/4] Computing overall metrics...")

    for name, fn in [('RMSE', rmse), ('MAE', mae), ('Bias', bias)]:
        est, lower, upper, dist = bootstrap_metric(y_true, y_pred, fn, n_bootstrap)
        results[f'overall_{name.lower()}'] = {
            'estimate': est,
            'ci_lower': lower,
            'ci_upper': upper,
            'std': np.std(dist),
            'distribution': dist
        }
        print(f"      {name}: {est:.3f} nm [95% CI: {lower:.3f}, {upper:.3f}]")

    # Per-distance metrics
    print("\n[2/4] Computing per-distance metrics...")

    for dist_val in [15, 20, 30]:
        mask = y_true == dist_val
        if mask.sum() < 100:
            continue

        y_true_dist = y_true[mask]
        y_pred_dist = y_pred[mask]

        est, lower, upper, dist = bootstrap_metric(
            y_true_dist, y_pred_dist, rmse, n_bootstrap
        )
        results[f'rmse_{dist_val}nm'] = {
            'estimate': est,
            'ci_lower': lower,
            'ci_upper': upper,
            'std': np.std(dist),
            'distribution': dist
        }
        print(f"      {dist_val}nm RMSE: {est:.3f} nm [95% CI: {lower:.3f}, {upper:.3f}]")

        # Bias per distance
        est, lower, upper, _ = bootstrap_metric(
            y_true_dist, y_pred_dist, bias, n_bootstrap
        )
        results[f'bias_{dist_val}nm'] = {
            'estimate': est,
            'ci_lower': lower,
            'ci_upper': upper
        }

    return results


def run_inference_speed_bootstrap(model, X: np.ndarray,
                                  n_bootstrap: int = 100) -> Dict:
    """Bootstrap confidence interval for inference speed."""
    import time

    print("\n[3/4] Computing inference speed CI...")

    times = []
    for _ in range(n_bootstrap):
        # Random sample of 100 predictions
        indices = np.random.choice(len(X), 100, replace=False)

        start = time.perf_counter()
        for i in indices:
            _ = model.predict(X[i:i+1])
        elapsed = (time.perf_counter() - start) / 100 * 1000  # ms per prediction

        times.append(elapsed)

    times = np.array(times)
    estimate = np.median(times)
    lower = np.percentile(times, 2.5)
    upper = np.percentile(times, 97.5)

    print(f"      Inference: {estimate:.3f} ms [95% CI: {lower:.3f}, {upper:.3f}]")

    speedup = 100 / estimate
    speedup_lower = 100 / upper
    speedup_upper = 100 / lower

    print(f"      Speedup:   {speedup:.0f}× [95% CI: {speedup_lower:.0f}×, {speedup_upper:.0f}×]")

    return {
        'inference_ms': {
            'estimate': estimate,
            'ci_lower': lower,
            'ci_upper': upper,
            'distribution': times
        },
        'speedup': {
            'estimate': speedup,
            'ci_lower': speedup_lower,
            'ci_upper': speedup_upper
        }
    }


def run_uq_coverage_bootstrap(y_true: np.ndarray, y_pred: np.ndarray,
                               y_lower: np.ndarray, y_upper: np.ndarray,
                               n_bootstrap: int = 1000) -> Dict:
    """Bootstrap confidence interval for UQ coverage."""
    print("\n[4/4] Computing UQ coverage CI...")

    def coverage(y_t, y_l, y_u):
        return np.mean((y_t >= y_l) & (y_t <= y_u))

    n = len(y_true)
    coverages = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        cov = coverage(y_true[indices], y_lower[indices], y_upper[indices])
        coverages.append(cov)

    coverages = np.array(coverages)
    estimate = coverage(y_true, y_lower, y_upper)
    lower = np.percentile(coverages, 2.5)
    upper = np.percentile(coverages, 97.5)

    print(f"      Coverage: {estimate*100:.1f}% [95% CI: {lower*100:.1f}%, {upper*100:.1f}%]")

    return {
        'coverage': {
            'estimate': estimate,
            'ci_lower': lower,
            'ci_upper': upper,
            'distribution': coverages
        }
    }


def print_results(results: Dict) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("Bootstrap Confidence Intervals (n=1000, 95% CI)")
    print("=" * 80)

    print("\n--- Overall Metrics ---")
    print(f"{'Metric':<20} {'Estimate':<15} {'95% CI':<25} {'Std':<10}")
    print("-" * 70)

    for key in ['overall_rmse', 'overall_mae', 'overall_bias']:
        if key in results:
            r = results[key]
            name = key.replace('overall_', '').upper()
            ci = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
            print(f"{name:<20} {r['estimate']:<15.3f} {ci:<25} {r['std']:.4f}")

    print("\n--- Per-Distance RMSE ---")
    print(f"{'Distance':<20} {'RMSE (nm)':<15} {'95% CI':<25}")
    print("-" * 60)

    for dist in [15, 20, 30]:
        key = f'rmse_{dist}nm'
        if key in results:
            r = results[key]
            ci = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
            print(f"{dist}nm{'':<17} {r['estimate']:<15.3f} {ci:<25}")

    print("\n--- Per-Distance Bias ---")
    print(f"{'Distance':<20} {'Bias (nm)':<15} {'95% CI':<25}")
    print("-" * 60)

    for dist in [15, 20, 30]:
        key = f'bias_{dist}nm'
        if key in results:
            r = results[key]
            ci = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
            print(f"{dist}nm{'':<17} {r['estimate']:+<15.3f} {ci:<25}")

    if 'inference_ms' in results:
        print("\n--- Inference Speed ---")
        r = results['inference_ms']
        ci = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
        print(f"Single inference:  {r['estimate']:.3f} ms {ci}")

        r = results['speedup']
        ci = f"[{r['ci_lower']:.0f}×, {r['ci_upper']:.0f}×]"
        print(f"Speedup vs MLE:    {r['estimate']:.0f}× {ci}")

    if 'coverage' in results:
        print("\n--- Uncertainty Quantification ---")
        r = results['coverage']
        ci = f"[{r['ci_lower']*100:.1f}%, {r['ci_upper']*100:.1f}%]"
        print(f"90% CI Coverage:   {r['estimate']*100:.1f}% {ci}")

    print("=" * 80)


def plot_results(results: Dict, output_dir: Path) -> None:
    """Generate bootstrap distribution plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Overall RMSE distribution
    ax = axes[0, 0]
    if 'overall_rmse' in results:
        r = results['overall_rmse']
        ax.hist(r['distribution'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(r['estimate'], color='red', linestyle='-', linewidth=2, label=f"Est: {r['estimate']:.3f}")
        ax.axvline(r['ci_lower'], color='orange', linestyle='--', linewidth=2, label=f"95% CI")
        ax.axvline(r['ci_upper'], color='orange', linestyle='--', linewidth=2)
        ax.set_xlabel('RMSE (nm)')
        ax.set_ylabel('Frequency')
        ax.set_title('Overall RMSE Bootstrap Distribution')
        ax.legend()

    # 2. Per-distance RMSE comparison
    ax = axes[0, 1]
    distances = [15, 20, 30]
    estimates = []
    errors_lower = []
    errors_upper = []

    for d in distances:
        key = f'rmse_{d}nm'
        if key in results:
            r = results[key]
            estimates.append(r['estimate'])
            errors_lower.append(r['estimate'] - r['ci_lower'])
            errors_upper.append(r['ci_upper'] - r['estimate'])

    if estimates:
        x = range(len(distances))
        ax.bar(x, estimates, yerr=[errors_lower, errors_upper],
               capsize=5, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{d}nm' for d in distances])
        ax.set_ylabel('RMSE (nm)')
        ax.set_title('Per-Distance RMSE with 95% CI')
        ax.grid(True, alpha=0.3, axis='y')

    # 3. Bias comparison
    ax = axes[0, 2]
    biases = []
    bias_errors_lower = []
    bias_errors_upper = []

    for d in distances:
        key = f'bias_{d}nm'
        if key in results:
            r = results[key]
            biases.append(r['estimate'])
            bias_errors_lower.append(r['estimate'] - r['ci_lower'])
            bias_errors_upper.append(r['ci_upper'] - r['estimate'])

    if biases:
        colors = ['red' if b > 0 else 'green' for b in biases]
        ax.bar(x, biases, yerr=[bias_errors_lower, bias_errors_upper],
               capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{d}nm' for d in distances])
        ax.set_ylabel('Bias (nm)')
        ax.set_title('Per-Distance Bias with 95% CI')
        ax.grid(True, alpha=0.3, axis='y')

    # 4. MAE distribution
    ax = axes[1, 0]
    if 'overall_mae' in results:
        r = results['overall_mae']
        ax.hist(r['distribution'], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(r['estimate'], color='red', linestyle='-', linewidth=2, label=f"Est: {r['estimate']:.3f}")
        ax.axvline(r['ci_lower'], color='orange', linestyle='--', linewidth=2)
        ax.axvline(r['ci_upper'], color='orange', linestyle='--', linewidth=2)
        ax.set_xlabel('MAE (nm)')
        ax.set_ylabel('Frequency')
        ax.set_title('Overall MAE Bootstrap Distribution')
        ax.legend()

    # 5. Inference time distribution
    ax = axes[1, 1]
    if 'inference_ms' in results:
        r = results['inference_ms']
        ax.hist(r['distribution'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(r['estimate'], color='red', linestyle='-', linewidth=2, label=f"Est: {r['estimate']:.3f}ms")
        ax.axvline(r['ci_lower'], color='orange', linestyle='--', linewidth=2)
        ax.axvline(r['ci_upper'], color='orange', linestyle='--', linewidth=2)
        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Inference Speed Bootstrap Distribution')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Inference speed\nnot measured', ha='center', va='center', transform=ax.transAxes)

    # 6. UQ Coverage distribution
    ax = axes[1, 2]
    if 'coverage' in results:
        r = results['coverage']
        ax.hist(r['distribution'] * 100, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax.axvline(r['estimate'] * 100, color='red', linestyle='-', linewidth=2,
                  label=f"Est: {r['estimate']*100:.1f}%")
        ax.axvline(90, color='green', linestyle='--', linewidth=2, label='Target: 90%')
        ax.set_xlabel('Coverage (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('UQ Coverage Bootstrap Distribution')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'UQ not measured', ha='center', va='center', transform=ax.transAxes)

    plt.suptitle('Bootstrap Confidence Intervals (n=1000, 95% CI)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'bootstrap_confidence_intervals.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Statistical Significance Analysis')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing extracted data')
    parser.add_argument('--model_path', type=str, default='models/xgboost_balanced.pkl',
                        help='Path to trained model')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples')
    parser.add_argument('--n_samples', type=int, default=50000,
                        help='Number of test samples')
    args = parser.parse_args()

    print("=" * 70)
    print("MINFLUX ML - Statistical Significance Analysis")
    print("=" * 70)

    # Load model
    import pickle
    print(f"\nLoading model: {args.model_path}")
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)

    # Load data
    data_dir = Path(args.data_dir)
    X_raw = np.load(data_dir / 'dynamic_data_X.npy')
    y = np.load(data_dir / 'dynamic_data_y.npy')

    print(f"Loaded {len(y):,} samples")

    # Use test subset
    np.random.seed(42)
    n_samples = min(args.n_samples, len(y))
    indices = np.random.choice(len(y), n_samples, replace=False)
    X_raw = X_raw[indices]
    y = y[indices]

    print(f"Using {n_samples:,} test samples")
    print(f"Bootstrap samples: {args.n_bootstrap}")

    # Engineer features
    X = engineer_features(X_raw)

    # Get predictions
    y_pred = model.predict(X)

    # Run bootstrap analysis
    results = run_bootstrap_analysis(y, y_pred, args.n_bootstrap)

    # Inference speed bootstrap
    speed_results = run_inference_speed_bootstrap(model, X, n_bootstrap=100)
    results.update(speed_results)

    # UQ analysis if MAPIE model exists
    mapie_path = Path('models/mapie_balanced.pkl')
    if mapie_path.exists():
        print("\nLoading MAPIE model for UQ analysis...")
        with open(mapie_path, 'rb') as f:
            mapie_model = pickle.load(f)

        _, y_intervals = mapie_model.predict_interval(X)
        y_lower = y_intervals[:, 0, 0]
        y_upper = y_intervals[:, 1, 0]

        uq_results = run_uq_coverage_bootstrap(y, y_pred, y_lower, y_upper, args.n_bootstrap)
        results.update(uq_results)

    # Print results
    print_results(results)

    # Generate plots
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)
    plot_results(results, output_dir)

    # LaTeX-ready output
    print("\n" + "-" * 70)
    print("LATEX-READY RESULTS:")
    print("-" * 70)

    r = results['overall_rmse']
    print(f"Overall RMSE: ${r['estimate']:.2f} \\pm {r['std']:.2f}$ nm (95\\% CI: [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}])")

    if 'speedup' in results:
        r = results['speedup']
        print(f"Speedup: ${r['estimate']:.0f}\\times$ (95\\% CI: [{r['ci_lower']:.0f}$\\times$, {r['ci_upper']:.0f}$\\times$])")

    if 'coverage' in results:
        r = results['coverage']
        print(f"UQ Coverage: ${r['estimate']*100:.1f}\\%$ (95\\% CI: [{r['ci_lower']*100:.1f}\\%, {r['ci_upper']*100:.1f}\\%])")


if __name__ == '__main__':
    main()
