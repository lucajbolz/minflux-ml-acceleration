#!/usr/bin/env python3
"""
Real-time MINFLUX distance estimation demo.

Simulates processing a MINFLUX trace in real-time, showing:
- Distance estimates over time
- Cumulative processing time (ML vs estimated MLE time)
- Live updating plot
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ml_inference import MINFLUXDistanceEstimator


def load_demo_trace(data_dir: str = 'data') -> tuple:
    """Load a demo trace from the dynamic dataset."""
    X_path = Path(data_dir) / 'dynamic_data_X.npy'
    y_path = Path(data_dir) / 'dynamic_data_y.npy'

    if not X_path.exists():
        raise FileNotFoundError(f"Data not found at {X_path}")

    X = np.load(X_path)
    y = np.load(y_path)

    # Get a continuous segment (simulating a trace)
    # Find 20nm samples (most common)
    mask_20nm = y == 20
    indices_20nm = np.where(mask_20nm)[0]

    # Take first 500 samples as a "trace"
    trace_length = min(500, len(indices_20nm))
    trace_indices = indices_20nm[:trace_length]

    return X[trace_indices], y[trace_indices]


def run_realtime_demo(
    model_path: str = 'models/xgboost_balanced.pkl',
    trace_length: int = 200,
    update_interval: int = 50,  # ms between updates
    save_path: str = None
):
    """
    Run real-time demo with animated plot.

    Args:
        model_path: Path to trained model
        trace_length: Number of measurements to process
        update_interval: Milliseconds between animation frames
        save_path: If provided, save animation as gif
    """
    print("=" * 60)
    print("MINFLUX ML - REAL-TIME DEMO")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {model_path}")
    estimator = MINFLUXDistanceEstimator(model_path, use_uncertainty=True)

    # Load trace data
    print("Loading demo trace...")
    X_trace, y_trace = load_demo_trace()
    X_trace = X_trace[:trace_length]
    y_trace = y_trace[:trace_length]
    print(f"Trace length: {len(X_trace)} measurements")
    print(f"Ground truth: {y_trace[0]:.0f} nm")

    # Storage for results
    times = []
    predictions = []
    lower_bounds = []
    upper_bounds = []
    ml_cumulative_time = []
    mle_estimated_time = []

    # MLE time estimate (100ms per measurement)
    MLE_TIME_PER_MEASUREMENT = 0.1  # seconds

    # Setup figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('MINFLUX ML Real-Time Distance Estimation', fontsize=14, fontweight='bold')

    # Plot 1: Distance over time
    ax1 = axes[0]
    ax1.set_xlabel('Measurement #')
    ax1.set_ylabel('Distance (nm)')
    ax1.set_xlim(0, trace_length)
    ax1.set_ylim(0, 40)
    ax1.axhline(y=y_trace[0], color='green', linestyle='--', alpha=0.7, label=f'Ground Truth ({y_trace[0]:.0f} nm)')

    line_pred, = ax1.plot([], [], 'b-', linewidth=1.5, label='ML Prediction')
    fill_ci = ax1.fill_between([], [], [], alpha=0.3, color='blue', label='90% CI')
    ax1.legend(loc='upper right')

    # Plot 2: Processing time comparison
    ax2 = axes[1]
    ax2.set_xlabel('Measurement #')
    ax2.set_ylabel('Cumulative Time (s)')
    ax2.set_xlim(0, trace_length)
    ax2.set_yscale('log')

    line_ml, = ax2.plot([], [], 'b-', linewidth=2, label='ML (actual)')
    line_mle, = ax2.plot([], [], 'r--', linewidth=2, label='MLE (estimated)')
    ax2.legend(loc='upper left')

    # Text annotations
    time_text = ax2.text(0.98, 0.95, '', transform=ax2.transAxes,
                          ha='right', va='top', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    speedup_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                             ha='left', va='top', fontsize=12, fontweight='bold',
                             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()

    def init():
        line_pred.set_data([], [])
        line_ml.set_data([], [])
        line_mle.set_data([], [])
        return line_pred, line_ml, line_mle

    def update(frame):
        if frame >= len(X_trace):
            return line_pred, line_ml, line_mle

        # Get measurement
        photons = X_trace[frame, :6]
        positions = X_trace[frame, 6:]

        # Predict with timing
        start = time.perf_counter()
        dist, lower, upper = estimator.predict(photons, positions)
        ml_time = time.perf_counter() - start

        # Store results
        times.append(frame)
        predictions.append(dist)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

        # Cumulative times
        if ml_cumulative_time:
            ml_cumulative_time.append(ml_cumulative_time[-1] + ml_time)
        else:
            ml_cumulative_time.append(ml_time)

        mle_estimated_time.append((frame + 1) * MLE_TIME_PER_MEASUREMENT)

        # Update prediction plot
        line_pred.set_data(times, predictions)

        # Update CI fill (recreate each frame)
        ax1.collections.clear()
        if len(times) > 1:
            ax1.fill_between(times, lower_bounds, upper_bounds, alpha=0.3, color='blue')
        ax1.axhline(y=y_trace[0], color='green', linestyle='--', alpha=0.7)

        # Update time plot
        line_ml.set_data(times, ml_cumulative_time)
        line_mle.set_data(times, mle_estimated_time)
        ax2.set_ylim(1e-4, max(mle_estimated_time[-1] * 2, 1))

        # Update text
        current_speedup = mle_estimated_time[-1] / ml_cumulative_time[-1] if ml_cumulative_time[-1] > 0 else 0
        time_text.set_text(f'ML: {ml_cumulative_time[-1]:.3f}s\nMLE: {mle_estimated_time[-1]:.1f}s')
        speedup_text.set_text(f'Speedup: {current_speedup:.0f}×')

        return line_pred, line_ml, line_mle

    # Run animation
    anim = FuncAnimation(fig, update, init_func=init,
                         frames=trace_length, interval=update_interval,
                         blit=False, repeat=False)

    if save_path:
        print(f"\nSaving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=20)
        print("Saved!")
    else:
        plt.show()

    # Print final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Measurements processed: {len(predictions)}")
    print(f"ML total time:          {ml_cumulative_time[-1]:.4f}s")
    print(f"MLE estimated time:     {mle_estimated_time[-1]:.1f}s")
    print(f"Speedup:                {mle_estimated_time[-1]/ml_cumulative_time[-1]:.0f}×")
    print(f"Mean prediction:        {np.mean(predictions):.2f} nm")
    print(f"Ground truth:           {y_trace[0]:.0f} nm")
    print(f"Mean error:             {np.mean(np.abs(np.array(predictions) - y_trace[0])):.2f} nm")


def run_static_demo(model_path: str = 'models/xgboost_balanced.pkl'):
    """Run a non-animated demo and save static plot."""
    print("=" * 60)
    print("MINFLUX ML - STATIC DEMO")
    print("=" * 60)

    # Load model
    estimator = MINFLUXDistanceEstimator(model_path, use_uncertainty=True)

    # Load trace
    X_trace, y_trace = load_demo_trace()
    X_trace = X_trace[:200]
    y_trace = y_trace[:200]

    # Process all at once
    print("\nProcessing trace...")
    start = time.perf_counter()

    predictions = []
    lower_bounds = []
    upper_bounds = []

    for i in range(len(X_trace)):
        photons = X_trace[i, :6]
        positions = X_trace[i, 6:]
        dist, lower, upper = estimator.predict(photons, positions)
        predictions.append(dist)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    ml_time = time.perf_counter() - start
    mle_time = len(X_trace) * 0.1

    predictions = np.array(predictions)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(predictions))
    ax.fill_between(x, lower_bounds, upper_bounds, alpha=0.3, color='blue', label='90% CI')
    ax.plot(x, predictions, 'b-', linewidth=1, label='ML Prediction')
    ax.axhline(y=y_trace[0], color='green', linestyle='--', linewidth=2, label=f'Ground Truth ({y_trace[0]:.0f} nm)')

    ax.set_xlabel('Measurement #', fontsize=12)
    ax.set_ylabel('Distance (nm)', fontsize=12)
    ax.set_title(f'MINFLUX ML Distance Estimation\n'
                 f'ML: {ml_time:.3f}s | MLE: {mle_time:.1f}s | Speedup: {mle_time/ml_time:.0f}×',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 40)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_realtime_result.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: demo_realtime_result.png")

    # Print summary
    print(f"\nML time:     {ml_time:.4f}s")
    print(f"MLE time:    {mle_time:.1f}s (estimated)")
    print(f"Speedup:     {mle_time/ml_time:.0f}×")
    print(f"Mean error:  {np.mean(np.abs(predictions - y_trace[0])):.2f} nm")


def main():
    parser = argparse.ArgumentParser(description='MINFLUX ML Real-Time Demo')
    parser.add_argument('--model', default='models/xgboost_balanced.pkl',
                        help='Path to trained model')
    parser.add_argument('--length', type=int, default=200,
                        help='Number of measurements to process')
    parser.add_argument('--static', action='store_true',
                        help='Generate static plot instead of animation')
    parser.add_argument('--save', type=str, default=None,
                        help='Save animation as gif')

    args = parser.parse_args()

    if args.static:
        run_static_demo(args.model)
    else:
        run_realtime_demo(args.model, args.length, save_path=args.save)


if __name__ == '__main__':
    main()
