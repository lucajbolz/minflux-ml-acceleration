#!/usr/bin/env python3
"""
MINFLUX ML Command-Line Interface.

Usage:
    minflux-ml predict --input data.csv --output results.csv
    minflux-ml benchmark
    minflux-ml info
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml_inference import MINFLUXDistanceEstimator


def cmd_predict(args):
    """Predict distances from input file."""
    print(f"Loading model: {args.model}")
    estimator = MINFLUXDistanceEstimator(args.model, use_uncertainty=args.uncertainty)

    # Load input data
    print(f"Loading data: {args.input}")
    input_path = Path(args.input)

    if input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
    elif input_path.suffix == '.npy':
        data = np.load(input_path)
        # Assume data is [photons(6), positions(6)]
        df = pd.DataFrame(data, columns=[
            'p1', 'p2', 'p3', 'p4', 'p5', 'p6',
            'x1', 'x2', 'x3', 'x4', 'x5', 'x6'
        ])
    else:
        print(f"Error: Unsupported file format: {input_path.suffix}")
        return 1

    # Extract photons and positions
    photon_cols = [c for c in df.columns if c.startswith('p') or 'photon' in c.lower()]
    pos_cols = [c for c in df.columns if c.startswith('x') or c.startswith('y') or 'pos' in c.lower()]

    if len(photon_cols) < 6 or len(pos_cols) < 6:
        print("Error: Input must have 6 photon columns and 6 position columns")
        print(f"Found photon columns: {photon_cols}")
        print(f"Found position columns: {pos_cols}")
        return 1

    photons = df[photon_cols[:6]].values
    positions = df[pos_cols[:6]].values

    print(f"Processing {len(df)} measurements...")

    # Predict
    start = time.perf_counter()

    if args.uncertainty:
        distances, lower, upper = estimator.predict_batch(photons, positions)
        results = pd.DataFrame({
            'distance_nm': distances,
            'ci_lower_nm': lower,
            'ci_upper_nm': upper,
            'ci_width_nm': upper - lower
        })
    else:
        distances = estimator.predict_batch(photons, positions)
        results = pd.DataFrame({'distance_nm': distances})

    elapsed = time.perf_counter() - start

    # Save results
    output_path = Path(args.output)
    if output_path.suffix == '.csv':
        results.to_csv(output_path, index=False)
    elif output_path.suffix == '.npy':
        np.save(output_path, results.values)
    else:
        results.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")
    print(f"Processed {len(df)} measurements in {elapsed:.3f}s")
    print(f"Throughput: {len(df)/elapsed:.0f} measurements/s")

    # Summary stats
    print(f"\nSummary:")
    print(f"  Mean distance: {distances.mean():.2f} nm")
    print(f"  Std:           {distances.std():.2f} nm")
    print(f"  Range:         [{distances.min():.2f}, {distances.max():.2f}] nm")

    return 0


def cmd_benchmark(args):
    """Benchmark inference speed."""
    print(f"Loading model: {args.model}")
    estimator = MINFLUXDistanceEstimator(args.model, use_uncertainty=args.uncertainty)

    print(f"\nBenchmarking with {args.samples} samples...")

    # Generate random test data
    np.random.seed(42)
    photons = np.random.poisson(100, (args.samples, 6)).astype(float)
    positions = np.random.uniform(-30, 30, (args.samples, 6))

    # Warmup
    _ = estimator.predict_batch(photons[:10], positions[:10])

    # Single prediction benchmark
    single_times = []
    for i in range(min(100, args.samples)):
        start = time.perf_counter()
        _ = estimator.predict(photons[i], positions[i])
        single_times.append(time.perf_counter() - start)

    # Batch prediction benchmark
    start = time.perf_counter()
    _ = estimator.predict_batch(photons, positions)
    batch_time = time.perf_counter() - start

    single_ms = np.median(single_times) * 1000
    batch_ms_per_sample = batch_time / args.samples * 1000

    print(f"\nResults:")
    print(f"  Single prediction:  {single_ms:.4f} ms")
    print(f"  Batch prediction:   {batch_ms_per_sample:.4f} ms/sample")
    print(f"  Throughput:         {1000/single_ms:.0f} predictions/s")
    print(f"  Speedup vs MLE:     {100/single_ms:.0f}× (MLE ~100ms)")

    if args.uncertainty:
        print(f"\n  Note: Uncertainty quantification enabled")

    return 0


def cmd_info(args):
    """Show model information."""
    import pickle

    model_path = Path(args.model)
    print(f"Model: {model_path}")

    if not model_path.exists():
        print("Error: Model file not found")
        return 1

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"\nModel Type: {type(model).__name__}")

    if hasattr(model, 'n_estimators'):
        print(f"Number of trees: {model.n_estimators}")
    if hasattr(model, 'max_depth'):
        print(f"Max depth: {model.max_depth}")
    if hasattr(model, 'feature_importances_'):
        print(f"Number of features: {len(model.feature_importances_)}")
    if hasattr(model, 'best_iteration'):
        print(f"Best iteration: {model.best_iteration}")

    # Check for MAPIE model
    mapie_path = model_path.parent / f'mapie_{model_path.stem.split("_")[-1]}.pkl'
    if mapie_path.exists():
        print(f"\nUQ Model available: {mapie_path}")
    else:
        print(f"\nNo UQ model found")

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='minflux-ml',
        description='MINFLUX ML Distance Estimation CLI'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Predict distances from file')
    pred_parser.add_argument('--input', '-i', required=True, help='Input file (CSV or NPY)')
    pred_parser.add_argument('--output', '-o', required=True, help='Output file (CSV or NPY)')
    pred_parser.add_argument('--model', '-m', default='models/xgboost_balanced.pkl',
                             help='Model path')
    pred_parser.add_argument('--uncertainty', '-u', action='store_true',
                             help='Include 90% confidence intervals')

    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark inference speed')
    bench_parser.add_argument('--model', '-m', default='models/xgboost_balanced.pkl',
                              help='Model path')
    bench_parser.add_argument('--samples', '-n', type=int, default=1000,
                              help='Number of samples')
    bench_parser.add_argument('--uncertainty', '-u', action='store_true',
                              help='Include uncertainty quantification')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--model', '-m', default='models/xgboost_balanced.pkl',
                             help='Model path')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == 'predict':
        return cmd_predict(args)
    elif args.command == 'benchmark':
        return cmd_benchmark(args)
    elif args.command == 'info':
        return cmd_info(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
