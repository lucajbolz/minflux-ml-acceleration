#!/usr/bin/env python3
"""
Bias correction for MINFLUX ML distance predictions.

The XGBoost models exhibit systematic bias due to regression-to-mean:
- 15nm: overestimated by ~3.3nm
- 20nm: near-optimal (+0.7nm)
- 30nm: underestimated by ~2.8nm

This module provides calibration methods to correct this bias.
"""

import argparse
import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


class BiasCorrector:
    """
    Calibrates ML predictions to remove systematic bias.

    Supports two correction methods:
    - 'linear': Simple linear recalibration (y_corr = a*y_pred + b)
    - 'isotonic': Non-parametric monotonic calibration

    Usage:
        corrector = BiasCorrector.load('models/bias_corrector_dynamic.pkl')
        y_corrected = corrector.correct(y_predicted)
    """

    def __init__(self, method: str = 'isotonic'):
        """
        Initialize bias corrector.

        Args:
            method: 'linear' or 'isotonic'
        """
        if method not in ('linear', 'isotonic'):
            raise ValueError(f"Method must be 'linear' or 'isotonic', got {method}")

        self.method = method
        self.calibrator = None
        self.is_fitted = False

        # Metrics
        self.rmse_before: Optional[float] = None
        self.rmse_after: Optional[float] = None
        self.bias_before: Optional[dict] = None
        self.bias_after: Optional[dict] = None

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> 'BiasCorrector':
        """
        Fit calibration on prediction-truth pairs.

        Args:
            y_pred: Model predictions (nm)
            y_true: Ground truth distances (nm)

        Returns:
            self
        """
        y_pred = np.asarray(y_pred).ravel()
        y_true = np.asarray(y_true).ravel()

        if len(y_pred) != len(y_true):
            raise ValueError("y_pred and y_true must have same length")

        # Store metrics before correction
        self.rmse_before = np.sqrt(np.mean((y_pred - y_true) ** 2))
        self.bias_before = self._compute_bias_by_distance(y_pred, y_true)

        # Fit calibrator
        if self.method == 'linear':
            self.calibrator = LinearRegression()
            self.calibrator.fit(y_pred.reshape(-1, 1), y_true)
        else:  # isotonic
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred, y_true)

        self.is_fitted = True

        # Compute metrics after correction
        y_corrected = self.correct(y_pred)
        self.rmse_after = np.sqrt(np.mean((y_corrected - y_true) ** 2))
        self.bias_after = self._compute_bias_by_distance(y_corrected, y_true)

        return self

    def correct(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply bias correction to predictions.

        Args:
            y_pred: Raw model predictions (nm)

        Returns:
            Corrected predictions (nm)
        """
        if not self.is_fitted:
            raise RuntimeError("BiasCorrector not fitted. Call fit() first.")

        y_pred = np.asarray(y_pred)
        original_shape = y_pred.shape
        y_flat = y_pred.ravel()

        if self.method == 'linear':
            y_corrected = self.calibrator.predict(y_flat.reshape(-1, 1))
        else:
            y_corrected = self.calibrator.predict(y_flat)

        return y_corrected.reshape(original_shape)

    def _compute_bias_by_distance(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        """Compute bias for each unique ground truth distance."""
        bias = {}
        for dist in np.unique(y_true):
            mask = y_true == dist
            if mask.sum() > 0:
                mean_pred = y_pred[mask].mean()
                bias[float(dist)] = float(mean_pred - dist)
        return bias

    def summary(self) -> str:
        """Return calibration summary."""
        if not self.is_fitted:
            return "BiasCorrector not fitted."

        lines = [
            f"Bias Correction Summary ({self.method})",
            "=" * 50,
            f"RMSE before: {self.rmse_before:.3f} nm",
            f"RMSE after:  {self.rmse_after:.3f} nm",
            f"Improvement: {(1 - self.rmse_after/self.rmse_before)*100:.1f}%",
            "",
            "Bias by distance (pred - true):",
            "  Distance | Before  | After",
            "  ---------|---------|--------"
        ]

        for dist in sorted(self.bias_before.keys()):
            before = self.bias_before.get(dist, 0)
            after = self.bias_after.get(dist, 0)
            lines.append(f"  {dist:5.0f} nm | {before:+6.2f} | {after:+6.2f}")

        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save calibrator to file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved bias corrector to {path}")

    @classmethod
    def load(cls, path: str) -> 'BiasCorrector':
        """Load calibrator from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def engineer_features(X_raw: np.ndarray) -> np.ndarray:
    """
    Apply feature engineering to raw MINFLUX data.

    Args:
        X_raw: Array of shape (n_samples, 12) with [photons(6), positions(6)]

    Returns:
        Engineered features of shape (n_samples, 15)
    """
    photons = X_raw[:, :6]
    positions = X_raw[:, 6:]

    total_photons = photons.sum(axis=1, keepdims=True)
    photon_ratios = photons / (total_photons + 1e-8)

    # Modulation depth
    mod_x = photons[:, 0] + photons[:, 2] - 2 * photons[:, 1]
    mod_y = photons[:, 3] + photons[:, 5] - 2 * photons[:, 4]
    modulation = np.stack([mod_x, mod_y], axis=1) / (total_photons + 1e-8)

    log_total = np.log(total_photons + 1)

    # Normalize positions
    pos_mean = positions.mean(axis=0, keepdims=True)
    pos_std = positions.std(axis=0, keepdims=True) + 1e-8
    positions_norm = (positions - pos_mean) / pos_std

    return np.concatenate([
        photon_ratios, positions_norm, modulation, log_total
    ], axis=1).astype(np.float32)


def calibrate_model(
    model_type: str = 'dynamic',
    method: str = 'isotonic',
    data_dir: str = 'data'
) -> BiasCorrector:
    """
    Calibrate bias correction for a model using validation data.

    Args:
        model_type: 'dynamic' or 'static'
        method: 'linear' or 'isotonic'
        data_dir: Directory containing extracted data

    Returns:
        Fitted BiasCorrector
    """
    from ml_inference import MINFLUXDistanceEstimator

    print("=" * 60)
    print(f"BIAS CALIBRATION - {model_type.upper()} MODEL")
    print("=" * 60)

    # Load model
    if model_type == 'dynamic':
        model_path = 'models/xgboost_dynamic.pkl'
        X_path = Path(data_dir) / 'dynamic_data_X.npy'
        y_path = Path(data_dir) / 'dynamic_data_y.npy'
    else:
        model_path = 'models/xgboost_optimized.pkl'
        X_path = Path(data_dir) / 'paper_data_X.npy'
        y_path = Path(data_dir) / 'paper_data_y.npy'

    print(f"\n[1] Loading model: {model_path}")
    estimator = MINFLUXDistanceEstimator(model_path)

    # Load data
    print(f"[2] Loading data...")
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Data files not found at {X_path} and {y_path}. "
            f"Run ml_extract_{model_type}.py first."
        )

    X_raw = np.load(X_path)
    y = np.load(y_path)
    print(f"    Loaded {len(y):,} samples")

    # Use subset for calibration (15% held out)
    np.random.seed(42)
    n_calib = min(200_000, len(y) // 5)
    indices = np.random.choice(len(y), n_calib, replace=False)
    X_calib_raw = X_raw[indices]
    y_calib = y[indices]
    print(f"    Using {n_calib:,} samples for calibration")

    # Engineer features if needed (raw data has 12 features, engineered has 15)
    print(f"    Input features: {X_calib_raw.shape[1]}")
    if X_calib_raw.shape[1] == 12:
        print(f"    Engineering features (12 -> 15)...")
        X_calib = engineer_features(X_calib_raw)
    else:
        X_calib = X_calib_raw

    # Generate predictions
    print(f"\n[3] Generating predictions...")
    y_pred = estimator.model.predict(X_calib)

    # Fit bias corrector
    print(f"\n[4] Fitting {method} calibration...")
    corrector = BiasCorrector(method=method)
    corrector.fit(y_pred, y_calib)

    # Print summary
    print(f"\n{corrector.summary()}")

    # Save
    save_path = f'models/bias_corrector_{model_type}.pkl'
    corrector.save(save_path)

    return corrector


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate bias correction for MINFLUX ML models'
    )
    parser.add_argument(
        '--model',
        choices=['dynamic', 'static', 'both'],
        default='dynamic',
        help='Which model to calibrate'
    )
    parser.add_argument(
        '--method',
        choices=['linear', 'isotonic'],
        default='isotonic',
        help='Calibration method'
    )
    parser.add_argument(
        '--data_dir',
        default='data',
        help='Directory containing extracted training data'
    )

    args = parser.parse_args()

    models = ['dynamic', 'static'] if args.model == 'both' else [args.model]

    for model in models:
        try:
            calibrate_model(model, args.method, args.data_dir)
            print()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Skipping {model} model.\n")


if __name__ == '__main__':
    main()
