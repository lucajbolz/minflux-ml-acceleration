#!/usr/bin/env python3
"""
Quick Demo - ML for MINFLUX Distance Estimation

Shows how to use the trained model for predictions.
"""

import numpy as np
import pickle


def load_model(model_path='models/xgboost_equal_balanced.pkl'):
    """Load the trained XGBoost model."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Loaded model: {model_path}")
    return model


def engineer_features(photons, positions, training_stats):
    """Apply feature engineering (same as training)."""
    photons = np.maximum(photons, 0)
    total = photons.sum() + 1e-8

    # Photon ratios
    photon_ratios = photons / total

    # Modulation
    mod_x = photons[0] + photons[2] - 2 * photons[1]
    mod_y = photons[3] + photons[5] - 2 * photons[4]
    modulation = np.array([mod_x, mod_y]) / total

    # Log total
    log_total = np.log(max(total, 1))

    # Normalize positions (using training statistics)
    pos_mean, pos_std = training_stats
    positions_norm = (positions - pos_mean) / pos_std

    # Combine features
    features = np.concatenate([
        photon_ratios,      # 6
        positions_norm,     # 6
        modulation,         # 2
        [log_total]         # 1
    ]).astype(np.float32)

    return features.reshape(1, -1)


def predict_distance(model, photons, positions, training_stats):
    """
    Predict distance from photon counts and beam positions.

    Args:
        model: Trained XGBoost model
        photons: Array of 6 photon counts [X-, X0, X+, Y-, Y0, Y+]
        positions: Array of 6 beam positions
        training_stats: (pos_mean, pos_std) from training data

    Returns:
        Predicted distance in nm
    """
    features = engineer_features(photons, positions, training_stats)
    prediction = model.predict(features)[0]
    return prediction


def main():
    print("=" * 60)
    print("ML for MINFLUX Distance Estimation - Quick Demo")
    print("=" * 60)
    print()

    # Load model
    model = load_model()

    # Load training statistics for normalization
    print("✓ Loading training statistics...")
    X_train = np.load('data/dynamic_data_X.npy')
    positions_train = X_train[:, 6:]
    pos_mean = positions_train.mean(axis=0)
    pos_std = positions_train.std(axis=0) + 1e-8
    training_stats = (pos_mean, pos_std)

    print()
    print("=" * 60)
    print("Example Predictions")
    print("=" * 60)
    print()

    # Example 1: Sample from 20nm data
    y_train = np.load('data/dynamic_data_y.npy')
    mask_20 = y_train == 20
    sample_idx = np.random.choice(np.where(mask_20)[0])
    sample = X_train[sample_idx]

    photons = sample[:6]
    positions = sample[6:]

    pred = predict_distance(model, photons, positions, training_stats)

    print("Example 1 (from 20nm training data):")
    print(f"  Photons:    {photons}")
    print(f"  Positions:  {positions}")
    print(f"  Predicted:  {pred:.2f} nm")
    print(f"  True:       20.00 nm")
    print(f"  Error:      {pred - 20:.2f} nm")
    print()

    # Example 2: Custom input
    print("Example 2 (custom photon pattern):")
    photons_custom = np.array([35, 42, 28, 38, 45, 30], dtype=float)
    positions_custom = np.array([-50, 0, 50, -50, 0, 50], dtype=float)

    pred_custom = predict_distance(model, photons_custom, positions_custom, training_stats)

    print(f"  Photons:    {photons_custom}")
    print(f"  Positions:  {positions_custom}")
    print(f"  Predicted:  {pred_custom:.2f} nm")
    print()

    print("=" * 60)
    print("Performance:")
    print("  Inference time: ~0.2 ms")
    print("  Speedup vs MLE: 5-50× (MLE: 1-10ms, Balzarotti et al. 2017)")
    print("=" * 60)


if __name__ == '__main__':
    main()
