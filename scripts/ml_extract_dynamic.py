"""
Extracts training data from MINFLUXDynamic for trace analysis

Loads all parsed.pkl files from datasets/MINFLUXDynamic/parsed/raw/
and creates a training dataset with:
- X: 6 photon values + 6 positions per measurement
- y: Ground-truth distance
"""

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def extract_measurements_from_pkl(pkl_path, distance_nm):
    """
    Extracts all measurements from a parsed.pkl file.

    IMPORTANT: Sorts photons by axis and position for consistency!
    Features: [n_x-, pos_x-, n_x0, pos_x0, n_x+, pos_x+,
               n_y-, pos_y-, n_y0, pos_y0, n_y+, pos_y+]

    Args:
        pkl_path: Path to parsed.pkl file
        distance_nm: Ground-truth distance in nm

    Returns:
        X: np.array (N, 12) - Photons + Positions (sorted!)
        y: np.array (N,) - Distances
    """
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)

    X = []
    y = []

    # Group by tuple (each measurement)
    for tuple_id in df['tuple'].unique():
        measurement = df[df['tuple'] == tuple_id]

        # Should have 6 values
        if len(measurement) != 6:
            continue

        # Sort by axis (0=x, 1=y) and then by position
        measurement_sorted = measurement.sort_values(['axis', 'pos'])
        photons = measurement_sorted['photons'].values
        positions = measurement_sorted['pos'].values

        # Interleave: [n_x-, pos_x-, n_x0, pos_x0, n_x+, pos_x+, n_y-, pos_y-, n_y0, pos_y0, n_y+, pos_y+]
        features = np.empty(12, dtype=np.float32)
        features[0::2] = photons  # photons at even indices
        features[1::2] = positions  # positions at odd indices

        X.append(features)
        y.append(distance_nm)

    return np.array(X), np.array(y)


def extract_all_dynamic_data(data_dir='datasets/MINFLUXDynamic/parsed/raw'):
    """
    Extracts all dynamic data.

    Returns:
        X: np.array (N, 12)
        y: np.array (N,)
    """

    print("="*70)
    print("MINFLUX DYNAMIC - DATA EXTRACTION")
    print("="*70)

    X_all = []
    y_all = []

    data_path = Path(data_dir)

    # Find all distance directories
    distance_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    print(f"\nFound distances: {[d.name for d in distance_dirs]}")
    print()

    stats = {}

    for dist_dir in distance_dirs:
        # Extract distance from directory name (e.g. "20nm" -> 20.0)
        dist_name = dist_dir.name
        distance_nm = float(dist_name.replace('nm', ''))

        print(f"[{dist_name}] Loading data...")

        # Find all parsed.pkl files
        pkl_files = list(dist_dir.rglob('parsed.pkl'))

        if len(pkl_files) == 0:
            print(f"  ⚠️  No pkl files found!")
            continue

        dist_X = []
        dist_y = []

        for pkl_file in tqdm(pkl_files, desc=f"  {dist_name}", leave=False):
            try:
                X, y = extract_measurements_from_pkl(pkl_file, distance_nm)
                dist_X.append(X)
                dist_y.append(y)
            except Exception as e:
                print(f"  ✗ Error at {pkl_file.name}: {e}")
                continue

        if dist_X:
            dist_X = np.concatenate(dist_X, axis=0)
            dist_y = np.concatenate(dist_y, axis=0)

            X_all.append(dist_X)
            y_all.append(dist_y)

            stats[dist_name] = len(dist_X)
            print(f"  ✓ {len(dist_X):,} measurements loaded")
        else:
            print(f"  ✗ No measurements loaded")

    # Combine all distances
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Statistics
    print(f"\n{'='*70}")
    print("STATISTICS")
    print(f"{'='*70}")
    print(f"\nTotal: {len(X_all):,} measurements")
    print(f"\nPer distance:")
    for dist_name, count in stats.items():
        print(f"  {dist_name}: {count:7,} measurements")

    print(f"\nData:")
    print(f"  X Shape: {X_all.shape}")
    print(f"  y Shape: {y_all.shape}")
    print(f"  Distance range: [{y_all.min():.1f}, {y_all.max():.1f}] nm")

    # Photon statistics
    photons = X_all[:, 0::2]
    total_photons = photons.sum(axis=1)
    print(f"\nPhotons per measurement:")
    print(f"  Mean: {total_photons.mean():.1f}")
    print(f"  Median: {np.median(total_photons):.1f}")
    print(f"  Std: {total_photons.std():.1f}")
    print(f"  Range: [{total_photons.min():.0f}, {total_photons.max():.0f}]")

    return X_all, y_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                       default='datasets/MINFLUXDynamic/parsed/raw',
                       help='Directory with MINFLUXDynamic data')
    parser.add_argument('--output_name', type=str,
                       default='dynamic_data',
                       help='Name for output files')
    args = parser.parse_args()

    # Extract data
    X, y = extract_all_dynamic_data(args.data_dir)

    # Save as .npy
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)

    X_path = output_dir / f'{args.output_name}_X.npy'
    y_path = output_dir / f'{args.output_name}_y.npy'

    np.save(X_path, X)
    np.save(y_path, y)

    print(f"\n{'='*70}")
    print("SUCCESSFULLY SAVED")
    print(f"{'='*70}")
    print(f"\nX: {X_path}")
    print(f"y: {y_path}")
    print(f"\nReady for training!")
