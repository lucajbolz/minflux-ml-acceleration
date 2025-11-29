"""
Extrahiert Trainingsdaten aus den echten Paper-Messungen

Lädt alle parsed.pkl Files aus datasets/MINFLUXStatic/parsed/
und erstellt einen Trainingsdatensatz mit:
- X: 6 Photonenwerte pro Messung
- y: Ground-truth Distanz (aus Verzeichnisnamen)
"""

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os


def extract_measurements_from_pkl(pkl_path, distance_nm):
    """
    Extrahiert alle Messungen aus einem parsed.pkl File.

    WICHTIG: Sortiert die Photonen nach axis und position für Konsistenz!
    JETZT MIT BEAM POSITIONEN als zusätzliche Features!

    Features: [n_x-, pos_x-, n_x0, pos_x0, n_x+, pos_x+,
               n_y-, pos_y-, n_y0, pos_y0, n_y+, pos_y+]

    Args:
        pkl_path: Path zum parsed.pkl File
        distance_nm: Ground-truth Distanz in nm

    Returns:
        X: np.array (N, 12) - Photonen + Positionen (sortiert!)
        y: np.array (N,) - Distanzen
    """
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)

    X = []
    y = []

    # Gruppiere nach tuple (jede Messung)
    for tuple_id in df['tuple'].unique():
        measurement = df[df['tuple'] == tuple_id]

        # Sollte 6 Werte haben
        if len(measurement) != 6:
            continue

        # Sortiere nach axis (0=x, 1=y) und dann nach position
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


def extract_all_paper_data(data_dir='datasets/MINFLUXStatic/parsed'):
    """
    Extrahiert alle Paper-Daten.

    Returns:
        X: np.array (N, 6)
        y: np.array (N,)
    """

    print("="*70)
    print("PAPER-DATEN EXTRAKTION")
    print("="*70)

    X_all = []
    y_all = []

    data_path = Path(data_dir)

    # Finde alle Distanz-Verzeichnisse
    distance_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    print(f"\nGefundene Distanzen: {[d.name for d in distance_dirs]}")
    print()

    stats = {}

    for dist_dir in distance_dirs:
        # Extrahiere Distanz aus Verzeichnisnamen (z.B. "08nm" -> 8.0)
        dist_name = dist_dir.name
        distance_nm = float(dist_name.replace('nm', ''))

        print(f"[{dist_name}] Lade Daten...")

        # Finde alle parsed.pkl Files
        pkl_files = list(dist_dir.rglob('parsed.pkl'))

        if len(pkl_files) == 0:
            print(f"  ⚠️  Keine pkl Files gefunden!")
            continue

        dist_X = []
        dist_y = []

        for pkl_file in tqdm(pkl_files, desc=f"  {dist_name}", leave=False):
            try:
                X, y = extract_measurements_from_pkl(pkl_file, distance_nm)
                dist_X.append(X)
                dist_y.append(y)
            except Exception as e:
                print(f"  ✗ Fehler bei {pkl_file.name}: {e}")
                continue

        if dist_X:
            dist_X = np.concatenate(dist_X, axis=0)
            dist_y = np.concatenate(dist_y, axis=0)

            X_all.append(dist_X)
            y_all.append(dist_y)

            stats[dist_name] = len(dist_X)
            print(f"  ✓ {len(dist_X)} Messungen geladen")
        else:
            print(f"  ✗ Keine Messungen geladen")

    # Kombiniere alle Distanzen
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Statistiken
    print(f"\n{'='*70}")
    print("STATISTIKEN")
    print(f"{'='*70}")
    print(f"\nGesamt: {len(X_all)} Messungen")
    print(f"\nPro Distanz:")
    for dist_name, count in stats.items():
        print(f"  {dist_name}: {count:5d} Messungen")

    print(f"\nDaten:")
    print(f"  X Shape: {X_all.shape}")
    print(f"  y Shape: {y_all.shape}")
    print(f"  Distanz Range: [{y_all.min():.1f}, {y_all.max():.1f}] nm")

    # Photonen-Statistik
    total_photons = X_all.sum(axis=1)
    print(f"\nPhotonen pro Messung:")
    print(f"  Mean: {total_photons.mean():.1f}")
    print(f"  Median: {np.median(total_photons):.1f}")
    print(f"  Min: {total_photons.min():.0f}")
    print(f"  Max: {total_photons.max():.0f}")
    print(f"  Std: {total_photons.std():.1f}")

    # Zeige Verteilung
    print(f"\nPhotonen pro Position (Mean):")
    for i in range(6):
        pos_names = ['n_x-', 'n_x0', 'n_x+', 'n_y-', 'n_y0', 'n_y+']
        zero_pct = 100 * (X_all[:, i] == 0).sum() / len(X_all)
        print(f"  {pos_names[i]:5s}: {X_all[:, i].mean():5.1f} (Zero: {zero_pct:4.1f}%)")

    return X_all, y_all


def save_paper_data(X, y, output_dir='data', output_name='paper_data'):
    """Speichert die extrahierten Daten."""

    os.makedirs(output_dir, exist_ok=True)

    np.save(f'{output_dir}/{output_name}_X.npy', X)
    np.save(f'{output_dir}/{output_name}_y.npy', y)

    print(f"\n{'='*70}")
    print("✓ DATEN GESPEICHERT")
    print(f"{'='*70}")
    print(f"  {output_dir}/{output_name}_X.npy")
    print(f"  {output_dir}/{output_name}_y.npy")
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/MINFLUXStatic/parsed')
    parser.add_argument('--output_name', type=str, default='paper_data')

    args = parser.parse_args()

    X, y = extract_all_paper_data(data_dir=args.data_dir)
    save_paper_data(X, y, output_name=args.output_name)
