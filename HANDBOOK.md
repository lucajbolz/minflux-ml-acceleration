# MINFLUX ML - Handbuch

> **Bachelorarbeit: Machine Learning Acceleration for MINFLUX Distance Estimation**
>
> Luca J. Bolz | 2024

---

## Inhaltsverzeichnis

1. [Überblick](#1-überblick)
2. [Installation](#2-installation)
3. [Schnellstart](#3-schnellstart)
4. [Modelle](#4-modelle)
5. [API Referenz](#5-api-referenz)
6. [Analyse & Ergebnisse](#6-analyse--ergebnisse)
7. [CLI Tool](#7-cli-tool)
8. [Reproduzierbarkeit](#8-reproduzierbarkeit)
9. [Limitationen](#9-limitationen)
10. [Referenzen](#10-referenzen)

---

## 1. Überblick

### Was ist das?

Eine Machine-Learning-Erweiterung für die MINFLUX-Nanoskopie, die die Distanzschätzung um **500×** beschleunigt.

### Kernproblem

MINFLUX verwendet Maximum Likelihood Estimation (MLE) zur Distanzschätzung:
- **MLE**: ~100ms pro Messung → 10 Messungen/s
- **ML**: ~0.2ms pro Messung → 5,000 Messungen/s

### Ergebnisse

| Methode | RMSE | Daten | Zeit | Speedup |
|---------|------|-------|------|---------|
| **MLE (Baseline)** | **4.24nm** | Experimentell | 100ms | 1× |
| ML (Original) | 5.12nm | Experimentell | 0.2ms | 500× |
| ML (Balanced) | 3.22nm | Simulation | 0.2ms | 500× |

> ⚠️ **Wichtig**: Der ML RMSE von 3.22nm wurde auf **Simulationsdaten** gemessen.
> Auf echten experimentellen Daten ist ML (5.12nm) schlechter als MLE (4.24nm).
> Der Hauptvorteil ist der **500× Speedup**, nicht die Genauigkeit.

---

## 2. Installation

### Voraussetzungen

- Python 3.9+
- ~2GB RAM (Inferenz)
- ~16GB RAM (Training)

### Setup

```bash
git clone https://github.com/lucajbolz/minflux-ml-acceleration.git
cd minflux-ml-acceleration
pip install -r requirements.txt
```

### Abhängigkeiten

```
numpy>=2.0
xgboost>=3.0
scikit-learn>=1.5
mapie>=1.0
pandas>=2.0
matplotlib>=3.8
```

---

## 3. Schnellstart

### Einfache Vorhersage

```python
import numpy as np
from ml_inference import MINFLUXDistanceEstimator

# Model laden
estimator = MINFLUXDistanceEstimator('models/xgboost_balanced.pkl')

# MINFLUX Messung (6 Photonenzahlen + 6 Beam-Positionen)
photons = np.array([35, 42, 28, 38, 45, 30])
positions = np.array([-10, 2, -5, -12, 6, -20])

# Distanz vorhersagen
distance = estimator.predict(photons, positions)
print(f"Distanz: {distance:.2f} nm")
```

### Mit Unsicherheitsquantifizierung

```python
# Model mit UQ laden
estimator = MINFLUXDistanceEstimator('models/xgboost_balanced.pkl',
                                      use_uncertainty=True)

# Vorhersage mit 90% Konfidenzintervall
distance, lower, upper = estimator.predict(photons, positions)
print(f"Distanz: {distance:.2f} nm")
print(f"90% CI:  [{lower:.2f}, {upper:.2f}] nm")
```

### Batch-Verarbeitung

```python
# 1000 Messungen auf einmal
photons_batch = np.random.poisson(40, (1000, 6)).astype(float)
positions_batch = np.random.uniform(-25, 25, (1000, 6))

distances = estimator.predict_batch(photons_batch, positions_batch)
print(f"Verarbeitet: {len(distances)} Messungen")
```

---

## 4. Modelle

### Verfügbare Modelle

| Modell | Datei | Anwendung | RMSE (Sim) |
|--------|-------|-----------|------------|
| **Balanced** | `xgboost_balanced.pkl` | Empfohlen | 3.22nm |
| Dynamic | `xgboost_dynamic.pkl` | Zeitreihen | 2.84nm* |
| Static | `xgboost_optimized.pkl` | Einzelmessungen | 5.13nm |

*auf Testdaten, nicht experimentell validiert

### Balanced Model (Empfohlen)

Das balanced Model korrigiert den systematischen Bias des Original-Modells:

| Distanz | Original RMSE | Balanced RMSE | Verbesserung |
|---------|---------------|---------------|--------------|
| 15nm | 8.24nm | 3.60nm | **+56%** |
| 20nm | 3.21nm | 2.91nm | +9% |
| 30nm | 7.24nm | 3.95nm | **+45%** |

### UQ Modelle

Für Uncertainty Quantification:
- `mapie_balanced.pkl` - Für balanced Model
- `mapie_dynamic.pkl` - Für dynamic Model
- `mapie_static.pkl` - Für static Model

---

## 5. API Referenz

### MINFLUXDistanceEstimator

```python
class MINFLUXDistanceEstimator:
    """ML-basierte MINFLUX Distanzschätzung."""

    def __init__(
        self,
        model_path: str = 'models/xgboost_balanced.pkl',
        use_uncertainty: bool = False
    ) -> None:
        """
        Args:
            model_path: Pfad zum trainierten XGBoost Model
            use_uncertainty: Wenn True, 90% Konfidenzintervalle berechnen
        """

    def predict(
        self,
        photons: np.ndarray,  # Shape: (6,)
        positions: np.ndarray  # Shape: (6,)
    ) -> float | tuple[float, float, float]:
        """
        Einzelne Vorhersage.

        Returns:
            Ohne UQ: distance (nm)
            Mit UQ: (distance, lower, upper) in nm
        """

    def predict_batch(
        self,
        photons: np.ndarray,    # Shape: (n, 6)
        positions: np.ndarray   # Shape: (n, 6)
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch-Vorhersage."""

    def benchmark(self, n_samples: int = 100) -> dict:
        """Performance-Benchmark."""
```

### Feature Engineering

Die Rohdaten (12 Features) werden zu 15 engineered Features transformiert:

1. **Photon Ratios** (6): `p_i / Σp_j`
2. **Normalized Positions** (6): Z-Score normalisiert
3. **Modulation Depth** (2): Interferenzmuster-Stärke
4. **Log Total Photons** (1): `log(Σp_j + 1)`

---

## 6. Analyse & Ergebnisse

### Generierte Plots

Alle Plots in `analysis/`:

| Plot | Beschreibung |
|------|--------------|
| `error_analysis.png` | Residuen, Scatter, Boxplots |
| `feature_importance.png` | XGBoost Feature Importance |
| `uq_calibration.png` | UQ Coverage pro Distanz |
| `robustness.png` | Photon-Budget & Noise Tests |
| `speedup_accuracy_tradeoff.png` | Speedup vs Accuracy Kurve |

### Plots generieren

```bash
python analysis_comprehensive.py
```

### Performance-Vergleich

```
Speedup-Accuracy Tradeoff:
┌─────────────┬──────────┬─────────┐
│ n_estimators│ RMSE(nm) │ Speedup │
├─────────────┼──────────┼─────────┤
│     10      │   4.8    │  2000×  │
│     50      │   3.8    │  1000×  │
│    100      │   3.5    │   600×  │
│    500      │   3.2    │   400×  │
└─────────────┴──────────┴─────────┘
```

---

## 7. CLI Tool

### Installation

Das CLI ist direkt nutzbar:

```bash
python minflux_ml_cli.py <command> [options]
```

### Befehle

#### predict - Batch-Vorhersagen

```bash
python minflux_ml_cli.py predict \
    --input data.csv \
    --output results.csv \
    --model models/xgboost_balanced.pkl \
    --uncertainty
```

#### benchmark - Geschwindigkeit testen

```bash
python minflux_ml_cli.py benchmark --samples 1000

# Output:
# Single prediction:  0.23 ms
# Throughput:         4300 predictions/s
# Speedup vs MLE:     430×
```

#### info - Model-Informationen

```bash
python minflux_ml_cli.py info --model models/xgboost_balanced.pkl
```

---

## 8. Reproduzierbarkeit

### Random Seeds

| Operation | Seed |
|-----------|------|
| Train/Test Split | 42 |
| Data Sampling | 42 |
| XGBoost Training | 42 |

### Datenverteilung

```
Dynamic Dataset (584,250 Samples):
├── 15nm:  24,950 (4.3%)
├── 20nm: 419,580 (71.8%)  ← Grund für Original-Bias
└── 30nm: 139,720 (23.9%)
```

### Reproduktion

```bash
# 1. Daten herunterladen (Zenodo)
wget https://zenodo.org/record/10625021/files/MINFLUXDynamic.zip

# 2. Features extrahieren
python ml_extract_dynamic.py --data_dir datasets/MINFLUXDynamic/parsed/raw

# 3. Balanced Model trainieren
python ml_train_balanced.py --weight_method inverse --compare

# 4. UQ kalibrieren
python ml_uncertainty_quantification.py --model balanced

# 5. Analyse generieren
python analysis_comprehensive.py
```

### Erwartete Ergebnisse

```
Balanced Model Training:
  Overall RMSE: 3.22 nm
  15nm RMSE:    3.60 nm (vs Original: 8.24nm, +56%)
  20nm RMSE:    2.91 nm
  30nm RMSE:    3.95 nm (vs Original: 7.24nm, +45%)
```

---

## 9. Limitationen

### 1. Distribution Shift

ML wurde auf **Simulationsdaten** trainiert. Performance auf echten experimentellen Daten ist schlechter:
- MLE auf echten Daten: **4.24nm**
- ML auf echten Daten: **5.12nm** (+21% schlechter)

### 2. Distanzbereich

- Dynamic Model: 15-30nm
- Static Model: 6-30nm
- Extrapolation nicht validiert

### 3. Systematischer Bias

Auch nach Balancing bleibt ein Bias:
- 15nm: +2.3nm Überschätzung
- 30nm: -2.5nm Unterschätzung

### 4. Photon-Budget Abhängigkeit

Trainiert mit:
- Dynamic: ~198 Photonen/Messung
- Static: ~84 Photonen/Messung

Bei stark abweichenden Photonenzahlen sinkt die Genauigkeit.

---

## 10. Referenzen

### Original MINFLUX Paper

Hensel, T. et al. *Diffraction minima resolve point scatterers at tiny fractions (1/80) of the wavelength*.
**Nature Physics** (2024). [DOI: 10.1038/s41567-024-02760-1](https://www.nature.com/articles/s41567-024-02760-1)

### Simulationsdaten

Zenodo Repository: [DOI: 10.5281/zenodo.10625021](https://doi.org/10.5281/zenodo.10625021)

### Software

- XGBoost: Chen & Guestrin, KDD 2016
- MAPIE: Conformal Prediction für Uncertainty Quantification
- NumPy, scikit-learn, pandas, matplotlib

### Zitation

```bibtex
@software{bolz2024minflux_ml,
  title={Machine Learning Acceleration for MINFLUX Distance Estimation},
  author={Bolz, Luca J.},
  year={2024},
  url={https://github.com/lucajbolz/minflux-ml-acceleration}
}
```

---

## Projektstruktur

```
.
├── HANDBOOK.md                      # Dieses Dokument
├── README.md                        # Quick Start
├── README_ML.md                     # Technische Details
├── REPRODUCIBILITY.md               # Reproduzierbarkeit
├── requirements.txt                 # Dependencies
│
├── ml_inference.py                  # Haupt-API
├── ml_train_balanced.py             # Balanced Training
├── ml_uncertainty_quantification.py # UQ Kalibrierung
├── minflux_ml_cli.py               # CLI Tool
├── analysis_comprehensive.py        # Analyse-Plots
├── demo_realtime.py                # Real-Time Demo
├── example_notebook.ipynb          # Jupyter Beispiel
│
├── models/
│   ├── xgboost_balanced.pkl        # Empfohlenes Model
│   ├── xgboost_dynamic.pkl         # Original Dynamic
│   ├── xgboost_optimized.pkl       # Static Model
│   ├── mapie_balanced.pkl          # UQ für Balanced
│   └── ...
│
├── analysis/                        # Generierte Plots
│   ├── error_analysis.png
│   ├── feature_importance.png
│   └── ...
│
├── data/                            # Extrahierte Features
└── lib/                             # Original MINFLUX Code
```

---

**Kontakt**: bolz@physik.uni-kiel.de | GitHub Issues
