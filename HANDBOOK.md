# MINFLUX ML - Handbook

> **Bachelor Thesis: Machine Learning Acceleration for MINFLUX Distance Estimation**
>
> Luca J. Bolz | 2024

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Models](#4-models)
5. [API Reference](#5-api-reference)
6. [Analysis & Results](#6-analysis--results)
7. [CLI Tool](#7-cli-tool)
8. [Reproducibility](#8-reproducibility)
9. [Limitations](#9-limitations)
10. [References](#10-references)

---

## 1. Overview

### What is this?

A machine learning extension for MINFLUX nanoscopy that accelerates distance estimation by **500×**.

### Core Problem

MINFLUX uses Maximum Likelihood Estimation (MLE) for distance estimation:
- **MLE**: ~100ms per measurement → 10 measurements/s
- **ML**: ~0.2ms per measurement → 5,000 measurements/s

### Results

| Method | RMSE | Data | Time | Speedup |
|--------|------|------|------|---------|
| **MLE (Baseline)** | **4.24nm** | Experimental | 100ms | 1× |
| ML (Original) | 5.12nm | Experimental | 0.2ms | 500× |
| ML (Balanced) | 3.22nm | Simulation | 0.2ms | 500× |

> **Important**: The ML RMSE of 3.22nm was measured on **simulation data**.
> On real experimental data, ML (5.12nm) is ~21% less accurate than MLE (4.24nm).
> The main advantage is the **500× speedup**, not accuracy.

---

## 2. Installation

### Requirements

- Python 3.9+
- ~2GB RAM (inference)
- ~16GB RAM (training)

### Setup

```bash
git clone https://github.com/lucajbolz/minflux-ml-acceleration.git
cd minflux-ml-acceleration
pip install -r requirements.txt
```

### Dependencies

```
numpy>=2.0
xgboost>=3.0
scikit-learn>=1.5
mapie>=1.0
pandas>=2.0
matplotlib>=3.8
```

---

## 3. Quick Start

### Basic Prediction

```python
import numpy as np
from ml_inference import MINFLUXDistanceEstimator

# Load model
estimator = MINFLUXDistanceEstimator('models/xgboost_balanced.pkl')

# MINFLUX measurement (6 photon counts + 6 beam positions)
photons = np.array([35, 42, 28, 38, 45, 30])
positions = np.array([-10, 2, -5, -12, 6, -20])

# Predict distance
distance = estimator.predict(photons, positions)
print(f"Distance: {distance:.2f} nm")
```

### With Uncertainty Quantification

```python
# Load model with UQ
estimator = MINFLUXDistanceEstimator('models/xgboost_balanced.pkl',
                                      use_uncertainty=True)

# Prediction with 90% confidence interval
distance, lower, upper = estimator.predict(photons, positions)
print(f"Distance: {distance:.2f} nm")
print(f"90% CI:  [{lower:.2f}, {upper:.2f}] nm")
```

### Batch Processing

```python
# 1000 measurements at once
photons_batch = np.random.poisson(40, (1000, 6)).astype(float)
positions_batch = np.random.uniform(-25, 25, (1000, 6))

distances = estimator.predict_batch(photons_batch, positions_batch)
print(f"Processed: {len(distances)} measurements")
```

---

## 4. Models

### Available Models

| Model | File | Use Case | RMSE (Sim) |
|-------|------|----------|------------|
| **Balanced** | `xgboost_balanced.pkl` | Recommended | 3.22nm |
| Dynamic | `xgboost_dynamic.pkl` | Time series | 2.84nm* |
| Static | `xgboost_optimized.pkl` | Single measurements | 5.13nm |

*on test data, not experimentally validated

### Balanced Model (Recommended)

The balanced model corrects the systematic bias of the original model:

| Distance | Original RMSE | Balanced RMSE | Improvement |
|----------|---------------|---------------|-------------|
| 15nm | 8.24nm | 3.60nm | **+56%** |
| 20nm | 3.21nm | 2.91nm | +9% |
| 30nm | 7.24nm | 3.95nm | **+45%** |

### UQ Models

For Uncertainty Quantification:
- `mapie_balanced.pkl` - For balanced model
- `mapie_dynamic.pkl` - For dynamic model
- `mapie_static.pkl` - For static model

---

## 5. API Reference

### MINFLUXDistanceEstimator

```python
class MINFLUXDistanceEstimator:
    """ML-based MINFLUX distance estimation."""

    def __init__(
        self,
        model_path: str = 'models/xgboost_balanced.pkl',
        use_uncertainty: bool = False
    ) -> None:
        """
        Args:
            model_path: Path to trained XGBoost model
            use_uncertainty: If True, compute 90% confidence intervals
        """

    def predict(
        self,
        photons: np.ndarray,  # Shape: (6,)
        positions: np.ndarray  # Shape: (6,)
    ) -> float | tuple[float, float, float]:
        """
        Single prediction.

        Returns:
            Without UQ: distance (nm)
            With UQ: (distance, lower, upper) in nm
        """

    def predict_batch(
        self,
        photons: np.ndarray,    # Shape: (n, 6)
        positions: np.ndarray   # Shape: (n, 6)
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch prediction."""

    def benchmark(self, n_samples: int = 100) -> dict:
        """Performance benchmark."""
```

### Feature Engineering

Raw data (12 features) is transformed to 15 engineered features:

1. **Photon Ratios** (6): `p_i / Σp_j`
2. **Normalized Positions** (6): Z-score normalized
3. **Modulation Depth** (2): Interference pattern strength
4. **Log Total Photons** (1): `log(Σp_j + 1)`

---

## 6. Analysis & Results

### Generated Plots

All plots in `analysis/`:

| Plot | Description |
|------|-------------|
| `error_analysis.png` | Residuals, scatter, boxplots |
| `feature_importance.png` | XGBoost feature importance |
| `uq_calibration.png` | UQ coverage per distance |
| `robustness.png` | Photon budget & noise tests |
| `speedup_accuracy_tradeoff.png` | Speedup vs accuracy curve |

### Generate Plots

```bash
python analysis_comprehensive.py
```

### Performance Comparison

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

### Usage

The CLI is directly usable:

```bash
python minflux_ml_cli.py <command> [options]
```

### Commands

#### predict - Batch predictions

```bash
python minflux_ml_cli.py predict \
    --input data.csv \
    --output results.csv \
    --model models/xgboost_balanced.pkl \
    --uncertainty
```

#### benchmark - Test speed

```bash
python minflux_ml_cli.py benchmark --samples 1000

# Output:
# Single prediction:  0.23 ms
# Throughput:         4300 predictions/s
# Speedup vs MLE:     430×
```

#### info - Model information

```bash
python minflux_ml_cli.py info --model models/xgboost_balanced.pkl
```

---

## 8. Reproducibility

### Random Seeds

| Operation | Seed |
|-----------|------|
| Train/Test Split | 42 |
| Data Sampling | 42 |
| XGBoost Training | 42 |

### Data Distribution

```
Dynamic Dataset (584,250 Samples):
├── 15nm:  24,950 (4.3%)
├── 20nm: 419,580 (71.8%)  ← Reason for original bias
└── 30nm: 139,720 (23.9%)
```

### Reproduction Steps

```bash
# 1. Download data (Zenodo)
wget https://zenodo.org/record/10625021/files/MINFLUXDynamic.zip

# 2. Extract features
python ml_extract_dynamic.py --data_dir datasets/MINFLUXDynamic/parsed/raw

# 3. Train balanced model
python ml_train_balanced.py --weight_method inverse --compare

# 4. Calibrate UQ
python ml_uncertainty_quantification.py --model balanced

# 5. Generate analysis
python analysis_comprehensive.py
```

### Expected Results

```
Balanced Model Training:
  Overall RMSE: 3.22 nm
  15nm RMSE:    3.60 nm (vs Original: 8.24nm, +56%)
  20nm RMSE:    2.91 nm
  30nm RMSE:    3.95 nm (vs Original: 7.24nm, +45%)
```

---

## 9. Limitations

### 1. Distribution Shift

ML was trained on **simulation data**. Performance on real experimental data is worse:
- MLE on real data: **4.24nm**
- ML on real data: **5.12nm** (+21% worse)

### 2. Distance Range

- Dynamic Model: 15-30nm
- Static Model: 6-30nm
- Extrapolation not validated

### 3. Systematic Bias

Even after balancing, some bias remains:
- 15nm: +2.3nm overestimation
- 30nm: -2.5nm underestimation

### 4. Photon Budget Dependency

Trained with:
- Dynamic: ~198 photons/measurement
- Static: ~84 photons/measurement

Accuracy decreases with significantly different photon counts.

---

## 10. References

### Original MINFLUX Paper

Hensel, T. et al. *Diffraction minima resolve point scatterers at tiny fractions (1/80) of the wavelength*.
**Nature Physics** (2024). [DOI: 10.1038/s41567-024-02760-1](https://www.nature.com/articles/s41567-024-02760-1)

### Simulation Data

Zenodo Repository: [DOI: 10.5281/zenodo.10625021](https://doi.org/10.5281/zenodo.10625021)

### Software

- XGBoost: Chen & Guestrin, KDD 2016
- MAPIE: Conformal Prediction for Uncertainty Quantification
- NumPy, scikit-learn, pandas, matplotlib

### Citation

```bibtex
@software{bolz2024minflux_ml,
  title={Machine Learning Acceleration for MINFLUX Distance Estimation},
  author={Bolz, Luca J.},
  year={2024},
  url={https://github.com/lucajbolz/minflux-ml-acceleration}
}
```

---

## Project Structure

```
.
├── HANDBOOK.md                      # This document
├── README.md                        # Quick Start
├── README_ML.md                     # Technical details
├── REPRODUCIBILITY.md               # Reproducibility guide
├── requirements.txt                 # Dependencies
│
├── ml_inference.py                  # Main API
├── ml_train_balanced.py             # Balanced training
├── ml_uncertainty_quantification.py # UQ calibration
├── minflux_ml_cli.py               # CLI tool
├── analysis_comprehensive.py        # Analysis plots
├── demo_realtime.py                # Real-time demo
├── example_notebook.ipynb          # Jupyter example
│
├── models/
│   ├── xgboost_balanced.pkl        # Recommended model
│   ├── xgboost_dynamic.pkl         # Original dynamic
│   ├── xgboost_optimized.pkl       # Static model
│   ├── mapie_balanced.pkl          # UQ for balanced
│   └── ...
│
├── analysis/                        # Generated plots
│   ├── error_analysis.png
│   ├── feature_importance.png
│   └── ...
│
├── data/                            # Extracted features
└── lib/                             # Original MINFLUX code
```

---

**Contact**: bolz@physik.uni-kiel.de | GitHub Issues
