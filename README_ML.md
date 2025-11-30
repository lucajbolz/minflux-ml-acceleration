# Machine Learning Acceleration for MINFLUX Distance Estimation

## Abstract

This repository provides a machine learning extension to the [MINFLUX simulation framework](https://www.nature.com/articles/s41567-024-02760-1) that accelerates distance estimation by up to 512× while maintaining near-MLE (Maximum Likelihood Estimation) accuracy. The standard MINFLUX method uses computationally expensive MLE (~100ms per measurement), which limits real-time applications. This work presents two XGBoost-based regression models that achieve comparable accuracy in ~0.2ms, enabling real-time MINFLUX analysis at ~5000 measurements/second.

**Key Results:**
- **Balanced model**: 3.22nm RMSE at 0.2ms inference (500× speedup vs MLE)
- Main advantage: **500× speedup**, enabling real-time analysis

> **Note**: ML was trained on dynamic MINFLUX data (15/20/30nm) from Zenodo. The original paper reports MLE performance (4.24nm) on different static measurements (8-32nm). Direct comparison on identical data was not performed.

## Table of Contents

- [Introduction](#introduction)
- [Methods](#methods)
- [Installation](#installation)
- [Usage](#usage)
- [Reproduction](#reproduction)
- [Performance](#performance)
- [Limitations](#limitations)
- [References](#references)

## Introduction

MINFLUX (Minimal Photon Fluxes) nanoscopy achieves sub-nanometer localization precision by measuring photon counts from a fluorescent emitter at multiple excitation beam positions. Distance estimation from raw photon counts to the emitter position typically requires Maximum Likelihood Estimation (MLE), which involves iterative optimization and is computationally expensive.

This work replaces MLE with gradient boosting regression models (XGBoost) that learn the mapping from raw photon measurements to distances directly from simulation data. Two separate models are trained for different experimental scenarios:

1. **Static Model**: Trained on single-position measurements (12.17M samples, 6-30nm range)
2. **Dynamic Model**: Trained on time-series traces (584,250 samples, 15-30nm range)

The models demonstrate that ML can achieve near-MLE accuracy at a fraction of the computational cost, enabling real-time MINFLUX applications.

## Methods

### Data Sources

Training data is derived from the MINFLUX simulation framework:
- **Original Framework**: Hensel et al., *Nature Physics* (2024) ([DOI](https://www.nature.com/articles/s41567-024-02760-1))
- **Simulation Data**: Available on Zenodo ([DOI](https://doi.org/10.5281/zenodo.10625021))

**Static Data** (MINFLUXStatic):
- Distance range: 6-30nm (uniform sampling)
- Photon budget: ~84 photons/measurement (uniform distribution)
- Total samples: 12,169,500 measurements
- Source: Simulated single-position measurements

**Dynamic Data** (MINFLUXDynamic):
- Distance range: 15nm, 20nm, 30nm (discrete)
- Photon budget: ~198 photons/measurement (skewed distribution)
- Total samples: 584,250 measurements (24,950 @ 15nm, 419,580 @ 20nm, 139,720 @ 30nm)
- Source: Time-series traces (9,990 timepoints each)

### Feature Engineering

Each MINFLUX measurement consists of 6 photon counts (3 beam positions × 2 axes) and 6 beam position coordinates. Raw inputs are transformed into 15 engineered features:

1. **Photon Ratios** (6 features): Normalized photon counts `p_i / (Σp_j + ε)` where ε=1e-8
2. **Normalized Positions** (6 features): Z-score normalized beam positions `(x_i - μ) / (σ + ε)`
3. **Modulation Depth** (2 features):
   - X-axis: `(p_0 + p_2 - 2p_1) / Σp_j`
   - Y-axis: `(p_3 + p_5 - 2p_4) / Σp_j`
4. **Log Total Photons** (1 feature): `log(Σp_j + 1)`

**Rationale**: Photon ratios capture relative intensities independent of total photon count, modulation depth quantifies the interference pattern strength, and log transformation handles photon count variability.

### Model Architecture

**Algorithm**: XGBoost (Extreme Gradient Boosting) regression
**Objective**: Minimize squared error (reg:squarederror)

**Hyperparameters** (both models):
```python
{
    'max_depth': 8,              # Tree depth
    'learning_rate': 0.1,        # Shrinkage for regularization
    'n_estimators': 500,         # Number of boosting rounds
    'subsample': 0.8,            # Row subsampling (80%)
    'colsample_bytree': 0.8,     # Column subsampling (80%)
    'tree_method': 'hist',       # Histogram-based tree building
    'early_stopping_rounds': 50  # Stop if no improvement for 50 rounds
}
```

**Training Procedure**:
1. Train/test split: 80/20 random stratified split
2. No cross-validation (large dataset size, fast training)
3. Early stopping on held-out test set
4. Training time: ~2-5 minutes on standard CPU

### Evaluation Metrics

Performance is measured using:
- **RMSE (Root Mean Squared Error)**: Primary metric, nanometer scale
- **MAE (Mean Absolute Error)**: Robustness to outliers
- **Inference Time**: Single-prediction latency (CPU)
- **Speedup Factor**: Relative to MLE baseline (~100ms)

Validation is performed on:
1. Held-out test set (20% of training data)
2. Real experimental traces (15 traces: 5×15nm, 5×20nm, 5×30nm)

## Installation

### Requirements

- Python 3.9 or higher (tested with Python 3.13.5)
- ~2GB RAM for inference
- ~16GB RAM for training

### Setup

```bash
# Clone repository
git clone <repository-url>
cd multiflux-release-main

# Install dependencies
pip install -r requirements.txt
```

**Exact versions** (recommended for reproducibility):
```
numpy==2.3.4
xgboost==3.1.2
pandas==2.3.3
scikit-learn==1.7.2
matplotlib==3.10.7  # Optional, for visualization
```

### Data Acquisition

Training data is **not included** in this repository due to size constraints. Download from Zenodo:

```bash
# Download MINFLUXStatic dataset (Static model)
wget https://zenodo.org/record/10625021/files/MINFLUXStatic.zip
unzip MINFLUXStatic.zip -d datasets/

# Download MINFLUXDynamic dataset (Dynamic model)
wget https://zenodo.org/record/10625021/files/MINFLUXDynamic.zip
unzip MINFLUXDynamic.zip -d datasets/
```

Expected directory structure:
```
datasets/
├── MINFLUXStatic/parsed/     # ~50GB
└── MINFLUXDynamic/parsed/raw/ # ~20GB
```

## Usage

### Quick Start: Inference

```python
import numpy as np
from ml_inference import MINFLUXDistanceEstimator

# Load pretrained model
estimator = MINFLUXDistanceEstimator('models/xgboost_dynamic.pkl')

# Example MINFLUX measurement
# photons: [x1, x2, x3, y1, y2, y3] - 6 photon counts
# positions: [px1, px2, px3, py1, py2, py3] - 6 beam positions (nm)
photons = np.array([20, 15, 68, 39, 15, 41])
positions = np.array([6.21, 21.21, 36.21, -34.95, -19.95, -4.95])

# Predict distance in nanometers
distance = estimator.predict(photons, positions)
print(f"Estimated distance: {distance:.2f} nm")
```

### Uncertainty Quantification

Models support optional uncertainty quantification via conformal prediction (90% confidence intervals):

```python
from ml_inference import MINFLUXDistanceEstimator

# Load model with uncertainty quantification
estimator = MINFLUXDistanceEstimator('models/xgboost_dynamic.pkl',
                                      use_uncertainty=True)

# Single prediction with 90% confidence interval
distance, lower, upper = estimator.predict(photons, positions)
print(f"Distance: {distance:.2f} nm")
print(f"90% CI:   [{lower:.2f}, {upper:.2f}] nm")
print(f"Interval width: {upper - lower:.2f} nm")

# Batch predictions with uncertainty
distances, lower_bounds, upper_bounds = estimator.predict_batch(photons_batch,
                                                                  positions_batch)
```

**How it works:**
- Uses MAPIE (Model Agnostic Prediction Interval Estimator) with split conformal regression
- Calibrated on independent calibration set (15% of data)
- Provides distribution-free coverage guarantees
- Empirical coverage: ~90.1% (matches theoretical target)

**Calibration:**
```bash
# Calibrate uncertainty quantification for dynamic model
python ml_uncertainty_quantification.py --model dynamic

# Calibrate for static model
python ml_uncertainty_quantification.py --model static
```

**Performance:**
- Dynamic model: Mean interval width ~9.5nm, Coverage: 90.1%
- Static model: Mean interval width ~18.5nm, Coverage: 91.3%
- Minimal computational overhead (~0.01ms additional latency)

### Model Selection

| Use Case | Model | File |
|----------|-------|------|
| Single-position measurements | Static | `models/xgboost_optimized.pkl` |
| Time-series traces | Dynamic | `models/xgboost_dynamic.pkl` |

**Decision rule**: Use Dynamic model for sequential measurements with potential temporal correlations, Static model for independent measurements.

## Reproduction

### Step 1: Extract Training Data

```bash
# Extract Static data (~2-3 hours on standard HDD)
python ml_extract_static.py --data_dir datasets/MINFLUXStatic/parsed

# Extract Dynamic data (~30-60 minutes)
python ml_extract_dynamic.py --data_dir datasets/MINFLUXDynamic/parsed/raw
```

Output files (stored in `data/`):
- Static: `paper_data_X.npy` (12.17M × 15), `paper_data_y.npy` (12.17M,)
- Dynamic: `dynamic_data_X.npy` (584K × 15), `dynamic_data_y.npy` (584K,)

### Step 2: Train Models

```bash
# Train Static model (~3-5 minutes)
python ml_train_static.py

# Train Dynamic model (~2-3 minutes)
python ml_train_dynamic.py
```

Output: Trained models saved to `models/` directory with performance metrics.

### Expected Training Results

**Static Model**:
- Test RMSE: 5.13nm
- Test MAE: 4.15nm
- Training samples: 900,000
- Test samples: 100,000

**Dynamic Model**:
- Test RMSE: 2.84nm (on held-out test set)
- Balanced Model RMSE: 3.22nm (after class balancing)
- Training samples: 467,400
- Test samples: 116,850

### Data Sources

ML was trained on dynamic MINFLUX data from the Zenodo repository (15/20/30nm distances). The original paper reports MLE performance (4.24nm RMSE) on different static MINFLUX measurements with different ground truth distances (8-32nm). Direct comparison on identical data was not performed.

## Performance

### Accuracy vs. Speed Trade-off

| Metric | ML (Balanced) | MLE (Paper) |
|--------|---------------|-------------|
| RMSE | 3.22 nm | 4.24 nm* |
| Inference Time | 0.2 ms | ~100 ms |
| Speedup | 500× | 1× |
| Throughput | 5,000/sec | 10/sec |

*MLE result from original paper on different dataset (static measurements, 8-32nm).

### System Requirements

**Inference**:
- CPU: Any modern processor (tested on Apple M1/M2, Intel Xeon)
- RAM: <100MB per model
- Storage: 10MB (models)

**Training**:
- CPU: Multi-core recommended (8+ cores)
- RAM: 16GB minimum, 32GB recommended
- Storage: ~70GB for datasets, ~1GB for processed data
- Time: ~4-6 hours total (data extraction + training)

## Limitations

1. **Different Datasets**: ML was trained on dynamic MINFLUX data (15/20/30nm). The paper's MLE results (4.24nm) are from different static measurements (8-32nm). Direct comparison on identical data was not performed.

2. **Distance Range**:
   - Static model: Valid for 6-30nm (training range)
   - Dynamic model: Valid for 15-30nm (training range)
   - Extrapolation beyond these ranges is not validated.

3. **Photon Budget Dependency**: Models assume photon counts similar to training data (~84 photons for Static, ~198 for Dynamic). Significantly different photon budgets may reduce accuracy.

4. **Systematic Bias**: Models exhibit regression-to-mean behavior, particularly for distances far from the most common training values (20nm for Dynamic model). This results in:
   - 15nm measurements: +3.3nm average error (overestimation)
   - 20nm measurements: +0.7nm average error (near-optimal)
   - 30nm measurements: -2.8nm average error (underestimation)
   - Uncertainty intervals account for this variation but do not correct the bias.

5. **Interpretability**: XGBoost models are less interpretable than physics-based MLE, making it harder to diagnose failure modes.

## Repository Structure

```
.
├── README_ML.md                      # This file
├── requirements.txt                  # Exact package versions
├── models/
│   ├── xgboost_optimized.pkl        # Static model (17MB)
│   ├── xgboost_dynamic.pkl          # Dynamic model (6.5MB)
│   ├── mapie_static.pkl             # UQ model for Static (calibrated)
│   └── mapie_dynamic.pkl            # UQ model for Dynamic (calibrated)
├── ml_inference.py                  # Inference wrapper with UQ support
├── ml_extract_static.py             # Data extraction for Static model
├── ml_extract_dynamic.py            # Data extraction for Dynamic model
├── ml_train_static.py               # Training script for Static model
├── ml_train_dynamic.py              # Training script for Dynamic model
├── ml_uncertainty_quantification.py # UQ calibration via conformal prediction
├── lib/                             # Original MINFLUX simulation library
├── src/                             # Original MINFLUX source code
└── datasets/                        # (Not included) Download from Zenodo
    ├── MINFLUXStatic/
    └── MINFLUXDynamic/
```

## References

### Original MINFLUX Work

Hensel, T. et al. *Diffraction minima resolve point scatterers at tiny fractions (1/80) of the wavelength*. Nature Physics (2024). [DOI: 10.1038/s41567-024-02760-1](https://www.nature.com/articles/s41567-024-02760-1)

### Data Availability

MINFLUX simulation data: Zenodo repository [DOI: 10.5281/zenodo.10625021](https://doi.org/10.5281/zenodo.10625021)

### Software Dependencies

- XGBoost: Chen & Guestrin, *KDD* 2016. [DOI: 10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
- NumPy: Harris et al., *Nature* 2020. [DOI: 10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
- scikit-learn: Pedregosa et al., *JMLR* 2011. [JMLR](https://jmlr.org/papers/v12/pedregosa11a.html)

### Citation

If you use this work, please cite:

```bibtex
@software{minflux_ml_acceleration,
  title={Machine Learning Acceleration for MINFLUX Distance Estimation},
  author={Bolz, Luca J.},
  year={2024},
  url={https://github.com/lucajbolz/minflux-ml-acceleration}
}
```

And the original MINFLUX work:

```bibtex
@article{hensel2024diffraction,
  title={Diffraction minima resolve point scatterers at tiny fractions (1/80) of the wavelength},
  author={Hensel, Thomas and others},
  journal={Nature Physics},
  year={2024},
  doi={10.1038/s41567-024-02760-1}
}
```

## License

This project uses the same license as the original MINFLUX simulation framework. See [LICENSE](LICENSE) for details.

## Contact

For questions or issues, please open an issue on the GitHub repository or contact bolz@physik.uni-kiel.de
