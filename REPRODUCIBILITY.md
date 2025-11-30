# Reproducibility Guide

This document describes how to reproduce all results in this repository.

## Environment

### Software Versions
```
Python: 3.9+
numpy: 2.3.4
xgboost: 3.1.2
scikit-learn: 1.7.2
mapie: 1.2.0
pandas: 2.3.3
matplotlib: 3.10.7
```

### Hardware
Results were generated on:
- Apple M1/M2 MacBook (ARM64)
- Also tested on Intel Xeon

## Random Seeds

All random operations use fixed seeds for reproducibility:

| Operation | Seed | Location |
|-----------|------|----------|
| Train/test split | 42 | ml_train_balanced.py:144 |
| Data sampling | 42 | analysis_comprehensive.py |
| Benchmark data | 42 | minflux_ml_cli.py |

## Data

### Source
Training data from: [Zenodo DOI: 10.5281/zenodo.10625021](https://doi.org/10.5281/zenodo.10625021)

### Extracted Data (in `data/`)
- `dynamic_data_X.npy`: 584,250 × 12 (photons + positions)
- `dynamic_data_y.npy`: 584,250 (ground truth distances)

### Data Distribution
```
15nm: 24,950 samples (4.3%)
20nm: 419,580 samples (71.8%)
30nm: 139,720 samples (23.9%)
```

## Models

### xgboost_balanced.pkl
Trained with inverse-frequency sample weighting to correct class imbalance.

**Hyperparameters:**
```python
{
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'random_state': 42
}
```

**Results (test set, seed=42):**
| Distance | RMSE (nm) | Bias (nm) |
|----------|-----------|-----------|
| 15nm | 3.60 | +2.29 |
| 20nm | 2.91 | +1.27 |
| 30nm | 3.95 | -2.45 |
| **Overall** | **3.22** | - |

### mapie_balanced.pkl
Conformal prediction model for uncertainty quantification.

**Coverage:** 88.3% (target: 90%)
**Mean interval width:** ~8nm

## Reproduction Steps

### 1. Setup Environment
```bash
git clone https://github.com/lucajbolz/minflux-ml-acceleration.git
cd minflux-ml-acceleration
pip install -r requirements.txt
```

### 2. Download Data
```bash
# From Zenodo
wget https://zenodo.org/record/10625021/files/MINFLUXDynamic.zip
unzip MINFLUXDynamic.zip -d datasets/
```

### 3. Extract Features
```bash
python ml_extract_dynamic.py --data_dir datasets/MINFLUXDynamic/parsed/raw
```

### 4. Train Model
```bash
python ml_train_balanced.py --weight_method inverse --compare
```

Expected output:
```
Overall RMSE: 3.217 nm
15nm: 3.601 nm (56% improvement vs original)
20nm: 2.905 nm
30nm: 3.950 nm (45% improvement vs original)
```

### 5. Calibrate UQ
```bash
python ml_uncertainty_quantification.py --model balanced
```

### 6. Run Analysis
```bash
python analysis_comprehensive.py
```

## Benchmarks

### Inference Speed (Apple M2)
| Metric | Value |
|--------|-------|
| Single prediction | 0.23 ms |
| Batch (1000 samples) | 0.0015 ms/sample |
| Throughput | 4,300 pred/s |
| Speedup vs MLE | 430× |

### MLE Baseline
- Time: ~100ms per measurement
- RMSE: 4.24nm (from original paper)

## Output Files

Running all scripts generates:
```
models/
  xgboost_balanced.pkl    # Trained model
  mapie_balanced.pkl      # UQ calibration

analysis/
  error_analysis.png      # Residual plots
  feature_importance.png  # Feature importance
  uq_calibration.png      # UQ per distance
  robustness.png          # Photon/noise tests
  speedup_accuracy_tradeoff.png

demo_realtime_result.png  # Demo output
```

## Verification

To verify your reproduction matches:
```bash
python -c "
import numpy as np
from ml_inference import MINFLUXDistanceEstimator

est = MINFLUXDistanceEstimator('models/xgboost_balanced.pkl')
X = np.load('data/dynamic_data_X.npy')[:1000]
y = np.load('data/dynamic_data_y.npy')[:1000]

# Feature engineering
photons = X[:, :6]
positions = X[:, 6:]
pred = est.predict_batch(photons, positions)
rmse = np.sqrt(np.mean((pred - y)**2))
print(f'RMSE: {rmse:.3f} nm (expected: ~3.2 nm)')
"
```

## Important Note on Metrics

The RMSE values reported above (3.22nm) are measured on **simulation data** (same distribution as training).

On **experimental data**, the ML model achieves:
- ML RMSE: 5.12nm
- MLE RMSE: 4.24nm (baseline)
- ML is ~21% less accurate than MLE

The primary advantage of ML is the **500× speedup**, not improved accuracy.

## Contact

For issues with reproducibility, open a GitHub issue or contact: bolz@physik.uni-kiel.de
