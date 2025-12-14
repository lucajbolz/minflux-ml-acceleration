# ML for MINFLUX Distance Estimation

XGBoost model that makes MINFLUX distance estimation **500× faster** than MLE.

## Results

| Metric | Value |
|--------|-------|
| RMSE | 3.22 nm |
| Inference time | 0.2 ms |
| Speedup vs MLE | ~500× |

**Note:** ML was trained on simulated data (15/20/30nm). MLE results (4.24nm) from the paper are from different measurements. Main contribution is the speedup.

## Usage

```python
import numpy as np
from ml_inference import MINFLUXDistanceEstimator

estimator = MINFLUXDistanceEstimator('models/xgboost_balanced.pkl')

photons = np.array([35, 42, 28, 38, 45, 30])
positions = np.array([-10, 2, -5, -12, 6, -20])

distance = estimator.predict(photons, positions)
print(f"Distance: {distance:.2f} nm")
```

## Installation

```bash
pip install -r requirements.txt
```

## Reproduction

```bash
# 1. Download data (Zenodo)
wget https://zenodo.org/record/10625021/files/MINFLUXDynamic.zip
unzip MINFLUXDynamic.zip -d datasets/

# 2. Extract features
python ml_extract_dynamic.py --data_dir datasets/MINFLUXDynamic/parsed/raw

# 3. Train model
python ml_train_balanced.py

# 4. Analysis
python analysis_comprehensive.py
```

## Limitations

- Trained on simulated data (15-30nm range)
- Systematic bias at 15nm (+2.3nm) and 30nm (-2.5nm)
- Not validated for different photon counts

## References

**Original Paper:** Hensel et al., *Nature Physics* (2024). [DOI: 10.1038/s41567-024-02760-1](https://www.nature.com/articles/s41567-024-02760-1)

**Data:** [Zenodo](https://doi.org/10.5281/zenodo.10625021)
