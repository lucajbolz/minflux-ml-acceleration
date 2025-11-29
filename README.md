# Machine Learning Acceleration for MINFLUX Distance Estimation

This repository provides a **machine learning extension** to the [MINFLUX simulation framework](https://www.nature.com/articles/s41567-024-02760-1) that accelerates distance estimation by up to **512×** while maintaining near-MLE accuracy.

## Quick Overview

The standard MINFLUX method uses Maximum Likelihood Estimation (MLE) which is computationally expensive (~100ms per measurement). This work presents two XGBoost-based regression models that achieve comparable accuracy in ~0.2ms, enabling real-time MINFLUX analysis.

### Key Results

| Method | RMSE (nm) | Inference Time | Speedup | Measurements/sec |
|--------|-----------|----------------|---------|------------------|
| MLE (baseline) | 4.24 | 100ms | 1× | 10 |
| **Dynamic ML** | **5.12** | **0.20ms** | **512×** | **5,000** |
| Static ML | 9.07 | 0.17ms | 576× | 5,880 |

### Quick Start

```python
import numpy as np
from ml_inference import MINFLUXDistanceEstimator

# Load pretrained model
estimator = MINFLUXDistanceEstimator('models/xgboost_dynamic.pkl')

# Example MINFLUX measurement (6 photon counts + 6 beam positions)
photons = np.array([20, 15, 68, 39, 15, 41])
positions = np.array([6.21, 21.21, 36.21, -34.95, -19.95, -4.95])

# Predict distance in nanometers
distance = estimator.predict(photons, positions)
print(f"Estimated distance: {distance:.2f} nm")
```

## Documentation

**Full documentation** including methods, installation, reproduction instructions, and performance analysis:

**→ [README_ML.md](README_ML.md)**

## Original MINFLUX Framework

This project builds upon the MINFLUX simulation framework:

**Publication**: Hensel, T. et al. *Diffraction minima resolve point scattlers at tiny fractions (1/80) of the wavelength*. Nature Physics (2024).
[![DOI](https://img.shields.io/badge/DOI-10.1038/s41567--024--02760--1-blue)](https://www.nature.com/articles/s41567-024-02760-1)

**Data**: MINFLUX simulation datasets available on Zenodo:
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.10625021.svg)](https://doi.org/10.5281/zenodo.10625021)

The original simulation framework (`lib/`, `src/`) is included in this repository for reproducibility and is used to generate training data.

## Repository Structure

```
.
├── README_ML.md                   # Complete documentation
├── requirements.txt               # Python dependencies
├── models/
│   ├── xgboost_mse.pkl           # Static model (3.2MB)
│   └── xgboost_dynamic.pkl       # Dynamic model (6.5MB)
├── ml_inference.py               # Inference wrapper
├── ml_extract_static.py          # Static data extraction
├── ml_extract_dynamic.py         # Dynamic data extraction
├── ml_train_static.py            # Static model training
├── ml_train_dynamic.py           # Dynamic model training
├── lib/                          # Original MINFLUX simulation library
└── src/                          # Original MINFLUX source code
```

## Installation

```bash
# Clone repository
git clone https://github.com/lucajbolz/minflux-ml-acceleration.git
cd minflux-ml-acceleration

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.9+, ~2GB RAM for inference

## Citation

If you use this work, please cite both this ML extension and the original MINFLUX framework:

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

This project uses the same license as the original MINFLUX simulation framework.
