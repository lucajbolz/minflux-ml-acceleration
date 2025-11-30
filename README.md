# Machine Learning Acceleration for MINFLUX Distance Estimation

This repository provides a **machine learning extension** to the [MINFLUX simulation framework](https://www.nature.com/articles/s41567-024-02760-1) that accelerates distance estimation by **500×** while maintaining near-MLE accuracy.

## Quick Overview

The standard MINFLUX method uses Maximum Likelihood Estimation (MLE) which is computationally expensive (~100ms per measurement). This work presents XGBoost-based regression models that achieve comparable accuracy in ~0.2ms, enabling real-time MINFLUX analysis.

### Key Results

| Method | RMSE | Data | Inference Time | Speedup |
|--------|------|------|----------------|---------|
| MLE (baseline) | 4.24nm | Experimental | 100ms | 1× |
| **ML (Balanced)** | **3.22nm** | Simulation | **0.2ms** | **500×** |

> **Note**: ML achieves 3.22nm RMSE on simulation data. On experimental data, ML (5.12nm) is ~21% less accurate than MLE (4.24nm). The main advantage is the **500× speedup**.

### Quick Start

```python
import numpy as np
from ml_inference import MINFLUXDistanceEstimator

# Load pretrained model
estimator = MINFLUXDistanceEstimator('models/xgboost_balanced.pkl')

# Example MINFLUX measurement (6 photon counts + 6 beam positions)
photons = np.array([35, 42, 28, 38, 45, 30])
positions = np.array([-10, 2, -5, -12, 6, -20])

# Predict distance in nanometers
distance = estimator.predict(photons, positions)
print(f"Estimated distance: {distance:.2f} nm")

# With uncertainty quantification (90% confidence intervals)
estimator_uq = MINFLUXDistanceEstimator('models/xgboost_balanced.pkl',
                                         use_uncertainty=True)
distance, lower, upper = estimator_uq.predict(photons, positions)
print(f"Distance: {distance:.2f} nm [90% CI: {lower:.2f}-{upper:.2f}]")
```

## Documentation

| Document | Description |
|----------|-------------|
| **[HANDBOOK.md](HANDBOOK.md)** | Complete documentation (recommended) |
| [README_ML.md](README_ML.md) | Technical details & methods |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | How to reproduce results |
| [example_notebook.ipynb](example_notebook.ipynb) | Jupyter tutorial |

## Repository Structure

```
├── ml_inference.py                  # Main API
├── minflux_ml_cli.py               # CLI tool
├── models/
│   ├── xgboost_balanced.pkl        # Recommended model
│   ├── mapie_balanced.pkl          # UQ calibration
│   └── ...
├── analysis/                        # Generated plots
└── HANDBOOK.md                      # Full documentation
```

## Installation

```bash
git clone https://github.com/lucajbolz/minflux-ml-acceleration.git
cd minflux-ml-acceleration
pip install -r requirements.txt
```

**Requirements**: Python 3.9+, ~2GB RAM

## Original MINFLUX Framework

**Publication**: Hensel, T. et al. *Diffraction minima resolve point scatterers at tiny fractions (1/80) of the wavelength*. Nature Physics (2024).
[![DOI](https://img.shields.io/badge/DOI-10.1038/s41567--024--02760--1-blue)](https://www.nature.com/articles/s41567-024-02760-1)

**Data**: [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.10625021.svg)](https://doi.org/10.5281/zenodo.10625021)

## Citation

```bibtex
@software{bolz2024minflux_ml,
  title={Machine Learning Acceleration for MINFLUX Distance Estimation},
  author={Bolz, Luca J.},
  year={2024},
  url={https://github.com/lucajbolz/minflux-ml-acceleration}
}
```

## License

This project uses the same license as the original MINFLUX simulation framework.
