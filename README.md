# ML for MINFLUX Distance Estimation

XGBoost model for fast MINFLUX distance estimation with sub-millisecond inference.

## Results

| Metric | Value |
|--------|-------|
| RMSE | 3.2 nm |
| Inference time | 0.2 ms |
| Speedup vs MLE | 5-50× (MLE: 1-10 ms, Balzarotti et al. 2017) |

**Note:** ML was trained on simulated data (15/20/30 nm) from Hensel et al. Nature Physics 2024. MLE timing is based on literature estimates. Main contribution is the fast, predictable inference time.

## Quick Start

```python
from ml_inference import MINFLUXDistanceEstimator
import numpy as np

# Load model
estimator = MINFLUXDistanceEstimator('models/xgboost_balanced.pkl')

# Predict distance
photons = np.array([35, 42, 28, 38, 45, 30])
positions = np.array([-10, 2, -5, -12, 6, -20])
distance = estimator.predict(photons, positions)
print(f"Distance: {distance:.2f} nm")
```

## Project Structure

```
├── ml_inference.py          # Main inference module (use this!)
├── quick_demo.py            # Quick demo script
├── requirements.txt         # Dependencies
│
├── models/                  # Trained models
│   ├── xgboost_balanced.pkl # Main model (recommended)
│   └── mapie_balanced.pkl   # Uncertainty quantification
│
├── scripts/                 # Training & analysis scripts
│   ├── ml_extract_dynamic.py
│   ├── ml_train_balanced.py
│   └── analysis_comprehensive.py
│
├── presentation/            # Bachelor thesis presentation
│   ├── main-2.tex          # Beamer slides
│   ├── handout.tex         # Detailed handout
│   └── karteikarten.tex    # Flashcards for Q&A
│
├── lib/                     # Original paper code (Hensel et al.)
└── src/                     # Original paper figures
```

## Installation

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test
python ml_inference.py
```

## Full Training Pipeline

```bash
# 1. Download data from Zenodo
# https://zenodo.org/record/10625021

# 2. Extract features
python scripts/ml_extract_dynamic.py --data_dir datasets/MINFLUXDynamic/parsed/raw

# 3. Train balanced model
python scripts/ml_train_balanced.py

# 4. Analysis
python scripts/analysis_comprehensive.py
```

## Features (15 total)

| Feature | Count | Description |
|---------|-------|-------------|
| Photon Ratios | 6 | Normalized photon counts per beam position |
| Beam Positions | 6 | Normalized spatial coordinates |
| Modulation Depth | 2 | Contrast between center and side beams |
| Log Total Photons | 1 | SNR proxy |

## Limitations

- Trained on simulated data only (15-30 nm range)
- Systematic bias at boundaries (+0.8 nm at 15 nm, -2.6 nm at 30 nm)
- Not validated on different photon budgets
- Experimental validation pending

## References

- **Original Paper:** Hensel et al., *Nature Physics* (2024). [DOI: 10.1038/s41567-024-02760-1](https://www.nature.com/articles/s41567-024-02760-1)
- **MLE Timing:** Balzarotti et al., *Science* (2017). [DOI: 10.1126/science.aak9913](https://science.sciencemag.org/content/355/6325/606)
- **Data:** [Zenodo](https://doi.org/10.5281/zenodo.10625021)

## License

MIT License - see [LICENSE](LICENSE)
