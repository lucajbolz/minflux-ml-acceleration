# Multiflux Simulation - Setup Complete! 🎉

## Environment Ready

- **Python**: 3.10.19 (Conda environment: `minflux`)
- **JAX**: 0.6.2 (GPU-accelerated computing)
- **NumPy**: 2.2.6
- **All core packages installed successfully!**

## How to Use

### Activate the environment:
```bash
conda activate minflux
```

### Run a test simulation:
```bash
python test_simulation.py
```

### Run a full simulation (from the original code):
```bash
python src/scripts/SimulateData.py
```

## What Was Changed

1. **Config paths** ([lib/config.py](lib/config.py:8-10)):
   - `ROOT_DIR`: Set to project directory
   - `DATA_DIR`: Created at `./data/`
   - `OUTPUT_DIR`: Created at `./output/`

2. **Package versions** (upgraded for Python 3.10 compatibility):
   - JAX: 0.4.7→0.6.2
   - JAXlib: 0.4.7→0.6.2
   - NumPy: 1.23.5→2.2.6
   - SciPy: 1.10.1→1.15.3
   - And other scientific packages

3. **Removed packages** (not critical for simulation):
   - `line-profiler` (Python 3.13 compatibility issue)
   - `PyQt5` (GUI tools - not needed for core simulation)
   - Documentation tools (Sphinx, etc.)

## Next Steps for Your Project

### Phase 1: Reproduce & Verify
1. Run simulations with different molecular separations
2. Compare outputs with paper results
3. Verify the Maximum-Likelihood Estimation works

### Phase 2: ML Model
1. **Data Collection**:
   - Generate training data: different separations (1-20nm)
   - Collect photon counts from simulations
   - Label with ground truth distances

2. **Feature Engineering**:
   - Input features: photon counts at measurement positions
   - Target: molecular distance (d)
   - Consider: phase offsets, iteration data

3. **Model Options**:
   - **Option A**: Simple Neural Network
     - Input: Photon counts (flattened)
     - Output: Distance prediction

   - **Option B**: Transformer/Attention Model
     - Better for sequential measurement data
     - Can learn spatial relationships

   - **Option C**: CNN
     - If you treat photon counts as "images"
     - Good for pattern recognition

4. **Training Pipeline**:
   ```python
   # Pseudo-code
   from lib.simulation.experiment import ExperimentGenerator

   # Generate training data
   for distance in range(1, 20):  # 1-20 nm
       exp = ExperimentGenerator().get_default(
           mol_pos=[[0,0], [distance, 0]],
           repetitions=1000
       )
       exp.perform()
       # Extract photon counts → features
       # Label: distance

   # Train model
   model.fit(features, distances)

   # Compare:
   # - ML prediction time vs. MLE optimization time
   # - Accuracy comparison
   ```

## Useful Files

- [lib/simulation/experiment.py](lib/simulation/experiment.py) - Core simulation
- [lib/simulation/MINFLUXMonteCarlo.py](lib/simulation/MINFLUXMonteCarlo.py) - MLE implementation (what you want to replace)
- [src/scripts/SimulateData.py](src/scripts/SimulateData.py) - Data generation script
- [docs/chapters/examples/](docs/chapters/examples/) - Jupyter notebooks with examples

## Troubleshooting

If you get module errors:
```bash
conda activate minflux
pip install <missing-package>
```

## Questions?

Check the original documentation:
- Paper: https://doi.org/10.1101/2024.01.24.576982
- Data: https://doi.org/10.5281/zenodo.10625021
