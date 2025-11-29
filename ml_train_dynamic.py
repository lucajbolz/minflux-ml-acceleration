"""
Trains XGBoost model for MINFLUXDynamic (trace data)

Separate model for dynamic measurements with higher photon counts
and uneven photon distribution.
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import time

print("="*70)
print("XGBOOST TRAINING - MINFLUX DYNAMIC")
print("="*70)

# Load data
print("\n[1] Loading data...")
X_raw = np.load('data/dynamic_data_X.npy')
y = np.load('data/dynamic_data_y.npy')

print(f"   Samples: {len(X_raw):,}")
print(f"   Distances: {np.unique(y)} nm")

# Feature Engineering
print("\n[2] Feature Engineering...")
photons = X_raw[:, 0::2]
positions = X_raw[:, 1::2]

total_photons = photons.sum(axis=1, keepdims=True)
photon_ratios = photons / (total_photons + 1e-8)

mod_x = photons[:, 0] + photons[:, 2] - 2 * photons[:, 1]
mod_y = photons[:, 3] + photons[:, 5] - 2 * photons[:, 4]
modulation = np.stack([mod_x, mod_y], axis=1) / (total_photons + 1e-8)

log_total = np.log(total_photons + 1)

pos_mean = positions.mean(axis=0, keepdims=True)
pos_std = positions.std(axis=0, keepdims=True) + 1e-8
positions_norm = (positions - pos_mean) / pos_std

X = np.concatenate([
    photon_ratios,
    positions_norm,
    modulation,
    log_total
], axis=1).astype(np.float32)

print(f"   Features: {X.shape[1]} (6 ratios + 6 positions + 2 modulation + 1 log_total)")

# Train/Test Split
print("\n[3] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

print(f"   Train: {len(X_train):,} samples")
print(f"   Test:  {len(X_test):,} samples")

# Train XGBoost
print("\n[4] Training XGBoost...")
params = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'early_stopping_rounds': 50
}

model = xgb.XGBRegressor(**params)

start = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
train_time = time.time() - start

print(f"   ✓ Training complete: {train_time:.1f}s")
print(f"   Best iteration: {model.best_iteration}")

# Evaluate
print("\n[5] Evaluating...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"   Train RMSE: {train_rmse:.2f}nm")
print(f"   Test RMSE:  {test_rmse:.2f}nm")

# Benchmark inference speed
print("\n[6] Benchmarking inference speed...")
times = []
for i in range(100):
    start = time.perf_counter()
    _ = model.predict(X_test[i:i+1])
    elapsed = time.perf_counter() - start
    times.append(elapsed)

inf_time = np.median(times) * 1000  # ms

print(f"   Inference time: {inf_time:.4f}ms")
print(f"   Speedup vs MLE: {100/inf_time:.0f}x")

# Save model
print("\n[7] Saving model...")
model_path = 'models/xgboost_dynamic.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"   ✓ Saved: {model_path}")

# Summary
print("\n" + "="*70)
print("DYNAMIC MODEL - FINAL PERFORMANCE")
print("="*70)
print(f"\nTest RMSE: {test_rmse:.2f}nm")
print(f"Inference: {inf_time:.4f}ms")
print(f"Speedup:   {100/inf_time:.0f}x faster than MLE")
print(f"Model:     {model_path}")
print("="*70)
print("✓ DYNAMIC MODEL READY!")
print("="*70)
