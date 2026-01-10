"""
XGBoost with PHYSICS-INFORMED LOSS (Poisson like MLE!)

Key Innovation: Uses Poisson objective instead of MSE → like paper's MLE
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle
import time

print("="*70)
print("MINFLUX XGBOOST - PHYSICS-INFORMED (Poisson Loss)")
print("="*70)

# 1. Load data (1M for speed)
print(f"\n[1] Loading data...")
X_train = np.load('data/paper_data_with_pos_X.npy')
y_train = np.load('data/paper_data_with_pos_y.npy')

print(f"   Total: X={X_train.shape}, y={y_train.shape}")

# 2. Sample 1M
n_samples = 1_000_000
print(f"\n[2] Sample {n_samples:,} for speed test...")
indices = np.random.RandomState(42).choice(len(X_train), n_samples, replace=False)
X_train = X_train[indices]
y_train = y_train[indices]

print(f"   Subset: X={X_train.shape}")
print(f"   Distance range: [{y_train.min():.1f}, {y_train.max():.1f}] nm")

# 3. Feature Engineering (same as NN)
print(f"\n[3] Engineered Features (paper-compliant)...")

photons = X_train[:, 0::2]
positions = X_train[:, 1::2]
total_photons = photons.sum(axis=1, keepdims=True)

# Features
photon_ratios = photons / (total_photons + 1e-8)
mod_x = photons[:, 0] + photons[:, 2] - 2 * photons[:, 1]
mod_y = photons[:, 3] + photons[:, 5] - 2 * photons[:, 4]
modulation = np.stack([mod_x, mod_y], axis=1) / (total_photons + 1e-8)
log_total = np.log(total_photons + 1)

# Normalize positions
pos_mean = positions.mean(axis=0, keepdims=True)
pos_std = positions.std(axis=0, keepdims=True) + 1e-8
positions_norm = (positions - pos_mean) / pos_std

X_features = np.concatenate([
    photon_ratios,
    positions_norm,
    modulation,
    log_total
], axis=1).astype(np.float32)

print(f"   Features: {X_features.shape[1]}")

# 4. Train/Val Split
print(f"\n[4] Train/Val Split (90/10)...")
n_val = int(len(X_features) * 0.1)
n_train = len(X_features) - n_val

indices = np.random.RandomState(42).permutation(len(X_features))
train_idx = indices[:n_train]
val_idx = indices[n_train:]

X_tr = X_features[train_idx]
y_tr = y_train[train_idx]
X_val = X_features[val_idx]
y_val = y_train[val_idx]

print(f"   Training: {n_train:,}")
print(f"   Validation: {n_val:,}")

# 5. XGBoost training with DIFFERENT objectives
print(f"\n[5] Training XGBoost models...")
print(f"   Comparing: reg:squarederror vs count:poisson")

results = {}

# 5a. Standard: Squared Error (MSE)
print(f"\n   [A] XGBoost with reg:squarederror (Standard MSE)...")
start = time.time()
xgb_mse = xgb.XGBRegressor(
    objective='reg:squarederror',  # Standard MSE
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    tree_method='hist'  # Fast histogram-based
)
xgb_mse.fit(X_tr, y_tr, verbose=False)
elapsed_mse = time.time() - start

y_pred_mse_train = xgb_mse.predict(X_tr)
y_pred_mse_val = xgb_mse.predict(X_val)
rmse_mse_train = np.sqrt(mean_squared_error(y_tr, y_pred_mse_train))
rmse_mse_val = np.sqrt(mean_squared_error(y_val, y_pred_mse_val))

print(f"      Train RMSE: {rmse_mse_train:.2f} nm")
print(f"      Val RMSE:   {rmse_mse_val:.2f} nm")
print(f"      Time: {elapsed_mse:.1f}s")

results['MSE'] = {
    'train_rmse': rmse_mse_train,
    'val_rmse': rmse_mse_val,
    'time': elapsed_mse
}

# 5b. PHYSICS-INFORMED: Poisson (like MLE!)
print(f"\n   [B] XGBoost with count:poisson (PHYSICS-INFORMED!)...")
start = time.time()
xgb_poisson = xgb.XGBRegressor(
    objective='count:poisson',  # POISSON like MLE!!!
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    tree_method='hist'
)
xgb_poisson.fit(X_tr, y_tr, verbose=False)
elapsed_poisson = time.time() - start

y_pred_poisson_train = xgb_poisson.predict(X_tr)
y_pred_poisson_val = xgb_poisson.predict(X_val)
rmse_poisson_train = np.sqrt(mean_squared_error(y_tr, y_pred_poisson_train))
rmse_poisson_val = np.sqrt(mean_squared_error(y_val, y_pred_poisson_val))

print(f"      Train RMSE: {rmse_poisson_train:.2f} nm")
print(f"      Val RMSE:   {rmse_poisson_val:.2f} nm")
print(f"      Time: {elapsed_poisson:.1f}s")

results['Poisson'] = {
    'train_rmse': rmse_poisson_train,
    'val_rmse': rmse_poisson_val,
    'time': elapsed_poisson
}

# 5c. Gamma Distribution (alternative physics-informed)
print(f"\n   [C] XGBoost with reg:gamma (Alternative Physics)...")
start = time.time()
xgb_gamma = xgb.XGBRegressor(
    objective='reg:gamma',  # Gamma distribution
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    tree_method='hist'
)
xgb_gamma.fit(X_tr, y_tr, verbose=False)
elapsed_gamma = time.time() - start

y_pred_gamma_train = xgb_gamma.predict(X_tr)
y_pred_gamma_val = xgb_gamma.predict(X_val)
rmse_gamma_train = np.sqrt(mean_squared_error(y_tr, y_pred_gamma_train))
rmse_gamma_val = np.sqrt(mean_squared_error(y_val, y_pred_gamma_val))

print(f"      Train RMSE: {rmse_gamma_train:.2f} nm")
print(f"      Val RMSE:   {rmse_gamma_val:.2f} nm")
print(f"      Time: {elapsed_gamma:.1f}s")

results['Gamma'] = {
    'train_rmse': rmse_gamma_train,
    'val_rmse': rmse_gamma_val,
    'time': elapsed_gamma
}

# 6. Comparison
print(f"\n{'='*70}")
print("RESULTS - OBJECTIVE COMPARISON")
print(f"{'='*70}")
print(f"\n{'Method':<20} {'Train RMSE':>12} {'Val RMSE':>12} {'Time':>10}")
print(f"{'-'*70}")
for name, res in results.items():
    print(f"{name:<20} {res['train_rmse']:>11.2f}nm {res['val_rmse']:>11.2f}nm {res['time']:>9.1f}s")

# Find best
best_method = min(results.items(), key=lambda x: x[1]['val_rmse'])
print(f"\n{'='*70}")
print(f"✓ BEST MODEL: {best_method[0]}")
print(f"  Val RMSE: {best_method[1]['val_rmse']:.2f} nm")
print(f"{'='*70}")

# 7. Feature Importance (best model)
print(f"\n[6] Feature Importance (Top 5) - {best_method[0]}:")
best_model = xgb_poisson if best_method[0] == 'Poisson' else (xgb_gamma if best_method[0] == 'Gamma' else xgb_mse)

feature_names = [
    'ratio_x-', 'ratio_x0', 'ratio_x+', 'ratio_y-', 'ratio_y0', 'ratio_y+',
    'pos_x-', 'pos_x0', 'pos_x+', 'pos_y-', 'pos_y0', 'pos_y+',
    'mod_x', 'mod_y', 'log_total'
]
importances = best_model.feature_importances_
top_idx = np.argsort(importances)[::-1][:5]
for i, idx in enumerate(top_idx):
    print(f"   {i+1}. {feature_names[idx]:12s}: {importances[idx]:.4f}")

# 8. Save model
print(f"\n[7] Saving best model...")
with open(f'models/xgboost_{best_method[0].lower()}.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"   ✓ models/xgboost_{best_method[0].lower()}.pkl")

print(f"\n{'='*70}")
print(f"✓ XGBOOST TRAINING COMPLETE!")
print(f"  Best Val RMSE: {best_method[1]['val_rmse']:.2f} nm ({best_method[0]})")
print(f"{'='*70}")
