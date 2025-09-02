#!/usr/bin/env python3
"""
Debug script to understand data format differences
"""

import sys
sys.path.append('.')

# Import model functions
from models import prepare_data_pipeline, get_fold_data

# Prepare data
print("Loading data...")
info = prepare_data_pipeline(csv_path="../data/mlogit_Train_wide.csv", verbose=False)
print(f"Data ready: {info['n_individuals']} individuals, {info['n_scenarios']} scenarios")

# Get data in both formats
X_train, y_train, X_val, y_val = get_fold_data(0)
(X_train_ext, y_train_ext, X_val_ext, y_val_ext,
 train_ids, val_ids, train_obs_ids, val_obs_ids,
 train_alts, val_alts) = get_fold_data(0, return_individuals=True)

print()
print("=== Data Format Comparison ===")
print(f"X_train shape: {X_train.shape}")
print(f"X_train_ext shape: {X_train_ext.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_val_ext shape: {X_val_ext.shape}")
print()
print(f"Sample X_train[0]: {X_train[0]}")
print(f"Sample X_train_ext[0:2]: {X_train_ext[0:2]}")
print()
print(f"val_obs_ids[:10]: {val_obs_ids[:10]}")
print(f"val_alts[:10]: {val_alts[:10]}")
