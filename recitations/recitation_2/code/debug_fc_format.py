#!/usr/bin/env python3
"""
Debug script to prepare correct FullyConnectedDNN data format
"""

import sys
sys.path.append('.')
import numpy as np
import pandas as pd

# Import model functions
from models import prepare_data_pipeline, get_fold_data

# Prepare data
print("Loading data...")
info = prepare_data_pipeline(csv_path="../data/mlogit_Train_wide.csv", verbose=False)

# Get extended data format
(X_train_ext, y_train_ext, X_val_ext, y_val_ext,
 train_ids, val_ids, train_obs_ids, val_obs_ids,
 train_alts, val_alts) = get_fold_data(0, return_individuals=True)

print("Original data format:")
print(f"X_train_ext shape: {X_train_ext.shape}")
print(f"Sample data: {X_train_ext[:4]}")
print(f"Obs IDs: {train_obs_ids[:4]}")
print(f"Alternatives: {train_alts[:4]}")

# Create FullyConnectedDNN format (choice sets with concatenated alternatives)
def create_fc_format(X, obs_ids, alts, y):
    """Convert individual alternative format to choice set format for FC-DNN"""
    
    # Create dataframe for easier manipulation
    df = pd.DataFrame({
        'obs_id': obs_ids,
        'alt': alts,
        'choice': y,
        'price': X[:, 0],
        'time': X[:, 1],
        'comfort': X[:, 2],
        'change': X[:, 3]
    })
    
    # Group by observation and create choice sets
    choice_sets = []
    choice_labels = []
    
    for obs_id, group in df.groupby('obs_id'):
        if len(group) == 2:  # Valid choice set
            # Sort by alternative to ensure consistent order
            group = group.sort_values('alt')
            
            # Get features for both alternatives
            alt1_features = group.iloc[0][['price', 'time', 'comfort', 'change']].values
            alt2_features = group.iloc[1][['price', 'time', 'comfort', 'change']].values
            
            # Concatenate features
            choice_set_features = np.concatenate([alt1_features, alt2_features])
            choice_sets.append(choice_set_features)
            
            # Determine which alternative was chosen (1 if alt2, 0 if alt1)
            chosen_alt = group[group['choice'] == 1]['alt'].iloc[0] if group['choice'].sum() == 1 else 1
            choice_label = 1 if chosen_alt == 2 else 0
            choice_labels.append(choice_label)
    
    return np.array(choice_sets), np.array(choice_labels)

# Create FC format
print("\nConverting to FullyConnectedDNN format...")
X_fc_train, y_fc_train = create_fc_format(X_train_ext, train_obs_ids, train_alts, y_train_ext)
X_fc_val, y_fc_val = create_fc_format(X_val_ext, val_obs_ids, val_alts, y_val_ext)

print(f"FC training data shape: {X_fc_train.shape}")
print(f"FC training labels shape: {y_fc_train.shape}")
print(f"Sample FC training data: {X_fc_train[0]}")
print(f"Sample FC training label: {y_fc_train[0]}")

print(f"\nFC validation data shape: {X_fc_val.shape}")
print(f"Choice distribution: {np.bincount(y_fc_train.astype(int))}")
