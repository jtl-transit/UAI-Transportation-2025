#!/usr/bin/env python3
"""
Test ASU-DNN with diverse synthetic data.
"""

import sys
sys.path.append('.')
import numpy as np

from models.data_loading import DataManager
from models import create_model

def test_diverse_data():
    """Test ASU-DNN with diverse synthetic data."""
    print("Testing ASU-DNN with Diverse Data")
    print("=" * 40)
    
    # Prepare data
    data_manager = DataManager()
    data_manager.prepare_data_pipeline(csv_path="../data/mlogit_Train_wide.csv", verbose=False)
    
    # Get extended data
    X_train_ext, y_train_ext, _, _, _, _, train_obs_ids, _, train_alts, _ = data_manager.get_fold_data(0, return_individuals=True)
    
    # Train ASU-DNN model
    print("Training ASU-DNN model...")
    asu_model = create_model("ASU_DNN")
    asu_model.fit(X_train_ext, y_train_ext, train_obs_ids, train_alts, epochs=30, verbose=0)
    print("ASU-DNN fitted successfully")
    
    # Test 1: Diverse choice set (different features for each alternative)
    print("\nTest 1: Diverse choice set...")
    diverse_X = np.array([
        [2000, 120, 1, 1],  # Alternative 1: cheaper, faster, has change, comfortable
        [3000, 140, 0, 0],  # Alternative 2: more expensive, slower, no change, uncomfortable
    ])
    diverse_obs_ids = np.array(['diverse_1', 'diverse_1'])
    diverse_alts = np.array([1, 2])
    
    diverse_probs = asu_model.predict_proba(diverse_X, diverse_obs_ids, diverse_alts)
    print(f"Diverse probabilities: {diverse_probs}")
    
    # Test 2: Extreme differences
    print("\nTest 2: Extreme differences...")
    extreme_X = np.array([
        [1000, 60, 0, 1],   # Very cheap, very fast, no change, comfortable
        [5000, 200, 1, 0],  # Very expensive, very slow, change, uncomfortable
    ])
    extreme_obs_ids = np.array(['extreme_1', 'extreme_1'])
    extreme_alts = np.array([1, 2])
    
    extreme_probs = asu_model.predict_proba(extreme_X, extreme_obs_ids, extreme_alts)
    print(f"Extreme probabilities: {extreme_probs}")
    
    # Test 3: Real training data sample
    print("\nTest 3: Real training data sample...")
    real_X = X_train_ext[:2]  # First choice set from training data
    real_obs_ids = train_obs_ids[:2]
    real_alts = train_alts[:2]
    
    print(f"Real X:\n{real_X}")
    print(f"Real obs_ids: {real_obs_ids}")
    print(f"Real alts: {real_alts}")
    
    real_probs = asu_model.predict_proba(real_X, real_obs_ids, real_alts)
    print(f"Real probabilities: {real_probs}")
    
    # Test 4: Check if the issue is with identical alternatives
    print("\nTest 4: Identical alternatives (should give 0.5, 0.5)...")
    identical_X = np.array([
        [2400, 130, 1, 1],  # Identical
        [2400, 130, 1, 1],  # Identical
    ])
    identical_obs_ids = np.array(['identical_1', 'identical_1'])
    identical_alts = np.array([1, 2])
    
    identical_probs = asu_model.predict_proba(identical_X, identical_obs_ids, identical_alts)
    print(f"Identical probabilities: {identical_probs}")

if __name__ == "__main__":
    test_diverse_data()
