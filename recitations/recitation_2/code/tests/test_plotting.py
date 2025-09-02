#!/usr/bin/env python3
"""
Test individual plotting functionality for each model.
"""

import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from models.data_loading import DataManager
from models import create_model

def test_individual_plots():
    """Test plotting for each model individually."""
    print("Testing Individual Model Plotting")
    print("=" * 40)
    
    # Prepare data
    print("Preparing data...")
    data_manager = DataManager()
    data_manager.prepare_data_pipeline(csv_path="../data/mlogit_Train_wide.csv", verbose=False)
    
    # Get extended data for all models
    X_train_ext, y_train_ext, _, _, _, _, train_obs_ids, _, train_alts, _ = data_manager.get_fold_data(0, return_individuals=True)
    print(f"Training data: {X_train_ext.shape}")
    
    # Define baseline scenario (based on actual data ranges)
    baseline_data = {
        'price': 2400,    # baseline price (cents)
        'time': 130,      # baseline time (minutes)
        'change': 1,      # baseline number of changes
        'comfort': 1      # baseline comfort level (0-1)
    }
    
    # Define price range to test (in cents)
    price_range = np.linspace(1500, 4500, 20)
    alternative_names = ['Public Transport', 'Private Car']
    
    print("\nTesting models individually...")
    
    # Test MNL model
    print("\n1. Testing MNL model...")
    try:
        mnl_model = create_model("MultinomialLogit")
        result = mnl_model.fit(X_train_ext, y_train_ext, train_obs_ids, train_alts)
        if mnl_model.fitted_:
            print("   MNL fitted successfully")
            probs = mnl_model.plot_choice_probabilities(
                variable_name='price',
                variable_range=price_range,
                baseline_data=baseline_data,
                alternative_names=alternative_names,
                figsize=(8, 5)
            )
            plt.savefig('mnl_test_plot.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   MNL plot saved. Probability range: {probs.min():.3f} - {probs.max():.3f}")
        else:
            print("   MNL failed to fit")
    except Exception as e:
        print(f"   MNL error: {e}")
    
    # Test ASU-DNN model
    print("\n2. Testing ASU-DNN model...")
    try:
        asu_model = create_model("ASU_DNN")
        asu_model.fit(X_train_ext, y_train_ext, train_obs_ids, train_alts, epochs=30, verbose=0)
        print("   ASU-DNN fitted successfully")
        probs = asu_model.plot_choice_probabilities(
            variable_name='price',
            variable_range=price_range,
            baseline_data=baseline_data,
            alternative_names=alternative_names,
            figsize=(8, 5)
        )
        plt.savefig('asu_test_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ASU-DNN plot saved. Probability range: {probs.min():.3f} - {probs.max():.3f}")
    except Exception as e:
        print(f"   ASU-DNN error: {e}")
    
    # Test FC-DNN model
    print("\n3. Testing FC-DNN model...")
    try:
        fc_model = create_model("FullyConnectedDNN")
        fc_model.fit(X_train_ext, y_train_ext, train_obs_ids, train_alts, epochs=20, verbose=0)
        print("   FC-DNN fitted successfully")
        probs = fc_model.plot_choice_probabilities(
            variable_name='price',
            variable_range=price_range,
            baseline_data=baseline_data,
            alternative_names=alternative_names,
            figsize=(8, 5)
        )
        plt.savefig('fc_test_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   FC-DNN plot saved. Probability range: {probs.min():.3f} - {probs.max():.3f}")
    except Exception as e:
        print(f"   FC-DNN error: {e}")
    
    print("\nTesting completed!")

if __name__ == "__main__":
    test_individual_plots()
