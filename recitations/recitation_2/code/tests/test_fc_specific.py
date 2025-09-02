#!/usr/bin/env python3
"""
Test FC-DNN plotting specifically to verify it's working.
"""

import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from models.data_loading import DataManager
from models import create_model

def test_fc_dnn_plotting():
    """Test FC-DNN plotting specifically."""
    print("Testing FC-DNN Plotting")
    print("=" * 30)
    
    # Prepare data
    data_manager = DataManager()
    data_manager.prepare_data_pipeline(csv_path="../data/mlogit_Train_wide.csv", verbose=False)
    
    # Get extended data
    X_train_ext, y_train_ext, _, _, _, _, train_obs_ids, _, train_alts, _ = data_manager.get_fold_data(0, return_individuals=True)
    
    print(f"Training data: {X_train_ext.shape}")
    
    # Train FC-DNN model
    print("Training FC-DNN model...")
    fc_model = create_model("FullyConnectedDNN")
    fc_model.fit(X_train_ext, y_train_ext, train_obs_ids, train_alts, epochs=20, verbose=0)
    print("FC-DNN fitted successfully")
    
    # Define baseline scenario
    baseline_data = {
        'price': 2400,
        'time': 130,
        'change': 1,
        'comfort': 1
    }
    
    # Test plotting
    print("Testing plotting...")
    price_range = np.linspace(1500, 4500, 20)
    alternative_names = ['Public Transport', 'Private Car']
    
    try:
        probs = fc_model.plot_choice_probabilities(
            variable_name='price',
            variable_range=price_range,
            baseline_data=baseline_data,
            alternative_names=alternative_names,
            figsize=(8, 5)
        )
        plt.savefig('fc_dnn_specific_test.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ FC-DNN plotting successful!")
        print(f"   Probability matrix shape: {probs.shape}")
        print(f"   Probability range: {probs.min():.3f} - {probs.max():.3f}")
        print(f"   Alt 1 (Public Transport) range: {probs[:, 0].min():.3f} - {probs[:, 0].max():.3f}")
        print(f"   Alt 2 (Private Car) range: {probs[:, 1].min():.3f} - {probs[:, 1].max():.3f}")
        
        # Check that probabilities sum to 1
        prob_sums = probs.sum(axis=1)
        print(f"   Probability sums: {prob_sums.min():.3f} - {prob_sums.max():.3f} (should be ~1.0)")
        
        # Show how probabilities change
        print(f"   At lowest price ({price_range[0]:.0f}): P(Public)={probs[0,0]:.3f}, P(Car)={probs[0,1]:.3f}")
        print(f"   At highest price ({price_range[-1]:.0f}): P(Public)={probs[-1,0]:.3f}, P(Car)={probs[-1,1]:.3f}")
        
    except Exception as e:
        print(f"❌ FC-DNN plotting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fc_dnn_plotting()
