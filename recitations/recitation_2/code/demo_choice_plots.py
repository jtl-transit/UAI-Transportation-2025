#!/usr/bin/env python3
"""
Demonstration of Choice Probability Plotting
============================================

This script demonstrates how to use the new plot_choice_probabilities 
function for both MNL and ASU-DNN models.
"""

import numpy as np
import matplotlib.pyplot as plt
from models.data_loading import DataManager
from models import create_model

def demo_choice_probability_plots():
    """
    Demonstrate choice probability plotting for both MNL and ASU-DNN models.
    """
    print("Choice Probability Plotting Demo")
    print("=" * 50)
    
    # Prepare data
    print("\nPreparing data...")
    data_manager = DataManager()
    data_manager.prepare_data_pipeline(csv_path="../data/mlogit_Train_wide.csv", verbose=False)
    
    # Get training data from first fold
    X_train, y_train, X_val, y_val = data_manager.get_fold_data(0)
    
    # Get extended data for models that need obs_ids and alternatives
    (X_train_ext, y_train_ext, X_val_ext, y_val_ext,
     train_ids, val_ids, train_obs_ids, val_obs_ids,
     train_alts, val_alts) = data_manager.get_fold_data(0, return_individuals=True)
    
    print(f"Training data: {X_train_ext.shape}")
    print(f"Features: price, time, change, comfort")
    
    # Train MNL model
    print("\nTraining MNL model...")
    mnl_model = create_model("MultinomialLogit")
    mnl_model.fit(X_train_ext, y_train_ext, train_obs_ids, train_alts)
    print(f"MNL model converged: {mnl_model.converged}")
    
    # Train ASU-DNN model  
    print("\nTraining ASU-DNN model...")
    asu_model = create_model("ASU_DNN")
    asu_model.fit(X_train_ext, y_train_ext, train_obs_ids, train_alts, epochs=50, verbose=0)
    print("ASU-DNN model trained")
    
    # Define baseline scenario and variable ranges
    baseline_data = {
        'price': 15.0,    # baseline price
        'time': 25.0,     # baseline time  
        'change': 1.0,    # baseline number of changes
        'comfort': 3.0    # baseline comfort level
    }
    
    # Test plotting for different variables
    test_scenarios = [
        {
            'variable': 'price',
            'range': np.linspace(5, 50, 50),
            'title': 'Choice Probabilities vs Price'
        },
        {
            'variable': 'time', 
            'range': np.linspace(10, 60, 50),
            'title': 'Choice Probabilities vs Travel Time'
        },
        {
            'variable': 'comfort',
            'range': np.linspace(1, 5, 50), 
            'title': 'Choice Probabilities vs Comfort Level'
        }
    ]
    
    alternative_names = ['Private Transport', 'Public Transport']
    
    # Generate plots for each scenario
    for scenario in test_scenarios:
        print(f"\nGenerating plots for {scenario['variable']}...")
        
        # MNL plot
        print(f"  MNL model - {scenario['title']}")
        try:
            mnl_probs = mnl_model.plot_choice_probabilities(
                variable_name=scenario['variable'],
                variable_range=scenario['range'],
                baseline_data=baseline_data,
                alternative_names=alternative_names,
                figsize=(10, 6)
            )
            print(f"    Generated MNL plot successfully")
        except Exception as e:
            print(f"    MNL plot failed: {e}")
        
        # ASU-DNN plot
        print(f"  ASU-DNN model - {scenario['title']}")
        try:
            asu_probs = asu_model.plot_choice_probabilities(
                variable_name=scenario['variable'],
                variable_range=scenario['range'],
                baseline_data=baseline_data,
                alternative_names=alternative_names,
                figsize=(10, 6)
            )
            print(f"    Generated ASU-DNN plot successfully")
        except Exception as e:
            print(f"    ASU-DNN plot failed: {e}")
    
    print("\nDemo completed!")
    print("\nUsage example:")
    print("model.plot_choice_probabilities(")
    print("    variable_name='price',")
    print("    variable_range=np.linspace(5, 50, 50),")
    print("    baseline_data={'time': 25, 'change': 1, 'comfort': 3},")
    print("    alternative_names=['PT', 'Car']")
    print(")")

if __name__ == "__main__":
    demo_choice_probability_plots()
