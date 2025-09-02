#!/usr/bin/env python3
"""
Simple Choice Probability Plotting Demo
======================================

This script demonstrates the new choice probability plotting functionality
for MNL, ASU-DNN, and FC-DNN models.
"""

import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Use interactive backend if available, otherwise fall back to Agg
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')

from models.data_loading import DataManager
from models import create_model

def plot_comparison_demo():
    """
    Create plots comparing choice probabilities across different models.
    """
    print("Choice Probability Plotting Demo")
    print("=" * 50)
    
    # Prepare data
    print("Preparing data...")
    data_manager = DataManager()
    data_manager.prepare_data_pipeline(csv_path="../data/mlogit_Train_wide.csv", verbose=False)
    
    # Get extended data for all models
    X_train_ext, y_train_ext, _, _, _, _, train_obs_ids, _, train_alts, _ = data_manager.get_fold_data(0, return_individuals=True)
    print(f"Training data: {X_train_ext.shape}")
    
    # Define baseline scenario (based on actual data ranges)
    baseline_data = {
        'price': 2400,    # baseline price (cents, from data inspection)
        'time': 130,      # baseline time (minutes)
        'change': 1,      # baseline number of changes
        'comfort': 1      # baseline comfort level (0-1 from data)
    }
    
    # Define price range to test (in cents to match data scale)
    price_range = np.linspace(1500, 4500, 30)
    alternative_names = ['Public Transport', 'Private Car']
    
    print("\nTraining models...")
    
    # Train and test MNL model
    print("  Training MNL model...")
    try:
        mnl_model = create_model("MultinomialLogit")
        result = mnl_model.fit(X_train_ext, y_train_ext, train_obs_ids, train_alts)
        mnl_success = hasattr(mnl_model, 'fitted_') and mnl_model.fitted_
        print(f"    MNL converged: {mnl_success}")
    except Exception as e:
        print(f"    MNL failed: {e}")
        mnl_model = None
        mnl_success = False
    
    # Train and test ASU-DNN model
    print("  Training ASU-DNN model...")
    try:
        asu_model = create_model("ASU_DNN")
        asu_model.fit(X_train_ext, y_train_ext, train_obs_ids, train_alts, epochs=30, verbose=0)
        asu_success = True
        print("    ASU-DNN trained successfully")
    except Exception as e:
        print(f"    ASU-DNN failed: {e}")
        asu_model = None
        asu_success = False
    
    # Train and test FC-DNN model
    print("  Training FC-DNN model...")
    try:
        fc_model = create_model("FullyConnectedDNN")
        fc_model.fit(X_train_ext, y_train_ext, train_obs_ids, train_alts, epochs=20, verbose=0)
        fc_success = True
        print("    FC-DNN trained successfully")
    except Exception as e:
        print(f"    FC-DNN failed: {e}")
        fc_model = None
        fc_success = False
    
    # Generate plots
    print("\nGenerating choice probability plots...")
    
    # Create a comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Choice Probabilities vs Price - Model Comparison', fontsize=16)
    
    colors = ['#FF6B6B', '#4ECDC4']
    
    # MNL Plot
    if mnl_success and mnl_model:
        try:
            # Get probabilities without plotting
            matplotlib.use('Agg')  # Temporarily use non-interactive backend
            mnl_probs = mnl_model.plot_choice_probabilities(
                variable_name='price',
                variable_range=price_range,
                baseline_data=baseline_data,
                alternative_names=alternative_names,
                figsize=(6, 4)
            )
            plt.close()  # Close the individual plot
            
            # Plot on subplot
            axes[0].plot(price_range, mnl_probs[:, 0], label=alternative_names[0], color=colors[0], linewidth=2)
            axes[0].plot(price_range, mnl_probs[:, 1], label=alternative_names[1], color=colors[1], linewidth=2)
            axes[0].set_title('MNL Model')
            axes[0].set_xlabel('Price (Cents Dutch Guilder)')
            axes[0].set_ylabel('Choice Probability')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 1)
            
        except Exception as e:
            axes[0].text(0.5, 0.5, f'MNL Plot Failed:\\n{str(e)[:50]}...', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('MNL Model (Failed)')
    else:
        axes[0].text(0.5, 0.5, 'MNL Model\\nNot Available', 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('MNL Model (Failed)')
    
    # ASU-DNN Plot
    if asu_success and asu_model:
        try:
            asu_probs = asu_model.plot_choice_probabilities(
                variable_name='price',
                variable_range=price_range,
                baseline_data=baseline_data,
                alternative_names=alternative_names,
                figsize=(6, 4)
            )
            plt.close()  # Close the individual plot
            
            axes[1].plot(price_range, asu_probs[:, 0], label=alternative_names[0], color=colors[0], linewidth=2)
            axes[1].plot(price_range, asu_probs[:, 1], label=alternative_names[1], color=colors[1], linewidth=2)
            axes[1].set_title('ASU-DNN Model')
            axes[1].set_xlabel('Price (Cents Dutch Guilder)')
            axes[1].set_ylabel('Choice Probability')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1)
            
        except Exception as e:
            axes[1].text(0.5, 0.5, f'ASU-DNN Plot Failed:\\n{str(e)[:50]}...', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('ASU-DNN Model (Failed)')
    else:
        axes[1].text(0.5, 0.5, 'ASU-DNN Model\\nNot Available', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('ASU-DNN Model (Failed)')
    
    # FC-DNN Plot
    if fc_success and fc_model:
        try:
            fc_probs = fc_model.plot_choice_probabilities(
                variable_name='price',
                variable_range=price_range,
                baseline_data=baseline_data,
                alternative_names=alternative_names,
                figsize=(6, 4)
            )
            plt.close()  # Close the individual plot
            
            axes[2].plot(price_range, fc_probs[:, 0], label=alternative_names[0], color=colors[0], linewidth=2)
            axes[2].plot(price_range, fc_probs[:, 1], label=alternative_names[1], color=colors[1], linewidth=2)
            axes[2].set_title('FC-DNN Model')
            axes[2].set_xlabel('Price (Cents Dutch Guilder)')
            axes[2].set_ylabel('Choice Probability')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim(0, 1)
            
        except Exception as e:
            axes[2].text(0.5, 0.5, f'FC-DNN Plot Failed:\\n{str(e)[:50]}...', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('FC-DNN Model (Failed)')
    else:
        axes[2].text(0.5, 0.5, 'FC-DNN Model\\nNot Available', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('FC-DNN Model (Failed)')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('choice_probability_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison plot saved as 'choice_probability_comparison.png'")
    
    # Try to show the plot if possible
    try:
        matplotlib.use('TkAgg')
        plt.show()
    except:
        print("Cannot display plot interactively, but saved to file.")
    
    print("\nDemo completed!")
    print("\nTo use the plotting functionality in your own code:")
    print("  model.plot_choice_probabilities(")
    print("      variable_name='price',")
    print("      variable_range=np.linspace(5, 50, 30),")
    print("      baseline_data={'time': 25, 'change': 1, 'comfort': 3},")
    print("      alternative_names=['Public Transport', 'Car']")
    print("  )")

if __name__ == "__main__":
    plot_comparison_demo()
