# Recitation 2: Discrete Choice Modeling Implementation Plan

## Overview
Create a comprehensive, standalone computational notebook for discrete choice modeling using the 1987 Netherlands Train dataset. The notebook will compare classical Multinomial Logit (MNL) models with modern Alternative-Specific Utility Deep Neural Networks (ASU-DNN).

## Dataset Context
- **Source**: 1987 Netherlands cross-sectional choice study
- **Observations**: 2,929 individuals with choice scenarios
- **Country**: Netherlands
- **Format**: Wide format with choice scenarios (choice1 vs choice2)
- **Storage**: Available in `mlogit_choice_data.pickle`

### Variable Definitions:
- **`id`**: Individual identifier
- **`choiceid`**: Choice scenario identifier  
- **`choice`**: Selected alternative (choice1 or choice2)
- **`pricez`**: Price of proposition z (z=1,2) in cents of guilders
- **`timez`**: Travel time of proposition z (z=1,2) in minutes
- **`comfortz`**: Comfort of proposition z (z=1,2), values 0, 1, or 2 in decreasing comfort order
- **`changez`**: Number of changes for proposition z (z=1,2)

## Implementation Structure

### 1. Data Loading and Preprocessing
**Cell Content:**
- Load the Train dataset from `mlogit_choice_data.pickle`
- Explore data structure and basic statistics (2,929 individuals)
- Convert from wide format to long format for choice modeling
- Create utility variables and alternative-specific variables
- Handle missing values and data validation
- Note: comfort variable has 3 levels (0, 1, 2) in decreasing comfort order

**Key Transformations:**
- Wide → Long format conversion
- Create alternative indicators (choice1=0, choice2=1)
- Generate choice indicators (chosen=1, not_chosen=0)
- Feature engineering for utility functions
- Handle ordinal comfort variable appropriately

### 2. Exploratory Data Analysis
**Cell Content:**
- Dataset overview and descriptive statistics
- Choice distribution analysis
- Attribute correlation analysis
- Visualization of choice patterns by attributes
- Value of Time (VOT) preliminary analysis

**Visualizations:**
- Choice distribution by individual characteristics
- Attribute distributions by chosen alternative
- Correlation heatmaps
- Box plots of attributes by choice

### 3. Data Splitting Strategy
**Cell Content:**
- Individual-level stratified splitting (not choice-level)
- 5-fold cross-validation setup
- Train/validation/test splits maintaining individual integrity
- Data standardization for neural networks

**Implementation Details:**
- Use `StratifiedKFold` on individual level
- Ensure same individual doesn't appear in train/test
- Separate scalers for different model types

### 4. Classical Baseline: Multinomial Logit (MNL)
**Cell Content:**
- Implement MNL using `scipy.optimize` or `statsmodels`
- Alternative-specific constants and attributes
- Utility function specification: U = ASC + β₁*price + β₂*time + β₃*change + β₄*comfort
- Parameter estimation via maximum likelihood
- Handle ordinal comfort variable (0, 1, 2 decreasing comfort)

**MNL Implementation:**
```python
class MultinomialLogit:
    def __init__(self):
        self.params = None
        self.convergence_info = None
    
    def fit(self, X, y, alternatives):
        # Maximum likelihood estimation
        # Handle alternative-specific variables
        # Account for ordinal comfort levels
        pass
    
    def predict_proba(self, X, alternatives):
        # Calculate choice probabilities
        pass
    
    def calculate_elasticities(self, X, alternatives):
        # Calculate direct and cross elasticities
        pass
```

### 5. ASU-DNN Implementation
**Cell Content:**
- Theory-informed neural architecture
- Alternative-specific utility heads
- Sparse connectivity based on choice theory
- TensorFlow/PyTorch implementation

**ASU-DNN Architecture:**
```python
class ASU_DNN(tf.keras.Model):
    def __init__(self, n_alternatives, feature_dims, hidden_units):
        super().__init__()
        # Alternative-specific branches
        self.alternative_branches = []
        for i in range(n_alternatives):
            branch = tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_units[0], activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(hidden_units[1], activation='relu'),
                tf.keras.layers.Dense(1)  # Utility output
            ])
            self.alternative_branches.append(branch)
    
    def call(self, inputs):
        # Process each alternative separately
        utilities = []
        for i, branch in enumerate(self.alternative_branches):
            alt_features = inputs[:, i, :]  # Features for alternative i
            utility = branch(alt_features)
            utilities.append(utility)
        
        # Stack and apply softmax
        utilities = tf.stack(utilities, axis=1)
        probabilities = tf.nn.softmax(utilities, axis=1)
        return probabilities
```

### 6. Fully Connected DNN Baseline
**Cell Content:**
- Standard fully connected architecture for comparison
- Same capacity as ASU-DNN but without structural constraints
- Feature flattening approach

### 7. Model Training and Validation
**Cell Content:**
- 5-fold cross-validation implementation
- Training procedures for each model type
- Hyperparameter optimization
- Early stopping and regularization

**Training Loop:**
```python
def cross_validate_models(X, y, cv_folds=5):
    results = {
        'mnl': {'accuracy': [], 'log_likelihood': []},
        'fc_dnn': {'accuracy': [], 'log_likelihood': []},
        'asu_dnn': {'accuracy': [], 'log_likelihood': []}
    }
    
    for fold, (train_idx, val_idx) in enumerate(cv_folds):
        # Train each model
        # Evaluate on validation set
        # Store metrics
        pass
    
    return results
```

### 8. Model Evaluation and Comparison
**Cell Content:**
- Accuracy metrics across folds
- Log-likelihood comparison
- Calibration analysis (reliability diagrams)
- Model interpretability analysis
- Statistical significance testing

**Evaluation Metrics:**
- Classification accuracy
- Log-likelihood
- AIC/BIC for model comparison
- Brier Score for calibration
- Confusion matrices

### 9. Economic Interpretation
**Cell Content:**
- Value of Time (VOT) calculation from MNL
- Elasticity analysis for all models
- Substitution patterns
- Policy simulation capabilities

**Economic Analysis:**
```python
def calculate_vot(mnl_model, price_coef, time_coef):
    """Calculate Value of Time in monetary units per time unit"""
    return -time_coef / price_coef

def simulate_policy_change(model, baseline_data, price_change_pct):
    """Simulate market share changes due to price modifications"""
    pass
```

### 10. Robustness Analysis
**Cell Content:**
- Sensitivity analysis to hyperparameters
- Out-of-sample prediction on held-out test set
- Model stability across CV folds
- Convergence diagnostics

### 11. Visualization and Results
**Cell Content:**
- Performance comparison plots
- Model architecture diagrams
- Utility function visualizations
- Economic interpretation plots

**Key Visualizations:**
- CV performance box plots
- ROC curves for each model
- Utility surface plots
- Elasticity heatmaps
- VOT distributions

### 12. Discussion and Insights
**Cell Content:**
- Trade-offs between model complexity and interpretability
- When to use theory-informed vs. black-box models
- Limitations and future extensions
- Connection to broader transportation modeling

## Technical Requirements

### Dependencies
```python
# Core packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import tensorflow as tf

# Scikit-learn utilities
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# Statistical modeling
import statsmodels.api as sm
from scipy.optimize import minimize
```

### Utility Functions
- Data preprocessing utilities
- Cross-validation helpers
- Plotting functions
- Economic calculation utilities

### File Structure
```
recitation_2/
├── Recitation_2_Code.ipynb       # Main notebook
├── code/
│   ├── util_nn_mlarch.py        # Neural network utilities (existing)
│   ├── baselines.py             # MNL implementation
│   └── plotting_utils.py        # Visualization functions
└── data/
    ├── mlogit_choice_data.pickle # Main dataset (2,929 individuals)
    ├── mlogit_Train_wide.csv    # Alternative CSV format
    └── train_metadata.pdf       # Documentation
```

## Expected Outcomes

### Educational Goals
1. **Understand discrete choice theory** and its connection to neural architectures
2. **Compare model performance** across classical and modern approaches
3. **Interpret economic parameters** from different model types
4. **Appreciate trade-offs** between flexibility and interpretability

### Technical Deliverables
1. **Fully functional notebook** with all implementations
2. **Comprehensive evaluation** across multiple metrics
3. **Economic interpretation** of results
4. **Reproducible results** with proper random seeds

### Key Insights Expected
- ASU-DNN should outperform MNL in prediction accuracy
- ASU-DNN should maintain better interpretability than FC-DNN
- Trade-offs between model complexity and estimation stability
- Economic parameters (VOT, elasticities) should be comparable across theory-based models

## Implementation Timeline

1. **Data Loading & EDA** (2-3 cells)
2. **MNL Implementation** (2-3 cells)  
3. **Neural Network Implementations** (3-4 cells)
4. **Cross-Validation Framework** (2-3 cells)
5. **Evaluation & Visualization** (3-4 cells)
6. **Economic Analysis** (2-3 cells)
7. **Discussion & Insights** (1-2 cells)

**Total: ~15-20 cells for comprehensive coverage**

## Success Criteria

- [ ] All models successfully train and converge
- [ ] Cross-validation results show expected performance hierarchy
- [ ] Economic parameters are reasonable and interpretable
- [ ] Visualizations clearly communicate model differences
- [ ] Notebook runs end-to-end without errors
- [ ] Educational narrative is clear and engaging
