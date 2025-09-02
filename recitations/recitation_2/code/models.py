# ===============================================================================
# DISCRETE CHOICE MODELING FOR TRANSPORTATION - MODELS.PY
# ===============================================================================
# 
# This module implements various discrete choice models for transportation analysis:
# - Classical models: Simple Logistic Regression, Multinomial Logit (MNL)
# - Neural network models: Fully Connected DNN, Alternative-Specific Utility DNN (ASU-DNN)
# 
# All models follow a unified API for easy comparison and cross-validation.
# Compatible with Google Colab, TensorFlow/Keras, and accessible for undergraduate students.
#
# REFACTORING PLAN - STEP-BY-STEP IMPLEMENTATION:
# ===============================================================================
#
# PHASE 1: CODE ORGANIZATION & CLEANUP
# -------------------------------------
# ✓ Step 1.1: Organize imports by category (standard, scientific, ML, viz)
# ✓ Step 1.2: Remove unused imports and variables
# ✓ Step 1.3: Add comprehensive docstrings to all classes and methods
# ✓ Step 1.4: Implement consistent error handling and validation
# ✓ Step 1.5: Add type hints for better code clarity
#
# PHASE 2: UNIFIED MODEL API DESIGN
# ----------------------------------
# ✓ Step 2.1: Define abstract base class or protocol for all models
# ✓ Step 2.2: Standardize method signatures: fit(X, y, obs_ids, alternatives, **kwargs)
# ✓ Step 2.3: Standardize output: predict_proba(X, obs_ids, alternatives) -> np.array
# ✓ Step 2.4: Add model summary/interpretation methods for each model type
# ✓ Step 2.5: Implement consistent random state handling
#
# PHASE 3: CLASSICAL MODELS IMPLEMENTATION
# -----------------------------------------
# ✓ Step 3.1: SimpleLogisticBaseline - Clean up and complete missing methods
# ✓ Step 3.2: MultinomialLogit - Fix incomplete utility computation and parameter splitting
# ✓ Step 3.3: Add elasticity computation and Value of Time (VOT) calculation
# ✓ Step 3.4: Implement proper convergence checking and model diagnostics
# ✓ Step 3.5: Add coefficient interpretation methods
#
# PHASE 4: NEURAL NETWORK MODELS
# -------------------------------
# ✓ Step 4.1: FullyConnectedDNN - Complete missing data preparation methods
# ✓ Step 4.2: ASUDNNGeneral - Fix incomplete utility computation and featurization
# ✓ Step 4.3: Implement proper data preprocessing and feature engineering
# ✓ Step 4.4: Add training monitoring and early stopping
# ✓ Step 4.5: Implement model interpretation methods (attention weights, utility curves)
#
# PHASE 5: CROSS-VALIDATION FRAMEWORK
# ------------------------------------
# ✓ Step 5.1: Create unified cross-validation function for all models
# ✓ Step 5.2: Implement person-level splitting (avoiding data leakage)
# ✓ Step 5.3: Add comprehensive evaluation metrics (accuracy, log-loss, AUC, calibration)
# ✓ Step 5.4: Create results comparison and visualization functions
# ✓ Step 5.5: Add statistical significance testing for model comparisons
#
# PHASE 6: UTILITY FUNCTIONS & HELPERS
# -------------------------------------
# ✓ Step 6.1: Data preparation utilities (wide-to-long conversion, validation)
# ✓ Step 6.2: Visualization functions (training curves, choice probabilities, elasticities)
# ✓ Step 6.3: Model diagnostics and interpretation tools
# ✓ Step 6.4: Export/import functions for trained models
# ✓ Step 6.5: Documentation and example usage
#
# IMPLEMENTATION PRIORITIES:
# - UG student accessibility (clear comments, intuitive variable names)
# - Google Colab compatibility (minimal dependencies, clear error messages)
# - Transportation domain knowledge (VOT, elasticities, choice modeling best practices)
# - Reproducibility (fixed random seeds, deterministic training)
# - Computational efficiency (vectorized operations, proper batching)
#
# ===============================================================================



# Imports

import os
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize
from scipy.special import logsumexp

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis






