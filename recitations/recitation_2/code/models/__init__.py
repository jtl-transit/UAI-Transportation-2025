"""
Transportation Choice Models Package
===================================

Unified API for discrete choice models in transportation analysis.
This package provides a common interface for:
- Classical models: Simple Logistic Regression, Multinomial Logit (MNL)
- Neural network models: Fully Connected DNN, Alternative-Specific Utility DNN (ASU-DNN)

All models follow the unified API:
- fit(X, y, obs_ids, alternatives, **kwargs)
- predict_proba(X, obs_ids, alternatives) -> np.array

Compatible with 5-fold cross-validation and easy model comparison.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import pandas as pd

# Import data loading utilities
from .data_loading import (
    load_transportation_data,
    transform_wide_to_long, 
    setup_cross_validation,
    get_fold_data,
    prepare_data_pipeline
)

# Global data variables are accessed via data_loading module when needed
import models.data_loading as data_module


class ChoiceModelProtocol(ABC):
    """
    Abstract base class defining the unified API for all choice models.
    
    All models must implement:
    - fit(X, y, obs_ids, alternatives, **kwargs) 
    - predict_proba(X, obs_ids, alternatives) -> np.array
    - get_model_name() -> str
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, 
            obs_ids: np.ndarray, alternatives: np.ndarray, 
            **kwargs) -> 'ChoiceModelProtocol':
        """
        Fit the choice model to training data.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_observations, n_features)
            Feature matrix where each row is an alternative
        y : np.ndarray, shape (n_observations,)
            Binary choice indicators (1 if chosen, 0 otherwise)
        obs_ids : np.ndarray, shape (n_observations,)
            Choice scenario identifiers
        alternatives : np.ndarray, shape (n_observations,)
            Alternative identifiers (e.g., 1, 2 for binary choice)
        **kwargs : dict
            Model-specific parameters
            
        Returns:
        --------
        self : fitted model instance
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray, 
                     obs_ids: np.ndarray, alternatives: np.ndarray) -> np.ndarray:
        """
        Predict choice probabilities for each alternative.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_observations, n_features)
            Feature matrix where each row is an alternative
        obs_ids : np.ndarray, shape (n_observations,)
            Choice scenario identifiers  
        alternatives : np.ndarray, shape (n_observations,)
            Alternative identifiers
            
        Returns:
        --------
        probabilities : np.ndarray, shape (n_observations,)
            Choice probability for each alternative
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return a descriptive name for this model."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters. Override in subclasses if needed."""
        return {}
    
    def set_params(self, **params) -> 'ChoiceModelProtocol':
        """Set model parameters. Override in subclasses if needed."""
        return self


def get_available_models() -> List[str]:
    """Get list of available model names."""
    available_models = []
    
    # Test SimpleLogistic
    try:
        from .simple_logistic_baseline import SimpleLogisticBaseline
        available_models.append('SimpleLogistic')
    except ImportError as e:
        print(f"Warning: SimpleLogistic not available: {e}")
        
    # Test MultinomialLogit
    try:
        from .multinomial_logit import MultinomialLogit
        available_models.append('MultinomialLogit')
    except ImportError as e:
        print(f"Warning: MultinomialLogit not available: {e}")
        
    # Test FullyConnectedDNN
    try:
        from .fc_dnn import FullyConnectedDNN
        available_models.append('FullyConnectedDNN')
    except ImportError as e:
        print(f"Warning: FullyConnectedDNN not available: {e}")
        
    # Test ASU_DNN
    try:
        from .asu_dnn_general import ASUDNNGeneral
        available_models.append('ASU_DNN')
    except ImportError as e:
        print(f"Warning: ASU_DNN not available: {e}")
        
    return available_models


def create_model(model_name: str, **params):
    """
    Create a model instance with optional parameter overrides.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    **params : dict
        Parameters to override defaults
        
    Returns:
    --------
    Model instance ready for training
    """
    if model_name == 'SimpleLogistic':
        from .simple_logistic_baseline import SimpleLogisticBaseline
        defaults = {'random_state': 42, 'max_iter': 1000}
        return SimpleLogisticBaseline(**{**defaults, **params})
    
    elif model_name == 'MultinomialLogit':
        from .multinomial_logit import MultinomialLogit
        defaults = {'include_constants': True, 'scale': False, 'random_state': 42, 'maxiter': 1000}
        return MultinomialLogit(**{**defaults, **params})
    
    elif model_name == 'FullyConnectedDNN':
        from .fc_dnn import FullyConnectedDNN  
        defaults = {'hidden_units': [64, 32, 16], 'dropout_rate': 0.3, 'learning_rate': 0.001}
        return FullyConnectedDNN(**{**defaults, **params})
    
    elif model_name == 'ASU_DNN':
        from .asu_dnn_general import ASUDNNGeneral
        defaults = {'hidden_units': [64, 32], 'dropout_rate': 0.2, 'learning_rate': 0.001}
        return ASUDNNGeneral(**{**defaults, **params})
    
    else:
        available = get_available_models()
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")


def list_models(detailed: bool = False) -> None:
    """
    Print available models and their descriptions.
    
    Parameters:
    -----------
    detailed : bool
        Whether to show detailed information about each model
    """
    print("Available Transportation Choice Models:")
    print("=" * 50)
    
    models_info = {
        'SimpleLogistic': 'Simple Logistic Baseline (binary choice)',
        'MultinomialLogit': 'Multinomial Logit Model',
        'FullyConnectedDNN': 'Fully Connected Deep Neural Network',
        'ASU_DNN': 'Alternative-Specific Utility DNN'
    }
    
    available = get_available_models()
    for model_name in available:
        description = models_info.get(model_name, 'No description available')
        print(f"- {model_name}: {description}")
        
        if detailed:
            try:
                model = create_model(model_name)
                print(f"   Status: PASS Available")
                print(f"   Class: {model.__class__.__name__}")
            except Exception as e:
                print(f"   Status: FAIL Error: {e}")
            print()


# Import evaluation functions (to be implemented)
def evaluate_all_models(*args, **kwargs):
    """Placeholder for evaluation framework (Phase 5)"""
    raise NotImplementedError("Evaluation framework not yet implemented (Phase 5)")

def evaluate_single_model(*args, **kwargs):
    """Placeholder for evaluation framework (Phase 5)"""
    raise NotImplementedError("Evaluation framework not yet implemented (Phase 5)")


# Package version and info
__version__ = "0.1.0"
__author__ = "Transportation AI Course"
__description__ = "Unified discrete choice modeling framework for transportation analysis"

# Main exports
__all__ = [
    # Core classes
    'ChoiceModelProtocol',
    
    # Model registry functions
    'get_available_models',
    'create_model',
    'list_models',
    
    # Data utilities
    'load_transportation_data',
    'transform_wide_to_long',
    'setup_cross_validation', 
    'get_fold_data',
    'prepare_data_pipeline',
    
    # Evaluation (when available)
    'evaluate_all_models',
    'evaluate_single_model',
]


if __name__ == "__main__":
    # Quick test/demo
    print(f"Transportation Choice Models v{__version__}")
    print(f"{__description__}\n")
    
    list_models(detailed=True)
    
    print("\n" + "="*50) 
    print("Quick API Test:")
    print("="*50)
    
    # Test data loading
    try:
        info = prepare_data_pipeline(verbose=False)
        print(f"PASS Data loading: {info['long_shape']} observations ready")
    except Exception as e:
        print(f"FAIL Data loading failed: {e}")
    
    # Test model creation
    available = get_available_models()
    if available:
        try:
            model = create_model(available[0])
            print(f"PASS Model creation: {model.__class__.__name__}")
        except Exception as e:
            print(f"FAIL Model creation failed: {e}")
    
    print(f"\nPackage ready! Available models: {available}")