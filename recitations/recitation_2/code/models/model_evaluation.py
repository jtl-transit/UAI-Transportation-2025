
"""
Model Evaluation Framework for Transportation Choice Models
==========================================================

This module implements unified cross-validation evaluation for all choice models.
Supports person-level splitting, comprehensive metrics, and model comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from dataclasses import dataclass
import warnings

# Import our model infrastructure
from .data_loading import DataManager, get_fold_data, prepare_data_pipeline


@dataclass
class ModelResult:
    """Results for a single model on a single fold."""
    model_name: str
    fold: int
    n_train_individuals: int
    n_val_individuals: int
    n_train_obs: int
    n_val_obs: int
    accuracy: float
    log_loss: float
    auc: Optional[float]
    training_time: float
    prediction_time: float
    additional_metrics: Dict[str, Any] = None


def evaluate_single_model(model_creator, model_name: str, 
                         data_path: str = None,
                         n_folds: int = 5,
                         model_params: Dict[str, Any] = None,
                         verbose: bool = True) -> List[ModelResult]:
    """
    Evaluate a single model using k-fold cross-validation.
    
    Parameters:
    -----------
    model_creator : callable
        Function that creates a model instance (e.g., lambda: create_model('SimpleLogistic'))
    model_name : str
        Name of the model for reporting
    data_path : str, optional
        Path to data file. If None, uses default.
    n_folds : int
        Number of CV folds
    model_params : dict, optional
        Parameters to pass to model creator
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    list : List of ModelResult objects, one per fold
    """
    import time
    
    if verbose:
        print(f"\nRunning {model_name} evaluation with {n_folds}-fold CV...")
    
    # Create data manager and prepare data
    data_manager = DataManager()
    
    if verbose:
        print("  Preparing data...")
    data_manager.prepare_data_pipeline(csv_path=data_path, n_folds=n_folds, verbose=False)
    
    results = []
    
    for fold in range(n_folds):
        if verbose:
            print(f"  Fold {fold + 1}/{n_folds}...")
        
        try:
            # Get fold data
            (X_train, y_train, X_val, y_val,
             train_ids, val_ids,
             train_obs_ids, val_obs_ids,
             train_alts, val_alts) = data_manager.get_fold_data(fold, return_individuals=True)
            
            # Create model instance
            if model_params:
                model = model_creator(**model_params)
            else:
                model = model_creator()
            
            # Train model
            start_time = time.time()
            
            # Use appropriate fit signature based on model
            if hasattr(model, 'fit'):
                # Try standard signature first
                try:
                    model.fit(X_train, y_train, train_obs_ids, train_alts)
                except TypeError:
                    # Fallback for models with different signatures
                    model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            
            try:
                # Try standard prediction signature
                y_pred_proba = model.predict_proba(X_val, val_obs_ids, val_alts)
            except TypeError:
                # Fallback for models with different signatures
                y_pred_proba = model.predict_proba(X_val)
            
            prediction_time = time.time() - start_time
            
            # Handle different prediction formats
            if len(y_pred_proba.shape) == 2:
                # Multi-class probabilities - take class 1 probability
                y_pred_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba[:, 0]
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, (y_pred_proba >= 0.5).astype(int))
            
            # Log loss with clipping to avoid numerical issues
            y_pred_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            logloss = log_loss(y_val, y_pred_clipped)
            
            # AUC (only if we have both classes)
            try:
                if len(np.unique(y_val)) > 1:
                    auc = roc_auc_score(y_val, y_pred_proba)
                else:
                    auc = None
            except ValueError:
                auc = None
            
            # Create result
            result = ModelResult(
                model_name=model_name,
                fold=fold + 1,
                n_train_individuals=len(train_ids),
                n_val_individuals=len(val_ids),
                n_train_obs=len(X_train),
                n_val_obs=len(X_val),
                accuracy=accuracy,
                log_loss=logloss,
                auc=auc,
                training_time=training_time,
                prediction_time=prediction_time
            )
            
            results.append(result)
            
            if verbose:
                auc_str = f"{auc:.3f}" if auc is not None else "N/A"
                print(f"    Success: Acc: {accuracy:.3f}, LogLoss: {logloss:.3f}, AUC: {auc_str}")
        
        except Exception as e:
            if verbose:
                print(f"    Failed: Fold {fold + 1} failed: {e}")
            warnings.warn(f"Fold {fold + 1} failed for {model_name}: {e}")
    
    return results


def evaluate_all_models(model_creators: Dict[str, callable],
                       data_path: str = None,
                       n_folds: int = 5,
                       verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate multiple models using k-fold cross-validation.
    
    Parameters:
    -----------
    model_creators : dict
        Dictionary mapping model names to creator functions
    data_path : str, optional
        Path to data file
    n_folds : int
        Number of CV folds
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    tuple : (results_df, summary_df)
        - results_df: Detailed results for each model and fold
        - summary_df: Aggregated results with mean ± std
    """
    if verbose:
        print("Starting Multi-Model Evaluation")
        print("=" * 50)
    
    # Prepare data once
    if verbose:
        print("Preparing data...")
    data_manager = DataManager()
    data_manager.prepare_data_pipeline(csv_path=data_path, n_folds=n_folds, verbose=verbose)
    
    all_results = []
    
    for model_name, model_creator in model_creators.items():
        try:
            model_results = evaluate_single_model(
                model_creator=model_creator,
                model_name=model_name,
                data_path=data_path,
                n_folds=n_folds,
                verbose=verbose
            )
            all_results.extend(model_results)
            
        except Exception as e:
            if verbose:
                print(f"Failed: {model_name} evaluation failed: {e}")
            warnings.warn(f"Model {model_name} evaluation failed: {e}")
    
    # Convert to DataFrame
    if all_results:
        results_df = pd.DataFrame([
            {
                'model': r.model_name,
                'fold': r.fold,
                'n_train_individuals': r.n_train_individuals,
                'n_val_individuals': r.n_val_individuals,
                'n_train_obs': r.n_train_obs,
                'n_val_obs': r.n_val_obs,
                'accuracy': r.accuracy,
                'log_loss': r.log_loss,
                'auc': r.auc,
                'training_time': r.training_time,
                'prediction_time': r.prediction_time
            }
            for r in all_results
        ])
        
        # Create summary
        summary_stats = results_df.groupby('model').agg({
            'accuracy': ['mean', 'std', 'count'],
            'log_loss': ['mean', 'std'],
            'auc': ['mean', 'std'],
            'training_time': ['mean', 'std'],
            'prediction_time': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = [f'{col[0]}_{col[1]}' for col in summary_stats.columns]
        summary_df = summary_stats.reset_index()
        
        if verbose:
            print("\nEvaluation Complete!")
            print("=" * 50)
            print("Summary Results:")
            print(summary_df)
        
        return results_df, summary_df
    
    else:
        if verbose:
            print("No successful model evaluations")
        return pd.DataFrame(), pd.DataFrame()


def quick_model_comparison(models_dict: Dict[str, callable] = None,
                          data_path: str = None,
                          n_folds: int = 5) -> None:
    """
    Quick comparison of available models with nice formatting.
    
    Parameters:
    -----------
    models_dict : dict, optional
        Model creators. If None, uses default available models.
    data_path : str, optional
        Data path
    n_folds : int
        Number of folds
    """
    if models_dict is None:
        # Import here to avoid circular imports
        from . import create_model
        
        models_dict = {
            'SimpleLogistic': lambda: create_model('SimpleLogistic'),
            'MultinomialLogit': lambda: create_model('MultinomialLogit'),
            'FullyConnectedDNN': lambda: create_model('FullyConnectedDNN'),
            'ASU_DNN': lambda: create_model('ASU_DNN')
        }
    
    results_df, summary_df = evaluate_all_models(
        model_creators=models_dict,
        data_path=data_path,
        n_folds=n_folds,
        verbose=True
    )
    
    if not summary_df.empty:
        print("\nModel Ranking by Accuracy:")
        ranking = summary_df.sort_values('accuracy_mean', ascending=False)
        for i, (_, row) in enumerate(ranking.iterrows(), 1):
            acc = row['accuracy_mean']
            acc_std = row['accuracy_std']
            model = row['model']
            print(f"  {i}. {model}: {acc:.3f} ± {acc_std:.3f}")
        
        print("\nDetailed Comparison:")
        print("-" * 70)
        print(f"{'Model':<15} {'Accuracy':<12} {'LogLoss':<12} {'AUC':<12} {'Time(s)':<10}")
        print("-" * 70)
        
        for _, row in ranking.iterrows():
            model = row['model']
            acc = f"{row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}"
            ll = f"{row['log_loss_mean']:.3f}±{row['log_loss_std']:.3f}"
            auc = f"{row['auc_mean']:.3f}±{row['auc_std']:.3f}" if not pd.isna(row['auc_mean']) else "N/A"
            time_total = row['training_time_mean'] + row['prediction_time_mean']
            
            print(f"{model:<15} {acc:<12} {ll:<12} {auc:<12} {time_total:.2f}")
    
    return results_df, summary_df


# Legacy function for backward compatibility
def evaluate_simple_logistic_cv(*args, **kwargs):
    """Legacy function - use evaluate_single_model instead."""
    warnings.warn("evaluate_simple_logistic_cv is deprecated. Use evaluate_single_model instead.")
    return None


if __name__ == "__main__":
    # Quick test of the evaluation framework
    print("Testing Evaluation Framework...")
    
    try:
        from . import create_model
        
        # Test single model evaluation
        results = evaluate_single_model(
            model_creator=lambda: create_model('SimpleLogistic'),
            model_name='SimpleLogistic',
            data_path='../data/mlogit_Train_wide.csv',
            n_folds=3,  # Smaller for testing
            verbose=True
        )
        
        print(f"✅ Single model test: {len(results)} fold results")
        
        # Test multi-model evaluation
        quick_model_comparison(
            data_path='../data/mlogit_Train_wide.csv',
            n_folds=3
        )
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")

# Evaluate across your existing 5 folds
# results_df, coefs_df = evaluate_simple_logistic_cv(get_fold_data)

# print("Fold metrics (choice-set level):")
# display(results_df)

# print("Per-fold coefficients on (alt2 - alt1):")
# display(coefs_df.groupby(['feature_on_(alt2-alt1)']).agg({ 'coef':'mean'}))