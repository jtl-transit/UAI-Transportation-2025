"""
Transportation Choice Data Loading and Preprocessing Module
============================================================

This module handles:
- Loading transportation choice data from CSV files
- Wide-to-long format transformation
- Person-level stratified cross-validation splits
- Data quality validation and feature engineering

Compatible with Google Colab and designed for undergraduate accessibility.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Union
from sklearn.model_selection import KFold, StratifiedKFold
import warnings

# Global configuration
RANDOM_STATE = 42
DEFAULT_FEATURE_COLS = ['price', 'time', 'comfort', 'change']


class DataManager:
    """
    Manages all data state and operations for transportation choice modeling.
    
    This class encapsulates data loading, transformation, and cross-validation
    setup to eliminate global variables and improve testability.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        """
        Initialize the DataManager.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        
        # Data state
        self.df_raw = None
        self.df_long = None
        self.cv_folds = None
        self.individual_ids = None
        
        # Configuration
        self.feature_cols = DEFAULT_FEATURE_COLS.copy()
    
    def load_transportation_data(self, csv_path: Optional[str] = None,
                               verbose: bool = True) -> pd.DataFrame:
        """
        Load transportation choice data from CSV file.
        
        Parameters:
        -----------
        csv_path : str, optional
            Full path to the CSV file. If None, will look for default file.
        verbose : bool
            Whether to print loading information
            
        Returns:
        --------
        pd.DataFrame : Raw data in wide format
        """
        if verbose:
            print(f"Loading data from: {csv_path}")
        
        self.df_raw = pd.read_csv(csv_path)
        
        if verbose:
            print(f"Dataset shape: {self.df_raw.shape[0]:,} observations × {self.df_raw.shape[1]} variables")
            print(f"Columns: {list(self.df_raw.columns)}")
        
        # Validate data structure
        self._validate_data_structure(self.df_raw, verbose=verbose)
        
        return self.df_raw

    def _validate_data_structure(self, df: pd.DataFrame, verbose: bool = True) -> None:
        """
        Validate that the dataset has the expected structure for choice modeling.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw transportation choice data
        verbose : bool
            Whether to print validation results
        """
        required_cols = ['id', 'choice']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if verbose:
            print("\nData Quality Checks:")
            print(f"- Missing values: {df.isnull().sum().sum()}")
            print(f"- Duplicate rows: {df.duplicated().sum()}")
            print(f"- Unique individuals: {df['id'].nunique():,}")
            
            # Check choice distribution
            choice_dist = df['choice'].value_counts()
            print("- Choice distribution:")
            for choice, count in choice_dist.items():
                percentage = (count / len(df)) * 100
                print(f"  {choice}: {count:,} ({percentage:.1f}%)")

    def transform_wide_to_long(self, df_wide: pd.DataFrame = None,
                              feature_cols: List[str] = None,
                              verbose: bool = True) -> pd.DataFrame:
        """
        Transform wide format choice data to long format for modeling.
        
        Parameters:
        -----------
        df_wide : pd.DataFrame, optional
            Wide format data. If None, uses self.df_raw.
        feature_cols : list, optional
            Feature column names. If None, uses default features.
        verbose : bool
            Whether to print transformation information
            
        Returns:
        --------
        pd.DataFrame : Long format data with columns:
            - id: Individual identifier
            - obs_id: Choice scenario identifier  
            - alternative: Alternative number (1, 2, ...)
            - chosen: Binary choice indicator (1 if chosen, 0 otherwise)
            - feature columns: price, time, comfort, change, etc.
        """
        if df_wide is None:
            df_wide = self.df_raw
            if df_wide is None:
                raise ValueError("No data available. Call load_transportation_data() first.")
        
        if feature_cols is None:
            feature_cols = self.feature_cols
        
        if verbose:
            print("Transforming data to long format...")
        
        # Auto-detect column naming convention
        sample_feature = feature_cols[0]  # Use 'price' as test feature
        
        # Check for numeric suffix (price1, price2) vs dot suffix (price.1, price.2)
        alt_cols_num = [col for col in df_wide.columns if col.startswith(sample_feature) and col[len(sample_feature):].isdigit()]
        alt_cols_dot = [col for col in df_wide.columns if col.startswith(f"{sample_feature}.")]
        
        if verbose:
            print(f"Numeric format columns found: {len(alt_cols_num)}")
            print(f"Dot format columns found: {len(alt_cols_dot)}")
        
        # Determine number of alternatives and format
        if alt_cols_num and not alt_cols_dot:
            alt_format = "num"
            n_alternatives = len(alt_cols_num)
            alternatives = sorted([int(col[len(sample_feature):]) for col in alt_cols_num])
        elif alt_cols_dot and not alt_cols_num:
            alt_format = "dot" 
            n_alternatives = len(alt_cols_dot)
            alternatives = sorted([int(col.split('.')[-1]) for col in alt_cols_dot])
        else:
            # Fallback detection
            if f"{sample_feature}1" in df_wide.columns:
                alt_format = "num"
            elif f"{sample_feature}.1" in df_wide.columns:
                alt_format = "dot"
            else:
                raise ValueError(f"Cannot detect alternative column format. Expected columns like '{sample_feature}1' or '{sample_feature}.1'")
            
            alternatives = [1, 2]  # Assume binary choice as fallback
            n_alternatives = 2
        
        if verbose:
            print(f"Detected alternatives: {alternatives} (format: {alt_format})")
        
        # Build long format data
        long_data = []
        
        for _, row in df_wide.iterrows():
            individual_id = row['id']
            chosen_alternative = row['choice']
            
            # Create observation ID (scenario within individual)
            if 'choiceid' in df_wide.columns:
                obs_id = f"{individual_id}_{row['choiceid']}"
            else:
                obs_id = f"{individual_id}_1"  # Single choice per individual
            
            # Create one row per alternative
            for alt in alternatives:
                # Build feature values for this alternative
                alt_data = {
                    'id': individual_id,
                    'obs_id': obs_id,
                    'alternative': alt,
                    'chosen': 1 if alt == chosen_alternative else 0
                }
                
                # Add feature columns
                for feature in feature_cols:
                    if alt_format == "num":
                        col_name = f"{feature}{alt}"
                    else:  # dot format
                        col_name = f"{feature}.{alt}"
                    
                    if col_name in df_wide.columns:
                        alt_data[feature] = row[col_name]
                    else:
                        print(f"Warning: Column {col_name} not found, setting to 0")
                        alt_data[feature] = 0
                
                long_data.append(alt_data)
        
        self.df_long = pd.DataFrame(long_data)
        
        # Validation
        if verbose:
            n_scenarios = self.df_long['obs_id'].nunique()
            n_observations = len(self.df_long)
            n_choices = self.df_long['chosen'].sum()
            
            print("Wide-to-long transformation validated:")
            print(f"  - {len(df_wide):,} scenarios → {n_observations:,} alternative observations")
            print(f"  - {n_alternatives} alternatives per scenario")
            print(f"  - {n_choices:,} total choices (1 per scenario)")
            
            # Check data integrity
            choices_per_scenario = self.df_long.groupby('obs_id')['chosen'].sum()
            invalid_scenarios = (choices_per_scenario != 1).sum()
            if invalid_scenarios > 0:
                print(f"  - Warning: {invalid_scenarios} scenarios have != 1 choice")
        
        return self.df_long

    def setup_cross_validation(self, df: pd.DataFrame = None,
                              n_splits: int = 5,
                              random_state: int = None,
                              stratify_by_choice_rate: bool = True,
                              verbose: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Set up person-level cross-validation splits to avoid data leakage.
        
        Parameters:
        -----------
        df : pd.DataFrame, optional
            Long format data. If None, uses self.df_long.
        n_splits : int
            Number of CV folds (default: 5)
        random_state : int
            Random seed for reproducibility
        stratify_by_choice_rate : bool
            Whether to stratify by individual choice rates
            
        Returns:
        --------
        list : List of (train_indices, val_indices) tuples for each fold
        """
        if random_state is None:
            random_state = self.random_state
            
        if df is None:
            df = self.df_long
            if df is None:
                raise ValueError("No data available. Call load_transportation_data() first.")
        
        # Calculate individual-level statistics
        individual_stats = df.groupby('id').agg(
            total_choices=('chosen', 'sum'),
            total_observations=('chosen', 'count'),
            unique_scenarios=('obs_id', 'nunique'),
        ).reset_index()
        
        # Calculate choice rate for stratification
        individual_stats['choice_rate'] = (
            individual_stats['total_choices'] / individual_stats['unique_scenarios']
        )
        
        # Get unique individual IDs
        self.individual_ids = individual_stats['id'].values
        
        # Set up cross-validation
        if stratify_by_choice_rate:
            # Try stratified split first
            try:
                # Create choice rate bins for stratification
                choice_rates = individual_stats['choice_rate'].values
                valid_rates = choice_rates[(choice_rates >= 0) & (choice_rates <= 1)]
                
                if len(np.unique(valid_rates)) > 1:
                    # Bin choice rates for stratification
                    rate_bins = pd.cut(valid_rates, bins=min(5, len(np.unique(valid_rates))), 
                                     labels=False, duplicates='drop')
                    
                    valid_individuals = individual_stats[
                        (individual_stats['choice_rate'] >= 0) & 
                        (individual_stats['choice_rate'] <= 1)
                    ]
                    
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                    cv_folds = list(cv.split(valid_individuals['id'], rate_bins))
                    
                    if verbose:
                        print("Using stratified K-fold splitting")
                else:
                    raise ValueError("Not enough variation in choice rates for stratification")
                    
            except (ValueError, Exception) as e:
                if verbose:
                    print(f"Warning: {len(individual_stats)} individuals had invalid choice rates, using standard K-fold")
                
                # Fallback to standard K-fold
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                cv_folds = list(cv.split(self.individual_ids))
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            cv_folds = list(cv.split(self.individual_ids))
        
        self.cv_folds = cv_folds
        
        if verbose:
            print("Using standard K-fold splitting")
            print(f"\nCross-validation setup ({n_splits} folds):")
            for i, (train_idx, val_idx) in enumerate(cv_folds):
                train_individuals = len(train_idx)
                val_individuals = len(val_idx)
                
                # Count observations
                train_ids = self.individual_ids[train_idx]
                val_ids = self.individual_ids[val_idx]
                train_obs = len(df[df['id'].isin(train_ids)])
                val_obs = len(df[df['id'].isin(val_ids)])
                
                print(f"  Fold {i+1}: {train_individuals} train individuals ({train_obs} obs) | {val_individuals} val individuals ({val_obs} obs)")
        
        return self.cv_folds

    def get_fold_data(self, fold_idx: int, 
                     feature_cols: List[str] = None,
                     return_individuals: bool = False) -> Tuple:
        """
        Get training and validation data for a specific cross-validation fold.
        
        Parameters:
        -----------
        fold_idx : int
            Index of the fold (0-based)
        feature_cols : list, optional
            Feature columns to include. If None, uses default features.
        return_individuals : bool
            Whether to return individual IDs along with data
            
        Returns:
        --------
        tuple : (X_train, y_train, X_val, y_val) or extended with individual info
        """
        if self.df_long is None:
            raise ValueError("No data available. Call load_transportation_data() first.")
        
        if self.cv_folds is None:
            raise ValueError("Cross-validation not set up. Call setup_cross_validation() first.")
        
        if feature_cols is None:
            feature_cols = self.feature_cols
        
        if fold_idx < 0 or fold_idx >= len(self.cv_folds):
            raise ValueError(f"fold_idx must be between 0 and {len(self.cv_folds)-1}")
        
        # Get individual IDs for this fold
        train_idx, val_idx = self.cv_folds[fold_idx]
        train_ids = self.individual_ids[train_idx]
        val_ids = self.individual_ids[val_idx]
        
        # Create masks for observations
        train_mask = self.df_long['id'].isin(train_ids)
        val_mask = self.df_long['id'].isin(val_ids)
        
        # Extract features and targets
        X_train = self.df_long.loc[train_mask, feature_cols].values
        X_val = self.df_long.loc[val_mask, feature_cols].values
        
        y_train = self.df_long.loc[train_mask, 'chosen'].values
        y_val = self.df_long.loc[val_mask, 'chosen'].values
        
        if return_individuals:
            train_obs_ids = self.df_long.loc[train_mask, 'obs_id'].values
            val_obs_ids = self.df_long.loc[val_mask, 'obs_id'].values
            train_alts = self.df_long.loc[train_mask, 'alternative'].values
            val_alts = self.df_long.loc[val_mask, 'alternative'].values
            
            return (X_train, y_train, X_val, y_val,
                   train_ids, val_ids,
                   train_obs_ids, val_obs_ids,
                   train_alts, val_alts)
        
        return X_train, y_train, X_val, y_val

    def prepare_data_pipeline(self, csv_path: Optional[str] = None,
                             feature_cols: List[str] = None,
                             n_folds: int = 5,
                             verbose: bool = True) -> dict:
        """
        Complete end-to-end data preparation pipeline.
        
        Parameters:
        -----------
        csv_path : str, optional
            Path to CSV file
        feature_cols : list, optional
            Feature columns to use
        n_folds : int
            Number of CV folds
        verbose : bool
            Whether to print progress information
            
        Returns:
        --------
        dict : Data information including shapes, column names, etc.
        """
        if verbose:
            print("Starting data preparation pipeline...")
        
        # Step 1: Load raw data
        self.load_transportation_data(csv_path, verbose)
        
        if verbose:
            print(f"\n{'='*60}")
            print("DATASET OVERVIEW")
            print('='*60)
            print(f"Dataset shape: {self.df_raw.shape[0]:,} observations × {self.df_raw.shape[1]} variables")
            print(f"Columns: {list(self.df_raw.columns)}")
        
        # Step 2: Transform to long format
        self.transform_wide_to_long(feature_cols=feature_cols, verbose=verbose)
        
        # Step 3: Set up cross-validation
        if verbose:
            print("\nSetting up cross-validation...")
        self.setup_cross_validation(n_splits=n_folds, verbose=verbose)
        
        # Summary
        info = {
            'raw_shape': self.df_raw.shape,
            'long_shape': self.df_long.shape,
            'n_individuals': len(self.individual_ids),
            'n_scenarios': self.df_long['obs_id'].nunique(),
            'n_alternatives': self.df_long['alternative'].nunique(),
            'n_choices': self.df_long['chosen'].sum(),
            'n_folds': len(self.cv_folds),
            'feature_cols': self.feature_cols
        }
        
        if verbose:
            print(f"\nData preparation complete!")
            print(f"   Raw data: {info['raw_shape']} | Long data: {info['long_shape']}")
            print(f"   {info['n_individuals']} individuals | {info['n_scenarios']:,} scenarios | {info['n_alternatives']} alternatives")
            print(f"   {info['n_choices']:,} choices | {info['n_folds']}-fold CV ready")
        
        return info


# Global instance for backward compatibility
_default_manager = None

def get_default_manager() -> DataManager:
    """Get or create the default DataManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = DataManager()
    return _default_manager


# Legacy functions for backward compatibility
def load_transportation_data(csv_path: Optional[str] = None,
                           verbose: bool = True) -> pd.DataFrame:
    """Legacy function for backward compatibility. Use DataManager.load_transportation_data() instead."""
    manager = get_default_manager()
    return manager.load_transportation_data(csv_path, verbose)


def transform_wide_to_long(df_wide: pd.DataFrame, 
                          feature_cols: List[str] = None,
                          verbose: bool = True) -> pd.DataFrame:
    """Legacy function for backward compatibility. Use DataManager.transform_wide_to_long() instead."""
    manager = get_default_manager()
    manager.df_raw = df_wide
    return manager.transform_wide_to_long(df_wide, feature_cols, verbose)


def setup_cross_validation(df: Optional[pd.DataFrame] = None,
                          n_splits: int = 5,
                          random_state: int = RANDOM_STATE,
                          stratify_by_choice_rate: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Legacy function for backward compatibility. Use DataManager.setup_cross_validation() instead."""
    manager = get_default_manager()
    if df is not None:
        manager.df_long = df
    return manager.setup_cross_validation(df, n_splits, random_state, stratify_by_choice_rate)


def get_fold_data(fold_idx: int, 
                 feature_cols: List[str] = None,
                 return_individuals: bool = False) -> Tuple:
    """Legacy function for backward compatibility. Use DataManager.get_fold_data() instead."""
    manager = get_default_manager()
    return manager.get_fold_data(fold_idx, feature_cols, return_individuals)


def prepare_data_pipeline(csv_path: Optional[str] = None,
                         feature_cols: List[str] = None,
                         n_folds: int = 5,
                         verbose: bool = True) -> dict:
    """Legacy function for backward compatibility. Use DataManager.prepare_data_pipeline() instead."""
    manager = get_default_manager()
    return manager.prepare_data_pipeline(csv_path, feature_cols, n_folds, verbose)


# Legacy global access for module-level imports (still needed by model_evaluation.py)
cv_folds = None
individual_ids = None
df_long = None
df_raw = None

def _update_globals_from_manager(manager: DataManager):
    """Update global variables for backward compatibility."""
    global cv_folds, individual_ids, df_long, df_raw
    cv_folds = manager.cv_folds
    individual_ids = manager.individual_ids
    df_long = manager.df_long  
    df_raw = manager.df_raw
