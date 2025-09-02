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
        
        # Load the data
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

# Legacy function - module level backward compatibility
def _validate_data_structure(df: pd.DataFrame, verbose: bool = True) -> None:
    """Legacy function for backward compatibility."""
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


def load_transportation_data(csv_path: Optional[str] = None,
                           verbose: bool = True) -> pd.DataFrame:
    """Legacy function for backward compatibility. Use DataManager.load_transportation_data() instead."""
    manager = DataManager()
    return manager.load_transportation_data(csv_path, verbose)


def transform_wide_to_long(df_wide: pd.DataFrame, 
                          feature_cols: List[str] = None,
                          verbose: bool = True) -> pd.DataFrame:
    """Legacy function for backward compatibility. Use DataManager.transform_wide_to_long() instead."""
    manager = DataManager()
    manager.df_raw = df_wide
    return manager.transform_wide_to_long(df_wide, feature_cols, verbose)


def _validate_data_structure(df: pd.DataFrame, verbose: bool = True) -> None:
    """
    Validate the structure of the loaded data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data to validate
    verbose : bool
        Whether to print validation results
    """
    required_cols = ['id', 'choiceid', 'choice']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for data quality issues
    if verbose:
        print(f"\nData Quality Checks:")
        print(f"- Missing values: {df.isnull().sum().sum()}")
        print(f"- Duplicate rows: {df.duplicated().sum()}")
        print(f"- Unique individuals: {df['id'].nunique():,}")
        
        # Choice distribution
        choice_dist = df['choice'].value_counts()
        print(f"- Choice distribution:")
        for choice, count in choice_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {choice}: {count:,} ({percentage:.1f}%)")


def transform_wide_to_long(df_wide: pd.DataFrame, 
                          alternatives: List[Union[str, int]] = None,
                          feature_cols: List[str] = None) -> pd.DataFrame:
    """
    Transform data from wide format to long format for discrete choice modeling.
    
    In wide format: each row is a choice scenario with alternatives as columns
    In long format: each row is an alternative within a choice scenario
    
    Parameters:
    -----------
    df_wide : pd.DataFrame
        Data in wide format
    alternatives : list, optional
        List of alternative identifiers. If None, inferred from data.
    feature_cols : list, optional
        List of feature column prefixes. If None, uses default.
        
    Returns:
    --------
    pd.DataFrame : Data in long format with columns:
        - id: individual identifier
        - obs_id: choice scenario identifier  
        - alternative: alternative identifier
        - chosen: binary indicator (1 if chosen, 0 otherwise)
        - price, time, comfort, change: feature values
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    
    # Detect alternatives from column names if not provided
    if alternatives is None:
        # Look for numbered columns (e.g., price1, price2 or price.1, price.2)
        sample_feature = feature_cols[0]
        
        # Try format: price1, price2, etc.
        alt_cols_num = [col for col in df_wide.columns if col.startswith(sample_feature) and col[len(sample_feature):].isdigit()]
        # Try format: price.1, price.2, etc.
        alt_cols_dot = [col for col in df_wide.columns if col.startswith(f"{sample_feature}.")]
        
        if alt_cols_num:
            alternatives = sorted([int(col[len(sample_feature):]) for col in alt_cols_num])
            col_format = "num"  # price1, price2
        elif alt_cols_dot:
            alternatives = sorted([int(col.split('.')[-1]) for col in alt_cols_dot])
            col_format = "dot"  # price.1, price.2
        else:
            raise ValueError(f"Could not detect alternatives from column names. "
                           f"Expected format: {sample_feature}1, {sample_feature}2 or "
                           f"{sample_feature}.1, {sample_feature}.2, etc.")
    else:
        # Determine format from existing columns
        sample_feature = feature_cols[0]
        if f"{sample_feature}1" in df_wide.columns:
            col_format = "num"
        elif f"{sample_feature}.1" in df_wide.columns:
            col_format = "dot"
        else:
            raise ValueError(f"Could not determine column format for features")
    
    print(f"Detected alternatives: {alternatives} (format: {col_format})")
    
    # Create long format data
    long_data = []
    
    for _, row in df_wide.iterrows():
        individual_id = row['id']
        choice_id = row['choiceid'] 
        chosen_alt_str = row['choice']
        
        # Parse chosen alternative (e.g., "choice1" -> 1, "choice2" -> 2)
        if chosen_alt_str.startswith('choice'):
            chosen_alt = int(chosen_alt_str.replace('choice', ''))
        else:
            chosen_alt = int(chosen_alt_str)  # In case it's already numeric
        
        # Create observation ID (unique across all choice scenarios)
        obs_id = f"{individual_id}_{choice_id}"
        
        for alt in alternatives:
            alt_row = {
                'id': individual_id,
                'obs_id': obs_id,
                'alternative': alt,
                'chosen': 1 if alt == chosen_alt else 0
            }
            
            # Add feature values for this alternative
            for feature in feature_cols:
                if col_format == "num":
                    col_name = f"{feature}{alt}"
                else:  # dot format
                    col_name = f"{feature}.{alt}"
                    
                if col_name in df_wide.columns:
                    alt_row[feature] = row[col_name]
                else:
                    warnings.warn(f"Column {col_name} not found, setting to NaN")
                    alt_row[feature] = np.nan
            
            long_data.append(alt_row)
    
    df_long = pd.DataFrame(long_data)
    
    # Validate the transformation
    _validate_long_format(df_wide, df_long)
    
    return df_long


def _validate_long_format(df_wide: pd.DataFrame, df_long: pd.DataFrame) -> None:
    """
    Validate the wide-to-long transformation.
    """
    n_scenarios = len(df_wide)
    n_alternatives = df_long['alternative'].nunique()
    expected_rows = n_scenarios * n_alternatives
    
    if len(df_long) != expected_rows:
        raise ValueError(f"Long format validation failed: expected {expected_rows} rows, got {len(df_long)}")
    
    # Check that each scenario has exactly one chosen alternative
    choices_per_scenario = df_long.groupby('obs_id')['chosen'].sum()
    invalid_scenarios = choices_per_scenario[choices_per_scenario != 1]
    
    if len(invalid_scenarios) > 0:
        raise ValueError(f"Found {len(invalid_scenarios)} scenarios without exactly 1 chosen alternative")
    
    print(f"Wide-to-long transformation validated:")
    print(f"  - {n_scenarios:,} scenarios → {len(df_long):,} alternative observations")
    print(f"  - {n_alternatives} alternatives per scenario")
    print(f"  - {df_long['chosen'].sum():,} total choices (1 per scenario)")


def setup_cross_validation(df: Optional[pd.DataFrame] = None,
                          n_splits: int = 5,
                          random_state: int = RANDOM_STATE,
                          stratify_by_choice_rate: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Set up person-level cross-validation splits to avoid data leakage.
    
    Parameters:
    -----------
    df : pd.DataFrame, optional
        Long format data. If None, uses global df_long.
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
    global cv_folds, individual_ids
    
    if df is None:
        df = df_long
        if df is None:
            raise ValueError("No data available. Call load_transportation_data() first.")
    
    # Calculate individual-level statistics
    individual_stats = df.groupby('id').agg(
        total_choices=('chosen', 'sum'),
        total_observations=('chosen', 'count'),
        unique_scenarios=('obs_id', 'nunique'),
    ).round(2)
    
    individual_stats['choice_rate'] = (
        individual_stats['total_choices'] / individual_stats['total_observations']
    )
    
    individual_ids = individual_stats.index.values
    
    # Set up cross-validation splitter
    if stratify_by_choice_rate and len(individual_stats) > n_splits * 2:
        try:
            # Create choice rate bins for stratification
            choice_rate_bins = pd.qcut(individual_stats['choice_rate'], 
                                     q=min(5, len(individual_stats) // n_splits),
                                     labels=False, duplicates='drop')
            
            # Remove any NaN values
            valid_mask = ~pd.isna(choice_rate_bins)
            if valid_mask.sum() < len(individual_stats):
                print(f"Warning: {(~valid_mask).sum()} individuals had invalid choice rates, using standard K-fold")
                splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                cv_folds = list(splitter.split(individual_ids))
                print(f"Using standard K-fold splitting")
            else:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                cv_folds = list(splitter.split(individual_ids, choice_rate_bins))
                print(f"Using stratified K-fold by choice rate ({min(5, len(individual_stats) // n_splits)} bins)")
        except Exception as e:
            print(f"Warning: Stratification failed ({e}), using standard K-fold")
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            cv_folds = list(splitter.split(individual_ids))
            print(f"Using standard K-fold splitting")
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_folds = list(splitter.split(individual_ids))
        print(f"Using standard K-fold splitting")
    
    # Verify the splits
    print(f"\n✓ Cross-validation setup ({n_splits} folds):")
    for fold_idx, (train_individuals, val_individuals) in enumerate(cv_folds):
        train_ids = individual_ids[train_individuals]
        val_ids = individual_ids[val_individuals]
        
        train_mask = df['id'].isin(train_ids)
        val_mask = df['id'].isin(val_ids)
        
        train_obs = train_mask.sum()
        val_obs = val_mask.sum()
        
        print(f"  Fold {fold_idx+1}: {len(train_ids):,} train individuals ({train_obs:,} obs) | "
              f"{len(val_ids):,} val individuals ({val_obs:,} obs)")
    
    return cv_folds


def get_fold_data(fold_idx: int, 
                  feature_cols: List[str] = None,
                  return_individuals: bool = False,
                  df: Optional[pd.DataFrame] = None) -> Tuple:
    """
    Get training and validation data for a specific fold.
    
    Parameters:
    -----------
    fold_idx : int
        Fold index (0 to n_folds-1)
    feature_cols : list, optional
        Feature columns to extract. If None, uses default.
    return_individuals : bool
        Whether to return individual IDs as well
    df : pd.DataFrame, optional
        Long format data. If None, uses global df_long.
        
    Returns:
    --------
    tuple : Training and validation data arrays
        If return_individuals=False: (X_train, y_train, X_val, y_val)
        If return_individuals=True: (X_train, y_train, X_val, y_val,
                                    train_ids, val_ids, 
                                    train_obs_ids, val_obs_ids,
                                    train_alts, val_alts)
    """
    if df is None:
        df = df_long
        if df is None:
            raise ValueError("No data available. Call load_transportation_data() first.")
    
    if cv_folds is None:
        raise ValueError("Cross-validation not set up. Call setup_cross_validation() first.")
    
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    
    if fold_idx < 0 or fold_idx >= len(cv_folds):
        raise ValueError(f"fold_idx must be between 0 and {len(cv_folds)-1}")
    
    # Get individual IDs for this fold
    train_idx, val_idx = cv_folds[fold_idx]
    train_ids = individual_ids[train_idx]
    val_ids = individual_ids[val_idx]
    
    # Create masks for observations
    train_mask = df['id'].isin(train_ids)
    val_mask = df['id'].isin(val_ids)
    
    # Extract features and targets
    X_train = df.loc[train_mask, feature_cols].values
    X_val = df.loc[val_mask, feature_cols].values
    
    y_train = df.loc[train_mask, 'chosen'].values
    y_val = df.loc[val_mask, 'chosen'].values
    
    if return_individuals:
        train_obs_ids = df.loc[train_mask, 'obs_id'].values
        val_obs_ids = df.loc[val_mask, 'obs_id'].values
        train_alts = df.loc[train_mask, 'alternative'].values
        val_alts = df.loc[val_mask, 'alternative'].values
        
        return (X_train, y_train, X_val, y_val,
                train_ids, val_ids,
                train_obs_ids, val_obs_ids,
                train_alts, val_alts)
    
    return X_train, y_train, X_val, y_val


def prepare_data_pipeline(csv_path: Optional[str] = None,
                         data_dir: Optional[str] = None,
                         n_splits: int = 5,
                         feature_cols: List[str] = None,
                         random_state: int = RANDOM_STATE,
                         verbose: bool = True) -> dict:
    """
    Complete data preparation pipeline: load, transform, and setup CV.
    
    Parameters:
    -----------
    csv_path : str, optional
        Path to CSV file
    data_dir : str, optional
        Directory containing data
    n_splits : int
        Number of CV folds
    feature_cols : list, optional
        Feature columns to use
    random_state : int
        Random seed
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    dict : Data information including shapes, column names, etc.
    """
    global df_raw, df_long
    
    if verbose:
        print("Starting data preparation pipeline...")
    
    # Step 1: Load raw data
    df_raw = load_transportation_data(csv_path, verbose)
    
    # Step 2: Transform to long format
    if verbose:
        print("\nTransforming data to long format...")
    df_long = transform_wide_to_long(df_raw, feature_cols=feature_cols)
    
    # Step 3: Setup cross-validation
    if verbose:
        print("\nSetting up cross-validation...")
    setup_cross_validation(df_long, n_splits, random_state)
    
    # Return summary information
    info = {
        'raw_shape': df_raw.shape,
        'long_shape': df_long.shape,
        'n_individuals': df_long['id'].nunique(),
        'n_scenarios': df_long['obs_id'].nunique(),
        'n_alternatives': df_long['alternative'].nunique(),
        'feature_columns': feature_cols or DEFAULT_FEATURE_COLS,
        'n_folds': n_splits,
        'total_choices': df_long['chosen'].sum()
    }
    
    if verbose:
        print("\nData preparation complete!")
        print(f"   Raw data: {info['raw_shape']} | Long data: {info['long_shape']}")
        print(f"   {info['n_individuals']:,} individuals | {info['n_scenarios']:,} scenarios | {info['n_alternatives']} alternatives")
        print(f"   {info['total_choices']:,} choices | {n_splits}-fold CV ready")
    
    return info


# Convenience function for backward compatibility
def load_and_prepare_data(*args, **kwargs):
    """Alias for prepare_data_pipeline for backward compatibility."""
    return prepare_data_pipeline(*args, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("Testing data loading pipeline...")
    info = prepare_data_pipeline(csv_path='/Users/riccardofiorista/Documents/teaching/UAI25/UAI-Transportation-2025/recitations/recitation_2/data/mlogit_Train_wide.csv', verbose=True)
    
    # Test getting fold data
    print("\nTesting fold data extraction...")
    X_train, y_train, X_val, y_val = get_fold_data(0)
    print(f"Fold 0: Train {X_train.shape}, Val {X_val.shape}")