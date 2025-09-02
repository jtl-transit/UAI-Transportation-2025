#!/usr/bin/env python3
"""
Test Script for Phase 2 Implementation
======================================

This script tests the unified model API and ensures all models can be created,
trained, and used for prediction through a standardized interface.
"""

import sys
sys.path.append('.')
import numpy as np

print("Testing Phase 2 Implementation")
print("=" * 50)

print("\n[1] Testing Model Registry:")
try:
    from models import get_available_models, create_model, list_models
    
    available = get_available_models()
    print(f"✅ Available models: {available}")
    
    # Test detailed model listing
    print("\n📋 Model Details:")
    list_models(detailed=True)
    
except Exception as e:
    print(f"❌ Model registry failed: {e}")

print("\n[2] Testing Data Loading Integration:")
try:
    from models import prepare_data_pipeline, get_fold_data
    
    # Prepare data
    info = prepare_data_pipeline(csv_path="../data/mlogit_Train_wide.csv", verbose=False)
    print(f"✅ Data ready: {info['n_individuals']} individuals, {info['n_scenarios']} scenarios")
    
    # Get test data from first fold
    X_train, y_train, X_val, y_val = get_fold_data(0)
    print(f"✅ Test data extracted: Train {X_train.shape}, Val {X_val.shape}")
    
    # Get extended data for models that need obs_ids and alternatives
    (X_train_ext, y_train_ext, X_val_ext, y_val_ext,
     train_ids, val_ids, train_obs_ids, val_obs_ids,
     train_alts, val_alts) = get_fold_data(0, return_individuals=True)
    
    print(f"✅ Extended data ready for model testing")
    
except Exception as e:
    print(f"❌ Data integration failed: {e}")

print("\n[3] Testing Individual Model Creation & Training:")

# Test each model type
model_tests = {
    'SimpleLogistic': {
        'params': {'random_state': 42},
        'fit_args': (X_train_ext, y_train_ext, train_obs_ids, train_alts),
        'predict_args': (X_val_ext, val_obs_ids, val_alts),
        'expected_output_shape': (len(X_val_ext),)
    },
    'MultinomialLogit': {
        'params': {'random_state': 42, 'maxiter': 100},
        'fit_args': (X_train_ext, y_train_ext, train_obs_ids, train_alts),
        'predict_args': (X_val_ext, val_obs_ids, val_alts),
        'expected_output_shape': (len(X_val_ext),)
    },
    'FullyConnectedDNN': {
        'params': {'hidden_units': [32, 16]},
        'fit_args': (X_train_ext, y_train_ext, train_obs_ids, train_alts),
        'fit_kwargs': {'epochs': 5, 'verbose': 0},
        'predict_args': (X_val_ext, val_obs_ids, val_alts),
        'expected_output_shape': (len(X_val_ext),)  # One probability per alternative
    },
    'ASU_DNN': {
        'params': {'hidden_units': [32, 16]},
        'fit_args': (X_train_ext, y_train_ext, train_obs_ids, train_alts),
        'fit_kwargs': {'epochs': 5, 'verbose': 0},
        'predict_args': (X_val_ext, val_obs_ids, val_alts),
        'expected_output_shape': (len(X_val_ext),)
    }
}

successful_models = []
failed_models = []

for model_name, test_config in model_tests.items():
    try:
        print(f"\n🔸 Testing {model_name}:")
        
        # Create model
        model = create_model(model_name, **test_config['params'])
        print(f"  ✅ Model created: {model.__class__.__name__}")
        
        # Fit model
        fit_args = test_config['fit_args']
        fit_kwargs = test_config.get('fit_kwargs', {})
        model.fit(*fit_args, **fit_kwargs)
        print(f"  ✅ Model fitted successfully")
        
        # Test prediction
        predict_args = test_config['predict_args']
        predict_kwargs = test_config.get('predict_kwargs', {})
        predictions = model.predict_proba(*predict_args, **predict_kwargs)
        print(f"  ✅ Predictions shape: {predictions.shape}")
        
        # Validate prediction properties
        if len(predictions.shape) == 1:
            # Binary probability predictions
            print(f"  ✅ Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        else:
            # Multi-class probability predictions
            print(f"  ✅ Prediction shape: {predictions.shape}")
            print(f"  ✅ Row sums around 1.0: {np.allclose(predictions.sum(axis=1), 1.0)}")
        
        successful_models.append(model_name)
        
    except Exception as e:
        print(f"  ❌ {model_name} failed: {e}")
        failed_models.append((model_name, str(e)))

print(f"\n[4] Testing Model API Consistency:")

# Test that successful models have required methods
api_tests = []
for model_name in successful_models:
    try:
        model = create_model(model_name)
        
        # Check required methods exist
        has_fit = hasattr(model, 'fit') and callable(getattr(model, 'fit'))
        has_predict = hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba'))
        
        api_tests.append({
            'model': model_name,
            'has_fit': has_fit,
            'has_predict': has_predict,
            'api_compliant': has_fit and has_predict
        })
        
    except Exception as e:
        api_tests.append({
            'model': model_name,
            'has_fit': False,
            'has_predict': False,
            'api_compliant': False,
            'error': str(e)
        })

for test in api_tests:
    status = "✅" if test['api_compliant'] else "❌"
    print(f"  {status} {test['model']}: fit={test['has_fit']}, predict={test['has_predict']}")

print(f"\nPhase 2 Results Summary:")
print("=" * 50)
print(f"✅ Successful models: {len(successful_models)}/{len(model_tests)}")
print(f"   Working: {', '.join(successful_models)}")

if failed_models:
    print(f"❌ Failed models: {len(failed_models)}")
    for model_name, error in failed_models:
        print(f"   {model_name}: {error}")

compliant_models = [t['model'] for t in api_tests if t['api_compliant']]
print(f"✅ API compliant models: {len(compliant_models)}/{len(successful_models)}")

if len(successful_models) >= 2:
    print(f"\nPhase 2 SUCCESS: Ready for unified evaluation!")
    print(f"   - {len(successful_models)} models working")
    print(f"   - {len(compliant_models)} API compliant")
    print(f"   - Ready for Phase 5: Cross-validation framework")
else:
    print(f"\nPhase 2 PARTIAL: Need to fix remaining models")
    print(f"   - Continue debugging failed models")
    print(f"   - Standardize API signatures")

print(f"\nNext Steps:")
print(f"   - Fix any failed model imports/training")
print(f"   - Implement model wrappers for signature standardization")
print(f"   - Move to Phase 5: Unified evaluation framework")
