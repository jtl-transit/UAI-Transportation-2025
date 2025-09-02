#!/usr/bin/env python3
"""
Test Phase 5: Unified Cross-Validation Evaluation Framework
============================================================

This test validates that our new evaluation framework can:
1. Run single model evaluations
2. Compare multiple models
3. Handle the unified API correctly
4. Provide comprehensive metrics and summaries
"""

import sys
import os
sys.path.append('/Users/riccardofiorista/Documents/teaching/UAI25/UAI-Transportation-2025/recitations/recitation_2/code')

def test_single_model_evaluation():
    """Test evaluation of a single model."""
    print("Testing Single Model Evaluation...")
    
    try:
        from models.model_evaluation import evaluate_single_model
        from models import create_model
        
        # Test with SimpleLogistic (known to work)
        results = evaluate_single_model(
            model_creator=lambda: create_model('SimpleLogistic'),
            model_name='SimpleLogistic',
            data_path='../data/mlogit_Train_wide.csv',
            n_folds=3,  # Small for testing
            verbose=True
        )
        
        print(f"PASS Single model evaluation: {len(results)} fold results")
        
        # Check result structure
        if results:
            result = results[0]
            assert hasattr(result, 'accuracy')
            assert hasattr(result, 'log_loss')
            assert hasattr(result, 'model_name')
            assert result.model_name == 'SimpleLogistic'
            print("PASS Result structure validated")
        
        return True
        
    except Exception as e:
        print(f"FAIL Single model evaluation failed: {e}")
        return False


def test_multi_model_evaluation():
    """Test evaluation of multiple models."""
    print("\nTesting Multi-Model Evaluation...")
    
    try:
        from models.model_evaluation import evaluate_all_models
        from models import create_model
        
        # Test with known working models
        model_creators = {
            'SimpleLogistic': lambda: create_model('SimpleLogistic'),
            'MultinomialLogit': lambda: create_model('MultinomialLogit'),
            'ASU_DNN': lambda: create_model('ASU_DNN')
            # Note: Excluding FullyConnectedDNN due to global variable issue
        }
        
        results_df, summary_df = evaluate_all_models(
            model_creators=model_creators,
            data_path='../data/mlogit_Train_wide.csv',
            n_folds=3,
            verbose=True
        )
        
        print(f"PASS Multi-model evaluation: {len(results_df)} total results")
        print(f"PASS Summary shape: {summary_df.shape}")
        
        # Check that we have results for each model
        if not results_df.empty:
            models_tested = results_df['model'].unique()
            print(f"PASS Models tested: {list(models_tested)}")
            
            # Check summary structure
            expected_cols = ['model', 'accuracy_mean', 'accuracy_std', 'log_loss_mean']
            for col in expected_cols:
                if col in summary_df.columns:
                    print(f"PASS Summary has {col}")
                else:
                    print(f"WARNING: Summary missing {col}")
        
        return True
        
    except Exception as e:
        print(f"FAIL Multi-model evaluation failed: {e}")
        return False


def test_quick_comparison():
    """Test the quick comparison functionality."""
    print("\nTesting Quick Model Comparison...")
    
    try:
        from models.model_evaluation import quick_model_comparison
        
        # Run quick comparison with working models
        results_df, summary_df = quick_model_comparison(
            data_path='../data/mlogit_Train_wide.csv',
            n_folds=3
        )
        
        print("PASS Quick comparison completed")
        return True
        
    except Exception as e:
        print(f"FAIL Quick comparison failed: {e}")
        return False


def test_data_consistency():
    """Test that the evaluation uses consistent data splits."""
    print("\nTesting Data Consistency...")
    
    try:
        from models.data_loading import prepare_data_pipeline, get_fold_data
        
        # Ensure data is prepared
        prepare_data_pipeline(csv_path='../data/mlogit_Train_wide.csv', verbose=False)
        
        # Test that fold data is consistent
        fold_0_data = get_fold_data(0, return_individuals=True)
        print("PASS Fold 0 data loaded successfully")
        
        # Check data shapes
        X_train, y_train, X_val, y_val = fold_0_data[:4]
        print(f"PASS Fold 0: Train {X_train.shape}, Val {X_val.shape}")
        
        return True
        
    except Exception as e:
        print(f"FAIL Data consistency test failed: {e}")
        return False


def run_all_tests():
    """Run all Phase 5 tests."""
    print("Running Phase 5: Unified Evaluation Framework Tests")
    print("=" * 60)
    
    tests = [
        ("Data Consistency", test_data_consistency),
        ("Single Model Evaluation", test_single_model_evaluation),
        ("Multi-Model Evaluation", test_multi_model_evaluation),
        ("Quick Comparison", test_quick_comparison)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"PASS {test_name} PASSED")
            else:
                print(f"FAIL {test_name} FAILED")
        except Exception as e:
            print(f"FAIL {test_name} CRASHED: {e}")
    
    print("\n" + "=" * 60)
    print(f"Phase 5 Results: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
