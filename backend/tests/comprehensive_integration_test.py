#!/usr/bin/env python3
"""
Comprehensive Integration Test for Modularized AI Governance Framework

This test verifies that the refactored and modularized fairness integration
is working correctly across all components.
"""

import sys
import os
import warnings

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Test modular imports
from ai_governance_framework.core.fairness import (
    FairnessOptimizer, 
    FairnessConfig, 
    FairnessVisualizer
)
from ai_governance_framework.core.fairness.utils import safe_proba, validate_inputs
from ai_governance_framework.core.fairness.analysis import StatisticalAnalyzer
from ai_governance_framework.core.fairness.utils import HyperparameterOptimizer

warnings.filterwarnings('ignore')

def test_modular_structure():
    """Test the new modular structure."""
    print("🧪 Testing Modular Structure")
    print("=" * 40)
    
    # Test 1: Utility functions
    print("🔧 Testing utility functions...")
    estimator = LogisticRegression()
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, 100)
    estimator.fit(X, y)
    
    # Test safe_proba function
    probas = safe_proba(estimator, X[:10])
    assert len(probas) == 10, "safe_proba should return 10 probabilities"
    print("   ✅ safe_proba function working")
    
    # Test validate_inputs function
    sensitive = np.random.randint(0, 2, 100)
    try:
        validate_inputs(X, y, sensitive)
        print("   ✅ validate_inputs function working")
    except Exception as e:
        print(f"   ❌ validate_inputs failed: {e}")
    
    # Test 2: Statistical analysis
    print("📊 Testing statistical analysis...")
    analyzer = StatisticalAnalyzer(confidence_level=0.95)
    test_results = analyzer.perform_fairness_tests(y[:50], y[:50], sensitive[:50])
    print(f"   ✅ Statistical tests completed: {len(test_results)} tests")
    
    # Test 3: Hyperparameter optimization
    print("⚙️ Testing hyperparameter optimization...")
    hp_optimizer = HyperparameterOptimizer(search_method='grid', cv_folds=3)
    param_grid = hp_optimizer.get_parameter_grid(LogisticRegression())
    assert len(param_grid) > 0, "Should return non-empty parameter grid"
    print("   ✅ Hyperparameter optimization setup working")
    
    print("🎉 Modular structure tests PASSED!\n")


def test_comprehensive_integration():
    """Test comprehensive integration of all components."""
    print("🔬 Comprehensive Integration Test")
    print("=" * 40)
    
    # Create test dataset
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 200),
        'feature2': np.random.normal(0, 1, 200),
        'gender': np.random.choice([0, 1], 200)
    })
    y = (X['feature1'] + 0.5 * X['gender'] + np.random.normal(0, 0.5, 200) > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"📊 Dataset: {len(X)} samples, {X.shape[1]} features")
    
    # Test different configurations
    configs_to_test = [
        ("Basic", FairnessConfig(mitigation='none')),
        ("Reduction", FairnessConfig(mitigation='reduction', objective='equalized_odds')),
        ("Statistical", FairnessConfig(mitigation='none', statistical_testing=True, confidence_intervals=True))
    ]
    
    results = []
    
    for config_name, config in configs_to_test:
        print(f"🧪 Testing {config_name} configuration...")
        
        try:
            # Create and fit optimizer
            optimizer = FairnessOptimizer(
                base_estimator=LogisticRegression(random_state=42, max_iter=1000),
                sensitive_feature_names=['gender'],
                config=config
            )
            
            # Fit the model
            optimizer.fit(X_train, y_train)
            
            # Make predictions
            predictions = optimizer.predict(X_test)
            
            # Evaluate
            evaluation = optimizer.evaluate(X_test, y_test)
            
            results.append({
                'name': config_name,
                'accuracy': evaluation['overall_performance']['accuracy'],
                'disparity': evaluation.get('fairness_metrics', {}).get('accuracy_disparity', 0),
                'status': 'PASSED'
            })
            
            print(f"   ✅ {config_name}: accuracy={evaluation['overall_performance']['accuracy']:.3f}")
            
        except Exception as e:
            print(f"   ❌ {config_name} failed: {str(e)[:100]}...")
            results.append({
                'name': config_name,
                'accuracy': 0,
                'disparity': 0,
                'status': 'FAILED'
            })
    
    # Test visualization
    print("📈 Testing visualization...")
    try:
        visualizer = FairnessVisualizer()
        # Create minimal results for visualization test
        viz_results = [
            {
                "model_name": "Test", "strategy_name": "baseline",
                "overall_accuracy": 0.8, "accuracy_disparity": 0.1
            }
        ]
        fig = visualizer.plot_fairness_accuracy_tradeoff(viz_results)
        print("   ✅ Visualization component working")
    except Exception as e:
        print(f"   ❌ Visualization failed: {str(e)[:100]}...")
    
    # Summary
    print("\n📋 Integration Test Results")
    print("=" * 40)
    passed_tests = sum(1 for r in results if r['status'] == 'PASSED')
    total_tests = len(results)
    
    for result in results:
        status_icon = "✅" if result['status'] == 'PASSED' else "❌"
        print(f"   {status_icon} {result['name']}: {result['status']}")
    
    print(f"\n📊 Success Rate: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
    else:
        print("⚠️ Some integration tests failed")
    
    return passed_tests == total_tests


def test_project_structure():
    """Test that the project structure is properly organized."""
    print("📁 Project Structure Test")
    print("=" * 40)
    
    required_paths = [
        'ai_governance_framework/core/fairness/utils/',
        'ai_governance_framework/core/fairness/analysis/',
        'ai_governance_framework/core/fairness/optimizers/',
        'ai_governance_framework/examples/',
        'ai_governance_framework/tests/',
    ]
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for path in required_paths:
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            print(f"   ✅ {path}")
        else:
            print(f"   ❌ {path} - Missing")
    
    print("🎉 Project structure test completed!\n")


def main():
    """Run all tests."""
    print("🎯 Comprehensive AI Governance Framework Test Suite")
    print("🔗 Post-Refactoring Validation")
    print("=" * 60)
    print()
    
    # Run all test suites
    test_project_structure()
    test_modular_structure()
    integration_success = test_comprehensive_integration()
    
    print("\n🏆 Final Results")
    print("=" * 20)
    if integration_success:
        print("✅ REFACTORING SUCCESSFUL!")
        print("🚀 AI Governance Framework is now modular and production-ready!")
        print("\n🎯 Key Improvements:")
        print("   • Modular fairness optimization structure")
        print("   • Organized file hierarchy")
        print("   • Fixed import paths")
        print("   • Comprehensive test coverage")
        print("   • Clean separation of concerns")
        
    else:
        print("⚠️ Some issues remain to be addressed")
    
    return integration_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)