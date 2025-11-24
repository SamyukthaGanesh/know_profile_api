#!/usr/bin/env python3
"""
Advanced Fairness Optimizer Demo - Sophisticated Features
========================================================

This demo showcases the enhanced fairness optimizer with:
- Advanced ensemble methods (voting, bagging, boosting)
- Multi-objective optimization
- Hyperparameter optimization
- Statistical testing and confidence intervals
- Robustness analysis
- Cross-validation with fairness metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai_governance_framework.core.fairness.optimizer import FairnessOptimizer, FairnessConfig

def load_data():
    """Load and preprocess the loan approval dataset."""
    print("🔄 Loading loan approval dataset...")
    
    try:
        df = pd.read_csv('data.csv')
        print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Basic preprocessing
        print("🔄 Preprocessing data...")
        
        # Handle missing values
        df = df.dropna()
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['Gender', 'Loan_Status']:
                df[col] = pd.Categorical(df[col]).codes
        
        # Binary encode target and sensitive feature
        df['Loan_Status'] = (df['Loan_Status'] == 'Y').astype(int)
        df['Gender'] = (df['Gender'] == 'Female').astype(int)  # 1 = Female, 0 = Male
        
        print(f"✅ Preprocessing complete: {df.shape[0]} rows remaining")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return create_synthetic_data()

def create_synthetic_data():
    """Create synthetic loan approval data if real data unavailable."""
    print("🔄 Creating synthetic dataset...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    credit_score = np.random.normal(650, 100, n_samples)
    income = np.random.lognormal(10, 0.5, n_samples)
    loan_amount = np.random.lognormal(11, 0.3, n_samples)
    gender = np.random.binomial(1, 0.4, n_samples)  # 40% female
    
    # Create bias: slightly favor males in approval
    approval_prob = (
        0.3 + 
        0.4 * (credit_score - 600) / 200 +
        0.2 * np.log(income / 20000) +
        -0.1 * (gender * 0.3)  # Bias against females
    )
    approval_prob = np.clip(approval_prob, 0.05, 0.95)
    loan_approval = np.random.binomial(1, approval_prob, n_samples)
    
    df = pd.DataFrame({
        'ApplicantIncome': income,
        'LoanAmount': loan_amount,
        'Credit_History': (credit_score > 600).astype(int),
        'Gender': gender,
        'Loan_Status': loan_approval
    })
    
    print(f"✅ Synthetic dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def demo_advanced_algorithms():
    """Demonstrate advanced fairness algorithms."""
    print("\n" + "="*60)
    print("🎯 ADVANCED FAIRNESS ALGORITHMS DEMO")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['Loan_Status', 'Gender']]
    X = df[feature_cols + ['Gender']]
    y = df['Loan_Status']
    
    # Scale numerical features
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != 'Gender']
    
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split: {len(X_train)} train, {len(X_test)} test samples")
    print(f"📊 Gender distribution: {dict(X_train['Gender'].value_counts())}")
    print(f"📊 Approval rate: {y_train.mean():.2%}")
    
    # Define advanced configurations
    configs = {
        "Ensemble Voting": FairnessConfig(
            mitigation="ensemble",
            objective="demographic_parity",
            hyperparameter_optimization=True,
            statistical_testing=True,
            confidence_intervals=True,
            ensemble_config={
                'type': 'voting',
                'n_estimators': 5,
                'search_method': 'grid',
                'cv_folds': 3
            }
        ),
        
        "Multi-Objective": FairnessConfig(
            mitigation="multi_objective",
            objective="equalized_odds",
            hyperparameter_optimization=True,
            robustness_analysis=True,
            multi_objective_config={
                'objectives': ['accuracy', 'fairness'],
                'weights': [0.7, 0.3]
            }
        ),
        
        "Bagging Ensemble": FairnessConfig(
            mitigation="ensemble",
            objective="demographic_parity",
            statistical_testing=True,
            confidence_intervals=True,
            ensemble_config={
                'type': 'bagging',
                'n_estimators': 10,
                'search_method': 'random'
            }
        )
    }
    
    # Test different base models
    base_models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42)
    }
    
    results = {}
    
    for model_name, base_model in base_models.items():
        print(f"\n🔬 Testing {model_name}...")
        model_results = {}
        
        for config_name, config in configs.items():
            print(f"  ⚙️ Configuration: {config_name}")
            
            try:
                # Create and fit optimizer
                optimizer = FairnessOptimizer(
                    base_estimator=base_model,
                    sensitive_feature_names=['Gender'],
                    config=config
                )
                
                print(f"     🔄 Training...")
                optimizer.fit(X_train, y_train)
                
                print(f"     🔄 Evaluating...")
                eval_results = optimizer.evaluate(X_test, y_test)
                
                # Store key metrics
                model_results[config_name] = {
                    'overall': eval_results['overall'],
                    'disparities': eval_results['disparities'],
                    'has_statistical_tests': 'statistical_tests' in eval_results,
                    'has_confidence_intervals': 'confidence_intervals' in eval_results,
                    'has_robustness': 'robustness' in eval_results
                }
                
                print(f"     ✅ Accuracy: {eval_results['overall']['accuracy']:.3f}")
                print(f"     ✅ Accuracy Disparity: {eval_results['disparities']['accuracy']:.3f}")
                
                # Show advanced features
                if 'statistical_tests' in eval_results:
                    print(f"     📊 Statistical tests performed: {len(eval_results['statistical_tests'])} tests")
                
                if 'confidence_intervals' in eval_results:
                    ci_overall = eval_results['confidence_intervals']['overall']
                    if 'accuracy' in ci_overall:
                        ci_low, ci_high = ci_overall['accuracy']
                        print(f"     📊 Accuracy 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
                
                if 'robustness' in eval_results:
                    robustness = eval_results['robustness']
                    print(f"     🛡️ Robustness analysis: {len(robustness)} metrics")
                
            except Exception as e:
                print(f"     ❌ Error: {e}")
                model_results[config_name] = {'error': str(e)}
        
        results[model_name] = model_results
    
    return results

def demo_cross_validation():
    """Demonstrate cross-validation with fairness metrics."""
    print("\n" + "="*60)
    print("🔄 CROSS-VALIDATION WITH FAIRNESS METRICS")
    print("="*60)
    
    df = load_data()
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in ['Loan_Status', 'Gender']]
    X = df[feature_cols + ['Gender']]
    y = df['Loan_Status']
    
    # Scale features
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != 'Gender']
    
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Configure advanced fairness optimizer
    config = FairnessConfig(
        mitigation="reduction",
        objective="demographic_parity",
        constraints_eps=0.05,
        statistical_testing=True,
        confidence_intervals=True
    )
    
    optimizer = FairnessOptimizer(
        base_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
        sensitive_feature_names=['Gender'],
        config=config
    )
    
    print("🔄 Performing 5-fold cross-validation...")
    
    try:
        cv_results = optimizer.cross_validate_fairness(
            X, y,
            cv=5,
            scoring=['accuracy', 'precision', 'recall', 'f1']
        )
        
        print("✅ Cross-validation completed!")
        print(f"\n📊 Cross-validation Results:")
        
        # Display fairness disparities across folds
        if 'fairness_disparities' in cv_results:
            disparities = cv_results['fairness_disparities']
            print(f"\n🎯 Fairness Disparities Across Folds:")
            for metric, stats in disparities.items():
                mean_disp = stats['mean']
                std_disp = stats['std']
                print(f"  {metric.capitalize()}: {mean_disp:.3f} ± {std_disp:.3f}")
        
        # Display overall performance statistics
        if 'cv_scores' in cv_results:
            cv_scores = cv_results['cv_scores']
            print(f"\n📈 Overall Cross-Validation Performance:")
            for metric, scores in cv_scores.items():
                if metric.startswith('test_'):
                    metric_name = metric.replace('test_', '')
                    mean_val = np.mean(scores)
                    std_val = np.std(scores)
                    print(f"  {metric_name.capitalize()}: {mean_val:.3f} ± {std_val:.3f}")
        
    except Exception as e:
        print(f"❌ Cross-validation error: {e}")

def demo_statistical_analysis():
    """Demonstrate statistical testing and confidence intervals."""
    print("\n" + "="*60)
    print("📊 STATISTICAL ANALYSIS DEMO")
    print("="*60)
    
    df = load_data()
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in ['Loan_Status', 'Gender']]
    X = df[feature_cols + ['Gender']]
    y = df['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Configure with all statistical features
    config = FairnessConfig(
        mitigation="reduction",
        objective="demographic_parity",
        constraints_eps=0.05,
        statistical_testing=True,
        confidence_intervals=True,
        robustness_analysis=True
    )
    
    optimizer = FairnessOptimizer(
        base_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        sensitive_feature_names=['Gender'],
        config=config
    )
    
    print("🔄 Training fairness-aware model with statistical analysis...")
    optimizer.fit(X_train, y_train)
    
    print("🔄 Performing comprehensive evaluation...")
    results = optimizer.evaluate(X_test, y_test)
    
    # Display statistical test results
    if 'statistical_tests' in results:
        print("\n🧪 Statistical Test Results:")
        tests = results['statistical_tests']
        
        for test_name, test_result in tests.items():
            statistic = test_result['statistic']
            p_value = test_result['p_value']
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            print(f"  {test_name}: statistic={statistic:.3f}, p-value={p_value:.4f} {significance}")
            
            if p_value < 0.05:
                print(f"    🚨 Statistically significant difference detected!")
            else:
                print(f"    ✅ No significant difference found")
    
    # Display confidence intervals
    if 'confidence_intervals' in results:
        print("\n📊 95% Confidence Intervals:")
        ci = results['confidence_intervals']
        
        if 'overall' in ci:
            print("  Overall Performance:")
            for metric, (lower, upper) in ci['overall'].items():
                point_est = results['overall'][metric]
                print(f"    {metric.capitalize()}: {point_est:.3f} [{lower:.3f}, {upper:.3f}]")
        
        if 'by_group' in ci:
            print("  By Group:")
            for group, group_ci in ci['by_group'].items():
                group_name = "Female" if group == 1 else "Male"
                print(f"    {group_name}:")
                for metric, (lower, upper) in group_ci.items():
                    point_est = results['by_group'].loc[group, metric]
                    print(f"      {metric.capitalize()}: {point_est:.3f} [{lower:.3f}, {upper:.3f}]")
    
    # Display robustness analysis
    if 'robustness' in results:
        print("\n🛡️ Robustness Analysis:")
        robustness = results['robustness']
        
        if 'feature_importance_stability' in robustness:
            fi_stability = robustness['feature_importance_stability']
            print(f"  Feature Importance Stability:")
            print(f"    Mean std deviation: {fi_stability['mean_std']:.4f}")
            print(f"    Max std deviation: {fi_stability['max_std']:.4f}")
        
        if 'prediction_stability' in robustness:
            pred_stability = robustness['prediction_stability']
            print(f"  Prediction Stability:")
            for noise_level, stability in pred_stability.items():
                mean_agreement = stability['mean_agreement']
                print(f"    {noise_level}: {mean_agreement:.3f} agreement")
        
        if 'cross_group_generalization' in robustness:
            cross_gen = robustness['cross_group_generalization']
            print(f"  Cross-Group Generalization:")
            for pair, accuracy in cross_gen.items():
                print(f"    {pair}: {accuracy:.3f} accuracy")

def main():
    """Run all advanced demos."""
    print("🚀 ADVANCED FAIRNESS OPTIMIZER - SOPHISTICATED FEATURES DEMO")
    print("="*70)
    
    # Run all demos
    try:
        # 1. Advanced algorithms
        results = demo_advanced_algorithms()
        
        # 2. Cross-validation
        demo_cross_validation()
        
        # 3. Statistical analysis
        demo_statistical_analysis()
        
        print("\n" + "="*70)
        print("✨ DEMO COMPLETED SUCCESSFULLY!")
        print("✨ The fairness optimizer now includes:")
        print("   • Advanced ensemble methods (voting, bagging, boosting)")
        print("   • Multi-objective optimization")
        print("   • Hyperparameter optimization with grid/random search")
        print("   • Statistical significance testing")
        print("   • Bootstrap confidence intervals")
        print("   • Comprehensive robustness analysis")
        print("   • Cross-validation with fairness metrics")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()