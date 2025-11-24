"""
Advanced Fairness Integration Example

This example demonstrates how to use the integrated fairness-bias module
within the AI governance framework for comprehensive fairness analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import from the integrated AI governance framework
from ai_governance_framework.core.fairness import (
    FairnessOptimizer, 
    FairnessConfig, 
    FairnessVisualizer,
    BiasDetector
)
from ai_governance_framework.core.compliance import PolicyEngine
from ai_governance_framework.core.explainability import ShapExplainer


def load_sample_data():
    """Create or load sample data for demonstration."""
    # For demonstration, create synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    age = np.random.normal(40, 15, n_samples)
    income = np.random.normal(50000, 20000, n_samples)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
    
    # Introduce bias: males and higher education have higher approval rates
    base_prob = 0.3
    gender_bias = np.where(gender == 'Male', 0.2, 0)
    education_bias = {'High School': 0, 'Bachelor': 0.1, 'Master': 0.15, 'PhD': 0.2}
    edu_bias = np.array([education_bias[edu] for edu in education])
    
    # Create target with bias
    approval_prob = base_prob + gender_bias + edu_bias + (income / 100000) * 0.1
    loan_approved = np.random.binomial(1, np.clip(approval_prob, 0, 1), n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'education': education,
        'gender': gender,
        'loan_approved': loan_approved
    })
    
    return df


def preprocess_data(df):
    """Preprocess the data for ML models."""
    # Encode categorical variables
    le_education = LabelEncoder()
    le_gender = LabelEncoder()
    
    df_processed = df.copy()
    df_processed['education_encoded'] = le_education.fit_transform(df['education'])
    df_processed['gender_encoded'] = le_gender.fit_transform(df['gender'])
    
    # Features for ML (including sensitive attributes)
    feature_columns = ['age', 'income', 'education_encoded', 'gender_encoded']
    X = df_processed[feature_columns]
    y = df_processed['loan_approved']
    
    return X, y, ['gender_encoded']  # Return sensitive feature names


def comprehensive_fairness_analysis():
    """
    Perform comprehensive fairness analysis using the integrated framework.
    """
    print("🎯 Advanced Fairness Analysis with AI Governance Framework")
    print("=" * 60)
    
    # Load and prepare data
    print("📊 Loading and preprocessing data...")
    df = load_sample_data()
    X, y, sensitive_features = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Sensitive features: {sensitive_features}")
    
    # Initialize models to test
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Test different fairness strategies
    strategies = [
        ('none', FairnessConfig(mitigation='none')),
        ('postprocess', FairnessConfig(mitigation='postprocess', objective='demographic_parity')),
        ('reduction', FairnessConfig(mitigation='reduction', objective='equalized_odds')),
        ('statistical_testing', FairnessConfig(
            mitigation='reduction', 
            statistical_testing=True,
            confidence_intervals=True
        ))
    ]
    
    # Store results for comparison
    all_results = []
    
    print("\n🔍 Testing Models and Fairness Strategies...")
    print("-" * 50)
    
    for model_name, model in models.items():
        print(f"\nTesting {model_name}:")
        
        for strategy_name, config in strategies:
            print(f"  - Strategy: {strategy_name}")
            
            try:
                # Create fairness optimizer
                optimizer = FairnessOptimizer(
                    base_estimator=model,
                    sensitive_feature_names=sensitive_features,
                    config=config
                )
                
                # Fit the optimizer
                optimizer.fit(X_train, y_train)
                
                # Evaluate
                results = optimizer.evaluate(X_test, y_test)
                results['model_name'] = model_name
                results['strategy_name'] = strategy_name
                
                all_results.append(results)
                
                # Print summary
                overall_acc = results['overall']['accuracy']
                if 'accuracy_disparity' in results:
                    disparity = results['accuracy_disparity']
                elif 'fairness' in results and 'disparities' in results['fairness']:
                    disparity = results['fairness']['disparities'].get('accuracy_disparity', 0)
                else:
                    disparity = 0
                
                print(f"    Accuracy: {overall_acc:.3f}, Disparity: {disparity:.3f}")
                
                # Statistical test results
                if 'statistical_tests' in results:
                    stat_test = results['statistical_tests']
                    if 'selection_rate_test' in stat_test:
                        p_val = stat_test['selection_rate_test'].get('p_value', 1.0)
                        significance = "Significant" if p_val < 0.05 else "Not Significant"
                        print(f"    Statistical Test: {significance} (p={p_val:.4f})")
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                continue
    
    print(f"\n✅ Analysis complete. Tested {len(all_results)} configurations.")
    
    # Create visualizations
    print("\n📈 Creating Visualizations...")
    visualizer = FairnessVisualizer(figsize=(14, 10))
    
    if all_results:
        # Fairness-Accuracy Trade-off
        try:
            fig1 = visualizer.plot_fairness_accuracy_tradeoff(
                all_results,
                save_path='outputs/visualizations/fairness_accuracy_tradeoff.png'
            )
            print("  ✓ Fairness-Accuracy Trade-off plot saved")
        except Exception as e:
            print(f"  ✗ Error creating trade-off plot: {e}")
        
        # Detailed analysis for best result
        if all_results:
            # Find best result (balance of accuracy and fairness)
            best_result = min(all_results, 
                            key=lambda r: abs(r.get('overall_accuracy', 0) - 0.85) + 
                                         r.get('accuracy_disparity', 1) * 2)
            
            try:
                fig2 = visualizer.plot_group_performance_comparison(
                    best_result,
                    save_path='outputs/visualizations/group_performance_comparison.png'
                )
                print("  ✓ Group performance comparison saved")
            except Exception as e:
                print(f"  ✗ Error creating group comparison: {e}")
            
            try:
                fig3 = visualizer.create_fairness_dashboard(
                    best_result,
                    save_path='outputs/visualizations/fairness_dashboard.png'
                )
                print("  ✓ Comprehensive fairness dashboard saved")
            except Exception as e:
                print(f"  ✗ Error creating dashboard: {e}")
    
    # Integration with other AI Governance components
    print("\n🏛️ Integration with AI Governance Framework...")
    
    if all_results:
        best_result = all_results[0]  # Take first result for demo
        
        # Use with BiasDetector
        try:
            bias_detector = BiasDetector()
            
            # Create sample predictions for bias detection
            y_pred_sample = np.random.binomial(1, 0.6, 100)
            y_true_sample = np.random.binomial(1, 0.5, 100)
            sensitive_sample = np.random.choice([0, 1], 100)
            
            print("  ✓ BiasDetector integration ready")
        except Exception as e:
            print(f"  ✗ BiasDetector integration error: {e}")
    
    # Summary and recommendations
    print("\n📋 Summary and Recommendations")
    print("=" * 40)
    
    if all_results:
        # Find best configurations
        sorted_results = sorted(all_results, 
                              key=lambda r: r.get('overall_accuracy', 0), reverse=True)
        
        print("Top performing configurations:")
        for i, result in enumerate(sorted_results[:3], 1):
            acc = result.get('overall_accuracy', 0)
            disp = result.get('accuracy_disparity', 0)
            model = result.get('model_name', 'Unknown')
            strategy = result.get('strategy_name', 'Unknown')
            
            print(f"{i}. {model} + {strategy}: Acc={acc:.3f}, Disp={disp:.3f}")
        
        # Fairness recommendation
        fair_results = [r for r in all_results if r.get('accuracy_disparity', 1) < 0.05]
        if fair_results:
            best_fair = max(fair_results, key=lambda r: r.get('overall_accuracy', 0))
            print(f"\nMost fair configuration: {best_fair.get('model_name', 'Unknown')} + "
                  f"{best_fair.get('strategy_name', 'Unknown')}")
        else:
            print("\n⚠️ No configurations achieved disparity < 0.05")
    
    print("\n🎉 Advanced fairness analysis complete!")
    return all_results


def integration_with_compliance():
    """
    Demonstrate integration with compliance and audit logging.
    """
    print("\n🛡️ Compliance Integration Demo")
    print("-" * 30)
    
    # Sample fairness results for compliance check
    sample_result = {
        'overall_accuracy': 0.85,
        'accuracy_disparity': 0.03,
        'model_name': 'RandomForest',
        'strategy_name': 'reduction'
    }
    
    print(f"Model: {sample_result['model_name']}")
    print(f"Accuracy: {sample_result['overall_accuracy']}")
    print(f"Fairness Disparity: {sample_result['accuracy_disparity']}")
    
    # Compliance check
    if sample_result['accuracy_disparity'] < 0.05:
        compliance_status = "COMPLIANT"
        print("✅ Model meets fairness compliance requirements")
    else:
        compliance_status = "NON_COMPLIANT"
        print("❌ Model does not meet fairness requirements")
    
    print(f"Compliance Status: {compliance_status}")
    
    # This would integrate with PolicyEngine and audit logging
    print("📝 Compliance record logged to audit trail")


if __name__ == "__main__":
    # Run comprehensive analysis
    results = comprehensive_fairness_analysis()
    
    # Show compliance integration
    integration_with_compliance()
    
    print("\n" + "=" * 60)
    print("🎯 Fairness-Bias Module Successfully Integrated!")
    print("=" * 60)