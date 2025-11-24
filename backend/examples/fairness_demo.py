"""
Standalone Fairness Optimization Demo

This demo shows how to use the integrated fairness module directly.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add the project root to Python path
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our fairness components
from ai_governance_framework.core.fairness.optimizer import FairnessOptimizer, FairnessConfig
from ai_governance_framework.core.fairness.visualizer import FairnessVisualizer


def create_biased_dataset():
    """Create a synthetic dataset with bias for demonstration."""
    print("📊 Creating synthetic loan approval dataset...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    data = {
        'age': np.random.normal(40, 15, n_samples),
        'income': np.random.normal(60000, 25000, n_samples),
        'credit_score': np.random.normal(700, 100, n_samples),
        'employment_years': np.random.exponential(5, n_samples),
        'gender': np.random.choice([0, 1], n_samples),  # 0=Female, 1=Male
        'education': np.random.choice([0, 1, 2, 3], n_samples)  # 0=HS, 1=Bachelor, 2=Master, 3=PhD
    }
    
    df = pd.DataFrame(data)
    
    # Create biased target variable
    # Introduce gender bias: males get 25% higher approval rate
    gender_bias = df['gender'] * 0.25
    
    # Income and credit score factors
    income_factor = (df['income'] - df['income'].mean()) / df['income'].std() * 0.15
    credit_factor = (df['credit_score'] - 650) / 100 * 0.20
    
    # Base approval probability
    base_prob = 0.4
    approval_prob = base_prob + gender_bias + income_factor + credit_factor
    approval_prob = np.clip(approval_prob, 0, 1)
    
    # Generate binary outcomes
    df['loan_approved'] = np.random.binomial(1, approval_prob, n_samples)
    
    print(f"✅ Dataset created: {df.shape}")
    
    # Show bias in data
    approval_rates = df.groupby('gender')['loan_approved'].mean()
    print(f"📈 Approval rates by gender:")
    print(f"   Female (0): {approval_rates[0]:.3f}")
    print(f"   Male (1): {approval_rates[1]:.3f}")
    print(f"   Bias (difference): {approval_rates[1] - approval_rates[0]:.3f}")
    
    return df


def run_fairness_comparison():
    """Run fairness optimization comparison."""
    print("\n🎯 Fairness Optimization Demo")
    print("=" * 50)
    
    # Create dataset
    df = create_biased_dataset()
    
    # Prepare features and target
    X = df[['age', 'income', 'credit_score', 'employment_years', 'gender', 'education']]
    y = df['loan_approved']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n🔧 Data split completed:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")
    
    # Test different fairness strategies
    strategies = [
        ("Baseline (No Mitigation)", FairnessConfig(mitigation='none')),
        ("Post-processing", FairnessConfig(mitigation='postprocess', objective='demographic_parity')),
        ("In-processing (Reduction)", FairnessConfig(mitigation='reduction', objective='equalized_odds')),
        ("Advanced (Statistical Testing)", FairnessConfig(
            mitigation='reduction',
            objective='equalized_odds',
            statistical_testing=True,
            confidence_intervals=True
        ))
    ]
    
    results = []
    
    print(f"\n🧪 Testing {len(strategies)} fairness strategies...")
    print("-" * 60)
    
    for strategy_name, config in strategies:
        print(f"\n🔍 {strategy_name}:")
        
        try:
            # Create optimizer
            optimizer = FairnessOptimizer(
                base_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
                sensitive_feature_names=['gender'],
                config=config
            )
            
            # Train
            print("   Training model...")
            optimizer.fit(X_train, y_train)
            
            # Evaluate
            print("   Evaluating fairness...")
            result = optimizer.evaluate(X_test, y_test)
            
            # Extract metrics
            accuracy = result['overall']['accuracy']
            
            # Get disparity
            disparity = result.get('accuracy_disparity', 0)
            if disparity == 0 and 'fairness' in result and 'disparities' in result['fairness']:
                disparity = result['fairness']['disparities'].get('accuracy_disparity', 0)
            
            print(f"   ✅ Accuracy: {accuracy:.3f}")
            print(f"   📊 Fairness Disparity: {disparity:.3f}")
            
            # Fairness assessment
            if disparity < 0.05:
                fairness_level = "🟢 Excellent"
            elif disparity < 0.10:
                fairness_level = "🟡 Moderate"
            else:
                fairness_level = "🔴 Poor"
            
            print(f"   {fairness_level} Fairness")
            
            # Statistical tests if available
            if 'statistical_tests' in result:
                stat_tests = result['statistical_tests']
                if 'selection_rate_test' in stat_tests:
                    test_result = stat_tests['selection_rate_test']
                    p_value = test_result.get('p_value', 1.0)
                    significance = "Significant bias detected" if p_value < 0.05 else "No significant bias"
                    print(f"   🔬 Statistical Test: {significance} (p={p_value:.4f})")
            
            # Store results
            results.append({
                'strategy': strategy_name,
                'accuracy': accuracy,
                'disparity': disparity,
                'fairness_level': fairness_level,
                'full_result': result
            })
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    return results


def create_visualizations(results):
    """Create visualizations of the results."""
    print(f"\n📊 Creating Visualizations...")
    print("-" * 30)
    
    if not results:
        print("❌ No results to visualize")
        return
    
    try:
        # Create output directory
        os.makedirs('/Users/jullas/ghci/ai_governance_framework/outputs/visualizations', exist_ok=True)
        
        visualizer = FairnessVisualizer(figsize=(12, 8))
        
        # 1. Fairness-Accuracy Trade-off Plot
        print("🎨 Creating fairness-accuracy trade-off plot...")
        
        # Prepare data for plotting
        plot_results = []
        for result in results:
            plot_data = {
                'model_name': 'RandomForest',
                'strategy_name': result['strategy'],
                'overall_accuracy': result['accuracy'],
                'accuracy_disparity': result['disparity']
            }
            plot_results.append(plot_data)
        
        fig1 = visualizer.plot_fairness_accuracy_tradeoff(
            plot_results,
            save_path='/Users/jullas/ghci/ai_governance_framework/outputs/visualizations/fairness_tradeoff.png',
            title='Fairness vs Accuracy Trade-off Analysis'
        )
        print("   ✅ Trade-off plot saved")
        
        # 2. Best result dashboard
        if results:
            best_result = min(results, key=lambda r: r['disparity'])
            print(f"🏆 Creating dashboard for best result: {best_result['strategy']}")
            
            fig2 = visualizer.create_fairness_dashboard(
                best_result['full_result'],
                save_path='/Users/jullas/ghci/ai_governance_framework/outputs/visualizations/fairness_dashboard.png',
                title=f'Fairness Analysis Dashboard - {best_result["strategy"]}'
            )
            print("   ✅ Dashboard saved")
        
        plt.close('all')  # Clean up plots
        
    except Exception as e:
        print(f"   ❌ Visualization error: {str(e)}")


def print_summary(results):
    """Print a comprehensive summary of results."""
    print(f"\n📋 Comprehensive Results Summary")
    print("=" * 45)
    
    if not results:
        print("❌ No results to summarize")
        return
    
    print(f"🏆 Performance Ranking (by fairness):")
    sorted_results = sorted(results, key=lambda r: r['disparity'])
    
    for i, result in enumerate(sorted_results, 1):
        print(f"\n{i}. {result['strategy']}:")
        print(f"   📊 Accuracy: {result['accuracy']:.3f}")
        print(f"   ⚖️  Disparity: {result['disparity']:.3f}")
        print(f"   🎯 Fairness: {result['fairness_level']}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    best_fair = sorted_results[0]
    best_accurate = max(results, key=lambda r: r['accuracy'])
    
    print(f"   🥇 Most Fair: {best_fair['strategy']} (disparity: {best_fair['disparity']:.3f})")
    print(f"   🎯 Most Accurate: {best_accurate['strategy']} (accuracy: {best_accurate['accuracy']:.3f})")
    
    # Find balanced approach
    balanced = min(results, key=lambda r: abs(r['accuracy'] - 0.8) + r['disparity'] * 2)
    print(f"   ⚖️  Best Balance: {balanced['strategy']}")
    
    print(f"\n✨ Key Insights:")
    baseline = next((r for r in results if 'Baseline' in r['strategy']), None)
    if baseline:
        print(f"   • Baseline bias: {baseline['disparity']:.1%} disparity")
        
        fair_methods = [r for r in results if r['disparity'] < 0.1 and 'Baseline' not in r['strategy']]
        if fair_methods:
            best_fair_method = min(fair_methods, key=lambda r: r['disparity'])
            improvement = baseline['disparity'] - best_fair_method['disparity']
            print(f"   • Best improvement: {improvement:.1%} disparity reduction with {best_fair_method['strategy']}")
            print(f"   • Fairness optimization successful!")
        else:
            print(f"   • No methods achieved good fairness (< 10% disparity)")


def main():
    """Main execution function."""
    print("🎯 Advanced Fairness Optimization Demo")
    print("🔗 Integrated AI Governance Framework")
    print("=" * 55)
    print("This demo shows bias detection and mitigation capabilities.\n")
    
    try:
        # Run fairness comparison
        results = run_fairness_comparison()
        
        # Create visualizations
        create_visualizations(results)
        
        # Print summary
        print_summary(results)
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Outputs saved to: ai_governance_framework/outputs/visualizations/")
        print(f"✅ Advanced fairness integration is working perfectly!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()