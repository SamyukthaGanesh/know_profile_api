# custom_fairness_demo.py
"""
🎯 SIMPLE GUIDE: How to use your own models and datasets with FairnessOptimizer
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification
import sys
import os

# Add the project root to Python path  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai_governance_framework.core.fairness.optimizer import FairnessOptimizer, FairnessConfig

# ================================
# METHOD 1: USING YOUR OWN DATA
# ================================

def use_your_own_data():
    """Example showing how to use your own dataset"""
    print("📊 METHOD 1: Using Your Own Data")
    print("="*50)
    
    # STEP 1: Load your data (replace this with your own data loading)
    # ================================================================
    
    # Option A: Load from CSV file
    # your_data = pd.read_csv("your_dataset.csv")
    
    # Option B: Load from any source and create DataFrame
    # your_data = pd.DataFrame(your_dict_or_array)
    
    # Option C: For this demo, we'll create sample data
    sample_data = {
        'age': [25, 35, 45, 30, 40, 28, 50, 33],
        'income': [50000, 75000, 90000, 60000, 80000, 55000, 100000, 65000],
        'education': ['High School', 'Bachelor', 'Master', 'Bachelor', 'Master', 'High School', 'PhD', 'Bachelor'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'approved': [1, 1, 1, 0, 1, 0, 1, 1]  # Your target variable
    }
    
    your_data = pd.DataFrame(sample_data)
    print("Your data looks like this:")
    print(your_data)
    
    # STEP 2: Prepare your data
    # =========================
    
    # Separate features and target
    X = your_data.drop('approved', axis=1)  # Features (including sensitive attribute)
    y = your_data['approved']               # Target variable (should be 0/1)
    
    # Handle categorical variables (if needed)
    le = LabelEncoder()
    X['education'] = le.fit_transform(X['education'])
    X['gender'] = le.fit_transform(X['gender'])  # 0=Female, 1=Male
    
    # STEP 3: Choose your model
    # =========================
    
    # Use ANY sklearn-compatible model:
    your_model = RandomForestClassifier(n_estimators=50, random_state=42)
    # your_model = LogisticRegression(max_iter=1000)
    # your_model = SVC(probability=True)
    # your_model = GradientBoostingClassifier()
    
    # STEP 4: Create fairness optimizer
    # =================================
    
    optimizer = FairnessOptimizer(
        base_estimator=your_model,               # Your model
        sensitive_feature_names=['gender'],      # Column name(s) for sensitive attributes
        config=FairnessConfig(
            objective="equalized_odds",          # or "demographic_parity"
            mitigation="reduction",              # "none", "postprocess", or "reduction"
            constraints_eps=0.05
        ),
        random_state=42
    )
    
    # STEP 5: Train and evaluate
    # ===========================
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"\nTraining your model with fairness optimization...")
    optimizer.fit(X_train, y_train)
    
    print(f"Evaluating fairness and performance...")
    results = optimizer.evaluate(X_test, y_test)
    
    # STEP 6: View results
    # ====================
    
    print(f"\n📊 RESULTS:")
    print(f"Overall Accuracy: {results['overall']['accuracy']:.3f}")
    print(f"Accuracy Disparity: {results['disparities']['accuracy']:.3f}")
    print(f"Selection Rate Disparity: {results['disparities']['selection_rate']:.3f}")
    
    print(f"\nPerformance by Group:")
    print(results['by_group'][['accuracy', 'precision', 'recall']].round(3))
    
    return optimizer, results

# ================================
# METHOD 2: USING DIFFERENT MODELS
# ================================

def compare_your_models():
    """Example showing how to test multiple models"""
    print(f"\n🤖 METHOD 2: Comparing Different Models")
    print("="*50)
    
    # Load sample data (use your own here)
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    X['protected_group'] = np.random.choice([0, 1], size=len(X))
    
    # Define multiple models to test
    models_to_test = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=30),
        "SVM": SVC(probability=True),  # probability=True needed for fairness metrics
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=30)
    }
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    results_comparison = []
    
    for model_name, model in models_to_test.items():
        print(f"\nTesting {model_name}...")
        
        try:
            # Create optimizer with your model
            optimizer = FairnessOptimizer(
                base_estimator=model,
                sensitive_feature_names=['protected_group'],
                config=FairnessConfig(mitigation="none"),  # Start with baseline
                random_state=42
            )
            
            optimizer.fit(X_train, y_train)
            results = optimizer.evaluate(X_test, y_test)
            
            results_comparison.append({
                'Model': model_name,
                'Accuracy': results['overall']['accuracy'],
                'Accuracy_Disparity': results['disparities']['accuracy'],
                'Selection_Rate_Disparity': results['disparities']['selection_rate']
            })
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    # Show comparison
    if results_comparison:
        comparison_df = pd.DataFrame(results_comparison)
        print(f"\n📊 MODEL COMPARISON:")
        print(comparison_df.round(4))
        
        # Find best model
        best_idx = comparison_df['Accuracy_Disparity'].idxmin()
        best_model = comparison_df.loc[best_idx, 'Model']
        print(f"\n🏆 Most Fair Model: {best_model}")
    
    return results_comparison

# ================================
# METHOD 3: STEP-BY-STEP TEMPLATE
# ================================

def step_by_step_template():
    """Template you can copy for your own use"""
    print(f"\n📝 METHOD 3: Step-by-Step Template")
    print("="*50)
    
    print("Copy this template for your own use:")
    print()
    
    template_code = """
# TEMPLATE: Using FairnessOptimizer with your data and model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # or your preferred model
from ai_governance_framework.core.fairness.optimizer import FairnessOptimizer, FairnessConfig

# 1. LOAD YOUR DATA
# Replace this with your actual data loading
data = pd.read_csv("your_dataset.csv")  # or however you load your data

# 2. PREPARE DATA
X = data.drop('target_column', axis=1)  # Replace 'target_column' with your target
y = data['target_column']               # Should be binary (0/1)

# Handle missing values if needed
X = X.fillna(X.mean())  # or use more sophisticated imputation

# 3. DEFINE YOUR MODEL
your_model = RandomForestClassifier(n_estimators=100, random_state=42)
# You can use ANY sklearn model: LogisticRegression, SVC, etc.

# 4. CREATE FAIRNESS OPTIMIZER
optimizer = FairnessOptimizer(
    base_estimator=your_model,
    sensitive_feature_names=['gender'],  # Replace with your sensitive attribute column(s)
    config=FairnessConfig(
        objective="equalized_odds",      # or "demographic_parity"
        mitigation="reduction",          # or "none", "postprocess"
        constraints_eps=0.05
    )
)

# 5. TRAIN AND EVALUATE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

optimizer.fit(X_train, y_train)
results = optimizer.evaluate(X_test, y_test)

# 6. VIEW RESULTS
print(f"Accuracy: {results['overall']['accuracy']:.3f}")
print(f"Fairness (lower disparity = more fair): {results['disparities']['accuracy']:.3f}")
"""
    
    print(template_code)

if __name__ == "__main__":
    print("🎯 FAIRNESS OPTIMIZER: Using Your Own Models & Data")
    print("="*60)
    
    try:
        # Show different methods
        optimizer1, results1 = use_your_own_data()
        comparison = compare_your_models()
        step_by_step_template()
        
        print(f"\n" + "="*60)
        print("✅ QUICK SUMMARY:")
        print("="*60)
        print("To use your own data and model:")
        print("1. Load your data into a pandas DataFrame")
        print("2. Separate features (X) and target (y)")
        print("3. Choose any sklearn-compatible model")
        print("4. Create FairnessOptimizer with your model")
        print("5. Call optimizer.fit(X_train, y_train)")
        print("6. Evaluate with optimizer.evaluate(X_test, y_test)")
        print(f"\n🎉 That's it! Your model now has fairness analysis!")
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("Make sure you have all required packages installed.")
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

from ai_governance_framework.core.fairness.optimizer import FairnessOptimizer, FairnessConfig
from ai_governance_framework.core.fairness.visualizer import FairnessVisualizer

class CustomFairnessAnalyzer:
    """
    Easy-to-use interface for analyzing fairness with custom models and datasets
    """
    
    def __init__(self):
        self.visualizer = FairnessVisualizer()
        self.results = []
        
    def analyze_fairness(
        self, 
        model, 
        X, 
        y, 
        sensitive_features,
        test_size=0.3,
        fairness_strategies=None,
        random_state=42,
        model_name=None
    ):
        """
        Analyze fairness for a custom model and dataset
        
        Parameters:
        -----------
        model : sklearn estimator
            The machine learning model to analyze
        X : pd.DataFrame or np.array
            Features dataset
        y : pd.Series or np.array
            Target variable
        sensitive_features : list
            Names of sensitive features (columns in X)
        test_size : float
            Proportion of test set (default: 0.3)
        fairness_strategies : list
            List of fairness strategies to test (default: all)
        random_state : int
            Random seed for reproducibility
        model_name : str
            Name for the model (for display purposes)
        """
        
        print(f"\n{'='*80}")
        print(f"ANALYZING FAIRNESS FOR CUSTOM MODEL: {model_name or type(model).__name__}")
        print(f"{'='*80}")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name='target')
            
        # Basic data info
        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        print(f"Sensitive features: {sensitive_features}")
        
        # Preprocess data
        X_processed, scaler, encoders = self._preprocess_data(X, sensitive_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, stratify=y, test_size=test_size, random_state=random_state
        )
        
        # Default fairness strategies
        if fairness_strategies is None:
            fairness_strategies = [
                ("No Mitigation", FairnessConfig(mitigation="none")),
                ("Equalized Odds (In-process)", FairnessConfig(
                    objective="equalized_odds", 
                    mitigation="reduction", 
                    constraints_eps=0.02
                )),
                ("Demographic Parity (In-process)", FairnessConfig(
                    objective="demographic_parity", 
                    mitigation="reduction", 
                    constraints_eps=0.02
                ))
            ]
        
        # Test each fairness strategy
        model_results = []
        for strategy_name, config in fairness_strategies:
            print(f"\n{'-'*60}")
            print(f"Testing Strategy: {strategy_name}")
            print(f"{'-'*60}")
            
            try:
                # Create fairness optimizer
                optimizer = FairnessOptimizer(
                    base_estimator=model,
                    sensitive_feature_names=sensitive_features,
                    config=config,
                    random_state=random_state
                )
                
                # Fit and evaluate
                optimizer.fit(X_train, y_train)
                report = optimizer.evaluate(X_test, y_test)
                
                # Store results
                result = {
                    "model_name": model_name or type(model).__name__,
                    "strategy_name": strategy_name,
                    "optimizer": optimizer,
                    "report": report,
                    "overall_accuracy": report["overall"]["accuracy"],
                    "accuracy_disparity": report["disparities"]["accuracy"],
                    "selection_rate_disparity": report["disparities"]["selection_rate"],
                    "X_test": X_test,
                    "y_test": y_test
                }
                model_results.append(result)
                
                # Print summary
                print(f"Accuracy: {report['overall']['accuracy']:.4f}")
                print(f"Accuracy Disparity: {report['disparities']['accuracy']:.4f}")
                print(f"Selection Rate Disparity: {report['disparities']['selection_rate']:.4f}")
                
            except Exception as e:
                print(f"Error with {strategy_name}: {str(e)}")
        
        self.results.extend(model_results)
        return model_results
    
    def compare_models(
        self, 
        models_dict, 
        X, 
        y, 
        sensitive_features,
        test_size=0.3,
        random_state=42
    ):
        """
        Compare multiple models for fairness
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of {model_name: sklearn_estimator}
        X, y, sensitive_features : as in analyze_fairness
        """
        
        print(f"\n{'='*80}")
        print(f"COMPARING MULTIPLE MODELS FOR FAIRNESS")
        print(f"{'='*80}")
        
        all_results = []
        for model_name, model in models_dict.items():
            results = self.analyze_fairness(
                model=model,
                X=X,
                y=y,
                sensitive_features=sensitive_features,
                test_size=test_size,
                random_state=random_state,
                model_name=model_name
            )
            all_results.extend(results)
        
        # Generate comparison report
        self._generate_comparison_report(all_results)
        
        # Generate visualizations
        self._generate_visualizations(all_results, X, y, sensitive_features)
        
        return all_results
    
    def _preprocess_data(self, X, sensitive_features):
        """Preprocess the data"""
        X_processed = X.copy()
        
        # Handle missing values
        numerical_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_processed.select_dtypes(include=['object']).columns.tolist()
        
        encoders = {}
        
        # Impute missing values
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_processed[categorical_cols] = cat_imputer.fit_transform(X_processed[categorical_cols])
        
        if numerical_cols:
            num_imputer = SimpleImputer(strategy='median')
            X_processed[numerical_cols] = num_imputer.fit_transform(X_processed[numerical_cols])
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            encoders[col] = le
        
        # Scale numerical features (excluding sensitive features if they're numerical)
        scaler = StandardScaler()
        numerical_features_to_scale = [col for col in numerical_cols if col not in sensitive_features]
        
        if numerical_features_to_scale:
            X_processed[numerical_features_to_scale] = scaler.fit_transform(
                X_processed[numerical_features_to_scale]
            )
        
        return X_processed, scaler, encoders
    
    def _generate_comparison_report(self, results):
        """Generate comparison report"""
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Create summary DataFrame
        summary_data = []
        for result in results:
            summary_data.append({
                "Model": result["model_name"],
                "Strategy": result["strategy_name"],
                "Accuracy": result["overall_accuracy"],
                "Accuracy_Disparity": result["accuracy_disparity"],
                "Selection_Rate_Disparity": result["selection_rate_disparity"],
                "Fairness_Score": 1 - result["accuracy_disparity"]
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nDetailed Results:")
        print(summary_df.round(4))
        
        # Best performers
        print(f"\n{'='*50}")
        print("TOP PERFORMERS")
        print(f"{'='*50}")
        
        best_accuracy = summary_df.loc[summary_df["Accuracy"].idxmax()]
        print(f"\nHighest Accuracy:")
        print(f"  {best_accuracy['Model']} with {best_accuracy['Strategy']}")
        print(f"  Accuracy: {best_accuracy['Accuracy']:.4f}")
        print(f"  Disparity: {best_accuracy['Accuracy_Disparity']:.4f}")
        
        most_fair = summary_df.loc[summary_df["Accuracy_Disparity"].idxmin()]
        print(f"\nMost Fair (Lowest Disparity):")
        print(f"  {most_fair['Model']} with {most_fair['Strategy']}")
        print(f"  Accuracy: {most_fair['Accuracy']:.4f}")
        print(f"  Disparity: {most_fair['Accuracy_Disparity']:.4f}")
        
        best_balance = summary_df.loc[summary_df["Fairness_Score"].idxmax()]
        print(f"\nBest Overall Balance:")
        print(f"  {best_balance['Model']} with {best_balance['Strategy']}")
        print(f"  Accuracy: {best_balance['Accuracy']:.4f}")
        print(f"  Fairness Score: {best_balance['Fairness_Score']:.4f}")
    
    def _generate_visualizations(self, results, X, y, sensitive_features):
        """Generate visualizations for the results"""
        print(f"\n{'='*50}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*50}")
        
        if not results:
            print("No results to visualize.")
            return
        
        # Find best result for detailed analysis
        best_result = max(results, key=lambda x: x["overall_accuracy"] - x["accuracy_disparity"])
        
        try:
            # 1. Fairness-Accuracy Trade-off
            print("Creating fairness-accuracy trade-off plot...")
            self.visualizer.plot_fairness_accuracy_tradeoff(results)
            
            # 2. Group comparison for best model
            print("Creating group comparison analysis...")
            self.visualizer.plot_group_comparison(best_result["report"])
            
            # 3. Dataset bias analysis
            print("Creating bias analysis...")
            self.visualizer.plot_bias_analysis(X, y, sensitive_col=sensitive_features[0])
            
            # 4. Comprehensive dashboard
            print("Creating comprehensive fairness dashboard...")
            self.visualizer.create_fairness_dashboard(
                results, X, y, best_result, sensitive_col=sensitive_features[0]
            )
            
            print("All visualizations generated! 📊")
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")


def demo_with_synthetic_data():
    """Demo with synthetic dataset"""
    print("DEMO: Synthetic Dataset with Custom Models")
    print("="*50)
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Add a synthetic sensitive feature (gender)
    np.random.seed(42)
    gender = np.random.choice([0, 1], size=len(y), p=[0.4, 0.6])  # 40% female, 60% male
    
    # Create correlation between gender and outcome (introduce bias)
    bias_mask = (gender == 1) & (np.random.random(len(y)) < 0.2)
    y[bias_mask] = 1  # Males slightly more likely to get positive outcome
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df['gender'] = gender
    
    # Define models to compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    
    # Analyze fairness
    analyzer = CustomFairnessAnalyzer()
    results = analyzer.compare_models(
        models_dict=models,
        X=X_df,
        y=pd.Series(y),
        sensitive_features=['gender'],
        test_size=0.3,
        random_state=42
    )
    
    return results


def demo_with_custom_model_and_data():
    """Demo showing how to use your own model and data"""
    print("\nDEMO: Using Custom Model with Loan Dataset")
    print("="*50)
    
    # Load your existing data
    X = pd.read_csv("data.csv")
    y = (X.pop("Loan_Status") == "Y").astype(int)
    
    # Define your custom model
    # You can use ANY sklearn-compatible model here!
    custom_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42
    )
    
    # Analyze fairness
    analyzer = CustomFairnessAnalyzer()
    results = analyzer.analyze_fairness(
        model=custom_model,
        X=X,
        y=y,
        sensitive_features=['Gender'],
        model_name="Custom Neural Network"
    )
    
    return results


if __name__ == "__main__":
    # Run synthetic data demo
    synthetic_results = demo_with_synthetic_data()
    
    # Run custom model demo
    custom_results = demo_with_custom_model_and_data()
    
    print(f"\n{'='*80}")
    print("CUSTOM FAIRNESS ANALYSIS COMPLETE! 🎯")
    print("You can now easily analyze any model with any dataset!")
    print(f"{'='*80}")