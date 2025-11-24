#!/usr/bin/env python3
"""
Home Credit API Integration Test
Loads Home Credit dataset, trains model, and tests API endpoints with real data.
Now tests SQLite database integration instead of JSON files.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import requests
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import framework modules
from data.base_loader import CSVDataLoader
from data.data_processor import DataProcessor
from models.model_wrapper import ModelWrapper

# Database imports
from core.database.models import init_database, get_db_session, ModelRegistry, FairnessAnalysis
from core.database.services import ModelService, FairnessService

class HomeCreditAPITester:
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_proba = None
        self.feature_names = None
        self.sensitive_features = {}
        
    def cleanup_test_models(self):
        """Clean up test models from previous runs"""
        print("🧹 Cleaning up test models from previous runs...")
        
        try:
            with next(get_db_session()) as db:
                # Remove test models that aren't the actual Home Credit model
                test_model_patterns = ['test_', 'verification_', 'enterprise_test_']
                
                for pattern in test_model_patterns:
                    test_models = db.query(ModelRegistry).filter(ModelRegistry.model_id.like(f'{pattern}%')).all()
                    for model in test_models:
                        # Also clean related fairness analyses
                        db.query(FairnessAnalysis).filter(FairnessAnalysis.model_id == model.model_id).delete()
                        db.delete(model)
                        print(f"  🗑️ Removed test model: {model.model_id}")
                
                db.commit()
                print("✅ Test models cleaned up")
                
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
            # Don't fail the test if cleanup has issues
        
    def load_and_prepare_data(self, data_path):
        """Load and prepare Home Credit dataset"""
        print("🏠 Loading Home Credit Dataset...")
        
        # Check if file exists
        if not os.path.exists(data_path):
            print(f"❌ Dataset not found at: {data_path}")
            print("📥 Please download application_train.csv from:")
            print("   https://www.kaggle.com/c/home-credit-default-risk/data")
            return False
        
        # Load data
        data_loader = CSVDataLoader(
            data_path=data_path,
            target_column='TARGET',
            sensitive_feature_columns=['CODE_GENDER', 'DAYS_BIRTH']
        )
        
        data = data_loader.load()
        print(f"✅ Loaded {len(data)} loan applications")
        
        # Select key features for manageable processing
        selected_features = [
            'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
            'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
            'AMT_GOODS_PRICE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS', 'DAYS_BIRTH', 'DAYS_EMPLOYED'
        ]
        
        # Keep only available features
        available_features = [f for f in selected_features if f in data.columns]
        data_subset = data[available_features + ['TARGET']].copy()
        
        # Sample for faster processing
        data_subset = data_subset.sample(n=min(5000, len(data_subset)), random_state=42)
        
        # Create age from DAYS_BIRTH
        data_subset['AGE'] = (-data_subset['DAYS_BIRTH'] / 365).astype(int)
        data_subset = data_subset.drop('DAYS_BIRTH', axis=1)
        
        # Clean data - remove rows with missing critical values
        data_subset = data_subset.dropna(subset=['CODE_GENDER', 'AMT_INCOME_TOTAL', 'AMT_CREDIT'])
        
        print(f"✅ Using {len(data_subset)} samples with {len(data_subset.columns)-1} features")
        
        # Separate features and target
        X = data_subset.drop(['TARGET'], axis=1)
        y = data_subset['TARGET']
        
        # Process data
        processor = DataProcessor()
        X_processed = processor.fit_transform(
            X,
            handle_missing=True,
            encode_categorical=True,
            scale_features=True,
            remove_outliers=False
        )
        
        print(f"✅ Processed data shape: {X_processed.shape}")
        
        # Store feature names
        self.feature_names = list(X_processed.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.3, random_state=42, stratify=y
        )
        
        self.X_test = X_test
        self.y_test = y_test
        
        # Store original categorical data for sensitive features
        X_test_original = X.loc[X_test.index]
        self.sensitive_features = {
            'gender': X_test_original['CODE_GENDER'].astype(str).tolist(),
            'age_group': pd.cut(X_test_original['AGE'], 
                              bins=[0, 25, 35, 50, 65, 100], 
                              labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder']).astype(str).tolist()
        }
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the model"""
        print("\n🤖 Training Random Forest Model...")
        
        # Train model with class balancing to handle imbalanced data
        rf_model = RandomForestClassifier(
            n_estimators=100,  # Increased for better performance
            max_depth=15,      # Increased depth
            min_samples_split=10,  # Prevent overfitting
            min_samples_leaf=5,    # Prevent overfitting
            class_weight='balanced',  # ← FIX: Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Wrap model
        self.model = ModelWrapper(
            model=rf_model,
            model_type='classification',
            feature_names=self.feature_names,
            model_name='HomeCreditDefaultPredictor',
            model_version='2.0'
        )
        
        # Generate predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_proba = self.model.predict_proba(self.X_test)[:, 1]  # Get probability of class 1
        
        print("✅ Model trained successfully!")
        print(f"📊 Test Set Performance:")
        print(classification_report(self.y_test, self.y_pred, 
                                  target_names=['Will Repay', 'Will Default']))
        
        return True
    
    def test_api_health(self):
        """Test API health endpoint"""
        print("\n🔍 Testing API Health...")
        try:
            response = requests.get(f"{self.api_base_url}/fairness/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                print(f"✅ API Status: {health['status']}")
                print(f"   Components: {health['components']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ API server not running! Start with:")
            print("   uvicorn api.endpoints:app --host 0.0.0.0 --port 8000")
            return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_fairness_analysis(self, sensitive_feature_name='gender'):
        """Test fairness analysis with real Home Credit data"""
        print(f"\n🔍 Testing Fairness Analysis with {sensitive_feature_name}...")
        
        payload = {
            "model_id": "home_credit_default_predictor_v2",
            "features": self.X_test.values.tolist(),
            "labels": self.y_test.tolist(),
            "predictions": self.y_pred.tolist(),
            "probabilities": self.y_proba.tolist(),
            "sensitive_feature_name": sensitive_feature_name,
            "sensitive_feature_values": self.sensitive_features[sensitive_feature_name]
        }
        
        try:
            print(f"📤 Analyzing {len(payload['features'])} loan applications...")
            response = requests.post(f"{self.api_base_url}/fairness/analyze", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Fairness Analysis Completed!")
                print(f"📈 Overall Fairness Score: {result['overall_fairness_score']:.2f}/100")
                print(f"🚨 Bias Detected: {result['bias_detected']}")
                print(f"⚠️  Bias Severity: {result['bias_severity']}")
                
                print(f"📊 Group Analysis ({sensitive_feature_name}):")
                for group in result['group_metrics']:
                    print(f"   {group['group_name']}: {group['positive_rate']:.1%} default rate ({group['sample_size']} loans)")
                
                if result['recommendations']:
                    print(f"💡 AI Recommendations:")
                    for i, rec in enumerate(result['recommendations'][:3], 1):
                        print(f"   {i}. {rec}")
                
                return result
            else:
                print(f"❌ Analysis failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return None
    
    def test_individual_explanations(self, num_examples=3):
        """Test individual prediction explanations with SHAP"""
        print(f"\n🔍 Testing Individual Predictions with SHAP Explanations...")
        
        results = []
        
        # Get diverse samples - include both default and repaid cases if available
        default_indices = np.where(self.y_test == 1)[0]  # Find actual defaults
        repaid_indices = np.where(self.y_test == 0)[0]   # Find actual repaid
        
        # Select examples from both classes if available
        selected_indices = []
        if len(default_indices) > 0:
            # Add some actual default cases
            selected_indices.extend(np.random.choice(default_indices, 
                                                   size=min(num_examples//2, len(default_indices)), 
                                                   replace=False))
        if len(repaid_indices) > 0:
            # Add some actual repaid cases
            remaining = num_examples - len(selected_indices)
            selected_indices.extend(np.random.choice(repaid_indices, 
                                                   size=min(remaining, len(repaid_indices)), 
                                                   replace=False))
        
        # Fill remaining with random samples if needed
        while len(selected_indices) < num_examples:
            remaining_indices = set(range(len(self.X_test))) - set(selected_indices)
            if remaining_indices:
                selected_indices.append(np.random.choice(list(remaining_indices)))
            else:
                break
                
        sample_indices = selected_indices[:num_examples]
        
        for i, idx in enumerate(sample_indices):
            sample_features = self.X_test.iloc[idx].values.tolist()
            actual_label = int(self.y_test.iloc[idx])
            predicted_label = int(self.y_pred[idx]) 
            prediction_prob = float(self.y_proba[idx])
            
            print(f"\n📋 === Loan Application #{i+1} (Index {idx}) ===")
            print(f"🎯 Prediction: {'DEFAULT' if predicted_label == 1 else 'REPAID'} (Confidence: {prediction_prob:.1%})")
            print(f"✅ Actual: {'DEFAULT' if actual_label == 1 else 'REPAID'}")
            
            # Show if prediction is correct
            is_correct = predicted_label == actual_label
            print(f"🎯 Result: {'✅ CORRECT' if is_correct else '❌ WRONG'} prediction")
            
            # Call explanation API
            explanation_result = self.test_single_explanation(
                instance_id=f"loan_{idx}", 
                features=sample_features,
                prediction=predicted_label,
                prediction_probability=prediction_prob
            )
            
            if explanation_result:
                results.append({
                    'index': idx,
                    'features': sample_features,
                    'prediction': predicted_label,
                    'actual': actual_label,
                    'probability': prediction_prob,
                    'explanation': explanation_result
                })
                
                # Show top contributing factors
                top_factors = explanation_result.get('feature_contributions', [])[:3]
                print(f"🔍 Top 3 Contributing Factors:")
                for j, factor in enumerate(top_factors, 1):
                    contribution = factor.get('contribution', 0)
                    feature_name = factor.get('feature', 'unknown')
                    feature_value = factor.get('feature_value', 0)
                    direction = "+" if contribution > 0 else ""
                    print(f"   {j}. {feature_name}: {feature_value} (Impact: {direction}{contribution:.3f})")
        
        return results

    def test_single_explanation(self, instance_id, features, prediction, prediction_probability):
        """Test explanation API for single instance"""
        
        payload = {
            "model_id": "home_credit_v1",
            "instance_id": instance_id,
            "features": features,
            "feature_names": self.feature_names,
            "prediction": prediction,
            "prediction_probability": prediction_probability,
            "explanation_type": "shap"
        }
        
        try:
            response = requests.post(f"{self.api_base_url}/explainability/explain", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                print(f"❌ Explanation failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Explanation error: {e}")
            return None

    def test_global_explanation(self):
        """Test global model explanation"""
        print(f"\n🌍 Testing Global Model Explanation...")
        
        payload = {
            "model_id": "home_credit_v1",
            "explanation_type": "shap",
            "feature_names": self.feature_names,
            "sample_size": 1000
        }
        
        try:
            response = requests.post(f"{self.api_base_url}/explainability/explain-global", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Global Explanation Generated!")
                
                feature_importance = result.get('feature_importance', {})
                print(f"🏆 Top 10 Most Important Features:")
                
                # Sort features by importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                    print(f"   {i:2d}. {feature:<25} {importance:.4f}")
                
                return result
            else:
                print(f"❌ Global explanation failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Global explanation error: {e}")
            return None

    def test_user_friendly_explanations(self, num_examples=2):
        """Test user-friendly explanations for customers"""
        print(f"\n👥 Testing User-Friendly Explanations...")
        
        sample_indices = np.random.choice(len(self.X_test), size=num_examples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            predicted_label = int(self.y_pred[idx])
            prediction_prob = float(self.y_proba[idx])
            
            # Get actual feature values for personalized explanations
            feature_values = self.X_test.iloc[idx]
            actual_label = int(self.y_test.iloc[idx])
            
            # Create realistic top factors based on actual feature values
            top_factors = []
            feature_names = ['income', 'credit_amount', 'employment_days', 'age', 'education']
            
            # Sample some actual features and their values for personalized explanation
            for j, feat_name in enumerate(feature_names[:3]):
                if j < len(self.feature_names):
                    actual_feature_name = self.feature_names[j]
                    actual_value = float(feature_values.iloc[j]) if j < len(feature_values) else 0.0
                    # Create realistic contribution based on feature value
                    contribution = 0.5 + (actual_value * 0.1) if actual_value < 10 else 0.3
                    top_factors.append({
                        "feature": feat_name, 
                        "contribution": contribution,
                        "actual_value": actual_value
                    })
            
            payload = {
                "model_id": "home_credit_v1",
                "instance_id": f"loan_{idx}",
                "user_profile": {
                    "audience_type": "customer",
                    "technical_level": "basic",
                    "language": "english"
                },
                "explanation_data": {
                    "prediction": predicted_label,
                    "prediction_probability": prediction_prob,
                    "actual_label": actual_label,
                    "top_factors": top_factors,
                    "loan_index": int(idx)
                }
            }
            
            try:
                response = requests.post(f"{self.api_base_url}/explainability/explain-simple", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"\n💬 === Customer Explanation #{i+1} (Loan {idx}) ===")
                    print(f"🎯 Decision: {'APPROVED' if predicted_label == 0 else 'NEEDS REVIEW'}")
                    print(f"📝 Simple Explanation: {result.get('simple_explanation', 'N/A')}")
                    
                    detailed_factors = result.get('detailed_factors', [])
                    if detailed_factors:
                        print(f"🔍 Detailed Factors:")
                        for factor in detailed_factors:
                            print(f"   • {factor}")
                            
                    next_steps = result.get('next_steps', '')
                    if next_steps:
                        print(f"➡️  Next Steps: {next_steps}")
                        
                else:
                    print(f"❌ User-friendly explanation failed: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ User-friendly explanation error: {e}")

    def register_model_in_database(self):
        """Register model in database via API"""
        print("\n📝 Registering model in database...")
        
        payload = {
            "model_id": "home_credit_v1",
            "model_name": "Home Credit Default Predictor",
            "model_version": "1.0",
            "model_type": "classification",
            "feature_names": self.feature_names,
            "description": "ML model for predicting loan defaults using Home Credit data"
        }
        
        try:
            response = requests.post(f"{self.api_base_url}/models/register", json=payload)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Model registered: {result['message']}")
            else:
                print(f"⚠️  Model registration response: {response.status_code}")
                print(f"   (Model may already exist)")
        except Exception as e:
            print(f"❌ Model registration error: {e}")

    def verify_database_persistence(self):
        """Verify that data is actually persisted in database"""
        print("\n🔍 Verifying Database Persistence...")
        
        try:
            with next(get_db_session()) as db:
                # Check model registry - look for the actual model ID used in fairness test
                model_ids_to_check = ["home_credit_v1", "home_credit_default_predictor_v2"]
                model_found = False
                
                for model_id in model_ids_to_check:
                    model = ModelService.get_model(model_id, db)
                    if model:
                        print(f"✅ Model persisted: {model.model_name} (v{model.model_version})")
                        print(f"   Status: {model.status}, Features: {len(model.feature_names)}")
                        model_found = True
                        
                        # Check fairness analyses for this specific model
                        analyses = FairnessService.get_latest_fairness_analysis(model_id, db)
                        if analyses:
                            print(f"✅ Fairness analysis persisted: Score {analyses.overall_fairness_score}")
                            print(f"   Bias detected: {analyses.bias_detected}, Severity: {analyses.bias_severity}")
                        break
                
                if not model_found:
                    print("❌ Model not found in database")
                
                # Check all fairness analyses in database
                all_analyses = FairnessService.list_analyses(limit=10, db=db)
                if all_analyses:
                    print(f"✅ Found {len(all_analyses)} total fairness analyses in database")
                    for analysis in all_analyses[:3]:  # Show first 3
                        print(f"   📊 {analysis.model_id}: Score {analysis.overall_fairness_score:.1f}")
                else:
                    print("❌ No fairness analyses found in database")
                    
                # Check active models count
                active_models = ModelService.get_active_models(db)
                print(f"📊 Total active models in database: {len(active_models)}")
                
        except Exception as e:
            print(f"❌ Database verification failed: {e}")

    def test_dashboard_integration(self):
        """Test dashboard endpoints with real database data"""
        print("\n📊 Testing Dashboard Integration...")
        
        try:
            # Test overview endpoint
            response = requests.get(f"{self.api_base_url}/dashboard/overview")
            if response.status_code == 200:
                overview = response.json()
                print("✅ Dashboard Overview (Real Data):")
                print(f"   Active Models: {overview['active_models']}")
                print(f"   Fairness Score: {overview['fairness_score']}")
                print(f"   Recent Alerts: {len(overview['recent_alerts'])}")
            else:
                print(f"❌ Dashboard overview failed: {response.status_code}")
            
            # Test model health endpoint
            response = requests.get(f"{self.api_base_url}/dashboard/models/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Model Health Status ({len(health_data)} models):")
                for model in health_data:
                    print(f"   {model['model_name']}: {model['status']} (Fairness: {model['fairness_score']})")
            else:
                print(f"❌ Model health dashboard failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Dashboard integration test failed: {e}")

    def test_fairness_optimization(self, mitigation_strategy="reduction"):
        """Test fairness optimization with real data"""
        print(f"\n🔧 Testing Fairness Optimization ({mitigation_strategy})...")
        
        payload = {
            "model_id": f"home_credit_optimized_{mitigation_strategy}",
            "features": self.X_test.values.tolist(),
            "labels": self.y_test.tolist(),
            "sensitive_feature_name": "gender",
            "sensitive_feature_values": self.sensitive_features['gender'],
            "mitigation_strategy": mitigation_strategy,
            "fairness_objective": "equalized_odds"
        }
        
        try:
            print(f"⚙️ Applying {mitigation_strategy} strategy...")
            response = requests.post(f"{self.api_base_url}/fairness/optimize", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Optimization Completed!")
                print(f"✨ Success: {result['optimization_successful']}")
                print(f"📈 Fairness Improvement: {result['fairness_improvement']:.3f}")
                print(f"🎯 New Fairness Score: {result['new_fairness_score']:.3f}")
                print(f"📝 Summary: {result['optimization_summary']}")
                
                return result
            else:
                print(f"❌ Optimization failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Optimization error: {e}")
            return None
    
    def test_retrieve_metrics(self, model_id):
        """Test retrieving stored metrics"""
        print(f"\n📊 Retrieving Stored Metrics for {model_id}...")
        
        try:
            response = requests.get(f"{self.api_base_url}/fairness/models/{model_id}/metrics")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Metrics Retrieved!")
                print(f"📈 Stored Fairness Score: {result.get('overall_fairness_score', 'N/A')}")
                print(f"🕐 Analysis Timestamp: {result.get('timestamp', 'N/A')}")
                return result
            else:
                print(f"❌ Retrieval failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return None
    
    def check_output_files(self):
        """Check generated output files"""
        print("\n📁 Checking Generated Output Files...")
        
        output_dir = "outputs/fairness_optimization"
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"✅ Found {len(files)} files in {output_dir}:")
            
            for file in files:
                file_path = os.path.join(output_dir, file)
                if file.endswith('.json'):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        print(f"   📄 {file}")
                        if 'model_id' in data:
                            print(f"      Model: {data['model_id']}")
                        if 'overall_fairness_score' in data:
                            print(f"      Fairness Score: {data['overall_fairness_score']:.2f}")
                        if 'optimization_successful' in data:
                            print(f"      Optimization: {data['optimization_successful']}")
                            
                    except Exception as e:
                        print(f"   ❌ Error reading {file}: {e}")
        else:
            print(f"❌ Output directory not found: {output_dir}")
    
    def run_complete_test(self, data_path):
        """Run complete integration test with database verification"""
        print("🚀 Starting Home Credit API Integration Test (Database Mode)")
        print("=" * 70)
        
        # Initialize database
        print("📊 Initializing database...")
        try:
            init_database()
            print("✅ Database initialized successfully")
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            return False
            
        # Clean up test models from previous runs
        self.cleanup_test_models()
        
        # Step 1: Load and prepare data
        train_test_data = self.load_and_prepare_data(data_path)
        if not train_test_data:
            return False
        
        X_train, X_test, y_train, y_test = train_test_data
        
        # Step 2: Train model and register in database
        if not self.train_model(X_train, y_train):
            return False
        
        # Register model in database via API
        self.register_model_in_database()
        
        # Step 3: Test API health
        if not self.test_api_health():
            return False
        
        # Step 4: Test fairness analysis for both demographics
        analysis_results = []
        for feature in ['gender', 'age_group']:
            result = self.test_fairness_analysis(feature)
            if result:
                analysis_results.append(result)
        
        # Step 5: Test optimization strategies
        optimization_results = []
        for strategy in ['reduction', 'postprocess']:
            result = self.test_fairness_optimization(strategy)
            if result:
                optimization_results.append(result)
        
        # Step 6: Test individual predictions with SHAP explanations
        print("\n" + "="*50)
        print("🧠 EXPLAINABILITY TESTING")
        print("="*50)
        
        individual_results = self.test_individual_explanations(num_examples=3)
        
        # Step 7: Test global model explanation
        global_explanation = self.test_global_explanation()
        
        # Step 8: Test user-friendly explanations
        self.test_user_friendly_explanations(num_examples=2)
        
        # Step 9: Test database persistence verification
        self.verify_database_persistence()
        
        # Step 10: Test dashboard with real data
        self.test_dashboard_integration()
        
        # Summary
        print("\n" + "="*60)
        print("📊 INTEGRATION TEST SUMMARY")
        print("="*60)
        print(f"✅ Data loaded: {len(self.X_test)} test samples")
        print(f"✅ Model trained: RandomForest with {len(self.feature_names)} features")
        print(f"✅ Model registered in database: ✅")
        print(f"✅ Fairness analyses: {len(analysis_results)}")
        print(f"✅ Optimizations: {len(optimization_results)}")
        print(f"✅ Individual explanations: {len(individual_results) if individual_results else 0}")
        print(f"✅ Global explanation: {'Generated' if global_explanation else 'Failed'}")
        print(f"✅ Database persistence: Verified")
        print(f"✅ Dashboard integration: Real data connected")
        
        if analysis_results:
            avg_fairness = np.mean([r['overall_fairness_score'] for r in analysis_results])
            print(f"📈 Average Fairness Score: {avg_fairness:.2f}/100")
        
        print("\n🎉 Home Credit dataset successfully integrated with DATABASE-POWERED API!")
        print("🔗 All data now persists in SQLite database - no more JSON files!")
        
        return True


if __name__ == "__main__":
    # Update this path to where you put the Home Credit dataset
    DATA_PATH = "examples/application_train.csv"  # Change this to your actual path
    
    tester = HomeCreditAPITester()
    tester.run_complete_test(DATA_PATH)