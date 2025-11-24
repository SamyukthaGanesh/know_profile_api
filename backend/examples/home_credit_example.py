"""
Home Credit Example
Demonstrates the complete AI governance framework with Home Credit dataset.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import logging

# Import our framework modules
from data.base_loader import CSVDataLoader
from data.data_processor import DataProcessor
from models.model_wrapper import ModelWrapper

from core.explainability.shap_explainer import SHAPExplainer
from core.explainability.lime_explainer import LIMEExplainer
from core.fairness.statistical_parity import StatisticalParity
from core.fairness.equal_opportunity import EqualOpportunity
from core.fairness.calibration import CalibrationMetric
from core.fairness.bias_detector import BiasDetector, BiasSeverity

from core.literacy.prompt_generator import ExplanationContext, AudienceType
from core.literacy.user_prompts import UserPromptGenerator, UserProfile
from core.literacy.banker_prompts import BankerPromptGenerator, BankerProfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main example workflow"""
    
    print("="*80)
    print("HOME CREDIT - AI GOVERNANCE FRAMEWORK DEMO")
    print("="*80)
    
    # ============================================================
    # STEP 1: LOAD DATA
    # ============================================================
    print("\n[STEP 1] Loading Home Credit dataset...")
    
    # Update this path to your actual data location
    data_path = "/Users/muvarma/Documents/ghci hackathon/ai_governance_framework/examples/application_train.csv"  # Change this to your path
    
    data_loader = CSVDataLoader(
        data_path=data_path,
        target_column='TARGET',
        sensitive_feature_columns=['CODE_GENDER', 'DAYS_BIRTH']  # For fairness analysis
    )
    
    # Load data
    data = data_loader.load()
    print(f"✓ Loaded {len(data)} loan applications")
    
    # Get data info
    data_info = data_loader.get_data_info()
    print(f"✓ Features: {data_info['n_features']}")
    print(f"✓ Target distribution: {data_info['target_distribution']}")
    
    # ============================================================
    # STEP 2: PREPROCESS DATA
    # ============================================================
    print("\n[STEP 2] Preprocessing data...")
    
    # Select relevant features (to keep example manageable)
    selected_features = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
        'AMT_GOODS_PRICE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
        'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
        'HOUR_APPR_PROCESS_START', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]
    
    # Keep only selected features + target
    available_features = [f for f in selected_features if f in data.columns]
    data_subset = data[available_features + ['TARGET']].copy()
    
    # Sample for faster processing (remove this for full dataset)
    data_subset = data_subset.sample(n=min(10000, len(data_subset)), random_state=42)
    print(f"✓ Using {len(data_subset)} samples with {len(available_features)} features")
    
    # Create derived features
    data_subset['AGE'] = -data_subset['DAYS_BIRTH'] // 365
    data_subset['YEARS_EMPLOYED'] = -data_subset['DAYS_EMPLOYED'] // 365
    data_subset['CREDIT_INCOME_RATIO'] = data_subset['AMT_CREDIT'] / (data_subset['AMT_INCOME_TOTAL'] + 1)
    
    # Create age groups for fairness analysis
    data_subset['AGE_GROUP'] = pd.cut(
        data_subset['AGE'], 
        bins=[0, 25, 35, 45, 55, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '55+']
    )
    
    # Initialize processor
    processor = DataProcessor()
    
    # Separate features and target
    X = data_subset.drop(['TARGET'], axis=1)
    y = data_subset['TARGET']
    
    # Process data
    X_processed = processor.fit_transform(
        X,
        handle_missing=True,
        encode_categorical=True,
        scale_features=True,
        remove_outliers=False
    )
    
    print(f"✓ Processed data shape: {X_processed.shape}")
    
    # ============================================================
    # STEP 3: TRAIN MODEL
    # ============================================================
    print("\n[STEP 3] Training model...")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Wrap model
    model = ModelWrapper(
        model=rf_model,
        model_type='classification',
        feature_names=list(X_processed.columns),
        model_name='HomeCreditDefaultPredictor',
        model_version='1.0'
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    print("\n✓ Model Performance:")
    print(classification_report(y_test, y_pred, target_names=['Repaid', 'Default']))
    
    # ============================================================
    # STEP 4: EXPLAINABILITY - SHAP
    # ============================================================
    print("\n[STEP 4] Running SHAP Explainability Analysis...")
    
    # Create SHAP explainer
    shap_explainer = SHAPExplainer(
        model=rf_model,
        data=X_train.sample(n=min(100, len(X_train)), random_state=42),
        feature_names=list(X_processed.columns),
        mode='classification'
    )
    
    # Explain a single instance
    instance_idx = 0
    instance = X_test.iloc[instance_idx:instance_idx+1]
    
    shap_result = shap_explainer.explain_instance(instance)
    
    print("\n✓ SHAP Explanation for Test Instance:")
    print(f"  Prediction: {'Default' if y_pred[instance_idx] == 1 else 'Repaid'}")
    print(f"  Confidence: {y_proba[instance_idx].max():.2%}")
    print(f"  Base Value: {shap_result.base_value:.4f}")
    print("\n  Top 5 Contributing Factors:")
    for i, (feature, importance) in enumerate(shap_result.get_top_features(5), 1):
        direction = "increases" if importance > 0 else "decreases"
        print(f"  {i}. {feature}: {importance:.4f} ({direction} default risk)")
    
    # Global explanation
    global_shap = shap_explainer.explain_global(n_samples=100)
    print("\n✓ Global Feature Importance (SHAP):")
    for i, (feature, importance) in enumerate(list(global_shap.feature_importance.items())[:10], 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # ============================================================
    # STEP 5: EXPLAINABILITY - LIME
    # ============================================================
    print("\n[STEP 5] Running LIME Explainability Analysis...")
    
    lime_explainer = LIMEExplainer(
        model=rf_model,
        data=X_train.sample(n=min(1000, len(X_train)), random_state=42),
        feature_names=list(X_processed.columns),
        mode='classification'
    )
    
    lime_result = lime_explainer.explain_instance(instance, num_features=5)
    
    print("\n✓ LIME Explanation:")
    print(f"  Prediction: {'Default' if y_pred[instance_idx] == 1 else 'Repaid'}")
    print(f"  Local Model R²: {lime_result.coverage:.4f}")
    print("\n  Rules:")
    for rule in lime_result.rules[:5]:
        print(f"  • {rule}")
    
    # ============================================================
    # STEP 6: MULTI-DEMOGRAPHIC FAIRNESS ANALYSIS
    # ============================================================
    print("\n[STEP 6] Running Multi-Demographic Fairness Analysis...")
    
    # Create age groups from DAYS_BIRTH for analysis
    data_subset['AGE_GROUP'] = pd.cut(
        (-data_subset['DAYS_BIRTH'] / 365).astype(int), 
        bins=[0, 25, 35, 50, 65, 100], 
        labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder']
    )
    
    # Define multiple sensitive attributes to analyze
    sensitive_attributes = {
        'Gender': 'CODE_GENDER',
        'Age_Group': 'AGE_GROUP',
        'Income_Type': 'NAME_INCOME_TYPE',
        'Education': 'NAME_EDUCATION_TYPE',
        'Family_Status': 'NAME_FAMILY_STATUS'
    }
    
    # Filter attributes that exist in the dataset
    available_attributes = {}
    for attr_name, column_name in sensitive_attributes.items():
        if column_name in data_subset.columns:
            available_attributes[attr_name] = column_name
        else:
            print(f"⚠️  {attr_name} ({column_name}) not available in dataset")
    
    print(f"\n🔍 Analyzing {len(available_attributes)} demographic attributes...")
    
    # Collect all sensitive features for comprehensive analysis
    sensitive_features_dict = {}
    bias_summary = {}
    
    # Analyze each sensitive attribute individually first
    for attr_name, column_name in available_attributes.items():
        try:
            print(f"\n📊 Analyzing Bias for: {attr_name}")
            
            # Get the protected attribute values for test set
            protected_attribute = data_subset.loc[X_test.index, column_name]
            sensitive_features_dict[attr_name] = protected_attribute
            
            # Quick statistical parity check for preview
            stat_parity = StatisticalParity(model=rf_model, threshold=0.8)
            sp_result = stat_parity.calculate(X_test, y_test, protected_attribute, y_pred=y_pred)
            
            print(f"  ✓ Statistical Parity Score: {sp_result.overall_score:.4f}")
            print(f"  ✓ Bias Detected: {sp_result.bias_detected}")
            print(f"  ✓ Passes Threshold: {sp_result.passes_threshold}")
            
            # Store results for summary
            bias_summary[attr_name] = {
                'score': sp_result.overall_score,
                'bias_detected': sp_result.bias_detected,
                'num_groups': len(sp_result.group_scores)
            }
            
            # Show group breakdown
            if len(sp_result.group_scores) <= 6:  # Only show if not too many groups
                print(f"  ✓ Group Approval Rates:")
                for group, rate in sp_result.group_scores.items():
                    print(f"    {group}: {rate:.2%}")
            else:
                print(f"  ✓ Groups analyzed: {len(sp_result.group_scores)}")
                
        except Exception as e:
            print(f"  ❌ Error analyzing {attr_name}: {str(e)}")
            continue
    
    # ============================================================
    # STEP 7: COMPREHENSIVE MULTI-DEMOGRAPHIC BIAS DETECTION
    # ============================================================
    print("\n[STEP 7] Running Comprehensive Multi-Demographic Bias Detection...")
    
    # Run comprehensive bias detection for all available demographics
    if sensitive_features_dict:
        bias_detector = BiasDetector(
            model=rf_model,
            fairness_threshold=0.8,
            calibration_threshold=0.1,
            enable_statistical_parity=True,
            enable_equal_opportunity=True,
            enable_calibration=True
        )
        
        # Detect bias for all sensitive features at once
        try:
            comprehensive_reports = bias_detector.detect_bias(
                X=X_test,
                y_true=y_test,
                sensitive_features=sensitive_features_dict,
                y_pred=y_pred,
                y_proba=y_proba[:, 1]
            )
            
            # Display comprehensive results
            print(f"\n📊 MULTI-DEMOGRAPHIC BIAS SUMMARY:")
            print(f"{'='*60}")
            
            for attr_name, report in comprehensive_reports.items():
                status_icon = "✅" if report.severity == BiasSeverity.NONE else "⚠️"
                severity_color = "FAIR" if report.severity == BiasSeverity.NONE else report.severity.value.upper()
                
                print(f"{status_icon} {attr_name}:")
                print(f"    Fairness Score: {report.overall_fairness_score:.1f}/100")
                print(f"    Severity: {severity_color}")
                print(f"    Metrics Passed: {report.metrics_passed}/{report.metrics_passed + report.metrics_failed}")
                if report.priority_actions:
                    print(f"    Priority Action: {report.priority_actions[0]}")
                print()
            
            # Overall summary
            all_scores = [report.overall_fairness_score for report in comprehensive_reports.values()]
            avg_fairness = sum(all_scores) / len(all_scores) if all_scores else 0
            total_biased = sum(1 for report in comprehensive_reports.values() if report.severity != BiasSeverity.NONE)
            
            print(f"🎯 OVERALL ASSESSMENT:")
            print(f"    Average Fairness Score: {avg_fairness:.1f}/100")
            print(f"    Demographics with Bias: {total_biased}/{len(comprehensive_reports)}")
            print(f"    Model Fairness Status: {'FAIR' if total_biased == 0 else 'NEEDS ATTENTION'}")
            
        except Exception as e:
            print(f"❌ Error in comprehensive bias detection: {str(e)}")
            print("Falling back to individual analysis...")
            
            # Fallback: analyze gender only for demonstration
            if 'Gender' in sensitive_features_dict:
                gender_feature = sensitive_features_dict['Gender']
                comprehensive_report = bias_detector.detect_bias(
                    X=X_test,
                    y_true=y_test,
                    sensitive_features={'Gender': gender_feature},
                    y_pred=y_pred,
                    y_proba=y_proba[:, 1]
                )
                
                gender_report = comprehensive_report['Gender']
                print(f"\n✓ Sample Bias Report (Gender):")
                print(f"  Overall Fairness Score: {gender_report.overall_fairness_score:.1f}/100")
                print(f"  Severity: {gender_report.severity.value.upper()}")
                print(f"  Executive Summary: {gender_report.executive_summary}")
    else:
        print("⚠️ No sensitive attributes available for bias detection")
    
    # ============================================================
    # STEP 8: AI LITERACY - USER PROMPT
    # ============================================================
    print("\n[STEP 8] Generating User-Facing Explanation Prompt...")
    
    # Create user profile
    user_profile = UserProfile(
        user_id="USER_12345",
        literacy_level='beginner',
        preferred_language='en',
        customer_segment='existing'
    )
    
    # Create explanation context
    top_factors = []
    for feature, importance in shap_result.get_top_features(5):
        top_factors.append({
            'name': feature,
            'importance': abs(importance),
            'impact': 'positive' if importance > 0 else 'negative',
            'value': shap_result.feature_values.get(feature, 'N/A'),
            'shap_value': importance
        })
    
    explanation_context = ExplanationContext(
        decision_id="LOAN_APP_12345",
        decision_type="loan_application",
        outcome="approved" if y_pred[instance_idx] == 0 else "denied",
        confidence=float(y_proba[instance_idx].max()),
        model_name="HomeCreditDefaultPredictor",
        model_version="1.0",
        top_factors=top_factors,
        feature_values=shap_result.feature_values,
        feature_importance=shap_result.feature_importance,
        audience=AudienceType.USER_BEGINNER
    )
    
    # Generate user prompt
    user_prompt_gen = UserPromptGenerator()
    user_prompt = user_prompt_gen.generate_decision_explanation_prompt(
        context=explanation_context,
        user_profile=user_profile,
        include_improvement_tips=True
    )
    
    print("\n✓ User Explanation Prompt Generated:")
    print(f"  Audience: {user_profile.literacy_level}")
    print(f"  Tone: {user_prompt['tone_guidelines']['primary_tone']}")
    print(f"  Includes improvement tips: Yes")
    print(f"\n  Sample prompt excerpt:")
    print(user_prompt['formatted_prompt'][:500] + "...")
    
    # ============================================================
    # STEP 9: AI LITERACY - BANKER PROMPT
    # ============================================================
    print("\n[STEP 9] Generating Banker Technical Analysis Prompt...")
    
    # Create banker profile
    banker_profile = BankerProfile(
        banker_id="BANKER_789",
        role='technical_analyst',
        department='risk',
        expertise_level='senior'
    )
    
    # Generate banker prompt
    banker_prompt_gen = BankerPromptGenerator()
    banker_prompt = banker_prompt_gen.generate_technical_analysis_prompt(
        context=explanation_context,
        banker_profile=banker_profile,
        include_model_details=True,
        include_risk_metrics=True
    )
    
    print("\n✓ Banker Technical Analysis Prompt Generated:")
    print(f"  Role: {banker_profile.role}")
    print(f"  Department: {banker_profile.department}")
    print(f"  Includes model details: Yes")
    print(f"  Includes risk metrics: Yes")
    print(f"\n  Sample prompt excerpt:")
    print(banker_prompt['formatted_prompt'][:500] + "...")
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*80)
    print("DEMO COMPLETE - SUMMARY")
    print("="*80)
    print("\n✓ Successfully demonstrated:")
    print("  1. Data loading and preprocessing")
    print("  2. Model training and evaluation")
    print("  3. SHAP explainability (local and global)")
    print("  4. LIME explainability")
    print("  5. Fairness analysis (Statistical Parity, Equal Opportunity, Calibration)")
    print("  6. Comprehensive bias detection")
    print("  7. User-facing explanation prompt generation")
    print("  8. Banker technical analysis prompt generation")
    print("\n✓ Framework is model-agnostic and data-agnostic")
    print("✓ All components are modular and extensible")
    print("✓ Ready for integration with LLMs for natural language explanations")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()