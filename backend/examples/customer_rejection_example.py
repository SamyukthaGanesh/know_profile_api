#!/usr/bin/env python3
"""
Example: Using AI Governance Framework for Customer Loan Rejection Explanations
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai_governance_framework.core.explainability.shap_explainer import SHAPExplainer
from ai_governance_framework.core.literacy.user_prompts import UserPromptGenerator, UserProfile, ExplanationContext, AudienceType
import pandas as pd
import numpy as np

def explain_loan_rejection(model, customer_data, feature_names, customer_id="CUST_12345"):
    """
    Generate customer-friendly explanation for loan rejection
    
    Args:
        model: Trained ML model
        customer_data: Single customer's feature values
        feature_names: List of feature names
        customer_id: Unique customer identifier
    
    Returns:
        dict: Customer explanation with plain language reasoning
    """
    
    # 1. Get model prediction
    prediction = model.predict(customer_data.reshape(1, -1))[0]
    probabilities = model.predict_proba(customer_data.reshape(1, -1))[0]
    
    # Only proceed if loan is rejected (prediction = 1 for default)
    if prediction == 0:  # Loan approved
        return {"status": "approved", "explanation": "Loan was approved"}
    
    print(f"🔍 Analyzing rejection for Customer {customer_id}...")
    
    # 2. Generate SHAP explanation
    shap_explainer = SHAPExplainer(
        model=model,
        data=pd.DataFrame(customer_data.reshape(1, -1), columns=feature_names),
        mode='classification'
    )
    
    # Get SHAP values for this customer
    shap_result = shap_explainer.explain_instance(customer_data)
    
    # 3. Extract top factors that led to rejection
    top_rejection_factors = []
    for feature, importance in shap_result.get_top_features(5):
        if importance > 0:  # Positive SHAP = increases default risk = bad for approval
            top_rejection_factors.append({
                'name': feature,
                'shap_value': importance,
                'customer_value': shap_result.feature_values.get(feature, 'N/A'),
                'impact': 'increases rejection risk'
            })
    
    # 4. Create user-friendly explanation context
    explanation_context = ExplanationContext(
        decision_id=f"LOAN_REJ_{customer_id}",
        decision_type="loan_application",
        outcome="denied",
        confidence=float(probabilities[1]),  # Probability of default
        model_name="LoanDecisionModel",
        model_version="1.0",
        top_factors=top_rejection_factors,
        feature_values=shap_result.feature_values,
        feature_importance=shap_result.feature_importance,
        audience=AudienceType.USER_BEGINNER
    )
    
    # 5. Create customer profile
    user_profile = UserProfile(
        user_id=customer_id,
        literacy_level='beginner',
        preferred_language='en',
        customer_segment='prospective'
    )
    
    # 6. Generate customer-friendly explanation prompt
    user_prompt_gen = UserPromptGenerator()
    explanation_prompt = user_prompt_gen.generate_decision_explanation_prompt(
        context=explanation_context,
        user_profile=user_profile,
        include_improvement_tips=True
    )
    
    # 7. Format final explanation for customer
    customer_explanation = {
        "status": "rejected",
        "confidence": f"{probabilities[1]*100:.1f}%",
        "primary_reasons": [
            {
                "factor": factor['name'],
                "explanation": f"Your {factor['name']} contributed to this decision",
                "impact_score": factor['shap_value'],
                "your_value": factor['customer_value']
            }
            for factor in top_rejection_factors[:3]  # Top 3 reasons
        ],
        "improvement_suggestions": extract_improvement_tips(top_rejection_factors),
        "full_prompt": explanation_prompt['formatted_prompt'],
        "next_steps": [
            "Review the factors that affected your application",
            "Consider improving the highlighted areas",
            "You may reapply after addressing these factors",
            "Contact our customer service for personalized guidance"
        ]
    }
    
    return customer_explanation

def extract_improvement_tips(rejection_factors):
    """Convert technical rejection factors into actionable customer advice"""
    tips = []
    
    for factor in rejection_factors:
        factor_name = factor['name'].lower()
        
        if 'credit' in factor_name or 'ext_source' in factor_name:
            tips.append("Consider improving your credit score by paying bills on time and reducing debt")
        elif 'income' in factor_name:
            tips.append("Provide additional documentation of stable income sources")
        elif 'employment' in factor_name or 'employed' in factor_name:
            tips.append("Ensure employment history is well-documented and stable")
        elif 'age' in factor_name or 'birth' in factor_name:
            tips.append("Age-related factors may improve with employment stability over time")
        elif 'annuity' in factor_name or 'amount' in factor_name:
            tips.append("Consider adjusting your loan amount to better match your income")
        else:
            tips.append(f"Work on improving your {factor['name']} profile")
    
    return list(set(tips))  # Remove duplicates

# Example usage
if __name__ == "__main__":
    print("="*60)
    print("CUSTOMER LOAN REJECTION EXPLANATION DEMO")
    print("="*60)
    
    # This would be called when a customer's loan is rejected
    # customer_explanation = explain_loan_rejection(
    #     model=your_trained_model,
    #     customer_data=rejected_customer_features,
    #     feature_names=your_feature_names,
    #     customer_id="CUST_789"
    # )
    
    # Sample output structure
    sample_explanation = {
        "status": "rejected",
        "confidence": "78.3%",
        "primary_reasons": [
            {
                "factor": "EXT_SOURCE_2",
                "explanation": "Your credit score contributed to this decision",
                "impact_score": 0.045,
                "your_value": -0.63
            },
            {
                "factor": "DAYS_EMPLOYED",
                "explanation": "Your employment history contributed to this decision", 
                "impact_score": 0.032,
                "your_value": 500
            }
        ],
        "improvement_suggestions": [
            "Consider improving your credit score by paying bills on time",
            "Ensure employment history is well-documented and stable"
        ],
        "next_steps": [
            "Review the factors that affected your application",
            "Consider improving the highlighted areas", 
            "You may reapply after addressing these factors"
        ]
    }
    
    print("\n📋 Sample Customer Explanation:")
    print(f"Status: {sample_explanation['status']}")
    print(f"Model Confidence: {sample_explanation['confidence']}")
    print("\n🔍 Primary Reasons for Rejection:")
    for i, reason in enumerate(sample_explanation['primary_reasons'], 1):
        print(f"  {i}. {reason['explanation']}")
        print(f"     Impact: {reason['impact_score']:.3f}")
    
    print("\n💡 Improvement Suggestions:")
    for tip in sample_explanation['improvement_suggestions']:
        print(f"  • {tip}")
        
    print("\n✅ Your framework is perfect for this use case!")