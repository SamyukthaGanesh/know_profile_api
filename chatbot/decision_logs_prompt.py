from datetime import datetime
from http.client import HTTPException

from fastapi import logger
from ai_governance_framework.api.endpoints import PromptRequest, PromptResponse
from ai_governance_framework.core.literacy.banker_prompts import BankerProfile, BankerPromptGenerator
from ai_governance_framework.core.literacy.prompt_generator import AudienceType, ExplanationContext
from ai_governance_framework.core.literacy.user_prompts import UserProfile, UserPromptGenerator

def generate_prompt(request: PromptRequest):
    """Generate explanation prompt for LLM"""
    try:
        context = ExplanationContext(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            outcome=request.outcome,
            confidence=request.confidence,
            model_name=request.model_name,
            model_version=request.model_version,
            top_factors=request.top_factors,
            feature_values=request.feature_values,
            feature_importance={f['name']: f.get('importance', 0) 
                              for f in request.top_factors},
            audience=AudienceType.USER_BEGINNER,
            fairness_metrics=request.fairness_metrics
        )
        
        if request.audience_type.startswith('user'):
            user_profile = UserProfile(
                user_id="API_USER",
                literacy_level=request.user_literacy_level or 'beginner',
                preferred_language='en'
            )
            prompt_gen = UserPromptGenerator()
            prompt_dict = prompt_gen.generate_decision_explanation_prompt(
                context=context,
                user_profile=user_profile,
                include_improvement_tips=request.include_improvement_tips
            )
        elif request.audience_type.startswith('banker'):
            banker_profile = BankerProfile(
                banker_id="API_BANKER",
                role=request.banker_role or 'technical_analyst',
                department='risk',
                expertise_level='senior'
            )
            prompt_gen = BankerPromptGenerator()
            prompt_dict = prompt_gen.generate_technical_analysis_prompt(
                context=context,
                banker_profile=banker_profile
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown audience type: {request.audience_type}"
            )
        
        return PromptResponse(
            prompt_id=f"{request.decision_id}_{request.audience_type}_{datetime.now().timestamp()}",
            formatted_prompt=prompt_dict['formatted_prompt'],
            audience=request.audience_type,
            prompt_metadata={
                "decision_id": request.decision_id,
                "outcome": request.outcome,
                "confidence": request.confidence,
                "num_factors": len(request.top_factors)
            },
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        print(f"Error generating prompt: {e}")
    

if __name__ == "__main__":
    sample_request = {
            "decision_id": "LOAN_12345",
            "decision_type": "loan_approval",
            "user_id": "USER_5678",
            "outcome": "approved",
            "confidence": 0.87,
            "timestamp": "2025-11-22T21:54:21.875704",
            "model": {
                "name": "Home Credit Default Predictor",
                "version": "1.0"
            },
            "explanation": {
                "method": "SHAP",
                "top_factors": [
                {
                    "name": "EXT_SOURCE_2",
                    "value": 0.75,
                    "importance": 0.032,
                    "impact": "positive",
                    "description": "External credit score is high"
                },
                {
                    "name": "AMT_INCOME_TOTAL",
                    "value": 65000,
                    "importance": 0.028,
                    "impact": "positive",
                    "description": "Income is above average"
                },
                {
                    "name": "DAYS_EMPLOYED",
                    "value": -2500,
                    "importance": 0.021,
                    "impact": "positive",
                    "description": "Long employment history"
                },
                {
                    "name": "AMT_CREDIT",
                    "value": 35000,
                    "importance": -0.015,
                    "impact": "negative",
                    "description": "Loan amount is moderate"
                },
                {
                    "name": "CODE_GENDER",
                    "value": "M",
                    "importance": 0.008,
                    "impact": "neutral",
                    "description": "Gender has minimal impact"
                }
                ]
            },
            "fairness": {
                "bias_detected": False,
                "fairness_score": 98.5,
                "protected_attributes_used": [
                "CODE_GENDER",
                "AGE"
                ],
                "message": "No significant bias detected"
            },
            "compliance": {
                "is_compliant": True,
                "policies_checked": 6,
                "violations": [],
                "audit_receipt_id": "AUDIT_abc123xyz"
            },
            "user_explanation": "Your loan application was approved based on your strong credit history and stable income. The main positive factors were your excellent external credit score and consistent employment record."
        }

    # Example usage
    sample_request = PromptRequest(
        decision_id=sample_request.get("decision_id"),
        decision_type=sample_request.get("decision_type"),
        outcome=sample_request.get("outcome"),
        confidence=sample_request.get("confidence"),
        model_name=sample_request.get("model", {}).get("name"),
        model_version=sample_request.get("model", {}).get("version"),
        top_factors=sample_request.get("explanation", {}).get("top_factors", []),
        feature_values={f['name']: f['value'] for f in sample_request.get("explanation", {}).get("top_factors", [])},
        audience_type="user_beginner",
        user_literacy_level="beginner",
        include_improvement_tips=True
    )
    
    response = generate_prompt(sample_request)
    print("Generated Prompt ID:", response.prompt_id)
    print("Formatted Prompt:\n", response.formatted_prompt)
