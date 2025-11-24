
from openai import OpenAI
from ai_governance_chatbot.routed_agent_gemini.config import OPENAI_API_BASE, OPENAI_API_KEY
from ai_governance_chatbot.decision_logs_prompt import generate_prompt
from ai_governance_framework.api.endpoints import PromptRequest

from typing import List, Optional
from pydantic import BaseModel

class ReportFactor(BaseModel):
    factor_name: str
    description: str
    impact: str             
    value: Optional[str]      
    importance: Optional[float]

class ReportGlossary(BaseModel):
    term: str
    definition: str

class ReportImprovementPlan(BaseModel):
    improvement_suggestion_title: str
    improvement_suggestion_description: str
    timeline: Optional[str]
    difficulty: Optional[str]
    impact: Optional[str]

class Report(BaseModel):
    user_id: str
    decision_id: str
    decision_type: str
    outcome: str
    confidence: Optional[float]
    timestamp: Optional[str]
    summary: Optional[str]
    main_factors: List[ReportFactor]
    glossary: Optional[List[ReportGlossary]]
    improvement_plan: Optional[List[ReportImprovementPlan]]

def call_llm_gemini_report(user_content,
                    system_content,
                    max_completion_tokens=10000,
                    temperature=0,
                    client=None,
                    format_json=Report,
                    model_name='gemini-2.5-pro'):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content}
            ],
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature, 
            "response_format": format_json
        }
        response = client.chat.completions.parse(**kwargs)
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error during LLM processing: {e}")
        return None

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
    print(call_llm_gemini_report(
        user_content=response.formatted_prompt,
        system_content="You are an AI assistant that helps users understand AI decision logs. The logs have been provided using SHAP. Generate a comprehensive report based on the provided decision log. Strictly adhere to the structure and fields outlined in the Report schema. Ensure clarity and conciseness in your explanations. Explain terms in the glossary section. Provide actionable improvement suggestions in the improvement plan section.",
    ))
