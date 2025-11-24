import sqlite3
from ai_governance_chatbot.decision_logs_prompt import generate_prompt
from ai_governance_chatbot.routed_agent_gemini.call_gemini_report import call_llm_gemini_report
from ai_governance_chatbot.routed_agent_gemini.fetch_tools import get_db_connection
from ai_governance_framework.api.endpoints import PromptRequest

def fetch_decision_prompt_request(decision_id: str):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Fetch decision log
        cursor.execute("""
            SELECT
                d.decision_id,
                d.user_id,
                d.outcome,
                d.confidence,
                d.timestamp,
                d.model_name,
                d.model_version,
                d.user_explanation
            FROM decision_logs d
            WHERE d.decision_id = ?
        """, (decision_id,))
        decision = cursor.fetchone()
        if not decision:
            raise ValueError(f"No decision found for id {decision_id}")

        # Fetch explanation factors
        cursor.execute("""
            SELECT e.explanation_id, e.method
            FROM explanations e
            WHERE e.decision_id = ?
        """, (decision_id,))
        explanation = cursor.fetchone()
        top_factors = []
        if explanation:
            explanation_id = explanation[0]
            cursor.execute("""
                SELECT name, value, importance, impact, description
                FROM explanation_factors
                WHERE explanation_id = ?
                ORDER BY ABS(importance) DESC
            """, (explanation_id,))
            top_factors = [
                {
                    "name": row[0],
                    "value": row[1],
                    "importance": row[2],
                    "impact": row[3],
                    "description": row[4]
                }
                for row in cursor.fetchall()
            ]
        feature_values = {f["name"]: f["value"] for f in top_factors}

        # You can similarly fetch fairness and compliance if needed
        # Build PromptRequest
        prompt_request = PromptRequest(
            decision_id=decision[1],
            decision_type="loan_approval", 
            outcome=decision[2],
            confidence=decision[3],
            model_name=decision[5],
            model_version=decision[6],
            top_factors=top_factors,
            feature_values=feature_values,
            audience_type="user_beginner",  # or fetch from user profile if available
            user_literacy_level="beginner", # or fetch from user profile if available
            include_improvement_tips=True
        )
        
        prompt_generated = generate_prompt(prompt_request)
        response = call_llm_gemini_report(
            user_content=prompt_generated.formatted_prompt,
            system_content="You are an AI assistant that helps users understand AI decision logs. The logs have been provided using SHAP. Generate a comprehensive report based on the provided decision log. Strictly adhere to the structure and fields outlined in the Report schema. Ensure clarity and conciseness in your explanations. Explain terms in the glossary section. Provide actionable improvement suggestions in the improvement plan section."
            )
        return response
    finally:
        conn.close()

if __name__ == "__main__":
    print(fetch_decision_prompt_request("LOAN_12345"))