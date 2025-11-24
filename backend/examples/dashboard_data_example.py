"""
Dashboard Data Example
Generates sample data in the exact format dashboard will receive from API.
Frontend team can use this to develop UI without backend running.
"""

import json
import os
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any


def generate_dashboard_overview() -> Dict[str, Any]:
    """Generate sample data for dashboard overview"""
    return {
        "total_decisions": 1247,
        "decisions_today": 53,
        "compliance_rate": 0.956,
        "fairness_score": 95.2,
        "active_models": 3,
        "pending_reviews": 7,
        "recent_alerts": [
            {
                "id": "ALERT_001",
                "type": "fairness",
                "severity": "medium",
                "message": "Slight drift detected in gender fairness metric",
                "timestamp": datetime.now().isoformat(),
                "action_required": True
            },
            {
                "id": "ALERT_002",
                "type": "compliance",
                "severity": "low",
                "message": "2 policies pending approval",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "action_required": False
            },
            {
                "id": "ALERT_003",
                "type": "model_health",
                "severity": "high",
                "message": "Fraud detection model accuracy dropped below threshold",
                "timestamp": (datetime.now() - timedelta(hours=5)).isoformat(),
                "action_required": True
            }
        ],
        "timestamp": datetime.now().isoformat()
    }


def generate_model_health() -> List[Dict[str, Any]]:
    """Generate sample model health data"""
    return [
        {
            "model_id": "home_credit_v1",
            "model_name": "Home Credit Default Predictor",
            "status": "healthy",
            "accuracy": 0.924,
            "fairness_score": 98.5,
            "last_prediction": (datetime.now() - timedelta(minutes=5)).isoformat(),
            "predictions_today": 53,
            "drift_detected": False,
            "requires_retraining": False,
            "metrics": {
                "precision": 0.89,
                "recall": 0.87,
                "f1_score": 0.88,
                "auc_roc": 0.92
            }
        },
        {
            "model_id": "fraud_detector_v2",
            "model_name": "Fraud Detection Model",
            "status": "warning",
            "accuracy": 0.887,
            "fairness_score": 92.1,
            "last_prediction": (datetime.now() - timedelta(hours=1)).isoformat(),
            "predictions_today": 28,
            "drift_detected": True,
            "requires_retraining": False,
            "metrics": {
                "precision": 0.85,
                "recall": 0.83,
                "f1_score": 0.84,
                "auc_roc": 0.89
            }
        },
        {
            "model_id": "credit_scorer_v3",
            "model_name": "Credit Score Predictor",
            "status": "healthy",
            "accuracy": 0.931,
            "fairness_score": 96.8,
            "last_prediction": (datetime.now() - timedelta(minutes=2)).isoformat(),
            "predictions_today": 67,
            "drift_detected": False,
            "requires_retraining": False,
            "metrics": {
                "precision": 0.91,
                "recall": 0.89,
                "f1_score": 0.90,
                "auc_roc": 0.94
            }
        }
    ]


def generate_compliance_dashboard() -> Dict[str, Any]:
    """Generate sample compliance dashboard data"""
    return {
        "total_policies": 6,
        "active_policies": 6,
        "compliance_rate": 0.956,
        "recent_violations": [
            {
                "decision_id": "LOAN_12847",
                "policy_id": "BASEL_III_CREDIT_001",
                "policy_name": "Minimum Credit Score for Large Loans",
                "timestamp": (datetime.now() - timedelta(hours=3)).isoformat(),
                "message": "Credit score below threshold for loan amount",
                "action_taken": "flagged_for_review"
            },
            {
                "decision_id": "LOAN_12851",
                "policy_id": "BASEL_III_INCOME_001",
                "policy_name": "Minimum Income Requirement",
                "timestamp": (datetime.now() - timedelta(hours=6)).isoformat(),
                "message": "Income below minimum for unsecured loan",
                "action_taken": "denied"
            }
        ],
        "policies_by_regulation": {
            "Basel III": 2,
            "Equal Credit Opportunity Act": 2,
            "GDPR": 1,
            "Fair Credit Reporting Act": 1
        },
        "audit_chain_status": "valid",
        "total_audit_receipts": 1247,
        "last_audit_verification": (datetime.now() - timedelta(hours=1)).isoformat()
    }


def generate_consent_dashboard() -> Dict[str, Any]:
    """Generate sample consent dashboard data"""
    return {
        "total_users_with_consent": 8542,
        "active_consents": 34168,
        "revoked_consents": 287,
        "consent_rate": 0.923,
        "consents_by_purpose": {
            "credit_decision": 8542,
            "fraud_detection": 7821,
            "marketing": 3245,
            "analytics": 6789,
            "model_training": 5432
        },
        "recent_consent_actions": [
            {
                "user_id": "USER_8901",
                "action": "granted",
                "data_field": "income",
                "purpose": "credit_decision",
                "timestamp": datetime.now().isoformat()
            },
            {
                "user_id": "USER_8902",
                "action": "revoked",
                "data_field": "employment_history",
                "purpose": "marketing",
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat()
            },
            {
                "user_id": "USER_8903",
                "action": "granted",
                "data_field": "credit_score",
                "purpose": "fraud_detection",
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat()
            }
        ]
    }


def generate_fairness_trend(days: int = 30) -> Dict[str, Any]:
    """Generate fairness trend chart data"""
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days-1, -1, -1)]
    
    # Generate realistic-looking trend data
    base_score = 95.0
    trend = [base_score + random.uniform(-2, 2) for _ in range(days)]
    
    gender_scores = [base_score + 3 + random.uniform(-1, 1) for _ in range(days)]
    age_scores = [base_score - 3 + random.uniform(-1, 1) for _ in range(days)]
    
    return {
        "labels": dates,
        "datasets": [
            {
                "label": "Overall Fairness",
                "data": trend,
                "borderColor": "#4CAF50",
                "backgroundColor": "rgba(76, 175, 80, 0.1)"
            },
            {
                "label": "Gender Fairness",
                "data": gender_scores,
                "borderColor": "#2196F3",
                "backgroundColor": "rgba(33, 150, 243, 0.1)"
            },
            {
                "label": "Age Fairness",
                "data": age_scores,
                "borderColor": "#FF9800",
                "backgroundColor": "rgba(255, 152, 0, 0.1)"
            }
        ]
    }


def generate_decision_explanation() -> Dict[str, Any]:
    """Generate sample decision explanation for UI"""
    return {
        "decision_id": "LOAN_12345",
        "user_id": "USER_5678",
        "outcome": "approved",
        "confidence": 0.87,
        "timestamp": datetime.now().isoformat(),
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
            "protected_attributes_used": ["CODE_GENDER", "AGE"],
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


def generate_user_consent_view(user_id: str) -> Dict[str, Any]:
    """Generate sample user consent view"""
    return {
        "user_id": user_id,
        "summary": {
            "total_consents": 12,
            "active_consents": 10,
            "revoked_consents": 2,
            "expired_consents": 0
        },
        "active_consents": [
            {
                "consent_id": "CONSENT_001",
                "data_field": "AMT_INCOME_TOTAL",
                "data_field_label": "Income Information",
                "purpose": "credit_decision",
                "purpose_label": "Credit Decisions",
                "granted_at": (datetime.now() - timedelta(days=30)).isoformat(),
                "expires_at": (datetime.now() + timedelta(days=335)).isoformat(),
                "can_revoke": True
            },
            {
                "consent_id": "CONSENT_002",
                "data_field": "DAYS_EMPLOYED",
                "data_field_label": "Employment History",
                "purpose": "credit_decision",
                "purpose_label": "Credit Decisions",
                "granted_at": (datetime.now() - timedelta(days=30)).isoformat(),
                "expires_at": (datetime.now() + timedelta(days=335)).isoformat(),
                "can_revoke": True
            },
            {
                "consent_id": "CONSENT_003",
                "data_field": "AMT_INCOME_TOTAL",
                "data_field_label": "Income Information",
                "purpose": "fraud_detection",
                "purpose_label": "Fraud Prevention",
                "granted_at": (datetime.now() - timedelta(days=15)).isoformat(),
                "expires_at": (datetime.now() + timedelta(days=350)).isoformat(),
                "can_revoke": True
            }
        ],
        "timeline": [
            {
                "timestamp": (datetime.now() - timedelta(days=30)).isoformat(),
                "action": "granted",
                "data_field": "Income Information",
                "purpose": "Credit Decisions"
            },
            {
                "timestamp": (datetime.now() - timedelta(days=15)).isoformat(),
                "action": "granted",
                "data_field": "Income Information",
                "purpose": "Fraud Prevention"
            }
        ]
    }


def save_all_examples():
    """Save all example data to JSON files for frontend team"""
    
    examples = {
        "dashboard_overview.json": generate_dashboard_overview(),
        "model_health.json": generate_model_health(),
        "compliance_dashboard.json": generate_compliance_dashboard(),
        "consent_dashboard.json": generate_consent_dashboard(),
        "fairness_trend.json": generate_fairness_trend(30),
        "decision_explanation.json": generate_decision_explanation(),
        "user_consent_view.json": generate_user_consent_view("USER_DEMO_001")
    }
    
    print("="*80)
    print("  GENERATING SAMPLE DATA FOR DASHBOARD TEAM")
    print("="*80)
    
    # Create organized output directory
    dashboard_output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'dashboard_data')
    os.makedirs(dashboard_output_dir, exist_ok=True)
    
    for filename, data in examples.items():
        output_path = os.path.join(dashboard_output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Created: {output_path}")
    
    print("\n" + "="*80)
    print("  ✅ All sample data files created!")
    print(f"  Location: {dashboard_output_dir}")
    print("  Frontend team can use these for UI development")
    print("="*80 + "\n")


def print_sample_structure():
    """Print structure of each endpoint response for documentation"""
    
    print("\n" + "="*80)
    print("  API RESPONSE STRUCTURES FOR DASHBOARD")
    print("="*80)
    
    print("\n1. GET /dashboard/overview")
    print("-" * 40)
    print(json.dumps(generate_dashboard_overview(), indent=2)[:300] + "...")
    
    print("\n2. GET /dashboard/models/health")
    print("-" * 40)
    print(json.dumps(generate_model_health()[0], indent=2)[:300] + "...")
    
    print("\n3. GET /dashboard/compliance")
    print("-" * 40)
    print(json.dumps(generate_compliance_dashboard(), indent=2)[:300] + "...")
    
    print("\n4. GET /dashboard/consent")
    print("-" * 40)
    print(json.dumps(generate_consent_dashboard(), indent=2)[:300] + "...")
    
    print("\n5. GET /dashboard/charts/fairness-trend")
    print("-" * 40)
    print(json.dumps(generate_fairness_trend(7), indent=2)[:300] + "...")
    
    print("\n6. GET /dashboard/user/{user_id}/consent-status")
    print("-" * 40)
    print(json.dumps(generate_user_consent_view("USER_001"), indent=2)[:400] + "...")


if __name__ == "__main__":
    import os
    
    # Create output directory
    os.makedirs("dashboard_samples", exist_ok=True)
    
    # Generate all sample files
    save_all_examples()
    
    # Print structure guide
    print_sample_structure()
    
    print("\n📚 For complete API documentation, see: DASHBOARD_API_GUIDE.md")