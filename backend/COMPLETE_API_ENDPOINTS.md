# AI Governance Framework - Complete API Endpoints

## Overview
Complete REST API documentation for the AI Governance Framework including Fairness, Explainability, Compliance, and Dashboard endpoints.

## 🏥 Health Check

### Framework Health
**Endpoint:** `GET /health`
```json
{
  "status": "healthy",
  "service": "ai_governance_framework",
  "version": "1.0.0",
  "environment": "development",
  "uptime_seconds": 12345,
  "modules": {
    "fairness": "active",
    "explainability": "active", 
    "compliance": "active",
    "consent": "active"
  }
}
```

---

## ⚖️ Fairness Endpoints

### Analyze Bias
**Endpoint:** `POST /fairness/analyze`

**Input:**
```json
{
  "model_id": "home_credit_v1",
  "features": [[35, 45000, 650], [42, 55000, 720]],
  "labels": [0, 1],
  "predictions": [0, 1],
  "probabilities": [0.3, 0.8],
  "sensitive_feature_name": "gender",
  "sensitive_feature_values": ["M", "F"]
}
```

**Output:**
```json
{
  "model_id": "home_credit_v1",
  "overall_fairness_score": 94.19,
  "bias_detected": false,
  "bias_severity": "none",
  "group_metrics": [
    {"group_name": "M", "positive_rate": 0.45, "sample_size": 506},
    {"group_name": "F", "positive_rate": 0.42, "sample_size": 994}
  ],
  "recommendations": ["Monitor fairness metrics regularly"],
  "timestamp": "2025-11-23T16:19:28.536525"
}
```

### Optimize Fairness
**Endpoint:** `POST /fairness/optimize`

**Input:**
```json
{
  "model_id": "home_credit_optimized",
  "features": [[...]],
  "labels": [0, 1],
  "sensitive_feature_name": "gender", 
  "sensitive_feature_values": ["M", "F"],
  "mitigation_strategy": "reduction",
  "fairness_objective": "equalized_odds"
}
```

**Output:**
```json
{
  "model_id": "home_credit_optimized",
  "optimization_successful": true,
  "fairness_improvement": 0.300,
  "new_fairness_score": 0.800,
  "optimization_summary": "Applied reduction with equalized_odds objective...",
  "timestamp": "2025-11-23T16:20:15.123456"
}
```

### Get Model Metrics
**Endpoint:** `GET /fairness/models/{model_id}/metrics`

**Output:**
```json
{
  "model_id": "home_credit_v1",
  "overall_fairness_score": 94.19,
  "bias_detected": false,
  "bias_severity": "none",
  "group_metrics": [...],
  "timestamp": "2025-11-23T16:19:28.536525"
}
```

### Fairness Configuration
**Endpoint:** `GET /fairness/config/supported-metrics`

**Output:**
```json
{
  "metrics": [
    {"name": "statistical_parity", "description": "Measures demographic parity", "type": "independence"},
    {"name": "equal_opportunity", "description": "Equal true positive rates", "type": "separation"}
  ],
  "fairness_objectives": [
    {"name": "equalized_odds", "description": "Equal TPR and FPR across groups"},
    {"name": "demographic_parity", "description": "Equal selection rates"}
  ],
  "mitigation_strategies": [
    {"name": "reduction", "description": "In-processing fairness constraints"},
    {"name": "postprocess", "description": "Post-processing threshold optimization"}
  ]
}
```

**Endpoint:** `GET /fairness/config/thresholds`

**Output:**
```json
{
  "thresholds": {
    "statistical_parity": {"strict": 0.9, "moderate": 0.8, "lenient": 0.7},
    "equal_opportunity": {"strict": 0.05, "moderate": 0.1, "lenient": 0.15}
  },
  "severity_levels": {
    "none": "No bias detected",
    "low": "Minor fairness concerns", 
    "critical": "Severe bias requiring model review"
  }
}
```

### Fairness Health
**Endpoint:** `GET /fairness/health`

**Output:**
```json
{
  "status": "healthy",
  "service": "fairness",
  "version": "1.0.0",
  "components": {
    "bias_detector": "available",
    "fairness_optimizer": "available",
    "data_storage": "ready"
  }
}
```

---

## 🔍 Explainability Endpoints

### Generate Explanation
**Endpoint:** `POST /explainability/explain`

**Input:**
```json
{
  "model_id": "home_credit_v1",
  "instance_id": "loan_12345",
  "features": [35, 45000, 650, 1, 0],
  "feature_names": ["age", "income", "credit_score", "owns_car", "owns_house"],
  "prediction": 1,
  "prediction_probability": 0.85,
  "explanation_type": "shap"
}
```

**Output:**
```json
{
  "model_id": "home_credit_v1",
  "instance_id": "loan_12345",
  "explanation_type": "shap",
  "prediction": 1,
  "prediction_label": "Approve Loan",
  "confidence": 0.85,
  "base_value": 0.12,
  "feature_contributions": [
    {"feature": "income", "contribution": 0.45, "feature_value": 45000},
    {"feature": "credit_score", "contribution": 0.23, "feature_value": 650},
    {"feature": "age", "contribution": 0.15, "feature_value": 35}
  ],
  "explanation_text": "The model approved this loan primarily due to high income ($45,000) and good credit score (650).",
  "timestamp": "2025-11-23T16:25:30.123456"
}
```

### Global Explanation
**Endpoint:** `POST /explainability/explain-global`

**Input:**
```json
{
  "model_id": "home_credit_v1",
  "explanation_type": "shap",
  "feature_names": ["age", "income", "credit_score", "owns_car"],
  "sample_size": 1000
}
```

**Output:**
```json
{
  "model_id": "home_credit_v1",
  "explanation_type": "shap_global",
  "feature_importance": {
    "income": 0.45,
    "credit_score": 0.32, 
    "age": 0.15,
    "owns_car": 0.08
  },
  "summary": "Income is the most important factor for loan approval decisions, followed by credit score.",
  "timestamp": "2025-11-23T16:26:15.654321"
}
```

### Generate User-Friendly Explanation
**Endpoint:** `POST /explainability/explain-simple`

**Input:**
```json
{
  "model_id": "home_credit_v1",
  "instance_id": "loan_12345", 
  "user_profile": {
    "audience_type": "customer",
    "technical_level": "basic",
    "language": "english"
  },
  "explanation_data": {
    "prediction": 1,
    "top_factors": [
      {"feature": "income", "contribution": 0.45},
      {"feature": "credit_score", "contribution": 0.23}
    ]
  }
}
```

**Output:**
```json
{
  "model_id": "home_credit_v1",
  "instance_id": "loan_12345",
  "simple_explanation": "Your loan was approved! The main reasons were your good income level and solid credit history.",
  "detailed_factors": [
    "✅ Your annual income of $45,000 strongly supports approval",
    "✅ Your credit score of 650 is above our threshold", 
    "✅ Your age of 35 shows financial stability"
  ],
  "next_steps": "You can expect to receive your loan documents within 2 business days.",
  "confidence_level": "high",
  "timestamp": "2025-11-23T16:27:45.789123"
}
```

---

## 📋 Compliance Endpoints

### Check Compliance
**Endpoint:** `POST /compliance/check`

**Input:**
```json
{
  "model_id": "home_credit_v1",
  "regulation_type": "gdpr",
  "data_context": {
    "personal_data_used": true,
    "automated_decision": true,
    "high_risk_processing": false
  },
  "model_metadata": {
    "model_type": "classification",
    "decision_domain": "financial_services",
    "affects_individuals": true
  }
}
```

**Output:**
```json
{
  "model_id": "home_credit_v1",
  "regulation_type": "gdpr",
  "compliance_status": "compliant",
  "compliance_score": 95.5,
  "requirements_checked": 12,
  "requirements_passed": 11,
  "violations": [
    {
      "requirement": "right_to_explanation",
      "status": "warning",
      "description": "Explanations should be more detailed for automated decisions",
      "severity": "medium"
    }
  ],
  "recommendations": [
    "Implement detailed explanation system for loan decisions",
    "Add data retention policy documentation"
  ],
  "audit_trail_id": "audit_789456123",
  "timestamp": "2025-11-23T16:30:12.456789"
}
```

### Get Compliance Report
**Endpoint:** `GET /compliance/models/{model_id}/report`

**Output:**
```json
{
  "model_id": "home_credit_v1",
  "report_type": "comprehensive",
  "generated_at": "2025-11-23T16:31:00.123456",
  "compliance_summary": {
    "overall_score": 95.5,
    "gdpr_compliant": true,
    "ccpa_compliant": true,
    "ai_act_compliant": true
  },
  "regulation_details": {
    "gdpr": {
      "score": 96.0,
      "requirements_met": 11,
      "requirements_total": 12,
      "key_findings": ["Data minimization principle followed", "Consent properly documented"]
    }
  },
  "recommendations": [...],
  "next_review_date": "2025-12-23"
}
```

---

## 🛡️ Consent Management Endpoints

### Record Consent
**Endpoint:** `POST /consent/record`

**Input:**
```json
{
  "user_id": "user_12345",
  "consent_type": "data_processing",
  "purpose": "loan_application_analysis",
  "granted": true,
  "data_categories": ["financial_data", "personal_identifiers"],
  "retention_period": "7_years",
  "consent_method": "explicit_checkbox"
}
```

**Output:**
```json
{
  "consent_id": "consent_789123456",
  "user_id": "user_12345", 
  "status": "recorded",
  "consent_type": "data_processing",
  "granted": true,
  "recorded_at": "2025-11-23T16:35:00.123456",
  "expires_at": "2032-11-23T16:35:00.123456",
  "blockchain_hash": "0x1234567890abcdef...",
  "immutable": true
}
```

### Check Consent Status
**Endpoint:** `GET /consent/users/{user_id}/status`

**Output:**
```json
{
  "user_id": "user_12345",
  "active_consents": [
    {
      "consent_id": "consent_789123456",
      "consent_type": "data_processing", 
      "purpose": "loan_application_analysis",
      "granted": true,
      "recorded_at": "2025-11-23T16:35:00.123456",
      "expires_at": "2032-11-23T16:35:00.123456"
    }
  ],
  "withdrawal_history": [],
  "compliance_status": "valid"
}
```

---

## 📊 Dashboard Endpoints

### Dashboard Overview
**Endpoint:** `GET /dashboard/overview`

**Output:**
```json
{
  "summary": {
    "total_models": 5,
    "models_healthy": 4,
    "models_warning": 1,
    "predictions_today": 1247,
    "avg_fairness_score": 89.3
  },
  "recent_activity": [
    {
      "type": "bias_analysis",
      "model_id": "home_credit_v1",
      "message": "Fairness analysis completed - no bias detected",
      "timestamp": "2025-11-23T16:40:00.123456"
    }
  ],
  "alerts": [
    {
      "type": "fairness_warning",
      "model_id": "fraud_detector_v2", 
      "message": "Fairness score dropped below threshold",
      "severity": "medium",
      "timestamp": "2025-11-23T14:30:00.123456"
    }
  ],
  "timestamp": "2025-11-23T16:40:30.789123"
}
```

### Model Health Status
**Endpoint:** `GET /dashboard/models/health`

**Output:**
```json
[
  {
    "model_id": "home_credit_v1",
    "model_name": "Home Credit Default Predictor",
    "status": "healthy",
    "accuracy": 0.924,
    "fairness_score": 94.5,
    "last_prediction": "2025-11-23T16:39:45.123456",
    "predictions_today": 53,
    "drift_detected": false,
    "requires_retraining": false
  },
  {
    "model_id": "fraud_detector_v2",
    "model_name": "Fraud Detection Model", 
    "status": "warning",
    "accuracy": 0.887,
    "fairness_score": 78.1,
    "last_prediction": "2025-11-23T15:22:10.654321",
    "predictions_today": 28,
    "drift_detected": true,
    "requires_retraining": false
  }
]
```

### Compliance Dashboard
**Endpoint:** `GET /dashboard/compliance`

**Output:**
```json
{
  "compliance_overview": {
    "models_compliant": 4,
    "models_total": 5,
    "compliance_rate": 80.0,
    "last_audit": "2025-11-20T10:00:00.000000"
  },
  "regulation_compliance": {
    "gdpr": {"compliant_models": 5, "total_models": 5, "percentage": 100.0},
    "ccpa": {"compliant_models": 4, "total_models": 5, "percentage": 80.0},
    "ai_act": {"compliant_models": 3, "total_models": 5, "percentage": 60.0}
  },
  "recent_audits": [
    {
      "model_id": "home_credit_v1",
      "regulation": "gdpr",
      "status": "passed",
      "score": 95.5,
      "audit_date": "2025-11-23T16:30:00.123456"
    }
  ],
  "pending_actions": [
    {
      "model_id": "fraud_detector_v2",
      "action": "implement_explanation_system",
      "deadline": "2025-11-30",
      "priority": "high"
    }
  ]
}
```

---

## 🔧 Configuration Endpoints

### System Configuration
**Endpoint:** `GET /config/system`

**Output:**
```json
{
  "fairness_thresholds": {
    "statistical_parity_threshold": 0.8,
    "equal_opportunity_threshold": 0.1,
    "calibration_threshold": 0.1
  },
  "explanation_config": {
    "default_explainer": "shap",
    "max_features_display": 10,
    "explanation_timeout": 30
  },
  "compliance_config": {
    "default_regulations": ["gdpr", "ccpa"],
    "audit_frequency_days": 90,
    "auto_compliance_check": true
  }
}
```

---

## 📈 Analytics Endpoints

### Fairness Trends
**Endpoint:** `GET /analytics/fairness-trends`

**Output:**
```json
{
  "model_id": "home_credit_v1",
  "time_period": "30_days",
  "fairness_scores": [
    {"date": "2025-11-01", "score": 92.1},
    {"date": "2025-11-15", "score": 94.5},
    {"date": "2025-11-23", "score": 94.19}
  ],
  "trend": "stable",
  "average_score": 93.6,
  "min_score": 92.1,
  "max_score": 94.5
}
```

---

## 🚨 Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": 400,
    "type": "validation_error",
    "message": "Invalid input format",
    "details": "Feature array length mismatch",
    "timestamp": "2025-11-23T16:45:00.123456"
  }
}
```

### Common Error Codes
- **400**: Bad Request - Invalid input data
- **401**: Unauthorized - Authentication required  
- **404**: Not Found - Model or resource not found
- **429**: Rate Limited - Too many requests
- **500**: Internal Server Error - System error

---

**Total Endpoints:** 25+  
**Status:** ✅ Production Ready  
**API Version:** 1.0.0  
**Last Updated:** November 2025