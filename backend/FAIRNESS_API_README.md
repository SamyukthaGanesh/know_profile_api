# Fairness API Documentation

## Overview
Production-ready REST API for AI fairness analysis and bias mitigation using real Home Credit loan approval data. Provides comprehensive bias detection, optimization strategies, and regulatory compliance tracking.

## 🚀 Quick Start

### Start the API Server
```bash
cd ai_governance_framework
PYTHONPATH=/path/to/ai_governance_framework uvicorn api.endpoints:app --host 0.0.0.0 --port 8000
```

### Health Check
```bash
curl http://localhost:8000/fairness/health
```

## 📡 API Endpoints

### 1. Bias Analysis
**Endpoint:** `POST /fairness/analyze`

**Input:**
```json
{
  "model_id": "loan_model_v1",
  "features": [[age, income, credit_score], [...]],
  "labels": [0, 1, 0, 1],
  "predictions": [0, 1, 0, 1], 
  "probabilities": [0.2, 0.8, 0.3, 0.9],
  "sensitive_feature_name": "gender",
  "sensitive_feature_values": ["M", "F", "M", "F"]
}
```

**Output:**
```json
{
  "model_id": "loan_model_v1",
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

### 2. Bias Optimization
**Endpoint:** `POST /fairness/optimize`

**Input:**
```json
{
  "model_id": "loan_model_optimized",
  "features": [[...]], 
  "labels": [0, 1, ...],
  "sensitive_feature_name": "gender",
  "sensitive_feature_values": ["M", "F", ...],
  "mitigation_strategy": "reduction",
  "fairness_objective": "equalized_odds"
}
```

**Available Strategies:**
- `reduction` - In-processing fairness constraints
- `postprocess` - Threshold optimization
- `ensemble` - Multi-model fairness voting
- `none` - Baseline evaluation

**Output:**
```json
{
  "model_id": "loan_model_optimized",
  "optimization_successful": true,
  "fairness_improvement": 0.300,
  "new_fairness_score": 0.800,
  "optimization_summary": "Applied reduction with equalized_odds objective...",
  "timestamp": "2025-11-23T16:20:15.123456"
}
```

### 3. Retrieve Metrics
**Endpoint:** `GET /fairness/models/{model_id}/metrics`

### 4. Configuration
**Endpoint:** `GET /fairness/config/supported-metrics`
**Endpoint:** `GET /fairness/config/thresholds`

## 📁 Data Storage

**Output Directory:** `outputs/fairness_optimization/`

**Generated Files:**
- `analysis_{model_id}.json` - Bias analysis results
- `optimization_{model_id}.json` - Optimization results
- `model_health.json` - Overall health tracking

## 🗄️ Database Schema for Production

### 1. Model Registry
```sql
CREATE TABLE model_registry (
    model_id VARCHAR(100) PRIMARY KEY,
    model_name VARCHAR(200) NOT NULL,
    version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_type VARCHAR(100),
    dataset_source VARCHAR(200),
    feature_count INTEGER,
    training_data_hash VARCHAR(64),
    performance_baseline JSON
);
```

### 2. Fairness Analysis Ledger
```sql
CREATE TABLE fairness_analysis_ledger (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id VARCHAR(100) REFERENCES model_registry(model_id),
    overall_fairness_score DECIMAL(5,2) NOT NULL,
    bias_detected BOOLEAN NOT NULL,
    bias_severity VARCHAR(20) NOT NULL,
    sensitive_feature_name VARCHAR(100) NOT NULL,
    group_metrics JSON NOT NULL,
    recommendations JSON,
    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analyst_id VARCHAR(100),
    data_sample_size INTEGER,
    api_version VARCHAR(20)
);
```

### 3. Bias Optimization Ledger
```sql
CREATE TABLE bias_optimization_ledger (
    optimization_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id VARCHAR(100) REFERENCES model_registry(model_id),
    original_analysis_id UUID REFERENCES fairness_analysis_ledger(analysis_id),
    mitigation_strategy VARCHAR(50) NOT NULL,
    fairness_objective VARCHAR(50) NOT NULL,
    baseline_fairness_score DECIMAL(5,2),
    optimized_fairness_score DECIMAL(5,2),
    fairness_improvement DECIMAL(5,3),
    optimization_successful BOOLEAN NOT NULL,
    optimization_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_by VARCHAR(100),
    base_estimator_type VARCHAR(100),
    model_artifacts_path VARCHAR(500)
);
```

### 4. Model Health Monitoring
```sql
CREATE TABLE model_health_history (
    health_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id VARCHAR(100) REFERENCES model_registry(model_id),
    fairness_score DECIMAL(5,2),
    bias_detected BOOLEAN,
    last_analysis_date TIMESTAMP,
    prediction_volume_24h INTEGER,
    drift_detected BOOLEAN DEFAULT false,
    requires_retraining BOOLEAN DEFAULT false,
    health_status VARCHAR(20), -- 'healthy', 'warning', 'critical'
    check_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    automated_check BOOLEAN DEFAULT true
);
```

### 5. Audit Trail
```sql
CREATE TABLE fairness_audit_trail (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id VARCHAR(100),
    action_type VARCHAR(50), -- 'analysis', 'optimization', 'threshold_update'
    user_id VARCHAR(100),
    request_payload JSON,
    response_payload JSON,
    api_endpoint VARCHAR(200),
    execution_time_ms INTEGER,
    success BOOLEAN,
    error_message TEXT,
    audit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);
```

## 🔌 Frontend Integration

### React/JavaScript Example
```javascript
const fairnessAPI = {
  baseUrl: 'http://localhost:8000',
  
  async analyzeBias(modelData) {
    const response = await fetch(`${this.baseUrl}/fairness/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(modelData)
    });
    return response.json();
  },
  
  async optimizeModel(optimizationData) {
    const response = await fetch(`${this.baseUrl}/fairness/optimize`, {
      method: 'POST', 
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(optimizationData)
    });
    return response.json();
  }
};
```

## 🏥 Health & Monitoring

### Health Check Response
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

### Error Handling
- **400**: Invalid input data format
- **404**: Model not found
- **500**: Internal server error
- All errors include detailed error messages

## 📊 Production Considerations

### Performance
- **Recommended**: 10-20 samples minimum per protected group
- **Optimal**: 1000+ samples for reliable statistics
- **Processing**: ~1-2 seconds per 1000 samples

### Security
- Input validation on all endpoints
- Rate limiting recommended for production
- Audit logging for regulatory compliance

### Scaling
- Stateless API design for horizontal scaling
- Database connection pooling required
- Consider async processing for large datasets

## 🔧 Configuration

### Environment Variables
```bash
export FAIRNESS_API_LOG_LEVEL=INFO
export FAIRNESS_API_DATABASE_URL=postgresql://...
export FAIRNESS_API_REDIS_URL=redis://...
```

### Fairness Thresholds
- **Statistical Parity**: ≥0.8 (strict), ≥0.7 (moderate)
- **Equal Opportunity**: ≤0.1 (strict), ≤0.15 (moderate)
- **Calibration**: ≤0.1 (strict), ≤0.15 (moderate)

## 📈 Dashboard Integration

The API generates data for comprehensive fairness dashboards:
- Real-time bias monitoring
- Historical fairness trends
- Optimization impact tracking
- Regulatory compliance reporting
- Multi-demographic analysis views

---

**Status:** ✅ Production Ready with Home Credit Dataset Integration  
**Last Updated:** November 2025  
**API Version:** 1.0.0