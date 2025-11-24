# AI Governance Framework - Technical Documentation

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Advanced Fairness Module](#advanced-fairness-module)
3. [API Endpoints](#api-endpoints)
4. [Data Schemas](#data-schemas)
5. [Integration Examples](#integration-examples)
6. [Sample Data](#sample-data)

---

## 🚀 Quick Start

### Step 1: Start the Backend API
```bash
cd ai_governance_framework
pip install -r requirements.txt
python api/endpoints.py
```

**API will run on:** `http://localhost:8000`

**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI)

---

### Step 2: Test API Connection
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-11-22T12:00:00",
  "models_loaded": 0
}
```

---

### Step 3: Use Sample Data (While Backend is Being Integrated)
```bash
cd ai_governance_framework
python examples/dashboard_data_example.py
```

Creates sample JSON files in `outputs/dashboard_data/` for testing.

---

## � Advanced Fairness Module

### 🚀 Enhanced Fairness Optimization

The AI Governance Framework now includes an advanced fairness optimization module that provides comprehensive bias detection, mitigation, and analysis capabilities.

#### Key Features

✨ **Multiple Mitigation Strategies**
- No mitigation (baseline)
- Post-processing optimization
- In-processing with fairlearn reductions
- Ensemble methods (voting, bagging, boosting)
- Multi-objective optimization

🔬 **Statistical Rigor**
- Bootstrap confidence intervals
- Statistical significance testing
- Cross-validation with fairness tracking
- Robustness analysis

📊 **Comprehensive Visualizations**
- Fairness-accuracy trade-off plots
- Group performance comparisons
- Bias detection summaries
- Interactive fairness dashboards

#### Quick Usage

```python
from ai_governance_framework.core.fairness import (
    FairnessOptimizer, 
    FairnessConfig, 
    FairnessVisualizer
)

# Configure fairness optimization
config = FairnessConfig(
    mitigation='reduction',
    objective='equalized_odds',
    statistical_testing=True,
    confidence_intervals=True
)

# Create optimizer
optimizer = FairnessOptimizer(
    base_estimator=your_model,
    sensitive_feature_names=['gender', 'race'],
    config=config
)

# Train and evaluate
optimizer.fit(X_train, y_train)
results = optimizer.evaluate(X_test, y_test)

# Visualize results
visualizer = FairnessVisualizer()
visualizer.create_fairness_dashboard(results, save_path='fairness_report.png')
```

#### Examples and Migration

- **Advanced Usage**: `examples/advanced_fairness_example.py`
- **Migration Guide**: `examples/fairness_migration_demo.py`
- **Integration Tests**: `tests/test_fairness_integration.py`

---

## �🏗️ Project Overview

### Framework Components

✅ **Explainability** - SHAP, LIME, Anchors, Integrated Gradients implementations  
✅ **Advanced Fairness** - Comprehensive bias optimization with statistical testing  
✅ **Compliance** - Regulatory policy engine (Basel III, GDPR, ECOA)  
✅ **Consent** - User data consent management with blockchain-style audit trails  
✅ **AI Literacy** - Natural language prompt generation for different audiences  
✅ **Blockchain** - Cryptographic audit chains with tamper detection

---

## 🏗️ Project Overview

### What This Backend Provides

✅ **Explainability** - Why AI made a decision (SHAP, LIME)  
✅ **Advanced Fairness** - Bias detection, optimization, and statistical analysis  
✅ **Compliance** - Regulatory policy checking (Basel III, GDPR, ECOA)  
✅ **Consent** - User data consent management  
✅ **AI Literacy** - Generate natural language explanations  
✅ **Blockchain** - Tamper-proof audit trails  

---

## 📡 API Endpoints

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | API health status |
| `/predict` | POST | Model predictions |
| `/explain` | POST | SHAP/LIME explanations |
| `/prompts/generate` | POST | LLM prompt generation |
| `/fairness/analyze` | POST | Bias detection analysis |
| `/compliance/check` | POST | Regulatory compliance checking |
| `/compliance/policies` | GET | Available compliance policies |
| `/dashboard/overview` | GET | Dashboard summary metrics |
| `/dashboard/models/health` | GET | Model health status |
| `/dashboard/compliance` | GET | Compliance dashboard data |
| `/dashboard/consent` | GET | Consent management data |
| `/dashboard/charts/fairness-trend` | GET | Fairness trend chart data |
| `/dashboard/user/{user_id}/wallet` | GET | User consent wallet |
| `/blockchain/compliance/blocks` | GET | Compliance blockchain data |
| `/blockchain/consent/blocks/{user_id}` | GET | User consent blockchain |
| `/blockchain/graph/compliance` | GET | Blockchain graph visualization data |

**Interactive Documentation:** `http://localhost:8000/docs`

---

## 📊 Data Schemas

### 1. Dashboard Overview Response
```typescript
interface DashboardOverview {
  total_decisions: number;
  decisions_today: number;
  compliance_rate: number;        // 0-1 (e.g., 0.956 = 95.6%)
  fairness_score: number;         // 0-100
  active_models: number;
  pending_reviews: number;
  recent_alerts: Alert[];
  timestamp: string;              // ISO 8601
}

interface Alert {
  id: string;
  type: 'fairness' | 'compliance' | 'model_health' | 'consent';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: string;
  action_required: boolean;
}
```

**API:** `GET /dashboard/overview`

### 2. Model Health Response
```typescript
interface ModelHealth {
  model_id: string;
  model_name: string;
  status: 'healthy' | 'warning' | 'critical';
  accuracy: number;               // 0-1
  fairness_score: number;         // 0-100
  last_prediction: string;        // ISO 8601
  predictions_today: number;
  drift_detected: boolean;
  requires_retraining: boolean;
}
```

**API:** `GET /dashboard/models/health`

### 3. Explanation Response
```typescript
interface Explanation {
  method: 'shap' | 'lime';
  prediction: number | string;
  confidence: number;             // 0-1
  feature_importance: Record<string, number>;
  feature_values: Record<string, any>;
  top_features: TopFactor[];
  rules?: string[];               // LIME only
  base_value?: number;            // SHAP only
}

interface TopFactor {
  name: string;
  importance: number;
  value: any;
  impact?: 'positive' | 'negative';
  description?: string;
}
```

**API:** `POST /explain`

### 4. Compliance Check Response
```typescript
interface ComplianceCheck {
  decision_id: string;
  is_compliant: boolean;
  timestamp: string;
  policies_checked: number;
  violations_count: number;
  violations: Violation[];
  required_actions: Action[];
  audit_receipt_id: string;       // Blockchain receipt
  audit_hash: string;             // Cryptographic hash
}

interface Violation {
  policy_id: string;
  policy_name: string;
  regulation_source: string;      // e.g., "Basel III"
  message: string;
  recommended_action: string;
}
```

**API:** `POST /compliance/check`

### 5. Consent Wallet Response
```typescript
interface ConsentWallet {
  summary: {
    user_id: string;
    total_consents: number;
    active_consents: number;
    revoked_consents: number;
  };
  active_consents: ConsentRecord[];
  timeline: ConsentEvent[];
  wallet_verified: boolean;       // Blockchain verification
}

interface ConsentRecord {
  consent_id: string;
  data_field: string;             // e.g., "income", "credit_score"
  purpose: string;                // e.g., "credit_decision"
  granted_at: string;
  expires_at: string;
  is_valid: boolean;
  can_revoke: boolean;
}

interface ConsentEvent {
  timestamp: string;
  action: 'granted' | 'revoked';
  data_field: string;
  purpose: string;
}
```

**API:** `GET /dashboard/user/{user_id}/wallet`

### 6. Blockchain Response
```typescript
interface BlockchainData {
  chain_type: 'compliance' | 'consent';
  total_blocks: number;
  chain_valid: boolean;
  blocks: Block[];
  chain_metadata: ChainMetadata;
}

interface Block {
  block_number: number;
  block_id: string;
  timestamp: string;
  content_hash: string;           // SHA-256
  previous_hash: string;          // Links to previous block
  data: Record<string, any>;
  is_valid: boolean;
}

interface ChainMetadata {
  total_decisions?: number;
  compliant_blocks?: number;
  chain_errors: string[];
}
```

**API:** `GET /blockchain/compliance/blocks`

### 7. Graph Visualization Data
```typescript
interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata: Record<string, any>;
}

interface GraphNode {
  id: string;
  label: string;
  timestamp: string;
  hash: string;
  compliant?: boolean;
  color: string;                  // Hex color code
  size: number;
  data: Record<string, any>;
}

interface GraphEdge {
  from: string;                   // Node ID
  to: string;                     // Node ID
  label?: string;
  color: string;
  arrows?: 'to' | 'from' | 'both';
}
```

**API:** `GET /blockchain/graph/compliance`

## � Integration Examples

### Basic API Usage
### Dashboard Example
```javascript
// Fetch dashboard data
const response = await fetch('http://localhost:8000/dashboard/overview');
const dashboardData = await response.json();

// Use the data
console.log('Total decisions:', dashboardData.total_decisions);
console.log('Compliance rate:', dashboardData.compliance_rate);
```

### Explanation Example
```javascript
// Get model explanation
const explainResponse = await fetch('http://localhost:8000/explain', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model_name: "home_credit_model",
    features: {
      "AMT_INCOME_TOTAL": 50000,
      "AMT_CREDIT": 25000,
      "DAYS_EMPLOYED": -1500
    },
    method: "shap"
  })
});
const explanation = await explainResponse.json();
```

### Prompt Generation Example  
```javascript
// Generate LLM prompt
const promptResponse = await fetch('http://localhost:8000/prompts/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    decision_id: "LOAN_12345",
    outcome: "approved", 
    confidence: 0.85,
    top_factors: [
      { name: "income", importance: 0.3, value: 50000 },
      { name: "credit_score", importance: 0.25, value: 0.7 }
    ],
    audience_type: "user_beginner"
  })
});
const promptData = await promptResponse.json();
// Use promptData.formatted_prompt with your LLM
```

## 📦 Sample Data

**Location:** `outputs/dashboard_data/`

### Available JSON Files
- `dashboard_overview.json` - Main dashboard metrics
- `model_health.json` - Model performance data  
- `compliance_dashboard.json` - Compliance status
- `consent_dashboard.json` - Consent metrics
- `fairness_trend.json` - Chart data for fairness trends
- `decision_explanation.json` - Sample model explanations
- `user_consent_view.json` - User consent details
