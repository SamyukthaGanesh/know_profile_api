# Backend Integration Guide for Teammates

## 🎯 Current State: Mock Data → Need Real APIs

All admin dashboard pages currently use **hardcoded mock data**. Your teammates need to create FastAPI endpoints to replace this.

---

## 📊 Required Backend Endpoints

### 1. **Model Health Monitor**

#### Endpoint: `GET /api/admin/models/health`
```python
# app/routers/admin.py

@router.get("/api/admin/models/health")
async def get_model_health():
    return {
        "models": [
            {
                "id": "model-1",
                "name": "Credit Scoring Model v2.1",
                "status": "healthy",  # "healthy" | "warning" | "critical"
                "accuracy": 94.3,
                "drift": 2.1,  # Percentage
                "lastTraining": "2024-11-15",
                "predictions": 45231,
                "avgLatency": 45  # milliseconds
            }
            # ... more models
        ],
        "performanceHistory": [
            {
                "date": "Nov 15",
                "accuracy": 93.5,
                "precision": 92.1,
                "recall": 94.2
            }
            # ... more history
        ],
        "driftAnalysis": [
            {
                "feature": "Income",
                "drift": 1.8  # Percentage
            }
            # ... more features
        ]
    }
```

**What they need to calculate:**
- Model accuracy (from validation set)
- Drift score (compare current data distribution vs training data)
- Predictions count (from logs/database)
- Avg latency (from monitoring)

---

### 2. **Fairness Monitor**

#### Endpoint: `GET /api/admin/fairness/metrics`
```python
@router.get("/api/admin/fairness/metrics")
async def get_fairness_metrics():
    return {
        "overallBiasScore": 0.023,  # Lower is better, < 0.03 is good
        "protectedGroups": [
            {
                "group": "Gender",
                "biasScore": 0.012,
                "approvalRate": 68.5,
                "trend": "improving"  # "improving" | "stable" | "worsening"
            },
            {
                "group": "Age",
                "biasScore": 0.031,
                "approvalRate": 65.2,
                "trend": "stable"
            }
            # Add: Income Level, Location, etc.
        ],
        "rlOptimizer": {
            "episodes": 1247,  # Total RL training episodes
            "lastAdjustment": {
                "action": "Reduced weight on age feature",
                "biasReduction": 2.8,  # Percentage
                "accuracyImpact": -0.3  # Percentage (negative = decreased)
            }
        },
        "biasHistory": [
            {
                "episode": 0,
                "bias": 0.089,
                "accuracy": 92.1
            }
            # ... history of RL training
        ]
    }
```

**What they need to calculate:**
- **Bias Score**: Disparate Impact Ratio or Statistical Parity Difference
  - Example: `abs(P(approve|group_A) - P(approve|group_B))`
- **Protected Groups**: Gender, Age (18-25, 26-40, 41+), Income Level, Location
- **RL Optimizer**: If they have an RL model for fairness, track episodes & adjustments

---

### 3. **Approvals Queue**

#### Endpoint: `GET /api/admin/approvals/queue`
```python
@router.get("/api/admin/approvals/queue")
async def get_approvals_queue():
    return {
        "approvals": [
            {
                "id": "appr-001",
                "type": "model_update",  # "model_update" | "policy_change" | "threshold_adjustment" | "feature_deployment"
                "title": "Credit Scoring Model v2.2 Deployment",
                "description": "New model with improved accuracy",
                "requestedBy": "ML Team",
                "requestedAt": "2024-11-21T10:30:00Z",
                "priority": "high",  # "high" | "medium" | "low"
                "status": "pending",  # "pending" | "approved" | "rejected"
                "details": {
                    "accuracyImprovement": 0.8,
                    "biasReduction": 1.2,
                    "backtestingPassed": True,
                    "fairnessPassed": True
                }
            }
            # ... more approvals
        ]
    }
```

#### Endpoint: `POST /api/admin/approvals/{approval_id}/approve`
```python
@router.post("/api/admin/approvals/{approval_id}/approve")
async def approve_request(approval_id: str):
    # Update approval status in database
    # Trigger the approved action (deploy model, change policy, etc.)
    return {"status": "approved", "approval_id": approval_id}
```

#### Endpoint: `POST /api/admin/approvals/{approval_id}/reject`
```python
@router.post("/api/admin/approvals/{approval_id}/reject")
async def reject_request(approval_id: str):
    # Update approval status in database
    return {"status": "rejected", "approval_id": approval_id}
```

**What they need:**
- Database/storage for approval requests
- Logic to queue model updates, policy changes
- Human-in-the-loop workflow

---

## 🔧 How to Replace Mock Data in Frontend

### Step 1: Update the Component to Fetch from API

**Example for Model Health** (`src/pages/admin/ModelHealth.tsx`):

```typescript
// BEFORE (Mock data):
useEffect(() => {
  const mockModels = [...];
  setModels(mockModels);
  setLoading(false);
}, []);

// AFTER (Real API):
useEffect(() => {
  const fetchModelHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/admin/models/health');
      const data = await response.json();
      setModels(data.models);
      setPerformanceData(data.performanceHistory);
      setDriftData(data.driftAnalysis);
    } catch (error) {
      console.error('Failed to fetch model health:', error);
    } finally {
      setLoading(false);
    }
  };
  
  fetchModelHealth();
}, []);
```

### Step 2: Create API Service (Optional, but cleaner)

**Create** `src/services/adminApi.ts`:
```typescript
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const adminApi = {
  async getModelHealth() {
    const response = await fetch(`${API_BASE}/api/admin/models/health`);
    return response.json();
  },
  
  async getFairnessMetrics() {
    const response = await fetch(`${API_BASE}/api/admin/fairness/metrics`);
    return response.json();
  },
  
  async getApprovalsQueue() {
    const response = await fetch(`${API_BASE}/api/admin/approvals/queue`);
    return response.json();
  },
  
  async approveRequest(approvalId: string) {
    const response = await fetch(`${API_BASE}/api/admin/approvals/${approvalId}/approve`, {
      method: 'POST'
    });
    return response.json();
  },
  
  async rejectRequest(approvalId: string) {
    const response = await fetch(`${API_BASE}/api/admin/approvals/${approvalId}/reject`, {
      method: 'POST'
    });
    return response.json();
  }
};
```

**Then use it in components:**
```typescript
import { adminApi } from '../../services/adminApi';

useEffect(() => {
  const loadData = async () => {
    try {
      const data = await adminApi.getModelHealth();
      setModels(data.models);
    } catch (error) {
      console.error('Failed to load:', error);
    } finally {
      setLoading(false);
    }
  };
  loadData();
}, []);
```

---

## 📝 What Your Teammates Need to Implement

### Priority 1: Model Monitoring (If they have ML models)
- [ ] Track model accuracy, precision, recall
- [ ] Calculate data drift (compare distributions)
- [ ] Log prediction counts and latency
- [ ] Create `/api/admin/models/health` endpoint

### Priority 2: Fairness Metrics (If doing fairness work)
- [ ] Calculate bias scores for protected groups
- [ ] Track approval rates by demographic
- [ ] If using RL for fairness, log episodes & adjustments
- [ ] Create `/api/admin/fairness/metrics` endpoint

### Priority 3: Approvals Queue (If implementing governance)
- [ ] Create database schema for approvals
- [ ] Workflow for submitting approval requests
- [ ] Endpoints for approve/reject actions
- [ ] Create `/api/admin/approvals/queue` endpoint

---

## 🚀 Quick Integration Steps

### For Your Teammates (Backend):
1. Create a new file: `app/routers/admin.py`
2. Add the router to FastAPI main app
3. Implement the endpoints above
4. Return the data in the exact format shown

### For You (Frontend):
1. Replace the mock data in `useEffect` with real API calls
2. Test with `http://localhost:8000/api/admin/...`
3. Handle loading states and errors

---

## 🧪 Testing the Integration

### Test Backend Endpoints First:
```bash
# Test Model Health
curl http://localhost:8000/api/admin/models/health

# Test Fairness Metrics
curl http://localhost:8000/api/admin/fairness/metrics

# Test Approvals Queue
curl http://localhost:8000/api/admin/approvals/queue
```

### Then Test Frontend:
1. Open browser console (F12)
2. Navigate to admin pages
3. Check Network tab for API calls
4. Verify data is loading correctly

---

## 💡 Tips for Integration

1. **Start Simple**: Get one endpoint working first, then expand
2. **Use Real Data**: If they have a trained model, use actual metrics
3. **Fake It If Needed**: For demo, they can return realistic but static data
4. **CORS**: Make sure FastAPI has CORS enabled (already done in your `main.py`)
5. **Error Handling**: Frontend already handles loading states and errors

---

## 📊 Example: Minimal Backend Implementation

```python
# app/routers/admin.py
from fastapi import APIRouter

router = APIRouter(prefix="/api/admin", tags=["admin"])

@router.get("/models/health")
async def get_model_health():
    # For now, return realistic static data
    # Later, connect to actual model metrics
    return {
        "models": [
            {
                "id": "credit-score-model",
                "name": "Credit Scoring Model",
                "status": "healthy",
                "accuracy": 94.3,
                "drift": 2.1,
                "lastTraining": "2024-11-15",
                "predictions": 45231,
                "avgLatency": 45
            }
        ],
        "performanceHistory": [
            {"date": "Nov 21", "accuracy": 94.3, "precision": 93.2, "recall": 95.1}
        ],
        "driftAnalysis": [
            {"feature": "Income", "drift": 1.8}
        ]
    }

# Then in app/main.py:
from app.routers import admin
app.include_router(admin.router)
```

---

## 🎯 Summary

**Current State:**
- ✅ Frontend UI is complete and beautiful
- ❌ All admin pages use hardcoded mock data
- ❌ No real model metrics yet

**Next Steps:**
1. Your teammates create the FastAPI endpoints
2. You replace mock data with API calls (5-10 lines per component)
3. Test integration
4. 🎉 Demo-ready!

**Good News:**
- The UI is already perfect for demo
- Mock data looks realistic
- Integration is straightforward (just swap data source)
- If teammates don't finish in time, **mock data works fine for demo!**

---

*Questions? Check the existing working example in `src/pages/user/KnowYourProfile.tsx` which already fetches real data from FastAPI!*

