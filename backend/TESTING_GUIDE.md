# 🚀 Simple Testing Guide

## Only 2 Useful Files:

1. **`home_credit_api_test.py`** - Complete API testing with real ML model
2. **`enterprise_features_test.py`** - Database testing (no API needed)

## How to Test:

### Full API Test (Recommended)
```bash
# Terminal 1: Start server
cd /Users/muvarma/Downloads/ghci/ai_governance_framework
uvicorn api.endpoints:app --reload

# Terminal 2: Test everything
python home_credit_api_test.py
```

### Database Only Test
```bash
cd /Users/muvarma/Downloads/ghci/ai_governance_framework
python enterprise_features_test.py
```

## What Gets Tested:
- ✅ Database replaces JSON files
- ✅ Real ML model + fairness analysis  
- ✅ Enterprise compliance features
- ✅ Dashboard with real data
- ✅ All endpoints working

## Quick Manual Test:
```bash
# Start server
uvicorn api.endpoints:app --reload

# Test endpoints
curl http://localhost:8000/dashboard/overview
curl http://localhost:8000/docs
```

That's it! Everything else was deleted because it was redundant.