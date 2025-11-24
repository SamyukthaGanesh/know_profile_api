# 🏦 TrustBank AI Governance Platform - Monorepo

## 📦 Unified Repository Structure

This is now a **monorepo** containing all backend services and the frontend in one place!

```
know_profile_api/  (Monorepo Root)
├── app/                          # TrustBank Backend API (Port 8000)
│   ├── main.py                   # FastAPI application
│   ├── schemas.py                # Pydantic models
│   └── services/                 # Business logic modules
│
├── backend/                      # AI Governance Framework - GHCI (Port 8001)
│   ├── api/                      # FastAPI endpoints
│   │   ├── blockchain_endpoints.py
│   │   ├── dashboard_endpoints.py
│   │   ├── explainability_endpoints.py
│   │   └── fairness_endpoints.py
│   ├── core/                     # Core business logic
│   │   ├── compliance/
│   │   ├── explainability/
│   │   ├── fairness/
│   │   └── database/
│   └── data/                     # SQLite databases
│
├── chatbot/                      # AI Chatbot Service (Port 8002)
│   └── routed_agent_gemini/
│       ├── api.py                # Chatbot API
│       ├── agent.py              # AI agent logic
│       ├── rag_system.py         # RAG retrieval
│       └── ghci_integration.py   # GHCI connector
│
├── trust-platform-ui/            # React Frontend (Port 3000)
│   ├── src/
│   │   ├── components/           # Reusable components
│   │   ├── pages/                # Page components
│   │   │   ├── admin/            # Admin pages
│   │   │   └── user/             # User pages
│   │   └── services/             # API integration
│   └── auth-server-simple.js     # Auth server (Port 3001)
│
├── data/                         # Shared data files
│   ├── users.csv                 # User profiles
│   └── transactions.csv          # Transaction history
│
├── logs/                         # Application logs
│   ├── trustbank.log
│   ├── ai_governance_db.log
│   ├── chatbot.log
│   ├── auth.log
│   └── frontend.log
│
├── START_ALL_COMPLETE.sh         # Master startup script
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation
```

---

## 🚀 Quick Start

### Start Everything at Once
```bash
cd /Users/samganesh/Downloads/know_profile_api
./START_ALL_COMPLETE.sh
```

This single command starts:
1. **TrustBank Backend** (Port 8000) - User profiles, transactions, consents
2. **AI Governance Framework** (Port 8001) - Model health, fairness, blockchain
3. **AI Chatbot** (Port 8002) - RAG assistant, explanations
4. **Auth Server** (Port 3001) - JWT authentication
5. **React Frontend** (Port 3000) - User interface

---

## 📍 Service Endpoints

| Service | Port | API Docs | Purpose |
|---------|------|----------|---------|
| **TrustBank Backend** | 8000 | http://localhost:8000/docs | User data, profiles, transactions |
| **AI Governance (GHCI)** | 8001 | http://localhost:8001/docs | Model health, fairness, policies |
| **AI Chatbot** | 8002 | http://localhost:8002/docs | RAG assistant, explanations |
| **Auth Server** | 3001 | - | JWT authentication |
| **React Frontend** | 3000 | http://localhost:3000 | User & Admin portals |

---

## 🔑 Demo Credentials

- **User Account**: `demo@trustbank.com` / `demo123`
- **Admin Account**: `admin@trustbank.com` / `admin123`

---

## 📂 Individual Service Details

### 1️⃣ TrustBank Backend (`app/`)
**Purpose:** Core user data and financial services
- User profiles and financial health metrics
- Transaction history
- Consent management
- Rewards system
- Synthetic data generation

**Key Endpoints:**
- `GET /generate_profile/{user_id}` - User profile & metrics
- `GET /get_transactions/{user_id}` - Transaction history
- `GET /user/{user_id}/consents` - Consent management
- `POST /bootstrap` - Generate synthetic data

### 2️⃣ AI Governance Framework (`backend/`)
**Purpose:** ML model governance and compliance
- Real-time model health monitoring
- Fairness analysis and bias detection
- Policy management (Basel III, GDPR, ECOA)
- Blockchain audit trail
- SHAP-based explainability

**Key Endpoints:**
- `GET /dashboard/overview` - System health overview
- `GET /dashboard/models/health` - Model performance
- `GET /fairness/analyze` - Fairness analysis
- `GET /blockchain/compliance/blocks` - Audit trail
- `POST /explainability/explain` - SHAP explanations

### 3️⃣ AI Chatbot (`chatbot/`)
**Purpose:** Conversational AI assistant
- RAG-based regulatory knowledge
- Decision explanations
- User query handling
- SHAP report generation

**Key Endpoints:**
- `POST /chat` - User chat interface
- `POST /regulation_chat` - Admin regulatory queries
- `GET /decisions/{user_id}` - Decision history
- `GET /decision_report/{decision_id}` - SHAP reports

---

## 🛠️ Development

### Start Individual Services

**TrustBank Backend:**
```bash
cd know_profile_api
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**AI Governance Framework:**
```bash
cd know_profile_api/backend
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

**AI Chatbot:**
```bash
cd know_profile_api/chatbot/routed_agent_gemini
uvicorn api:app --host 0.0.0.0 --port 8002 --reload
```

**React Frontend:**
```bash
cd know_profile_api/trust-platform-ui
npm start
```

---

## 📝 Logs

All logs are centralized in `logs/` directory:
```bash
tail -f logs/trustbank.log          # TrustBank backend
tail -f logs/ai_governance_db.log   # GHCI
tail -f logs/chatbot.log            # Chatbot
tail -f logs/auth.log               # Auth server
tail -f logs/frontend.log           # React app
```

---

## 🛑 Stop All Services

```bash
pkill -f 'uvicorn'
pkill -f 'node.*auth-server'
pkill -f 'react-scripts'
```

---

## 🎯 Key Features

### User View
- 👤 **Know Your Profile** - Financial health dashboard with AI-driven insights
- 🧠 **AI Explanations** - Global model feature importance
- 🔐 **Privacy Settings** - Granular consent management
- 💬 **AI Assistant** - RAG-based financial chatbot

### Admin View
- 📊 **AI Governance Overview** - Real-time system health
- 🧠 **Model Health Monitor** - ML performance tracking
- ⚖️ **Fairness Monitor** - Bias detection and mitigation
- 📋 **Approvals Queue** - Human-in-the-loop decisions
- 📜 **Policy Manager** - Compliance policy management
- 🔗 **Blockchain Graph** - Cryptographic audit trail
- 🗄️ **Data Management** - Synthetic data generation

---

## 🔧 Tech Stack

- **Backend Framework:** FastAPI (Python 3.12)
- **Frontend Framework:** React 18 + TypeScript
- **ML Libraries:** Scikit-learn, SHAP, LIME, Fairlearn
- **Database:** SQLite (GHCI), CSV (TrustBank)
- **Visualization:** Recharts, Plotly
- **Auth:** JWT tokens
- **AI:** Gemini LLM, RAG with FAISS

---

## 📊 Monorepo Benefits

✅ **Single Codebase** - All services in one place  
✅ **Unified Startup** - One command to start everything  
✅ **Shared Dependencies** - Consistent Python environment  
✅ **Centralized Logs** - All logs in one directory  
✅ **Easy Development** - No context switching between repos  
✅ **Simple Deployment** - One repo to deploy  

---

**Version:** 2.0 (Monorepo)  
**Last Updated:** November 24, 2025  
**Status:** Production Ready ✅

