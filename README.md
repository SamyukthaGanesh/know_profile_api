# 🏦 TrustBank AI Governance Platform - Monorepo

A complete AI-powered financial trust platform featuring explainable AI, fairness monitoring, human-in-the-loop approvals, blockchain audit trails, and comprehensive consent management.

**🎯 All Backend Services Included in One Repository!**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ with virtual environment
- Node.js 16+ and npm
- Git

### Start All Services (One Command!)

```bash
cd /Users/samganesh/Downloads/know_profile_api
./START_ALL_COMPLETE.sh
```

This **single command** starts all 5 services:
1. **TrustBank Backend API** (Port 8000) - User profiles, transactions, consents
2. **AI Governance Framework (GHCI)** (Port 8001) - Model health, fairness, blockchain
3. **AI Chatbot** (Port 8002) - RAG-based assistant, explanations
4. **Auth Server** (Port 3001) - JWT authentication
5. **React Frontend** (Port 3000) - User & Admin interfaces

### Access the Platform

🌐 **User Portal:** http://localhost:3000  
👔 **Admin Portal:** http://localhost:3000/admin  
📡 **TrustBank API Docs:** http://localhost:8000/docs  
📡 **GHCI API Docs:** http://localhost:8001/docs  
📡 **Chatbot API Docs:** http://localhost:8002/docs

### Default Login Credentials

**Admin:**
- Email: `admin@trustbank.com`
- Password: `admin123`

**Regular User:**
- Email: `demo@trustbank.com`
- Password: `demo123`

---

## 📁 Monorepo Structure

```
know_profile_api/  (🎯 Monorepo Root - Everything in one place!)
│
├── app/                          # 🏦 TrustBank Backend API (Port 8000)
│   ├── main.py                   # FastAPI application
│   ├── schemas.py                # Pydantic models
│   └── services/                 # Business logic
│       ├── data.py               # User profiles & transactions
│       ├── model.py              # ML model training
│       ├── explain.py            # SHAP/LIME explanations
│       ├── fairness.py           # Bias detection
│       ├── consent_manager.py    # Consent management
│       ├── recommend.py          # Recommendations
│       ├── reports.py            # Report generation
│       └── visuals.py            # Visualizations
│
├── backend/                      # 🤖 AI Governance Framework (Port 8001)
│   ├── api/                      # FastAPI endpoints
│   │   ├── blockchain_endpoints.py
│   │   ├── dashboard_endpoints.py
│   │   ├── explainability_endpoints.py
│   │   ├── fairness_endpoints.py
│   │   └── enterprise_endpoints.py
│   ├── core/                     # Core business logic
│   │   ├── compliance/           # Regulatory policies (Basel III, GDPR, ECOA)
│   │   ├── consent/              # Consent blockchain
│   │   ├── database/             # SQLite persistence
│   │   ├── explainability/       # SHAP explainer
│   │   └── fairness/             # Fairness analysis & optimization
│   ├── data/                     # SQLite databases
│   └── outputs/                  # Generated reports & visualizations
│
├── chatbot/                      # 💬 AI Chatbot Service (Port 8002)
│   ├── api.py                    # Chatbot FastAPI endpoints
│   ├── routed_agent_gemini/      # Gemini-powered agent
│   │   ├── api.py                # Chat API
│   │   ├── agent.py              # AI agent logic
│   │   ├── rag_system.py         # RAG retrieval (FAISS)
│   │   ├── ghci_integration.py   # GHCI connector
│   │   └── fetch_tools.py        # Tool calling
│   ├── regulatory_companion/     # Regulatory knowledge base
│   ├── user_data.db              # Decision logs & explanations
│   └── configs/                  # Configuration files
│
├── trust-platform-ui/            # ⚛️ React Frontend (Port 3000)
│   ├── src/
│   │   ├── components/           # Reusable UI components
│   │   │   ├── shared/           # Buttons, Cards, Badges, etc.
│   │   │   ├── layout/           # UserLayout, AdminLayout
│   │   │   └── chatbot/          # ChatWidget, AdminRegulationChat
│   │   ├── pages/
│   │   │   ├── admin/            # Admin dashboard pages
│   │   │   │   ├── Overview.tsx  # AI Governance Overview
│   │   │   │   ├── ModelHealth.tsx
│   │   │   │   ├── FairnessMonitor.tsx
│   │   │   │   ├── ApprovalsQueue.tsx
│   │   │   │   ├── PolicyManager.tsx
│   │   │   │   ├── BlockchainGraph.tsx
│   │   │   │   └── DataManagement.tsx
│   │   │   └── user/             # User portal pages
│   │   │       ├── KnowYourProfile.tsx
│   │   │       ├── GlobalExplanations.tsx
│   │   │       └── ConsentWallet.tsx
│   │   ├── services/             # API integration
│   │   │   ├── api.ts            # TrustBank API
│   │   │   └── ghciApi.ts        # GHCI API
│   │   ├── context/              # React Context (Auth)
│   │   └── types/                # TypeScript types
│   ├── auth-server-simple.js     # JWT Auth Server (Port 3001)
│   └── db.json                   # User credentials database
│
├── data/                         # 📊 Shared Data Files
│   ├── users.csv                 # 200 synthetic user profiles
│   └── transactions.csv          # 38,000+ transactions
│
├── logs/                         # 📝 Centralized Logs
│   ├── trustbank.log
│   ├── ai_governance_db.log
│   ├── chatbot.log
│   ├── auth.log
│   └── frontend.log
│
├── START_ALL_COMPLETE.sh         # 🚀 Master startup script
├── MONOREPO_STRUCTURE.md         # 📖 Detailed monorepo guide
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🎯 Features

### 👤 User Features

**Know Your Profile Dashboard**
- Real-time credit score monitoring (300-850)
- Debt-to-income ratio calculation
- Asset breakdown (mutual funds, stocks, FDs, properties)
- Spending analytics by category
- AI-powered insights and recommendations
- Interactive transaction visualizations (Plotly charts)
- Rewards and achievements system

**AI Explanations**
- Global model feature importance
- SHAP value visualizations
- Plain-English interpretations
- Model transparency and trust

**Privacy Settings (Consent Wallet)**
- Granular data sharing controls
- Real-time consent management
- Blockchain-backed audit trail
- Per-institution consent toggles
- Data usage tracking

**AI Assistant**
- RAG-based financial chatbot
- Natural language queries
- Transaction insights
- Regulatory knowledge
- Decision explanations

### 👔 Admin Features

**AI Governance Overview**
- Real-time system health dashboard
- Active model monitoring
- Fairness alerts
- Compliance warnings
- Quick actions panel

**Model Health Monitor**
- Accuracy tracking (94.3% default)
- Fairness score monitoring
- Prediction volume metrics
- Drift detection
- Model retraining controls
- Performance trend charts

**Fairness Monitor**
- Bias detection across protected groups (gender, age, location)
- Disparate impact metrics
- Fairness optimization with RL
- Before/After bias comparison
- Trade-off visualization (accuracy vs fairness)
- Automated bias mitigation

**Approvals Queue**
- Human-in-the-loop workflow
- High-risk decision review
- Uncertainty flagging
- Bulk approval tools
- SHAP explanations for each decision

**Policy Manager**
- Regulatory policy management (Basel III, GDPR, ECOA)
- Policy enable/disable controls
- Compliance checking
- Violation tracking
- Audit reports

**Blockchain Graph**
- Cryptographic audit trail visualization
- Chain view & timeline view
- Block verification
- Tamper-proof compliance records
- Consent blockchain explorer

**Data Management**
- Synthetic data generation (200 users, 38K+ transactions)
- Bootstrap endpoint integration
- Model training triggers

---

## 🛠️ Technology Stack

### Backend Services

**TrustBank Backend:**
- FastAPI - High-performance Python API
- Pandas - Data manipulation
- Scikit-learn - ML models
- SHAP & LIME - Explainable AI
- Pydantic - Data validation

**AI Governance Framework (GHCI):**
- FastAPI - API framework
- SQLAlchemy - Database ORM
- SQLite - Persistence
- Fairlearn - Fairness metrics
- SHAP - Model explainability
- Hashlib - Blockchain cryptography

**AI Chatbot:**
- FastAPI - API framework
- Gemini LLM - Language model
- FAISS - Vector search (RAG)
- Sentence Transformers - Embeddings
- SQLite - Decision logs

### Frontend
- React 18 - Modern UI framework
- TypeScript - Type safety
- React Router 6 - Client routing
- Recharts - Data visualizations
- Context API - State management

### Auth & Data
- Express.js - Auth server
- bcryptjs - Password hashing
- JWT - Session management
- SQLite - Structured data
- CSV - User data persistence

---

## 📚 API Endpoints

### TrustBank Backend (Port 8000)

**User Profile:**
- `GET /generate_profile/{user_id}` - User profile with AI insights
- `GET /get_transactions/{user_id}` - Transaction history
- `GET /rewards/{user_id}` - Rewards data
- `GET /get_charts/{user_id}` - Chart data links

**Consent Management:**
- `GET /user/{user_id}/consents` - Get user consents
- `PUT /user/{user_id}/consents/{consent_id}?action=grant|revoke` - Update consent

**Data Generation:**
- `POST /bootstrap` - Generate synthetic data
- `POST /model-retrain` - Retrain ML model

**Fairness:**
- `GET /get_fairness_snapshot` - Fairness metrics snapshot
- `GET /report/bias_reduction` - Bias reduction report

### AI Governance Framework (Port 8001)

**Dashboard:**
- `GET /dashboard/overview` - System health overview
- `GET /dashboard/models/health` - Model performance
- `GET /dashboard/charts/fairness-trend` - Fairness trend data
- `GET /dashboard/compliance` - Compliance metrics
- `GET /dashboard/consent` - Consent metrics
- `GET /dashboard/user/{user_id}/wallet` - User consent wallet

**Model Management:**
- `POST /predict` - Make prediction
- `GET /models/list` - List registered models
- `POST /models/register` - Register new model

**Explainability:**
- `POST /explainability/explain` - Instance-level SHAP explanation
- `POST /explainability/explain-global` - Global feature importance
- `POST /explainability/explain-simple` - Simplified explanation

**Fairness:**
- `POST /fairness/analyze` - Analyze model fairness
- `GET /fairness/reports` - Get fairness reports
- `POST /fairness/optimize` - Optimize for fairness

**Compliance:**
- `GET /compliance/policies` - List policies
- `POST /compliance/policies` - Create policy
- `POST /compliance/check` - Check compliance

**Blockchain:**
- `GET /blockchain/compliance/blocks` - Get compliance blocks
- `GET /blockchain/timeline/compliance` - Get timeline
- `GET /blockchain/consent/blocks/{user_id}` - User consent chain

### AI Chatbot (Port 8002)

**Chat:**
- `POST /chat` - User chat interface
- `POST /regulation_chat` - Admin regulatory queries

**Explanations:**
- `GET /decisions/{user_id}` - Decision history
- `GET /decision_report/{decision_id}` - SHAP report for decision

Full interactive API documentation:
- TrustBank: http://localhost:8000/docs
- GHCI: http://localhost:8001/docs
- Chatbot: http://localhost:8002/docs

---

## 🔧 Development

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

**Auth Server:**
```bash
cd know_profile_api/trust-platform-ui
node auth-server-simple.js
```

**React Frontend:**
```bash
cd know_profile_api/trust-platform-ui
npm start
```

### Install Dependencies

**Backend (all Python services):**
```bash
cd know_profile_api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Frontend:**
```bash
cd know_profile_api/trust-platform-ui
npm install
```

### View Logs

```bash
cd know_profile_api
tail -f logs/trustbank.log          # TrustBank backend
tail -f logs/ai_governance_db.log   # AI Governance
tail -f logs/chatbot.log            # AI Chatbot
tail -f logs/auth.log               # Auth server
tail -f logs/frontend.log           # React frontend
```

---

## 📊 Data

The platform includes:
- **200 synthetic user profiles** with realistic financial data
- **38,000+ transactions** across multiple categories (groceries, bills, entertainment, etc.)
- **Random profile generation** for testing and demos
- **CSV-based storage** for easy inspection and modification
- **SQLite databases** for GHCI and Chatbot persistence

---

## 🎨 UI Design

- **Consistent light mode** throughout all views
- **Bank-like professional design** inspired by ICICI/HDFC
- **Clean white cards** with subtle shadows and borders
- **Professional blue gradients** (#1e3c72 → #2a5298)
- **High contrast** for excellent readability
- **Smooth animations** and hover effects
- **Responsive layout** for all screen sizes
- **Muted, consistent colors** across admin and user views

---

## 🔒 Security

- Password hashing with bcrypt (cost factor: 10)
- JWT-based session management with expiration
- CORS configured for localhost development
- Granular consent management with blockchain audit trail
- Cryptographic hash chains for tamper-proof records
- Audit logging for all data access and decisions

---

## 📦 Monorepo Benefits

✅ **Single Repository** - All services in one place  
✅ **One Command Startup** - `./START_ALL_COMPLETE.sh`  
✅ **Centralized Logging** - All logs in `logs/` directory  
✅ **Shared Dependencies** - Consistent Python environment  
✅ **Easier Development** - No context switching between repos  
✅ **Simplified Deployment** - One repo to clone and deploy  
✅ **Unified Version Control** - Track all changes together

---

## 🚀 Deployment Tips

1. **Update API URLs** in `trust-platform-ui/src/services/*.ts`
2. **Set environment variables** for production (OpenAI API keys, etc.)
3. **Use production build:**
   ```bash
   cd trust-platform-ui
   npm run build
   ```
4. **Serve with nginx** or similar reverse proxy
5. **Use PostgreSQL** instead of CSV/SQLite in production
6. **Enable HTTPS** for all endpoints
7. **Implement proper authentication** and role-based access control
8. **Set up monitoring** with Prometheus/Grafana
9. **Configure backup** for databases and audit logs
10. **Use container orchestration** (Docker/Kubernetes) for scalability

---

## 📄 Documentation

- `MONOREPO_STRUCTURE.md` - Detailed monorepo structure guide
- `PROJECT_STRUCTURE.txt` - Legacy structure reference
- http://localhost:8000/docs - TrustBank API (Swagger UI)
- http://localhost:8001/docs - GHCI API (Swagger UI)
- http://localhost:8002/docs - Chatbot API (Swagger UI)

---

## 🤝 Contributing

This is a hackathon/demo project demonstrating AI governance and transparency features for financial services. The codebase is designed to be:
- **Modular** - Easy to extend with new features
- **Well-documented** - Clear code and comprehensive docs
- **Demo-ready** - Includes sample data and realistic scenarios
- **Production-quality** - Professional UI and robust backend architecture

---



---


This platform demonstrates:
- ✅ **Explainable AI** - Every decision is transparent with SHAP/LIME
- ✅ **Fairness First** - Real-time bias detection and automated mitigation
- ✅ **Human-in-the-Loop** - Critical decisions require human approval
- ✅ **Privacy by Design** - Granular consent management with blockchain
- ✅ **Regulatory Ready** - Basel III, GDPR, ECOA compliance built-in
- ✅ **Production-Quality UI** - Professional bank-like design
- ✅ **Blockchain Audit Trail** - Tamper-proof compliance records
- ✅ **AI Assistant** - RAG-based chatbot for regulatory queries

**Perfect for showcasing responsible AI in financial services!** 🏆

---

## 📊 Key Metrics

- **3 Backend Services** (TrustBank, GHCI, Chatbot)
- **5 Ports** (8000, 8001, 8002, 3000, 3001)
- **~1,000 lines** of Python backend code (TrustBank)
- **~15,000 lines** of TypeScript/React frontend code
- **200 synthetic users** with complete financial profiles
- **38,000+ transactions** for realistic demo scenarios
- **10+ Admin pages** with real-time dashboards
- **4 User pages** with interactive visualizations
- **30+ API endpoints** across all services

---

**Made with ❤️ for the future of transparent, fair, and trustworthy AI in banking**
