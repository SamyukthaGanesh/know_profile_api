# 🏦 TrustBank AI Platform

A complete AI-powered financial trust platform featuring explainable AI, fairness monitoring, human-in-the-loop approvals, and comprehensive consent management.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ with virtual environment
- Node.js 16+ and npm
- Git

### Start All Services (Recommended)

```bash
# From the know_profile_api directory
chmod +x START_ALL_WITH_AUTH.sh
./START_ALL_WITH_AUTH.sh
```

This starts:
- **FastAPI Backend** (Port 8000)
- **Auth Server** (Port 3001)  
- **React Frontend** (Port 3000)

### Access the Platform

🌐 **Landing Page:** http://localhost:3000  
👤 **User Dashboard:** http://localhost:3000/user/dashboard  
👤 **Know Your Profile:** http://localhost:3000/user/profile  
👔 **Admin Panel:** http://localhost:3000/admin/overview  
📡 **API Docs:** http://localhost:8000/docs

### Default Login Credentials

**Admin:**
- User ID: `admin`
- Password: `password`

**Regular User:**
- User ID: `user1`
- Password: `password`

---

## 📁 Project Structure

```
know_profile_api/
├── app/                          # FastAPI Backend
│   ├── main.py                   # Main API application
│   ├── schemas.py                # Pydantic models
│   └── services/                 # Business logic
│       ├── data.py               # User profile & transaction data
│       ├── model.py              # AI model predictions
│       ├── explain.py            # SHAP/LIME explanations
│       ├── fairness.py           # Bias detection & fairness
│       ├── consent.py            # Consent management
│       ├── recommend.py          # Recommendations
│       ├── reports.py            # Reports generation
│       └── visuals.py            # Data visualizations
├── data/                         # CSV data files
│   ├── users.csv                 # 200 synthetic user profiles
│   └── transactions.csv          # 38,000+ transactions
├── trust-platform-ui/            # React Frontend
│   ├── src/
│   │   ├── components/           # Reusable UI components
│   │   ├── pages/                # User & Admin pages
│   │   ├── context/              # React Context (Auth)
│   │   ├── services/             # API service layer
│   │   └── types/                # TypeScript types
│   ├── auth-server-simple.js     # Authentication server
│   └── db.json                   # User auth database
├── START_ALL_WITH_AUTH.sh        # Start all services
├── START_ALL.sh                  # Start without auth (dev mode)
└── run.sh                        # Backend only
```

---

## 🎯 Features

### 👤 User Features

**Know Your Profile Dashboard**
- Real-time credit score monitoring with visual gauge
- Debt-to-income ratio calculation
- Asset breakdown (mutual funds, stocks, FDs, properties)
- Spending analytics by category
- AI-powered insights and recommendations
- Interactive transaction visualizations

**AI Explanations**
- SHAP feature importance for every decision
- LIME local explanations
- Plain-English interpretations
- What-if scenario analysis
- Decision transparency

**Consent Wallet**
- Granular data sharing controls
- Per-institution consent management
- Data usage tracking
- Audit trail of all data access
- Consent revocation

**AI Chatbot**
- Natural language queries
- Financial advice
- Profile analysis
- Transaction insights

### 👔 Admin Features

**Model Health Monitoring**
- Real-time accuracy tracking
- Data drift detection
- Retraining schedules
- Feature importance analysis
- Version control

**Fairness Monitor**
- Bias detection (Gender, Age, Location)
- Protected group analysis
- Disparate impact metrics
- RL-based fairness optimizer
- Compliance reporting

**Approvals Queue**
- Human-in-the-loop workflow
- High-risk decision review
- Uncertainty flagging
- Bulk approval tools
- Audit logging

**Regulatory Dashboard**
- RBI compliance tracking
- GDPR/privacy metrics
- Audit report generation
- Regulatory alerts

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** - High-performance Python API framework
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning models
- **SHAP & LIME** - Explainable AI
- **Pydantic** - Data validation

### Frontend
- **React 18** - Modern UI framework
- **TypeScript** - Type-safe JavaScript
- **React Router 6** - Client-side routing
- **Recharts** - Data visualizations
- **Context API** - State management

### Auth & Data
- **Express** - Auth server
- **bcryptjs** - Password hashing
- **JWT** - Session management
- **CSV** - Data persistence

---

## 📚 API Endpoints

### User Profile
- `GET /generate_profile/{user_id}` - Get user profile with AI insights
- `GET /get_transactions/{user_id}` - Get user transactions
- `GET /health` - Health check

### Admin
- `GET /api/admin/overview` - Platform overview metrics
- `GET /api/admin/models/health` - Model health status
- `GET /api/admin/fairness/metrics` - Fairness metrics
- `GET /api/admin/approvals/queue` - Pending approvals

### AI & Explanations
- Model prediction endpoints
- SHAP explanation generation
- LIME local explanations
- Counterfactual generation

Full API documentation: http://localhost:8000/docs

---

## 🔧 Development

### Backend Only

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

### Frontend Only

```bash
cd trust-platform-ui
npm start
```

### Auth Server Only

```bash
cd trust-platform-ui
node auth-server-simple.js
```

### Install Dependencies

**Backend:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Frontend:**
```bash
cd trust-platform-ui
npm install
```

---

## 📊 Data

The platform includes:
- **200 synthetic user profiles** with realistic financial data
- **38,000+ transactions** across multiple categories
- **Random profile generation** for testing and demos
- **CSV-based storage** for easy inspection and modification

---

## 🎨 UI Design

- **Consistent light mode** throughout
- **Bank-like professional design** (ICICI/HDFC inspired)
- **Clean white cards** with subtle shadows
- **Professional blue gradients** (#1e3c72 → #2a5298)
- **High contrast** for excellent readability
- **Smooth animations** and hover effects
- **Responsive layout** for all screen sizes

---

## 🔒 Security

- Password hashing with bcrypt
- JWT-based session management
- CORS configured for localhost development
- Granular consent management
- Audit logging for all data access

---

## 🚀 Deployment Tips

1. **Update API URLs** in `trust-platform-ui/src/services/api.ts`
2. **Set environment variables** for production
3. **Use production build:**
   ```bash
   cd trust-platform-ui
   npm run build
   ```
4. **Serve with nginx** or similar
5. **Use PostgreSQL** instead of CSV in production
6. **Enable HTTPS** for all endpoints
7. **Implement proper authentication** beyond demo credentials

---

## 📄 Documentation

- `trust-platform-ui/README.md` - Frontend documentation
- `trust-platform-ui/FEATURES.md` - Detailed feature list
- `trust-platform-ui/BACKEND_INTEGRATION_GUIDE.md` - Integration guide
- `http://localhost:8000/docs` - Interactive API documentation

---

## 🤝 Contributing

This is a hackathon project demonstrating AI trust and transparency features for financial services. The codebase is designed to be:
- **Modular** - Easy to extend with new features
- **Well-documented** - Clear code and comprehensive docs
- **Demo-ready** - Includes sample data and mock services

---

## 📝 License

MIT License - Feel free to use for learning, demos, and hackathons!

---

## 🎉 Built For Hackathons

This platform demonstrates:
- ✅ **Explainable AI** - Every decision is transparent
- ✅ **Fairness First** - Bias detection and mitigation
- ✅ **Human-in-the-Loop** - Critical decisions need approval
- ✅ **Privacy by Design** - Granular consent management
- ✅ **Regulatory Ready** - Built with compliance in mind
- ✅ **Production-Quality UI** - Professional bank-like design

**Perfect for showcasing responsible AI in financial services!** 🏆

---

**Made with ❤️ for the future of transparent and fair AI in banking**
