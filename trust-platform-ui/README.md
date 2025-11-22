# TrustBank AI Platform - React UI

A comprehensive React-based user interface for the TrustBank AI Platform, featuring both User and Bank Administrator dashboards with complete transparency and trust features.

## 🚀 Features

### User Dashboard
- **Dashboard Home**: Overview of trust scores, loan applications, and quick stats
- **AI Explanations**: Multi-level (beginner/intermediate/advanced) explanations with SHAP-based factor analysis
- **Consent Provenance Wallet**: Blockchain-verified consent management with cryptographic receipts
- **Fairness Monitor**: Real-time fairness metrics and comparisons
- **Audit Trail**: Immutable decision logs with hash verification

### Bank Admin Dashboard
- **Governance Overview**: System health, real-time metrics, and critical alerts
- **Model Health Monitoring**: Track accuracy, drift detection, and performance metrics
- **Dynamic Fairness Optimizer**: RL-based bias mitigation with episode tracking
- **Human-in-the-Loop Approvals**: Review and approve model updates, policy changes, and fairness adjustments
- **Regulatory Compliance**: AI-powered regulatory companion with parsed requirements
- **Blockchain Ledger**: Immutable audit trail with hash chain verification
- **Alert Center**: Real-time monitoring and alert management
- **Explainability Lab**: Test and generate explanations for different audiences

## 📦 Installation

```bash
cd trust-platform-ui
npm install
```

## 🔧 Development

```bash
npm start
```

Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

## 🏗️ Build

```bash
npm run build
```

Builds the app for production to the `build` folder.

## 📁 Project Structure

```
src/
├── components/
│   ├── shared/        # Reusable UI components
│   │   ├── Card.tsx
│   │   ├── Button.tsx
│   │   ├── Modal.tsx
│   │   ├── Badge.tsx
│   │   ├── ProgressBar.tsx
│   │   └── ToggleSwitch.tsx
│   └── layout/        # Layout components
│       ├── UserLayout.tsx
│       └── AdminLayout.tsx
├── pages/
│   ├── user/          # User dashboard pages
│   │   ├── Dashboard.tsx
│   │   ├── Explanations.tsx
│   │   └── ConsentWallet.tsx
│   └── admin/         # Admin dashboard pages
│       └── Overview.tsx
├── services/
│   └── api.ts         # API service layer with mock data
├── types/
│   └── api.ts         # TypeScript interfaces from API spec
├── styles/
│   └── global.css     # Global styles and animations
├── App.tsx            # Main app with routing
└── index.tsx          # Entry point
```

## 🎨 Design Features

- **Modern Gradient UI**: Beautiful gradients and smooth animations
- **Dark Mode Support**: Separate dark theme for admin dashboard
- **Responsive Design**: Works on desktop and mobile devices
- **Accessibility**: ARIA labels and keyboard navigation
- **Performance**: Optimized rendering and lazy loading

## 🔗 API Integration

The application uses a mock API service (`src/services/api.ts`) that can be easily replaced with real API calls to your backend:

```typescript
// Replace this:
const data = await api.getUserDashboard();

// With actual API call:
const response = await fetch('http://your-backend/api/user/dashboard');
const data = await response.json();
```

## 🌐 Routing

- **User Dashboard**: `/user/*`
  - `/user/dashboard` - Main dashboard
  - `/user/explanations` - AI explanations
  - `/user/consent` - Consent wallet
  - `/user/fairness` - Fairness report
  - `/user/audit` - Audit trail

- **Admin Dashboard**: `/admin/*`
  - `/admin/overview` - System overview
  - `/admin/models` - Model health
  - `/admin/fairness` - Fairness monitor
  - `/admin/approvals` - Approval queue
  - `/admin/regulatory` - Regulatory compliance
  - `/admin/audit` - Blockchain ledger
  - `/admin/alerts` - Alert center
  - `/admin/human-loop` - Human review cases
  - `/admin/explainability` - Explainability lab

## 🔑 Key Technologies

- **React 18** - UI framework
- **TypeScript** - Type safety
- **React Router 6** - Client-side routing
- **CSS Modules** - Scoped styling
- **Recharts** - Data visualization (ready to integrate)

## 📊 Novel Features

1. **Consent Provenance Wallet**: Blockchain-verified consent receipts with cryptographic hashes
2. **Dynamic Fairness Optimizer**: Real-time RL-based bias mitigation
3. **Regulatory AI Companion**: Auto-parsed machine-readable regulations
4. **Immutable Decision Ledger**: Hash-chained audit trail
5. **Multi-Level Explanations**: Adaptive explanations for different literacy levels

## 🚦 Next Steps

1. Connect to your FastAPI backend
2. Implement remaining admin pages
3. Add chart visualizations using Recharts
4. Implement real-time WebSocket updates
5. Add authentication and authorization
6. Implement export/download functionality
7. Add comprehensive testing

## 📝 License

This project is part of the TrustBank AI Platform.

