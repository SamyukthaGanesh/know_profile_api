from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict
import numpy as np
import pandas as pd

from .schemas import (
    ProfileSummary, ChartLinks, FairnessSnapshot, RewardsSummary, 
    BiasReductionReport, ConsentState, UserDashboardData, ConsentItem, ConsentReceipt
)
from .services import data as data_svc
from .services import model as model_svc
from .services import explain as xai_svc
from .services import fairness as fair_svc
from .services import consent as consent_svc
from .services import reports as reports_svc
from .services import recommend as rec_svc
from .services import visuals as vis_svc
from .services.consent_manager import CONSENT_MANAGER
from datetime import datetime

app = FastAPI(title="TrustLayer — Know Your Profile API", version="0.1.0")

# Enable CORS for React app - Allow all origins for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=False,  # Must be False when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: Dict[str, object] = {
    "bootstrapped": False,
    "history": {"statistical_parity": [], "equal_opportunity": [], "tpr_gap": [], "fpr_gap": [], "group_shap_parity": []}
}

@app.on_event("startup")
async def startup_event():
    """Auto-bootstrap on startup for hackathon demo - no manual bootstrap needed!"""
    print("🚀 Starting up TrustLayer API...")
    try:
        users, tx = data_svc.generate_synthetic(seed=42, n_users=200)
        features = data_svc.build_features()
        y = model_svc.label_eligibility(features).values
        X = features.drop(columns=["loan_eligible_label"], errors="ignore")
        model_svc.MODEL.fit(X, y)
        model_svc.ARTIFACTS["val_accuracy"] = None
        STATE["bootstrapped"] = True
        print(f"✅ Auto-bootstrapped: {len(users)} users, {len(tx)} transactions")
    except Exception as e:
        print(f"⚠️ Auto-bootstrap failed: {e}")
        print("   You can manually call /bootstrap endpoint")

@app.get("/health")
def health():
    return {"status": "ok", "bootstrapped": STATE["bootstrapped"]}

@app.post("/bootstrap")
def bootstrap(seed: int = 42, n_users: int = 200):
    users, tx = data_svc.generate_synthetic(seed=seed, n_users=n_users)
    features = data_svc.build_features()

    # label and fit a tiny model
    y = model_svc.label_eligibility(features).values
    X = features.drop(columns=["loan_eligible_label"], errors="ignore")
    model_svc.MODEL.fit(X, y)
    model_svc.ARTIFACTS["val_accuracy"] = None  # simple demo; no split here to keep API quick
    STATE["bootstrapped"] = True
    return {"users": len(users), "tx": len(tx), "features": len(features)}

@app.get("/generate_profile/{user_id}", response_model=ProfileSummary)
def generate_profile(user_id: str):
    if not STATE["bootstrapped"]:
        raise HTTPException(400, "Call /bootstrap first.")
    features = data_svc.DATA.features
    if user_id not in features.index:
        raise HTTPException(404, "User not found.")
    row = features.loc[user_id]
    X = features.drop(columns=["loan_eligible_label"], errors="ignore")
    proba = float(model_svc.MODEL.predict_proba(X.loc[[user_id]])[0])

    # importance (global proxy)
    importance = model_svc.MODEL.feature_importance()
    # NLG reasons (fallback to importances; SHAP optional)
    shap_reasons = xai_svc.nlg_from_importances(importance, X.loc[user_id])

    # tips & rewards
    tips = rec_svc.tips_for_user(row)
    rewards = rec_svc.rewards_for_user(row)

    # trust score (simple average of calibrated items for demo)
    # In real use, include conformal prediction intervals, calibration error, fairness pass flag.
    ai_trust_score = float(np.clip(0.7*proba + 0.3, 0, 1))

    # consent
    consent = consent_svc.get_consent(user_id)

    # top spend
    spend_cols = [c for c in X.columns if c in ["Shopping","Dining","Groceries","Transport","Utilities","Health","Entertainment","Rent","Savings","Investments"]]
    top_spend = row[spend_cols].sort_values(ascending=False).head(5).to_dict()

    return ProfileSummary(
        user_id=user_id,
        credit_score=int(row["credit_score"]),
        dti_ratio=float(row["dti_ratio"]),
        annual_interest_inr=int(row["annual_interest_inr"]),
        loan_probability=proba,
        top_spend=top_spend,
        tips=tips,
        ai_trust_score=ai_trust_score,
        shap_reasons=shap_reasons,
        consent=ConsentState(**consent.dict()),
        annual_income_inr=int(row["annual_income_inr"]),
        savings_balance_inr=int(row["savings_balance_inr"]),
        fd_balance_inr=int(row["fd_balance_inr"]),
        rd_balance_inr=int(row["rd_balance_inr"]),
        mf_value_inr=int(row["mf_value_inr"]),
        demat_value_inr=int(row["demat_value_inr"]),
        total_assets_inr=int(row["total_assets_inr"]),
    )

@app.get("/get_transactions/{user_id}")
def get_transactions(user_id: str, limit: int = 25):
    if not STATE["bootstrapped"]:
        raise HTTPException(400, "Call /bootstrap first.")
    tx = data_svc.DATA.tx
    df = tx[tx["user_id"]==user_id].sort_values("date", ascending=False).head(limit)
    return {"transactions": df.to_dict(orient="records")}

@app.get("/get_charts/{user_id}", response_model=ChartLinks)
def get_charts(user_id: str):
    if not STATE["bootstrapped"]:
        raise HTTPException(400, "Call /bootstrap first.")
    feats = data_svc.DATA.features
    if user_id not in feats.index:
        raise HTTPException(404, "User not found.")
    row = feats.loc[user_id]
    X = feats.drop(columns=["loan_eligible_label"], errors="ignore")
    importance = model_svc.MODEL.feature_importance()

    top_cols = [c for c in X.columns if c in ["Shopping","Dining","Groceries","Transport","Utilities","Health","Entertainment","Rent","Savings","Investments"]]
    top_spend = row[top_cols].sort_values(ascending=False).head(5)

    top_spend_url = vis_svc.chart_top_spend(top_spend)
    dti_url = vis_svc.chart_dti(float(row["dti_ratio"]))
    importance_url = vis_svc.chart_importance(importance)
    return ChartLinks(top_spend_url=top_spend_url, dti_url=dti_url, importance_url=importance_url)

@app.get("/get_fairness_snapshot", response_model=FairnessSnapshot)
def get_fairness_snapshot(threshold: float = 0.5):
    if not STATE["bootstrapped"]:
        raise HTTPException(400, "Call /bootstrap first.")
    feats = data_svc.DATA.features
    y_true = model_svc.label_eligibility(feats).values
    X = feats.drop(columns=["loan_eligible_label"], errors="ignore")
    proba = model_svc.MODEL.predict_proba(X)
    y_pred = (proba >= threshold).astype(int)
    groups = data_svc.DATA.users["region"].values

    sp = fair_svc.statistical_parity(y_pred, groups)
    eopp = fair_svc.equal_opportunity(y_true, y_pred, groups)
    eod = fair_svc.equalized_odds(y_true, y_pred, groups)

    # SHAP-parity proxy: reuse global importances to simulate explanation vectors
    importance = model_svc.MODEL.feature_importance()
    imp_vec = np.array(list(importance.values()))
    shap_matrix = np.tile(imp_vec, (len(groups),1))
    from .services import explain as xai
    gsp = xai.groupwise_shap_parity(shap_matrix, groups.tolist())

    # Append to history
    STATE["history"]["statistical_parity"].append(sp)
    STATE["history"]["equal_opportunity"].append(eopp)
    STATE["history"]["tpr_gap"].append(eod["tpr_gap"])
    STATE["history"]["fpr_gap"].append(eod["fpr_gap"])
    STATE["history"]["group_shap_parity"].append(gsp)

    return FairnessSnapshot(
        statistical_parity=float(sp),
        equal_opportunity=float(eopp),
        equalized_odds_tpr=float(eod["tpr_gap"]),
        equalized_odds_fpr=float(eod["fpr_gap"]),
        group_shap_parity=float(gsp)
    )

@app.get("/report/bias_reduction", response_model=BiasReductionReport)
def report_bias_reduction():
    hist = STATE["history"]
    return reports_svc.bias_reduction_report(hist)  # type: ignore

@app.get("/get_consent_status/{user_id}", response_model=ConsentState)
def get_consent_status(user_id: str):
    return consent_svc.get_consent(user_id)

@app.post("/set_consent/{user_id}", response_model=ConsentState)
def set_consent(user_id: str, state: ConsentState):
    return consent_svc.set_consent(user_id, state)

@app.get("/rewards/{user_id}", response_model=RewardsSummary)
def rewards(user_id: str):
    feats = data_svc.DATA.features
    if feats is None or user_id not in feats.index:
        raise HTTPException(404, "User not found.")
    r = rec_svc.rewards_for_user(feats.loc[user_id])
    return r

# ============================================================
# NEW: User Dashboard & Consent Management APIs
# ============================================================

@app.get("/user/dashboard/{user_id}")
def get_user_dashboard(user_id: str):
    """Get comprehensive dashboard data for a user"""
    if not STATE["bootstrapped"]:
        raise HTTPException(400, "Call /bootstrap first.")
    
    # Get user profile data
    features = data_svc.DATA.features
    if user_id not in features.index:
        raise HTTPException(404, "User not found.")
    
    row = features.loc[user_id]
    X = features.drop(columns=["loan_eligible_label"], errors="ignore")
    proba = float(model_svc.MODEL.predict_proba(X.loc[[user_id]])[0])
    
    # Calculate trust score components
    trust_components = {
        'accuracy': int(85 + (proba * 10)),  # Based on model confidence
        'fairness': 94,  # High fairness in demo
        'transparency': 88,
        'privacy': 90,
        'explainability': int(75 + (proba * 15)),
        'compliance': 95
    }
    
    # Generate user name from user_id (e.g., U1000 -> "User 1000")
    user_number = user_id.replace("U", "")
    names = ["Priya", "Raj", "Ananya", "Arjun", "Sneha", "Vikram", "Ishita", "Rohan"]
    name_idx = int(user_number) % len(names)
    user_name = f"{names[name_idx]} {names[(name_idx + 3) % len(names)]}"
    
    # Simulate active loan application (if credit score is decent)
    active_loan = None
    if row["credit_score"] > 600:
        active_loan = {
            'applicationId': f'APP-2024-{user_number}',
            'amount': int(row["annual_income_inr"] * 2.5),
            'currency': 'INR',
            'status': 'under_review' if proba < 0.7 else 'approved',
            'submittedAt': datetime.now().isoformat(),
            'confidence': round(proba, 2),
            'riskLevel': 'low' if proba > 0.8 else ('medium' if proba > 0.5 else 'high')
        }
    
    # Get consent stats
    consent_stats = CONSENT_MANAGER.get_consent_statistics(user_id)
    
    return {
        'user': {
            'userId': user_id,
            'name': user_name,
            'lastLogin': datetime.now().isoformat(),
            'profilePictureUrl': None
        },
        'trustScore': {
            'overall': int(sum(trust_components.values()) / len(trust_components)),
            'components': trust_components,
            'lastUpdated': datetime.now().isoformat(),
            'trend': 'up' if proba > 0.6 else 'stable'
        },
        'activeLoanApplication': active_loan,
        'quickStats': {
            'activeConsents': consent_stats['totalActive'],
            'fairnessRating': 94,
            'pendingActions': 2 if active_loan and active_loan['status'] == 'under_review' else 0
        }
    }

@app.get("/user/{user_id}/consents")
def get_user_consents(user_id: str):
    """Get all consents for a user"""
    if not STATE["bootstrapped"]:
        raise HTTPException(400, "Call /bootstrap first.")
    
    # Verify user exists
    features = data_svc.DATA.features
    if user_id not in features.index:
        raise HTTPException(404, "User not found.")
    
    consents = CONSENT_MANAGER.get_user_consents(user_id)
    statistics = CONSENT_MANAGER.get_consent_statistics(user_id)
    
    return {
        'consents': consents,
        'statistics': statistics
    }

@app.post("/user/{user_id}/consents/{consent_id}")
def update_user_consent(user_id: str, consent_id: str, action: str):
    """Grant or revoke a consent"""
    if action not in ['grant', 'revoke']:
        raise HTTPException(400, "Action must be 'grant' or 'revoke'")
    
    if not STATE["bootstrapped"]:
        raise HTTPException(400, "Call /bootstrap first.")
    
    try:
        receipt = CONSENT_MANAGER.update_consent(consent_id, action)
        return {'receipt': receipt}
    except ValueError as e:
        raise HTTPException(404, str(e))
