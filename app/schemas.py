from pydantic import BaseModel
from typing import List, Dict, Optional

class ConsentState(BaseModel):
    fraud: bool = True
    personalization: bool = True
    training: bool = True

class ProfileSummary(BaseModel):
    # Core fields
    user_id: str
    credit_score: int
    dti_ratio: float
    annual_interest_inr: int
    loan_probability: float
    top_spend: Dict[str, float]
    tips: List[str]
    ai_trust_score: float
    shap_reasons: List[str]
    consent: ConsentState
    # Asset fields
    annual_income_inr: int
    savings_balance_inr: int
    fd_balance_inr: int
    rd_balance_inr: int
    mf_value_inr: int
    demat_value_inr: int
    total_assets_inr: int

class ChartLinks(BaseModel):
    top_spend_url: str
    dti_url: str
    importance_url: str

class FairnessSnapshot(BaseModel):
    statistical_parity: float
    equal_opportunity: float
    equalized_odds_tpr: float
    equalized_odds_fpr: float
    group_shap_parity: float

class RewardsSummary(BaseModel):
    points: int
    cashback_eligible_inr: int
    unused_benefits: List[str]

class BiasReductionReport(BaseModel):
    metrics: Dict[str, List[float]]
    notes: List[str]
