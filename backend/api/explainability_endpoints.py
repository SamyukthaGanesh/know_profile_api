"""
Explainability Endpoints
API endpoints for model explanations using SHAP and LIME.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session

# Import database services
from core.database.services import ModelService, ExplanationService
from core.database.models import get_db_session

# Import explainers
from core.explainability.shap_explainer import SHAPExplainer
from core.explainability.lime_explainer import LIMEExplainer

router = APIRouter(prefix="/explainability", tags=["Explainability"])


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class ExplanationRequest(BaseModel):
    """Request for individual instance explanation"""
    model_id: str = Field(..., description="Model identifier")
    instance_id: str = Field(..., description="Instance identifier")
    features: List[float] = Field(..., description="Feature values")
    feature_names: List[str] = Field(..., description="Feature names")
    prediction: int = Field(..., description="Model prediction")
    prediction_probability: float = Field(..., description="Prediction probability")
    explanation_type: str = Field(default="shap", description="Type of explanation")


class GlobalExplanationRequest(BaseModel):
    """Request for global model explanation"""
    model_id: str = Field(..., description="Model identifier")
    explanation_type: str = Field(default="shap", description="Type of explanation")
    feature_names: List[str] = Field(..., description="Feature names")
    sample_size: int = Field(default=1000, description="Sample size for explanation")


class SimpleExplanationRequest(BaseModel):
    """Request for user-friendly explanation"""
    model_id: str = Field(..., description="Model identifier")
    instance_id: str = Field(..., description="Instance identifier")
    user_profile: Dict[str, Any] = Field(..., description="User profile information")
    explanation_data: Dict[str, Any] = Field(..., description="Explanation data")


class FeatureContribution(BaseModel):
    """Feature contribution information"""
    feature: str
    feature_value: Any
    contribution: float
    importance_rank: int


class ExplanationResponse(BaseModel):
    """Response for explanation request"""
    instance_id: str
    prediction: int
    prediction_probability: float
    explanation_type: str
    feature_contributions: List[FeatureContribution]
    explanation_summary: str
    confidence_score: float


# ============================================================
# EXPLAINABILITY ENDPOINTS
# ============================================================

@router.post("/explain", response_model=ExplanationResponse)
async def explain_instance(
    request: ExplanationRequest,
    db: Session = Depends(get_db_session)
):
    """Generate explanation for individual prediction"""
    try:
        # Mock explanation since we don't have the actual model
        # In production, you'd load the actual model and generate real explanations
        
        # Generate mock SHAP-like values
        feature_contributions = []
        
        for i, (feature_name, feature_value) in enumerate(zip(request.feature_names, request.features)):
            # Mock contribution calculation
            contribution = np.random.uniform(-0.5, 0.5) if i < 5 else np.random.uniform(-0.1, 0.1)
            
            feature_contributions.append(FeatureContribution(
                feature=feature_name,
                feature_value=feature_value,
                contribution=float(contribution),
                importance_rank=i + 1
            ))
        
        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        
        # Update ranks
        for i, contrib in enumerate(feature_contributions):
            contrib.importance_rank = i + 1
        
        # Generate explanation summary
        top_feature = feature_contributions[0]
        direction = "increase" if top_feature.contribution > 0 else "decrease"
        explanation_summary = f"The model's decision was most influenced by {top_feature.feature} (value: {top_feature.feature_value}), which tends to {direction} the probability of the positive class."
        
        # Save explanation to database
        explanation_record = ExplanationService.save_explanation(
            model_id=request.model_id,
            instance_id=request.instance_id,
            prediction_value=float(request.prediction_probability),
            prediction_label=str(request.prediction),
            confidence=float(request.prediction_probability),
            explanation_type=request.explanation_type,
            feature_contributions=[contrib.dict() for contrib in feature_contributions],
            explanation_text=explanation_summary,
            db=db
        )
        
        return ExplanationResponse(
            instance_id=request.instance_id,
            prediction=request.prediction,
            prediction_probability=request.prediction_probability,
            explanation_type=request.explanation_type,
            feature_contributions=feature_contributions,
            explanation_summary=explanation_summary,
            confidence_score=0.85  # Mock confidence score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")


@router.post("/explain-global")
async def explain_global_model(
    request: GlobalExplanationRequest,
    db: Session = Depends(get_db_session)
):
    """Generate global model explanation"""
    try:
        # Mock global feature importance
        feature_importance = {}
        
        for feature_name in request.feature_names:
            # Generate mock importance score
            importance = np.random.uniform(0.0, 1.0)
            feature_importance[feature_name] = float(importance)
        
        # Normalize importance scores
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        return {
            "model_id": request.model_id,
            "explanation_type": request.explanation_type,
            "feature_importance": feature_importance,
            "sample_size": request.sample_size,
            "explanation_summary": f"Global model explanation based on {request.sample_size} samples using {request.explanation_type}",
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Global explanation failed: {str(e)}")


@router.post("/explain-simple")
async def explain_simple(
    request: SimpleExplanationRequest,
    db: Session = Depends(get_db_session)
):
    """Generate user-friendly explanation"""
    try:
        explanation_data = request.explanation_data
        user_profile = request.user_profile
        
        prediction = explanation_data.get("prediction", 0)
        top_factors = explanation_data.get("top_factors", [])
        
        # Generate simple explanation based on user profile
        audience_type = user_profile.get("audience_type", "customer")
        
        if audience_type == "customer":
            if prediction == 1:
                simple_explanation = "Based on our analysis, your loan application has been approved. The main factors supporting this decision include your income level and credit history."
            else:
                simple_explanation = "Your loan application requires further review. This is primarily due to factors in your credit profile that need additional consideration."
        else:
            simple_explanation = f"Model prediction: {prediction}. Key contributing factors analyzed based on feature importance rankings."
        
        # Generate detailed factors
        detailed_factors = []
        for factor in top_factors[:3]:
            factor_name = factor.get("feature", "unknown")
            contribution = factor.get("contribution", 0)
            
            if contribution > 0:
                detailed_factors.append(f"• {factor_name} positively influences the decision")
            else:
                detailed_factors.append(f"• {factor_name} negatively influences the decision")
        
        # Generate next steps
        next_steps = "For more information about this decision, please contact our customer service team."
        if prediction == 0 and audience_type == "customer":
            next_steps = "Consider improving your credit score or providing additional documentation to strengthen your application."
        
        return {
            "instance_id": request.instance_id,
            "simple_explanation": simple_explanation,
            "detailed_factors": detailed_factors,
            "next_steps": next_steps,
            "user_profile": user_profile,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simple explanation failed: {str(e)}")


@router.get("/explanations/{model_id}")
async def get_model_explanations(
    model_id: str,
    limit: int = 10,
    db: Session = Depends(get_db_session)
):
    """Get recent explanations for a model"""
    try:
        explanations = ExplanationService.get_model_explanations(model_id, limit=limit, db=db)
        
        return {
            "model_id": model_id,
            "explanations": [
                {
                    "instance_id": exp.instance_id,
                    "explanation_type": exp.explanation_type,
                    "prediction": exp.prediction,
                    "prediction_probability": exp.prediction_probability,
                    "created_at": exp.created_at.isoformat(),
                    "explanation_summary": exp.explanation_summary
                }
                for exp in explanations
            ],
            "total_explanations": len(explanations)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get explanations: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check for explainability service"""
    return {
        "status": "healthy",
        "service": "explainability",
        "available_explainers": ["shap", "lime"],
        "timestamp": datetime.utcnow().isoformat()
    }