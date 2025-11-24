"""
FastAPI Endpoints
REST API for AI Governance Framework - Explainability, Fairness, Literacy, and Compliance.
Now using SQLite database for all persistence instead of JSON files.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import logging
import json
import os
from datetime import datetime
from sqlalchemy.orm import Session

# Import framework modules
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import get_settings

# Import database components
from core.database.models import init_database, get_db_session
from core.database.services import ModelService, HealthService

# Import model wrapper
from models.model_wrapper import ModelWrapper

# Import routers
from api.dashboard_endpoints import router as dashboard_router
from api.blockchain_endpoints import router as blockchain_router
from api.fairness_endpoints import router as fairness_router
from api.enterprise_endpoints import router as enterprise_router
from api.explainability_endpoints import router as explainability_router

from core.explainability.shap_explainer import SHAPExplainer
from core.explainability.lime_explainer import LIMEExplainer

from core.literacy.prompt_generator import ExplanationContext, AudienceType, ExplanationType
from core.literacy.user_prompts import UserPromptGenerator, UserProfile
from core.literacy.banker_prompts import BankerPromptGenerator, BankerProfile

# Import compliance modules
from core.compliance.policy_manager import PolicyManager
from core.compliance.policy_engine import PolicyEngine
from core.compliance.audit_logger import AuditLogger
from core.compliance.policy_schema import Policy, PolicyType, PolicyAction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title=settings.api.api_title,
    version=settings.api.api_version,
    description=settings.api.api_description
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.allow_origins,
    allow_credentials=settings.api.allow_credentials,
    allow_methods=settings.api.allow_methods,
    allow_headers=settings.api.allow_headers,
)

# Include routers
app.include_router(dashboard_router)
app.include_router(blockchain_router)
app.include_router(fairness_router)
app.include_router(enterprise_router)
app.include_router(explainability_router)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and core services"""
    try:
        # Initialize database tables
        init_database()
        logger.info("✅ Database initialized successfully")
        
        # Record framework health
        with next(get_db_session()) as db:
            HealthService.record_health_check(
                service_name="framework",
                status="healthy",
                db=db
            )
        
        logger.info("✅ AI Governance Framework started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize framework: {e}")
        raise e

# Global storage (deprecated - being replaced by database)
MODELS = {}
EXPLAINERS = {}

# Initialize compliance components
POLICY_MANAGER = PolicyManager()
POLICY_ENGINE = PolicyEngine(policies=POLICY_MANAGER.list_policies(enabled_only=True))
AUDIT_LOGGER = AuditLogger(storage_path="audit_ledger.json")


# ============================================================
# REQUEST/RESPONSE MODELS (Previous + New Compliance Models)
# ============================================================

# ... [Keep all previous request/response models] ...

class PredictionRequest(BaseModel):
    """Request for making predictions"""
    model_id: str = Field(..., description="Model identifier")
    features: Dict[str, Any] = Field(..., description="Feature values")
    return_probabilities: bool = Field(default=True, description="Return probability scores")


class PredictionResponse(BaseModel):
    """Response for predictions"""
    prediction: Union[int, float, str]
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    model_id: str
    timestamp: str


class ExplanationRequest(BaseModel):
    """Request for explanation"""
    model_id: str = Field(..., description="Model identifier")
    features: Dict[str, Any] = Field(..., description="Feature values")
    method: str = Field(default="shap", description="Explanation method: shap, lime")
    num_features: int = Field(default=10, description="Number of top features to return")


class ExplanationResponse(BaseModel):
    """Response for explanation"""
    method: str
    prediction: Union[int, float, str]
    confidence: Optional[float]
    feature_importance: Dict[str, float]
    feature_values: Dict[str, float]
    top_features: List[Dict[str, Any]]
    rules: Optional[List[str]] = None
    base_value: Optional[float] = None


class PromptRequest(BaseModel):
    """Request for generating explanation prompt"""
    decision_id: str = Field(..., description="Unique decision identifier")
    decision_type: str = Field(..., description="Type of decision")
    outcome: str = Field(..., description="Decision outcome")
    confidence: float = Field(..., description="Prediction confidence")
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    top_factors: List[Dict[str, Any]] = Field(..., description="Top contributing factors")
    feature_values: Dict[str, Any] = Field(..., description="Feature values")
    audience_type: str = Field(default="user_beginner", description="Audience type")
    user_literacy_level: Optional[str] = Field(default="beginner")
    banker_role: Optional[str] = Field(default="technical_analyst")
    fairness_metrics: Optional[Dict[str, Any]] = None
    include_improvement_tips: bool = Field(default=True)


class PromptResponse(BaseModel):
    """Response with generated prompt"""
    prompt_id: str
    formatted_prompt: str
    audience: str
    prompt_metadata: Dict[str, Any]
    timestamp: str


# ============================================================
# NEW COMPLIANCE REQUEST/RESPONSE MODELS
# ============================================================

class ComplianceCheckRequest(BaseModel):
    """Request for compliance checking"""
    decision_id: str = Field(..., description="Unique decision identifier")
    feature_values: Dict[str, Any] = Field(..., description="Feature values for the decision")
    decision_outcome: Optional[str] = Field(None, description="Decision outcome (approved/denied)")
    model_id: Optional[str] = Field(None, description="Model identifier")
    check_policies: Optional[List[str]] = Field(None, description="Specific policy IDs to check")
    create_audit_receipt: bool = Field(default=True, description="Create audit receipt")
    created_by: Optional[str] = Field(None, description="User making the decision")


class ComplianceCheckResponse(BaseModel):
    """Response for compliance check"""
    decision_id: str
    is_compliant: bool
    timestamp: str
    policies_checked: int
    violations_count: int
    warnings_count: int
    violations: List[Dict[str, Any]]
    required_actions: List[Dict[str, Any]]
    audit_receipt_id: Optional[str] = None
    audit_hash: Optional[str] = None


class PolicyCreateRequest(BaseModel):
    """Request to create a new policy"""
    policy_id: str = Field(..., description="Unique policy identifier")
    name: str = Field(..., description="Policy name")
    regulation_source: str = Field(..., description="Source regulation")
    policy_type: str = Field(..., description="Policy type")
    description: str = Field(..., description="Policy description")
    condition: Dict[str, Any] = Field(..., description="Policy condition")
    action: str = Field(..., description="Action when violated")
    priority: int = Field(default=0, description="Priority (higher checked first)")
    enabled: bool = Field(default=True, description="Whether policy is enabled")
    jurisdiction: Optional[str] = None
    rationale: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class PolicyResponse(BaseModel):
    """Response with policy details"""
    policy_id: str
    name: str
    regulation_source: str
    policy_type: str
    description: str
    enabled: bool
    priority: int
    version: str
    created_at: str


class AuditReceiptResponse(BaseModel):
    """Response with audit receipt"""
    receipt_id: str
    decision_id: str
    timestamp: str
    policies_checked: List[str]
    is_compliant: bool
    content_hash: str
    previous_hash: str
    chain_valid: bool


class ModelRegistrationRequest(BaseModel):
    """Request to register a new model"""
    model_id: str = Field(..., description="Unique model identifier")
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(default="1.0", description="Model version")
    model_type: str = Field(default="classification", description="classification or regression")
    feature_names: List[str] = Field(..., description="List of feature names")
    description: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    models_loaded: int


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_model(model_id: str) -> ModelWrapper:
    """Get model from registry"""
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return MODELS[model_id]


# ============================================================
# EXISTING API ENDPOINTS (Keep all previous endpoints)
# ============================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "AI Governance Framework API",
        "version": settings.api.api_version,
        "docs": "/docs",
        "health": "/health",
        "features": [
            "Explainability (SHAP, LIME)",
            "Fairness Analysis",
            "AI Literacy Prompts",
            "Regulatory Compliance"
        ]
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check(db: Session = Depends(get_db_session)):
    """Health check endpoint - now uses database for service health"""
    try:
        # Get service health summary from database
        health_summary = HealthService.get_service_health_summary(db)
        
        # Count active models from database
        active_models = len(ModelService.get_active_models(db))
        
        # Determine overall status
        overall_status = "healthy"
        for service, health in health_summary.items():
            if health['status'] == 'critical':
                overall_status = "critical"
                break
            elif health['status'] == 'warning' and overall_status != "critical":
                overall_status = "warning"
        
        # Record this health check
        HealthService.record_health_check(
            service_name="framework",
            status=overall_status,
            db=db
        )
        
        return HealthResponse(
            status=overall_status,
            service="ai_governance_framework",
            version=settings.api.api_version,
            environment="development",
            uptime_seconds=0,  # TODO: Calculate actual uptime
            modules=health_summary,
            timestamp=datetime.now().isoformat(),
            models_loaded=active_models
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="critical",
            service="ai_governance_framework", 
            version=settings.api.api_version,
            environment="development",
            uptime_seconds=0,
            modules={},
            timestamp=datetime.now().isoformat(),
            models_loaded=0
        )


# ... [Keep all previous endpoints: /models/register, /predict, /explain, /fairness/analyze, /prompts/generate] ...

@app.post("/models/register", tags=["Models"])
async def register_model(request: ModelRegistrationRequest, db: Session = Depends(get_db_session)):
    """Register a model in database"""
    try:
        logger.info(f"Registering model: {request.model_id}")
        
        # Register model in database
        model = ModelService.register_model(
            model_id=request.model_id,
            model_name=request.model_name,
            model_type=request.model_type,
            algorithm="unknown",  # Add missing required parameter
            version=request.model_version,  # Fix parameter name
            framework="unknown",  # Add missing required parameter
            performance_metrics={},  # Add missing required parameter
            fairness_metrics={},  # Add missing required parameter
            training_data_info={},  # Add missing required parameter
            feature_names=request.feature_names,
            db=db
        )
        
        logger.info(f"✅ Model '{request.model_id}' registered successfully in database")
        
        return {
            "message": f"Model '{request.model_id}' registered successfully",
            "model_id": request.model_id,
            "status": "registered",
            "registered_at": model.created_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/list", tags=["Models"])
async def list_models():
    """List all registered models"""
    return {
        "models": list(MODELS.keys()),
        "count": len(MODELS)
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """Make prediction (placeholder)"""
    try:
        if request.model_id not in MODELS:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")
        
        prediction = 0
        probabilities = {"class_0": 0.7, "class_1": 0.3} if request.return_probabilities else None
        confidence = 0.7 if request.return_probabilities else None
        
        return PredictionResponse(
            prediction=prediction,
            probabilities=probabilities,
            confidence=confidence,
            model_id=request.model_id,
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
async def explain(request: ExplanationRequest):
    """Generate explanation (placeholder)"""
    try:
        if request.model_id not in MODELS:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")
        
        mock_explanation = {
            "method": request.method,
            "prediction": 0,
            "confidence": 0.85,
            "feature_importance": {
                feature: float(np.random.random()) 
                for feature in list(request.features.keys())[:request.num_features]
            },
            "feature_values": {k: float(v) if isinstance(v, (int, float)) else v 
                             for k, v in request.features.items()},
            "top_features": [
                {
                    "name": feature,
                    "importance": float(np.random.random()),
                    "value": request.features.get(feature, 'N/A')
                }
                for feature in list(request.features.keys())[:request.num_features]
            ],
            "rules": [f"When {feature} = {request.features.get(feature)}" 
                     for feature in list(request.features.keys())[:3]] if request.method == "lime" else None,
            "base_value": 0.5 if request.method == "shap" else None
        }
        
        return ExplanationResponse(**mock_explanation)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prompts/generate", response_model=PromptResponse, tags=["AI Literacy"])
async def generate_prompt(request: PromptRequest):
    """Generate explanation prompt for LLM"""
    try:
        context = ExplanationContext(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            outcome=request.outcome,
            confidence=request.confidence,
            model_name=request.model_name,
            model_version=request.model_version,
            top_factors=request.top_factors,
            feature_values=request.feature_values,
            feature_importance={f['name']: f.get('importance', 0) 
                              for f in request.top_factors},
            audience=AudienceType.USER_BEGINNER,
            fairness_metrics=request.fairness_metrics
        )
        
        if request.audience_type.startswith('user'):
            user_profile = UserProfile(
                user_id="API_USER",
                literacy_level=request.user_literacy_level or 'beginner',
                preferred_language='en'
            )
            prompt_gen = UserPromptGenerator()
            prompt_dict = prompt_gen.generate_decision_explanation_prompt(
                context=context,
                user_profile=user_profile,
                include_improvement_tips=request.include_improvement_tips
            )
        elif request.audience_type.startswith('banker'):
            banker_profile = BankerProfile(
                banker_id="API_BANKER",
                role=request.banker_role or 'technical_analyst',
                department='risk',
                expertise_level='senior'
            )
            prompt_gen = BankerPromptGenerator()
            prompt_dict = prompt_gen.generate_technical_analysis_prompt(
                context=context,
                banker_profile=banker_profile
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown audience type: {request.audience_type}"
            )
        
        return PromptResponse(
            prompt_id=f"{request.decision_id}_{request.audience_type}_{datetime.now().timestamp()}",
            formatted_prompt=prompt_dict['formatted_prompt'],
            audience=request.audience_type,
            prompt_metadata={
                "decision_id": request.decision_id,
                "outcome": request.outcome,
                "confidence": request.confidence,
                "num_factors": len(request.top_factors)
            },
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# NEW COMPLIANCE API ENDPOINTS
# ============================================================

@app.post("/compliance/check", response_model=ComplianceCheckResponse, tags=["Compliance"])
async def check_compliance(request: ComplianceCheckRequest):
    """
    Check decision compliance against regulatory policies.
    
    This endpoint:
    1. Checks feature values against all enabled policies
    2. Identifies violations and required actions
    3. Creates cryptographic audit receipt (if requested)
    """
    try:
        logger.info(f"Checking compliance for decision: {request.decision_id}")
        
        # Check compliance
        is_compliant, results = POLICY_ENGINE.check_compliance(
            feature_values=request.feature_values,
            decision=request.decision_outcome,
            policy_ids=request.check_policies
        )
        
        # Extract violations
        violations = [r for r in results if not r.compliant]
        
        # Format violations
        violations_list = [
            {
                "policy_id": v.policy.policy_id,
                "policy_name": v.policy.name,
                "regulation_source": v.policy.regulation_source,
                "message": v.message,
                "recommended_action": v.recommended_action
            }
            for v in violations
        ]
        
        # Required actions
        required_actions = [
            {
                "policy": v.policy.policy_id,
                "action": v.recommended_action,
                "reason": v.message
            }
            for v in violations if v.recommended_action
        ]
        
        # Create audit receipt if requested
        audit_receipt_id = None
        audit_hash = None
        
        if request.create_audit_receipt:
            receipt = AUDIT_LOGGER.create_receipt(
                decision_id=request.decision_id,
                compliance_results=results,
                decision_outcome=request.decision_outcome,
                feature_values=request.feature_values,
                model_id=request.model_id,
                created_by=request.created_by
            )
            audit_receipt_id = receipt.receipt_id
            audit_hash = receipt.content_hash
        
        return ComplianceCheckResponse(
            decision_id=request.decision_id,
            is_compliant=is_compliant,
            timestamp=datetime.now().isoformat(),
            policies_checked=len(results),
            violations_count=len(violations),
            warnings_count=0,
            violations=violations_list,
            required_actions=required_actions,
            audit_receipt_id=audit_receipt_id,
            audit_hash=audit_hash
        )
    
    except Exception as e:
        logger.error(f"Error checking compliance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compliance/policies", tags=["Compliance"])
async def list_policies(
    enabled_only: bool = False,
    regulation_source: Optional[str] = None,
    policy_type: Optional[str] = None
):
    """
    List all policies with optional filters.
    """
    try:
        policies = POLICY_MANAGER.list_policies(
            enabled_only=enabled_only,
            regulation_source=regulation_source,
            policy_type=PolicyType(policy_type) if policy_type else None
        )
        
        return {
            "policies": [
                {
                    "policy_id": p.policy_id,
                    "name": p.name,
                    "regulation_source": p.regulation_source,
                    "policy_type": p.policy_type.value,
                    "enabled": p.enabled,
                    "priority": p.priority,
                    "version": p.version
                }
                for p in policies
            ],
            "count": len(policies)
        }
    
    except Exception as e:
        logger.error(f"Error listing policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compliance/policies/{policy_id}", response_model=PolicyResponse, tags=["Compliance"])
async def get_policy(policy_id: str):
    """Get specific policy details"""
    try:
        policy = POLICY_MANAGER.get_policy(policy_id)
        
        if not policy:
            raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
        
        return PolicyResponse(
            policy_id=policy.policy_id,
            name=policy.name,
            regulation_source=policy.regulation_source,
            policy_type=policy.policy_type.value,
            description=policy.description,
            enabled=policy.enabled,
            priority=policy.priority,
            version=policy.version,
            created_at=policy.created_at
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compliance/policies", tags=["Compliance"])
async def create_policy(request: PolicyCreateRequest):
    """Create a new regulatory policy"""
    try:
        from core.compliance.policy_schema import PolicyCondition
        
        # Create policy object
        policy = Policy(
            policy_id=request.policy_id,
            name=request.name,
            regulation_source=request.regulation_source,
            policy_type=PolicyType(request.policy_type),
            description=request.description,
            condition=PolicyCondition.from_dict(request.condition),
            action=PolicyAction(request.action),
            priority=request.priority,
            enabled=request.enabled,
            jurisdiction=request.jurisdiction,
            rationale=request.rationale,
            tags=request.tags
        )
        
        # Add to policy manager
        created_policy = POLICY_MANAGER.create_policy(policy)
        
        # Refresh policy engine
        POLICY_ENGINE.add_policy(created_policy)
        
        return {
            "message": f"Policy '{request.policy_id}' created successfully",
            "policy_id": request.policy_id,
            "status": "created"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/compliance/policies/{policy_id}/enable", tags=["Compliance"])
async def enable_policy(policy_id: str):
    """Enable a policy"""
    try:
        success = POLICY_MANAGER.enable_policy(policy_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
        
        # Refresh engine
        POLICY_ENGINE.policies = POLICY_MANAGER.list_policies(enabled_only=True)
        POLICY_ENGINE._sort_policies()
        
        return {"message": f"Policy '{policy_id}' enabled", "status": "enabled"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/compliance/policies/{policy_id}/disable", tags=["Compliance"])
async def disable_policy(policy_id: str):
    """Disable a policy"""
    try:
        success = POLICY_MANAGER.disable_policy(policy_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
        
        return {"message": f"Policy '{policy_id}' disabled", "status": "disabled"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compliance/audit/receipts", tags=["Compliance"])
async def list_audit_receipts(limit: int = 10):
    """Get recent audit receipts"""
    try:
        receipts = AUDIT_LOGGER.get_recent_receipts(n=limit)
        
        return {
            "receipts": [
                {
                    "receipt_id": r.receipt_id,
                    "decision_id": r.decision_id,
                    "timestamp": r.timestamp,
                    "policies_checked": len(r.policies_checked),
                    "is_compliant": all(cr.compliant for cr in r.compliance_results),
                    "content_hash": r.content_hash[:32] + "..."
                }
                for r in receipts
            ],
            "count": len(receipts)
        }
    
    except Exception as e:
        logger.error(f"Error listing receipts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compliance/audit/receipts/{receipt_id}", response_model=AuditReceiptResponse, tags=["Compliance"])
async def get_audit_receipt(receipt_id: str):
    """Get specific audit receipt"""
    try:
        receipt = AUDIT_LOGGER.get_receipt(receipt_id)
        
        if not receipt:
            raise HTTPException(status_code=404, detail=f"Receipt '{receipt_id}' not found")
        
        # Verify receipt
        is_valid = AUDIT_LOGGER.verify_receipt(receipt)
        
        return AuditReceiptResponse(
            receipt_id=receipt.receipt_id,
            decision_id=receipt.decision_id,
            timestamp=receipt.timestamp,
            policies_checked=receipt.policies_checked,
            is_compliant=all(r.compliant for r in receipt.compliance_results),
            content_hash=receipt.content_hash,
            previous_hash=receipt.previous_hash,
            chain_valid=is_valid
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting receipt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compliance/audit/verify", tags=["Compliance"])
async def verify_audit_chain():
    """Verify entire audit chain integrity"""
    try:
        is_valid, errors = AUDIT_LOGGER.verify_chain()
        
        stats = AUDIT_LOGGER.get_statistics()
        
        return {
            "chain_valid": is_valid,
            "total_receipts": stats['total_receipts'],
            "errors": errors if not is_valid else [],
            "statistics": stats
        }
    
    except Exception as e:
        logger.error(f"Error verifying chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compliance/summary", tags=["Compliance"])
async def get_compliance_summary():
    """Get compliance system summary"""
    try:
        policy_summary = POLICY_MANAGER.get_policy_summary()
        audit_stats = AUDIT_LOGGER.get_statistics()
        
        return {
            "policies": policy_summary,
            "audit_trail": audit_stats,
            "system_status": "operational"
        }
    
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info(f"Starting {settings.api.api_title} v{settings.api.api_version}")
    logger.info(f"Environment: {settings.env}")
    logger.info(f"Compliance module initialized with {len(POLICY_MANAGER.policies)} policies")
    logger.info(f"API running on {settings.api.host}:{settings.api.port}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API server")


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "endpoints:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_level=settings.logging.log_level.lower()
    )