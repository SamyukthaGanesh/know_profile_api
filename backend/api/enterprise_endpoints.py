"""
Enterprise Compliance Endpoints
Complete enterprise-grade compliance, audit, and cryptographic verification endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

# Import database services
from core.database.services import (
    ConsentService, CryptoReceiptService, PolicyEnforcementService,
    ComplianceAuditService, DataLineageService, RegulatoryReportingService
)
from core.database.models import get_db_session

router = APIRouter(prefix="/compliance", tags=["Enterprise Compliance"])


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class ConsentRequest(BaseModel):
    """Request model for consent recording"""
    user_id: str = Field(..., description="Unique user identifier")
    consent_type: str = Field(..., description="Type of consent")
    purpose: str = Field(..., description="Specific purpose for data processing")
    data_categories: List[str] = Field(..., description="Categories of data to be processed")
    granted: bool = Field(..., description="Whether consent is granted")
    consent_method: str = Field(..., description="Method of consent collection")
    consent_text: str = Field(..., description="Exact consent text shown to user")
    expires_at: Optional[datetime] = Field(None, description="Consent expiration date")
    ip_address: Optional[str] = Field(None, description="User's IP address")
    user_agent: Optional[str] = Field(None, description="User's browser information")
    geolocation: Optional[str] = Field(None, description="User's location")


class ConsentResponse(BaseModel):
    """Response model for consent operations"""
    consent_id: str
    status: str
    blockchain_hash: str
    digital_signature: str
    crypto_receipt_id: str
    recorded_at: str


class ConsentWithdrawalRequest(BaseModel):
    """Request model for consent withdrawal"""
    consent_id: str = Field(..., description="ID of consent to withdraw")
    withdrawal_reason: str = Field(..., description="Reason for withdrawal")
    performed_by: str = Field(..., description="Who is withdrawing the consent")


class ComplianceAuditRequest(BaseModel):
    """Request model for compliance audit"""
    audit_type: str = Field(..., description="Type of audit")
    target_type: str = Field(..., description="Type of target being audited")
    target_id: str = Field(..., description="ID of target being audited")
    auditor_id: str = Field(..., description="ID of auditor")
    audit_framework: str = Field(..., description="Compliance framework")
    compliance_status: str = Field(..., description="Audit result status")
    risk_level: str = Field(..., description="Risk level assessment")
    compliance_score: float = Field(..., description="Compliance score (0-100)")
    violations_found: Optional[List[Dict]] = Field(None, description="List of violations")
    recommendations: Optional[List[str]] = Field(None, description="Remediation recommendations")


class PolicyEnforcementRequest(BaseModel):
    """Request model for policy enforcement logging"""
    policy_id: str = Field(..., description="ID of policy being enforced")
    model_id: str = Field(..., description="ID of model affected")
    enforcement_action: str = Field(..., description="Action taken")
    trigger_condition: Dict = Field(..., description="Condition that triggered enforcement")
    action_result: str = Field(..., description="Result of enforcement action")
    affected_predictions: Optional[int] = Field(0, description="Number of predictions affected")
    business_impact: Optional[str] = Field(None, description="Business impact description")
    risk_score: Optional[float] = Field(0.0, description="Risk score (0-100)")


class DataLineageRequest(BaseModel):
    """Request model for data lineage recording"""
    model_id: str = Field(..., description="ID of model")
    prediction_id: str = Field(..., description="ID of prediction")
    data_sources: List[str] = Field(..., description="List of data sources")
    processing_steps: List[Dict] = Field(..., description="Data processing pipeline")
    personal_data_used: Optional[bool] = Field(False, description="Whether personal data was used")
    consent_references: Optional[List[str]] = Field(None, description="Related consent IDs")


class RegulatoryReportRequest(BaseModel):
    """Request model for regulatory report generation"""
    regulation_type: str = Field(..., description="Type of regulation")
    reporting_period_start: datetime = Field(..., description="Report period start date")
    reporting_period_end: datetime = Field(..., description="Report period end date")
    regulatory_authority: str = Field(..., description="Target regulatory authority")
    models_covered: List[str] = Field(..., description="List of model IDs to include")


# ============================================================
# CONSENT MANAGEMENT ENDPOINTS
# ============================================================

@router.post("/consent/record")
async def record_consent(request: ConsentRequest, db: Session = Depends(get_db_session)):
    """Record user consent with complete cryptographic audit trail"""
    try:
        consent_record = ConsentService.record_consent(
            user_id=request.user_id,
            consent_type=request.consent_type,
            purpose=request.purpose,
            data_categories=request.data_categories,
            granted=request.granted,
            consent_method=request.consent_method,
            consent_text=request.consent_text,
            expires_at=request.expires_at,
            ip_address=request.ip_address,
            user_agent=request.user_agent,
            geolocation=request.geolocation,
            db=db
        )
        
        return ConsentResponse(
            consent_id=consent_record.consent_id,
            status="recorded",
            blockchain_hash=consent_record.blockchain_hash,
            digital_signature=consent_record.digital_signature,
            crypto_receipt_id=f"receipt_consent_granted_{consent_record.consent_id[:12]}",
            recorded_at=consent_record.granted_at.isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record consent: {str(e)}")


@router.post("/consent/withdraw")
async def withdraw_consent(request: ConsentWithdrawalRequest, db: Session = Depends(get_db_session)):
    """Withdraw user consent with audit trail"""
    try:
        success = ConsentService.withdraw_consent(
            consent_id=request.consent_id,
            withdrawal_reason=request.withdrawal_reason,
            performed_by=request.performed_by,
            db=db
        )
        
        if success:
            return {"status": "withdrawn", "consent_id": request.consent_id, "timestamp": datetime.utcnow().isoformat()}
        else:
            raise HTTPException(status_code=404, detail="Consent not found or already withdrawn")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to withdraw consent: {str(e)}")


@router.get("/consent/{user_id}/status")
async def get_user_consent_status(user_id: str, db: Session = Depends(get_db_session)):
    """Get all active consents for a user"""
    try:
        consents = ConsentService.get_user_consents(user_id, db)
        
        return {
            "user_id": user_id,
            "active_consents": [
                {
                    "consent_id": c.consent_id,
                    "consent_type": c.consent_type,
                    "purpose": c.purpose,
                    "data_categories": c.data_categories,
                    "granted_at": c.granted_at.isoformat(),
                    "expires_at": c.expires_at.isoformat() if c.expires_at else None,
                    "blockchain_hash": c.blockchain_hash
                }
                for c in consents
            ],
            "total_consents": len(consents)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get consent status: {str(e)}")


@router.get("/consent/{consent_id}/audit-chain")
async def get_consent_audit_chain(consent_id: str, db: Session = Depends(get_db_session)):
    """Get complete blockchain-style audit chain for a consent"""
    try:
        audit_chain = ConsentService.get_consent_audit_chain(consent_id, db)
        chain_verification = ConsentService.verify_consent_chain(consent_id, db)
        
        return {
            "consent_id": consent_id,
            "audit_chain": audit_chain,
            "chain_verification": chain_verification
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get audit chain: {str(e)}")


# ============================================================
# CRYPTOGRAPHIC VERIFICATION ENDPOINTS
# ============================================================

@router.get("/crypto-receipt/{receipt_id}/verify")
async def verify_crypto_receipt(receipt_id: str, db: Session = Depends(get_db_session)):
    """Verify cryptographic receipt integrity"""
    try:
        verification_result = CryptoReceiptService.verify_receipt(receipt_id, db)
        return verification_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify receipt: {str(e)}")


# ============================================================
# POLICY ENFORCEMENT ENDPOINTS
# ============================================================

@router.post("/policy/enforce")
async def log_policy_enforcement(request: PolicyEnforcementRequest, db: Session = Depends(get_db_session)):
    """Log policy enforcement action with cryptographic proof"""
    try:
        enforcement_log = PolicyEnforcementService.log_policy_enforcement(
            policy_id=request.policy_id,
            model_id=request.model_id,
            enforcement_action=request.enforcement_action,
            trigger_condition=request.trigger_condition,
            action_result=request.action_result,
            affected_predictions=request.affected_predictions,
            business_impact=request.business_impact,
            risk_score=request.risk_score,
            db=db
        )
        
        return {
            "log_id": enforcement_log.log_id,
            "verification_hash": enforcement_log.verification_hash,
            "timestamp": enforcement_log.enforcement_timestamp.isoformat(),
            "status": "logged"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log policy enforcement: {str(e)}")


@router.get("/policy/violations")
async def get_policy_violations(
    policy_id: Optional[str] = None,
    model_id: Optional[str] = None,
    days: int = 30,
    db: Session = Depends(get_db_session)
):
    """Get recent policy violations"""
    try:
        violations = PolicyEnforcementService.get_policy_violations(
            policy_id=policy_id,
            model_id=model_id,
            days=days,
            db=db
        )
        
        return {
            "violations": violations,
            "total_violations": len(violations),
            "period_days": days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get violations: {str(e)}")


# ============================================================
# COMPLIANCE AUDIT ENDPOINTS
# ============================================================

@router.post("/audit/create")
async def create_compliance_audit(request: ComplianceAuditRequest, db: Session = Depends(get_db_session)):
    """Create comprehensive compliance audit with cryptographic receipt"""
    try:
        audit = ComplianceAuditService.create_compliance_audit(
            audit_type=request.audit_type,
            target_type=request.target_type,
            target_id=request.target_id,
            auditor_id=request.auditor_id,
            audit_framework=request.audit_framework,
            compliance_status=request.compliance_status,
            risk_level=request.risk_level,
            compliance_score=request.compliance_score,
            violations_found=request.violations_found,
            recommendations=request.recommendations,
            db=db
        )
        
        return {
            "audit_id": audit.audit_id,
            "compliance_score": audit.compliance_score,
            "risk_level": audit.risk_level,
            "next_audit_date": audit.next_audit_date.isoformat(),
            "follow_up_required": audit.follow_up_required,
            "crypto_receipt_generated": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create audit: {str(e)}")


@router.get("/audit/summary")
async def get_compliance_summary(db: Session = Depends(get_db_session)):
    """Get overall compliance summary"""
    try:
        summary = ComplianceAuditService.get_compliance_summary(db)
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get compliance summary: {str(e)}")


# ============================================================
# DATA LINEAGE ENDPOINTS
# ============================================================

@router.post("/data-lineage/record")
async def record_data_lineage(request: DataLineageRequest, db: Session = Depends(get_db_session)):
    """Record data lineage for AI model prediction"""
    try:
        lineage = DataLineageService.record_data_lineage(
            model_id=request.model_id,
            prediction_id=request.prediction_id,
            data_sources=request.data_sources,
            processing_steps=request.processing_steps,
            personal_data_used=request.personal_data_used,
            consent_references=request.consent_references,
            db=db
        )
        
        return {
            "lineage_id": lineage.lineage_id,
            "model_id": lineage.model_id,
            "prediction_id": lineage.prediction_id,
            "data_quality_score": lineage.data_accuracy_score,
            "retention_expires_at": lineage.retention_expires_at.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record data lineage: {str(e)}")


@router.get("/data-lineage/{prediction_id}")
async def get_prediction_lineage(prediction_id: str, db: Session = Depends(get_db_session)):
    """Get complete data lineage for a specific prediction"""
    try:
        lineage = DataLineageService.get_prediction_lineage(prediction_id, db)
        
        if not lineage:
            raise HTTPException(status_code=404, detail="Data lineage not found")
        
        return {
            "lineage_id": lineage.lineage_id,
            "model_id": lineage.model_id,
            "prediction_id": lineage.prediction_id,
            "data_sources": lineage.data_sources,
            "processing_steps": lineage.data_processing_steps,
            "personal_data_used": lineage.personal_data_used,
            "consent_references": lineage.consent_references,
            "data_quality": {
                "completeness": lineage.data_completeness,
                "accuracy_score": lineage.data_accuracy_score
            },
            "retention_expires_at": lineage.retention_expires_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get data lineage: {str(e)}")


# ============================================================
# REGULATORY REPORTING ENDPOINTS
# ============================================================

@router.post("/regulatory-report/generate")
async def generate_regulatory_report(request: RegulatoryReportRequest, db: Session = Depends(get_db_session)):
    """Generate regulatory compliance report with cryptographic signature"""
    try:
        report = RegulatoryReportingService.generate_regulatory_report(
            regulation_type=request.regulation_type,
            reporting_period_start=request.reporting_period_start,
            reporting_period_end=request.reporting_period_end,
            regulatory_authority=request.regulatory_authority,
            models_covered=request.models_covered,
            db=db
        )
        
        return {
            "report_id": report.report_id,
            "regulation_type": report.regulation_type,
            "report_data": report.report_data,
            "submission_status": report.submission_status,
            "report_hash": report.report_hash,
            "crypto_receipt_generated": True,
            "ready_for_submission": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.post("/regulatory-report/{report_id}/submit")
async def submit_regulatory_report(report_id: str, submission_reference: str, db: Session = Depends(get_db_session)):
    """Mark regulatory report as submitted to authority"""
    try:
        success = RegulatoryReportingService.submit_regulatory_report(
            report_id=report_id,
            submission_reference=submission_reference,
            db=db
        )
        
        if success:
            return {
                "report_id": report_id,
                "submission_status": "submitted",
                "submission_reference": submission_reference,
                "submission_timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Report not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit report: {str(e)}")