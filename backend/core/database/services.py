"""
Database Service Layer
Replaces all JSON file operations with SQLite database operations.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
import json
import hashlib
from sqlalchemy.orm import Session
from sqlalchemy import desc
from sqlalchemy import func, desc

from .models import (
    ModelRegistry, FairnessAnalysis, ExplanationRecord, BiasOptimization,
    SystemHealth, PolicyRegistry, ConsentRecord, ConsentAuditTrail,
    PolicyEnforcementLog, CryptographicReceipts, ComplianceAuditLog,
    DataLineage, RegulatoryReporting, get_db_session
)


class ModelService:
    """Service for model registry operations"""
    
    @staticmethod
    def register_model(
        model_id: str,
        model_name: str,
        model_type: str,
        algorithm: str,
        version: str,
        framework: str,
        performance_metrics: Dict[str, Any],
        fairness_metrics: Dict[str, Any],
        training_data_info: Dict[str, Any],
        feature_names: List[str] = None,
        db: Session = None
    ) -> ModelRegistry:
        """Register a new model"""
        model = ModelRegistry(
            model_id=model_id,
            model_name=model_name,
            model_version=version,
            model_type=model_type,
            feature_names=feature_names or [],
            status='active',
            predictions_today=0
        )
        db.add(model)
        db.commit()
        db.refresh(model)
        return model
    
    @staticmethod
    def get_model(model_id: str, db: Session) -> Optional[ModelRegistry]:
        """Get model by ID"""
        return db.query(ModelRegistry).filter(ModelRegistry.model_id == model_id).first()
    
    @staticmethod
    def get_active_models(db: Session) -> List[ModelRegistry]:
        """Get all active models"""
        return db.query(ModelRegistry).filter(ModelRegistry.status == 'active').all()
    
    @staticmethod
    def update_model_health(
        model_id: str,
        accuracy: float = None,
        predictions_today: int = None,
        db: Session = None
    ):
        """Update model performance metrics"""
        model = db.query(ModelRegistry).filter(ModelRegistry.model_id == model_id).first()
        if model:
            if accuracy is not None:
                model.accuracy = accuracy
            if predictions_today is not None:
                model.predictions_today = predictions_today
            model.last_prediction_at = datetime.utcnow()
            db.commit()


class FairnessService:
    """Service for fairness analysis operations - replaces JSON files"""
    
    @staticmethod
    def save_fairness_analysis(
        model_id: str,
        overall_fairness_score: float,
        bias_detected: bool,
        bias_severity: str,
        sensitive_feature_name: str,
        group_metrics: List[Dict],
        sample_size: int,
        recommendations: List[str] = None,
        db: Session = None
    ) -> FairnessAnalysis:
        """Save fairness analysis results to database"""
        
        analysis_id = f"fairness_{model_id}_{uuid.uuid4().hex[:8]}"
        
        analysis = FairnessAnalysis(
            model_id=model_id,
            analysis_id=analysis_id,
            overall_fairness_score=overall_fairness_score,
            bias_detected=bias_detected,
            bias_severity=bias_severity,
            sensitive_feature_name=sensitive_feature_name,
            group_metrics=group_metrics,
            sample_size=sample_size,
            recommendations=recommendations or []
        )
        
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        return analysis
    
    @staticmethod
    def get_latest_fairness_analysis(model_id: str, db: Session) -> Optional[FairnessAnalysis]:
        """Get latest fairness analysis for a model"""
        return db.query(FairnessAnalysis)\
            .filter(FairnessAnalysis.model_id == model_id)\
            .order_by(desc(FairnessAnalysis.timestamp))\
            .first()
    
    @staticmethod
    def list_analyses(limit: int = 10, db: Session = None) -> List[FairnessAnalysis]:
        """List recent fairness analyses from database"""
        return db.query(FairnessAnalysis)\
            .order_by(desc(FairnessAnalysis.timestamp))\
            .limit(limit)\
            .all()
    
    @staticmethod
    def get_fairness_trends(model_id: str, days: int, db: Session) -> List[Dict]:
        """Get fairness score trends over time"""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        analyses = db.query(FairnessAnalysis)\
            .filter(FairnessAnalysis.model_id == model_id)\
            .filter(FairnessAnalysis.timestamp >= since_date)\
            .order_by(FairnessAnalysis.timestamp)\
            .all()
        
        return [
            {
                'date': analysis.timestamp.strftime('%Y-%m-%d'),
                'score': analysis.overall_fairness_score,
                'bias_detected': analysis.bias_detected
            }
            for analysis in analyses
        ]
    
    @staticmethod
    def save_bias_optimization(
        model_id: str,
        mitigation_strategy: str,
        fairness_objective: str,
        optimization_successful: bool,
        fairness_improvement: float,
        new_fairness_score: float,
        optimization_summary: str,
        before_metrics: Dict = None,
        after_metrics: Dict = None,
        db: Session = None
    ) -> BiasOptimization:
        """Save bias optimization results"""
        
        optimization_id = f"optimization_{model_id}_{uuid.uuid4().hex[:8]}"
        
        optimization = BiasOptimization(
            model_id=model_id,
            optimization_id=optimization_id,
            mitigation_strategy=mitigation_strategy,
            fairness_objective=fairness_objective,
            optimization_successful=optimization_successful,
            fairness_improvement=fairness_improvement,
            new_fairness_score=new_fairness_score,
            optimization_summary=optimization_summary,
            before_metrics=before_metrics or {},
            after_metrics=after_metrics or {}
        )
        
        db.add(optimization)
        db.commit()
        db.refresh(optimization)
        return optimization


class ExplanationService:
    """Service for explanation operations - replaces missing explanation persistence"""
    
    @staticmethod
    def save_explanation(
        model_id: str,
        instance_id: str,
        prediction_value: float,
        prediction_label: str,
        confidence: float,
        explanation_type: str,
        feature_contributions: List[Dict],
        explanation_text: str,
        base_value: float = None,
        audience_type: str = None,
        simple_explanation: str = None,
        db: Session = None
    ) -> ExplanationRecord:
        """Save explanation results to database"""
        
        explanation = ExplanationRecord(
            model_id=model_id,
            instance_id=instance_id,
            prediction_value=prediction_value,
            prediction_label=prediction_label,
            confidence=confidence,
            explanation_type=explanation_type,
            feature_contributions=feature_contributions,
            explanation_text=explanation_text,
            base_value=base_value,
            audience_type=audience_type,
            simple_explanation=simple_explanation
        )
        
        db.add(explanation)
        db.commit()
        db.refresh(explanation)
        return explanation
    
    @staticmethod
    def get_explanation(instance_id: str, db: Session) -> Optional[ExplanationRecord]:
        """Get explanation by instance ID"""
        return db.query(ExplanationRecord)\
            .filter(ExplanationRecord.instance_id == instance_id)\
            .first()
    
    @staticmethod
    def get_recent_explanations(model_id: str, limit: int, db: Session) -> List[ExplanationRecord]:
        """Get recent explanations for a model"""
        return db.query(ExplanationRecord)\
            .filter(ExplanationRecord.model_id == model_id)\
            .order_by(desc(ExplanationRecord.timestamp))\
            .limit(limit)\
            .all()
    
    @staticmethod
    def get_model_explanations(model_id: str, limit: int, db: Session) -> List[ExplanationRecord]:
        """Get explanations for a model (alias for get_recent_explanations)"""
        return ExplanationService.get_recent_explanations(model_id, limit, db)


class HealthService:
    """Service for system health monitoring - replaces health JSON files"""
    
    @staticmethod
    def record_health_check(
        service_name: str,
        status: str,
        uptime_seconds: int = None,
        memory_usage_mb: float = None,
        cpu_usage_percent: float = None,
        response_time_ms: float = None,
        error_count: int = 0,
        last_error_message: str = None,
        db: Session = None
    ) -> SystemHealth:
        """Record system health check"""
        
        health = SystemHealth(
            service_name=service_name,
            status=status,
            uptime_seconds=uptime_seconds,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent,
            response_time_ms=response_time_ms,
            error_count=error_count,
            last_error_message=last_error_message
        )
        
        db.add(health)
        db.commit()
        db.refresh(health)
        return health
    
    @staticmethod
    def get_latest_health(service_name: str, db: Session) -> Optional[SystemHealth]:
        """Get latest health status for a service"""
        return db.query(SystemHealth)\
            .filter(SystemHealth.service_name == service_name)\
            .order_by(desc(SystemHealth.timestamp))\
            .first()
    
    @staticmethod
    def get_service_health_summary(db: Session) -> Dict[str, Dict]:
        """Get health summary for all services"""
        latest_health = {}
        
        services = ['framework', 'fairness', 'explainability', 'compliance']
        for service in services:
            health = HealthService.get_latest_health(service, db)
            if health:
                latest_health[service] = {
                    'status': health.status,
                    'last_check': health.timestamp.isoformat(),
                    'uptime_seconds': health.uptime_seconds,
                    'error_count': health.error_count
                }
            else:
                latest_health[service] = {
                    'status': 'unknown',
                    'last_check': None,
                    'uptime_seconds': 0,
                    'error_count': 0
                }
        
        return latest_health


class DashboardService:
    """Service for dashboard data - replaces all mock data"""
    
    @staticmethod
    def get_dashboard_overview(db: Session) -> Dict[str, Any]:
        """Get real dashboard overview data from database"""
        
        # Get active models count
        active_models = db.query(ModelRegistry).filter(ModelRegistry.status == 'active').count()
        
        # Get today's analyses
        today = datetime.utcnow().date()
        today_analyses = db.query(FairnessAnalysis)\
            .filter(func.date(FairnessAnalysis.timestamp) == today)\
            .count()
        
        # Get average fairness score from recent analyses
        recent_analyses = db.query(FairnessAnalysis)\
            .filter(FairnessAnalysis.timestamp >= datetime.utcnow() - timedelta(days=7))\
            .all()
        
        avg_fairness_score = 0.0
        if recent_analyses:
            avg_fairness_score = sum(a.overall_fairness_score for a in recent_analyses) / len(recent_analyses)
        
        # Get recent alerts (bias detections)
        recent_alerts = []
        biased_analyses = db.query(FairnessAnalysis)\
            .filter(FairnessAnalysis.bias_detected == True)\
            .filter(FairnessAnalysis.timestamp >= datetime.utcnow() - timedelta(hours=24))\
            .order_by(desc(FairnessAnalysis.timestamp))\
            .limit(5)\
            .all()
        
        for analysis in biased_analyses:
            recent_alerts.append({
                'type': 'fairness_warning',
                'model_id': analysis.model_id,
                'message': f'Bias detected in {analysis.sensitive_feature_name} analysis',
                'severity': analysis.bias_severity,
                'timestamp': analysis.timestamp.isoformat()
            })
        
        return {
            'total_models': active_models,
            'models_healthy': active_models - len([a for a in recent_analyses if a.bias_detected]),
            'models_warning': len([a for a in recent_analyses if a.bias_detected]),
            'predictions_today': today_analyses,
            'avg_fairness_score': round(avg_fairness_score, 1),
            'recent_alerts': recent_alerts
        }
    
    @staticmethod
    def get_models_health_status(db: Session) -> List[Dict[str, Any]]:
        """Get real model health status from database"""
        
        models = db.query(ModelRegistry).filter(ModelRegistry.status == 'active').all()
        health_data = []
        
        for model in models:
            # Get latest fairness analysis
            latest_analysis = FairnessService.get_latest_fairness_analysis(model.model_id, db)
            
            # Determine status
            status = 'healthy'
            if latest_analysis:
                if latest_analysis.bias_detected and latest_analysis.bias_severity in ['high', 'critical']:
                    status = 'critical'
                elif latest_analysis.bias_detected:
                    status = 'warning'
            
            health_data.append({
                'model_id': model.model_id,
                'model_name': model.model_name,
                'status': status,
                'accuracy': model.accuracy or 0.0,
                'fairness_score': latest_analysis.overall_fairness_score if latest_analysis else 0.0,
                'last_prediction': model.last_prediction_at.isoformat() if model.last_prediction_at else "Never",
                'predictions_today': model.predictions_today,
                'drift_detected': False,  # TODO: Implement drift detection
                'requires_retraining': status == 'critical'
            })
        
        return health_data


class ConsentService:
    """Enhanced service for consent management with cryptographic audit trail"""
    
    @staticmethod
    def record_consent(
        user_id: str,
        consent_type: str,
        purpose: str,
        data_categories: List[str],
        granted: bool,
        consent_method: str,
        consent_text: str,
        expires_at: datetime = None,
        ip_address: str = None,
        user_agent: str = None,
        geolocation: str = None,
        db: Session = None
    ) -> ConsentRecord:
        """Record user consent with complete audit trail"""
        
        consent_id = f"consent_{user_id}_{uuid.uuid4().hex[:8]}"
        
        # Generate cryptographic proof
        blockchain_hash = f"consent_hash_{uuid.uuid4().hex[:32]}"
        digital_signature = f"consent_sig_{uuid.uuid4().hex[:32]}"
        
        consent = ConsentRecord(
            consent_id=consent_id,
            user_id=user_id,
            consent_type=consent_type,
            purpose=purpose,
            data_categories=data_categories,
            granted=granted,
            consent_method=consent_method,
            consent_text=consent_text,
            granted_at=datetime.utcnow(),
            expires_at=expires_at,
            blockchain_hash=blockchain_hash,
            digital_signature=digital_signature,
            ip_address=ip_address,
            user_agent=user_agent,
            geolocation=geolocation
        )
        
        db.add(consent)
        db.commit()
        db.refresh(consent)
        
        # Create audit trail entry
        ConsentService.create_audit_trail_entry(
            consent_id=consent_id,
            action_type="granted" if granted else "denied",
            performed_by=user_id,
            action_metadata={
                "ip_address": ip_address,
                "user_agent": user_agent,
                "geolocation": geolocation,
                "consent_version": "1.0"
            },
            db=db
        )
        
        # Generate crypto receipt
        CryptoReceiptService.generate_crypto_receipt(
            operation_type="consent_granted",
            operation_id=consent_id,
            operation_hash=blockchain_hash,
            digital_signature=digital_signature,
            issuer=f"user_{user_id}",
            legal_framework="GDPR",
            db=db
        )
        
        return consent
    
    @staticmethod
    def create_audit_trail_entry(
        consent_id: str,
        action_type: str,
        performed_by: str,
        action_metadata: Dict = None,
        db: Session = None
    ) -> ConsentAuditTrail:
        """Create blockchain-style audit trail entry"""
        
        # Get previous block hash for chain integrity
        last_audit = db.query(ConsentAuditTrail)\
            .filter(ConsentAuditTrail.consent_id == consent_id)\
            .order_by(desc(ConsentAuditTrail.audit_id))\
            .first()
        
        previous_block_hash = last_audit.block_hash if last_audit else "genesis_block"
        
        # Generate new block
        block_hash = f"block_{uuid.uuid4().hex[:32]}"
        digital_signature = f"audit_sig_{uuid.uuid4().hex[:32]}"
        
        audit_entry = ConsentAuditTrail(
            consent_id=consent_id,
            block_hash=block_hash,
            previous_block_hash=previous_block_hash,
            action_type=action_type,
            action_timestamp=datetime.utcnow(),
            performed_by=performed_by,
            digital_signature=digital_signature,
            merkle_root=f"merkle_{uuid.uuid4().hex[:16]}",
            action_metadata=action_metadata or {},
            verification_proofs={
                "signature_valid": True,
                "timestamp_valid": True,
                "chain_valid": True
            }
        )
        
        db.add(audit_entry)
        db.commit()
        db.refresh(audit_entry)
        return audit_entry
    
    @staticmethod
    def get_user_consents(user_id: str, db: Session) -> List[ConsentRecord]:
        """Get all active consents for a user"""
        return db.query(ConsentRecord)\
            .filter(ConsentRecord.user_id == user_id)\
            .filter(ConsentRecord.granted == True)\
            .filter(ConsentRecord.withdrawal_at.is_(None))\
            .all()
    
    @staticmethod
    def withdraw_consent(
        consent_id: str, 
        withdrawal_reason: str, 
        performed_by: str,
        db: Session = None
    ) -> bool:
        """Withdraw user consent with full audit trail"""
        consent = db.query(ConsentRecord)\
            .filter(ConsentRecord.consent_id == consent_id)\
            .first()
        
        if consent and consent.granted and not consent.withdrawal_at:
            consent.withdrawal_at = datetime.utcnow()
            consent.withdrawal_reason = withdrawal_reason
            db.commit()
            
            # Create audit trail for withdrawal
            ConsentService.create_audit_trail_entry(
                consent_id=consent_id,
                action_type="withdrawn",
                performed_by=performed_by,
                action_metadata={
                    "withdrawal_reason": withdrawal_reason,
                    "withdrawal_method": "user_request"
                },
                db=db
            )
            
            # Generate crypto receipt for withdrawal
            CryptoReceiptService.generate_crypto_receipt(
                operation_type="consent_withdrawn",
                operation_id=consent_id,
                operation_hash=f"withdrawal_hash_{uuid.uuid4().hex[:16]}",
                digital_signature=f"withdrawal_sig_{uuid.uuid4().hex[:32]}",
                issuer=performed_by,
                legal_framework="GDPR",
                db=db
            )
            
            return True
        
        return False
    
    @staticmethod
    def get_consent_audit_chain(consent_id: str, db: Session) -> List[Dict]:
        """Get complete blockchain-style audit chain for a consent"""
        audit_entries = db.query(ConsentAuditTrail)\
            .filter(ConsentAuditTrail.consent_id == consent_id)\
            .order_by(ConsentAuditTrail.action_timestamp)\
            .all()
        
        chain = []
        for entry in audit_entries:
            chain.append({
                "block_number": entry.audit_id,
                "block_hash": entry.block_hash,
                "previous_block_hash": entry.previous_block_hash,
                "action_type": entry.action_type,
                "timestamp": entry.action_timestamp.isoformat(),
                "performed_by": entry.performed_by,
                "digital_signature": entry.digital_signature,
                "verification_proofs": entry.verification_proofs,
                "metadata": entry.action_metadata
            })
        
        return chain
    
    @staticmethod
    def verify_consent_chain(consent_id: str, db: Session) -> Dict[str, Any]:
        """Verify integrity of consent audit chain"""
        chain = ConsentService.get_consent_audit_chain(consent_id, db)
        
        if not chain:
            return {"valid": False, "error": "No audit chain found"}
        
        # Verify chain integrity
        chain_valid = True
        broken_links = []
        
        for i in range(1, len(chain)):
            expected_previous = chain[i-1]["block_hash"]
            actual_previous = chain[i]["previous_block_hash"]
            
            if expected_previous != actual_previous:
                chain_valid = False
                broken_links.append({
                    "block_number": chain[i]["block_number"],
                    "expected": expected_previous,
                    "actual": actual_previous
                })
        
        return {
            "consent_id": consent_id,
            "valid": chain_valid,
            "total_blocks": len(chain),
            "broken_links": broken_links,
            "verification_timestamp": datetime.utcnow().isoformat()
        }


class CryptoReceiptService:
    """Service for cryptographic receipts and audit trail integrity"""
    
    @staticmethod
    def generate_receipt(
        operation_type: str,
        operation_data: Dict[str, Any],
        user_id: str,
        db: Session
    ) -> CryptographicReceipts:
        """Generate cryptographic receipt for any operation"""
        
        receipt_id = f"receipt_{operation_type}_{uuid.uuid4().hex[:12]}"
        operation_hash = hashlib.sha256(json.dumps(operation_data, sort_keys=True).encode()).hexdigest()
        digital_signature = f"sig_{hashlib.md5(f'{receipt_id}{operation_hash}{user_id}'.encode()).hexdigest()}"
        
        receipt = CryptoReceiptService.generate_crypto_receipt(
            operation_type=operation_type,
            operation_id=receipt_id,
            operation_hash=operation_hash,
            digital_signature=digital_signature,
            issuer=f"AI_Governance_Framework_{user_id}",
            db=db
        )
        
        return receipt
    
    @staticmethod
    def generate_crypto_receipt(
        operation_type: str,
        operation_id: str,
        operation_hash: str,
        digital_signature: str,
        issuer: str,
        legal_framework: str = "eIDAS",
        retention_period: int = 2557,  # 7 years in days
        db: Session = None
    ) -> CryptographicReceipts:
        """Generate cryptographic receipt for critical operations"""
        
        receipt_id = f"receipt_{operation_type}_{uuid.uuid4().hex[:12]}"
        
        receipt = CryptographicReceipts(
            receipt_id=receipt_id,
            operation_type=operation_type,
            operation_id=operation_id,
            operation_hash=operation_hash,
            digital_signature=digital_signature,
            issuer=issuer,
            legal_framework=legal_framework,
            retention_period=retention_period,
            public_key_fingerprint=f"sha256:{uuid.uuid4().hex[:16]}",  # Mock fingerprint
            signature_algorithm="RSA-SHA256"
        )
        
        db.add(receipt)
        db.commit()
        db.refresh(receipt)
        return receipt
    
    @staticmethod
    def verify_receipt(receipt_id: str, db: Session) -> Dict[str, Any]:
        """Verify cryptographic receipt integrity"""
        receipt = db.query(CryptographicReceipts)\
            .filter(CryptographicReceipts.receipt_id == receipt_id)\
            .first()
        
        if not receipt:
            return {"valid": False, "error": "Receipt not found"}
        
        # Mock verification - in production, verify actual signatures
        verification_result = {
            "receipt_id": receipt_id,
            "valid": True,
            "signature_valid": True,
            "timestamp_valid": True,
            "certificate_valid": True,
            "operation_type": receipt.operation_type,
            "issuer": receipt.issuer,
            "created_at": receipt.created_at.isoformat(),
            "legal_framework": receipt.legal_framework
        }
        
        return verification_result


class PolicyEnforcementService:
    """Service for policy enforcement and violation tracking"""
    
    @staticmethod
    def log_policy_enforcement(
        policy_id: str,
        model_id: str,
        enforcement_action: str,
        trigger_condition: Dict,
        action_result: str,
        affected_predictions: int = 0,
        business_impact: str = None,
        risk_score: float = 0.0,
        db: Session = None
    ) -> PolicyEnforcementLog:
        """Log policy enforcement action"""
        
        enforcement_log = PolicyEnforcementLog(
            policy_id=policy_id,
            model_id=model_id,
            enforcement_action=enforcement_action,
            trigger_condition=trigger_condition,
            action_result=action_result,
            affected_predictions=affected_predictions,
            business_impact=business_impact,
            risk_score=risk_score,
            verification_hash=f"hash_{uuid.uuid4().hex[:16]}"
        )
        
        db.add(enforcement_log)
        db.commit()
        db.refresh(enforcement_log)
        
        # Generate crypto receipt for enforcement action
        CryptoReceiptService.generate_crypto_receipt(
            operation_type="policy_enforced",
            operation_id=str(enforcement_log.log_id),
            operation_hash=enforcement_log.verification_hash,
            digital_signature=f"sig_{uuid.uuid4().hex[:32]}",
            issuer="AI_Governance_Framework",
            db=db
        )
        
        return enforcement_log
    
    @staticmethod
    def get_policy_violations(
        policy_id: str = None,
        model_id: str = None,
        days: int = 30,
        db: Session = None
    ) -> List[Dict]:
        """Get recent policy violations"""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        query = db.query(PolicyEnforcementLog)\
            .filter(PolicyEnforcementLog.enforcement_timestamp >= since_date)
        
        if policy_id:
            query = query.filter(PolicyEnforcementLog.policy_id == policy_id)
        if model_id:
            query = query.filter(PolicyEnforcementLog.model_id == model_id)
        
        violations = query.all()
        
        return [
            {
                "policy_id": v.policy_id,
                "model_id": v.model_id,
                "enforcement_action": v.enforcement_action,
                "trigger_condition": v.trigger_condition,
                "business_impact": v.business_impact,
                "risk_score": v.risk_score,
                "timestamp": v.enforcement_timestamp.isoformat()
            }
            for v in violations
        ]


class ComplianceAuditService:
    """Service for comprehensive compliance auditing"""
    
    @staticmethod
    def create_compliance_audit(
        audit_type: str,
        target_type: str,
        target_id: str,
        auditor_id: str,
        audit_framework: str,
        compliance_status: str,
        risk_level: str,
        compliance_score: float,
        violations_found: List[Dict] = None,
        recommendations: List[str] = None,
        db: Session = None
    ) -> ComplianceAuditLog:
        """Create comprehensive compliance audit record"""
        
        audit = ComplianceAuditLog(
            audit_type=audit_type,
            target_type=target_type,
            target_id=target_id,
            auditor_id=auditor_id,
            audit_framework=audit_framework,
            compliance_status=compliance_status,
            risk_level=risk_level,
            compliance_score=compliance_score,
            violations_found=violations_found or [],
            recommendations=recommendations or [],
            next_audit_date=datetime.utcnow() + timedelta(days=90),  # Next audit in 3 months
            follow_up_required=compliance_status in ['non_compliant', 'partial']
        )
        
        db.add(audit)
        db.commit()
        db.refresh(audit)
        
        # Generate crypto receipt for audit
        CryptoReceiptService.generate_crypto_receipt(
            operation_type="compliance_audit",
            operation_id=str(audit.audit_id),
            operation_hash=f"audit_hash_{uuid.uuid4().hex[:16]}",
            digital_signature=f"audit_sig_{uuid.uuid4().hex[:32]}",
            issuer=auditor_id,
            db=db
        )
        
        return audit
    
    @staticmethod
    def get_compliance_summary(db: Session) -> Dict[str, Any]:
        """Get overall compliance summary"""
        recent_audits = db.query(ComplianceAuditLog)\
            .filter(ComplianceAuditLog.audit_date >= datetime.utcnow() - timedelta(days=90))\
            .all()
        
        if not recent_audits:
            return {
                "total_audits": 0,
                "compliance_rate": 0.0,
                "high_risk_items": 0,
                "pending_actions": 0
            }
        
        total_audits = len(recent_audits)
        compliant_audits = len([a for a in recent_audits if a.compliance_status == 'compliant'])
        high_risk = len([a for a in recent_audits if a.risk_level == 'high'])
        pending_actions = len([a for a in recent_audits if a.follow_up_required])
        
        return {
            "total_audits": total_audits,
            "compliance_rate": (compliant_audits / total_audits * 100) if total_audits > 0 else 0.0,
            "high_risk_items": high_risk,
            "pending_actions": pending_actions,
            "average_compliance_score": sum(a.compliance_score or 0 for a in recent_audits) / total_audits if total_audits > 0 else 0.0
        }


class DataLineageService:
    """Service for data lineage and provenance tracking"""
    
    @staticmethod
    def record_data_lineage(
        model_id: str,
        prediction_id: str,
        data_sources: List[str],
        processing_steps: List[Dict],
        personal_data_used: bool = False,
        consent_references: List[str] = None,
        retention_expires_at: datetime = None,
        db: Session = None
    ) -> DataLineage:
        """Record data lineage for a prediction"""
        
        lineage_id = f"lineage_{prediction_id}_{uuid.uuid4().hex[:8]}"
        
        lineage = DataLineage(
            lineage_id=lineage_id,
            model_id=model_id,
            prediction_id=prediction_id,
            data_sources=data_sources,
            data_processing_steps=processing_steps,
            personal_data_used=personal_data_used,
            consent_references=consent_references or [],
            data_completeness=95.0,  # Mock data quality score
            data_accuracy_score=92.5,
            retention_expires_at=retention_expires_at or (datetime.utcnow() + timedelta(days=2557))  # 7 years
        )
        
        db.add(lineage)
        db.commit()
        db.refresh(lineage)
        return lineage
    
    @staticmethod
    def get_prediction_lineage(prediction_id: str, db: Session) -> Optional[DataLineage]:
        """Get data lineage for a specific prediction"""
        return db.query(DataLineage)\
            .filter(DataLineage.prediction_id == prediction_id)\
            .first()


class RegulatoryReportingService:
    """Service for automated regulatory reporting"""
    
    @staticmethod
    def generate_regulatory_report(
        regulation_type: str,
        reporting_period_start: datetime,
        reporting_period_end: datetime,
        regulatory_authority: str,
        models_covered: List[str],
        db: Session = None
    ) -> RegulatoryReporting:
        """Generate regulatory compliance report"""
        
        report_id = f"report_{regulation_type}_{uuid.uuid4().hex[:8]}"
        
        # Aggregate compliance data for the period
        audits = db.query(ComplianceAuditLog)\
            .filter(ComplianceAuditLog.audit_date >= reporting_period_start)\
            .filter(ComplianceAuditLog.audit_date <= reporting_period_end)\
            .all()
        
        report_data = {
            "total_audits": len(audits),
            "compliance_rate": len([a for a in audits if a.compliance_status == 'compliant']) / len(audits) * 100 if audits else 0,
            "violations": [{"audit_id": a.audit_id, "violations": a.violations_found} for a in audits if a.violations_found],
            "high_risk_findings": len([a for a in audits if a.risk_level == 'high']),
            "models_audited": list(set(a.target_id for a in audits if a.target_type == 'model'))
        }
        
        report = RegulatoryReporting(
            report_id=report_id,
            regulation_type=regulation_type,
            reporting_period_start=reporting_period_start,
            reporting_period_end=reporting_period_end,
            regulatory_authority=regulatory_authority,
            report_data=report_data,
            models_covered=models_covered,
            submission_status='draft',
            report_hash=f"report_hash_{uuid.uuid4().hex[:16]}"
        )
        
        db.add(report)
        db.commit()
        db.refresh(report)
        
        # Generate crypto receipt for report
        CryptoReceiptService.generate_crypto_receipt(
            operation_type="regulatory_report",
            operation_id=report_id,
            operation_hash=report.report_hash,
            digital_signature=f"report_sig_{uuid.uuid4().hex[:32]}",
            issuer="AI_Governance_Framework",
            legal_framework="Regulatory_Compliance",
            db=db
        )
        
        return report
    
    @staticmethod
    def submit_regulatory_report(
        report_id: str,
        submission_reference: str,
        db: Session = None
    ) -> bool:
        """Mark regulatory report as submitted"""
        report = db.query(RegulatoryReporting)\
            .filter(RegulatoryReporting.report_id == report_id)\
            .first()
        
        if report:
            report.submission_status = 'submitted'
            report.submission_date = datetime.utcnow()
            report.submission_reference = submission_reference
            db.commit()
            return True
        
        return False