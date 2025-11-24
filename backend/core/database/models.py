"""
Database Models for AI Governance Framework
Clean, modular SQLite database schema replacing all JSON persistence.
"""

from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Integer, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os

Base = declarative_base()

# ============================================================
# CORE MODELS - Essential for all operations
# ============================================================

class ModelRegistry(Base):
    """Central registry for all AI models"""
    __tablename__ = 'models_registry'
    
    model_id = Column(String(255), primary_key=True)
    model_name = Column(String(255), nullable=False)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # 'classification', 'regression'
    feature_names = Column(JSON, nullable=False)
    status = Column(String(20), default='active')  # 'active', 'inactive', 'deprecated'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Performance metrics
    accuracy = Column(Float)
    last_prediction_at = Column(DateTime)
    predictions_today = Column(Integer, default=0)
    
    # Relationships
    fairness_analyses = relationship("FairnessAnalysis", back_populates="model")
    explanations = relationship("ExplanationRecord", back_populates="model")


class FairnessAnalysis(Base):
    """Fairness analysis results - replaces analysis_{model_id}.json"""
    __tablename__ = 'fairness_analysis'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(255), ForeignKey('models_registry.model_id'), nullable=False)
    analysis_id = Column(String(255), unique=True, nullable=False)
    
    # Core fairness metrics
    overall_fairness_score = Column(Float, nullable=False)
    bias_detected = Column(Boolean, nullable=False)
    bias_severity = Column(String(20), nullable=False)  # 'none', 'low', 'moderate', 'high', 'critical'
    
    # Sensitive feature analysis
    sensitive_feature_name = Column(String(100), nullable=False)
    group_metrics = Column(JSON, nullable=False)  # Store group-wise metrics
    
    # Analysis context
    sample_size = Column(Integer, nullable=False)
    recommendations = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("ModelRegistry", back_populates="fairness_analyses")


class ExplanationRecord(Base):
    """Individual prediction explanations - replaces scattered explanation data"""
    __tablename__ = 'explanations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(255), ForeignKey('models_registry.model_id'), nullable=False)
    instance_id = Column(String(255), nullable=False)
    
    # Prediction details
    prediction_value = Column(Float, nullable=False)
    prediction_label = Column(String(100))
    confidence = Column(Float)
    
    # Explanation details
    explanation_type = Column(String(50), nullable=False)  # 'shap', 'lime', 'anchors'
    feature_contributions = Column(JSON, nullable=False)
    explanation_text = Column(Text)
    base_value = Column(Float)
    
    # User context (for user-friendly explanations)
    audience_type = Column(String(50))  # 'customer', 'loan_officer', 'data_scientist'
    simple_explanation = Column(Text)
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("ModelRegistry", back_populates="explanations")


class BiasOptimization(Base):
    """Bias optimization results - replaces optimization_{model_id}.json"""
    __tablename__ = 'bias_optimization'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(255), ForeignKey('models_registry.model_id'), nullable=False)
    optimization_id = Column(String(255), unique=True, nullable=False)
    
    # Optimization configuration
    mitigation_strategy = Column(String(50), nullable=False)  # 'reduction', 'postprocess', 'preprocessing'
    fairness_objective = Column(String(100), nullable=False)
    
    # Results
    optimization_successful = Column(Boolean, nullable=False)
    fairness_improvement = Column(Float)
    new_fairness_score = Column(Float)
    optimization_summary = Column(Text)
    
    # Metrics comparison
    before_metrics = Column(JSON)
    after_metrics = Column(JSON)
    
    timestamp = Column(DateTime, default=datetime.utcnow)


class SystemHealth(Base):
    """System health monitoring - replaces scattered health JSON files"""
    __tablename__ = 'system_health'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    service_name = Column(String(100), nullable=False)  # 'fairness', 'explainability', 'framework'
    
    # Health metrics
    status = Column(String(20), nullable=False)  # 'healthy', 'warning', 'critical'
    uptime_seconds = Column(Integer)
    memory_usage_mb = Column(Float)
    cpu_usage_percent = Column(Float)
    response_time_ms = Column(Float)
    
    # Error tracking
    error_count = Column(Integer, default=0)
    last_error_message = Column(Text)
    
    timestamp = Column(DateTime, default=datetime.utcnow)


# ============================================================
# POLICIES & COMPLIANCE MODELS
# ============================================================

class PolicyRegistry(Base):
    """Policy definitions - for compliance management"""
    __tablename__ = 'policies'
    
    policy_id = Column(String(255), primary_key=True)
    policy_name = Column(String(255), nullable=False)
    policy_type = Column(String(100), nullable=False)
    regulation_source = Column(String(50), nullable=False)  # 'gdpr', 'ccpa', 'ai_act', 'internal'
    
    policy_content = Column(JSON, nullable=False)
    effective_date = Column(DateTime, nullable=False)
    expiry_date = Column(DateTime)
    status = Column(String(20), default='active')
    
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ConsentRecord(Base):
    """User consent tracking - for GDPR compliance"""
    __tablename__ = 'consent_records'
    
    consent_id = Column(String(255), primary_key=True)
    user_id = Column(String(255), nullable=False)
    
    # Consent details
    consent_type = Column(String(100), nullable=False)
    purpose = Column(String(500), nullable=False)
    data_categories = Column(JSON, nullable=False)
    
    # Consent status
    granted = Column(Boolean, nullable=False)
    consent_method = Column(String(100), nullable=False)
    consent_text = Column(Text, nullable=False)
    
    # Timing
    granted_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime)
    withdrawal_at = Column(DateTime)
    withdrawal_reason = Column(String(500))
    
    # Audit trail and crypto verification
    blockchain_hash = Column(String(255))
    digital_signature = Column(String(500))  # Digital signature for authenticity
    legal_basis = Column(String(50), default='consent')
    
    # Compliance metadata
    ip_address = Column(String(45))
    user_agent = Column(Text)
    geolocation = Column(String(100))
    consent_version = Column(String(50), default='1.0')


class ConsentAuditTrail(Base):
    """Detailed audit trail for consent changes - immutable blockchain-style records"""
    __tablename__ = 'consent_audit_trail'
    
    audit_id = Column(Integer, primary_key=True, autoincrement=True)
    consent_id = Column(String(255), ForeignKey('consent_records.consent_id'), nullable=False)
    block_hash = Column(String(255), unique=True, nullable=False)
    previous_block_hash = Column(String(255))
    
    # Action details
    action_type = Column(String(50), nullable=False)  # 'granted', 'withdrawn', 'expired', 'renewed'
    action_timestamp = Column(DateTime, nullable=False)
    performed_by = Column(String(255))  # User ID or system identifier
    
    # Verification data
    digital_signature = Column(String(500))
    timestamp_signature = Column(String(500))  # Timestamping authority signature
    merkle_root = Column(String(255))  # For batch verification
    
    # Context
    action_metadata = Column(JSON)
    verification_proofs = Column(JSON)  # Additional cryptographic proofs


class PolicyEnforcementLog(Base):
    """Policy enforcement actions and violations"""
    __tablename__ = 'policy_enforcement_log'
    
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    policy_id = Column(String(255), ForeignKey('policies.policy_id'), nullable=False)
    model_id = Column(String(255), ForeignKey('models_registry.model_id'))
    
    # Enforcement details
    enforcement_action = Column(String(100), nullable=False)  # 'block_prediction', 'require_explanation', 'flag_review'
    trigger_condition = Column(JSON, nullable=False)  # What triggered the policy
    action_result = Column(String(50), nullable=False)  # 'success', 'failure', 'partial'
    
    # Impact assessment
    affected_predictions = Column(Integer, default=0)
    business_impact = Column(String(500))
    risk_score = Column(Float)
    
    # Timestamps and signatures
    enforcement_timestamp = Column(DateTime, default=datetime.utcnow)
    verification_hash = Column(String(255))
    compliance_officer_approval = Column(String(255))


class CryptographicReceipts(Base):
    """Cryptographic receipts for all critical operations"""
    __tablename__ = 'crypto_receipts'
    
    receipt_id = Column(String(255), primary_key=True)
    operation_type = Column(String(100), nullable=False)  # 'consent_granted', 'policy_enforced', 'model_prediction'
    operation_id = Column(String(255), nullable=False)  # References to other table records
    
    # Cryptographic proofs
    operation_hash = Column(String(255), nullable=False)
    merkle_proof = Column(JSON)  # Merkle tree proof for batch operations
    digital_signature = Column(String(500), nullable=False)
    timestamp_signature = Column(String(500))  # Trusted timestamping
    
    # Verification data
    public_key_fingerprint = Column(String(255))
    certificate_chain = Column(JSON)
    signature_algorithm = Column(String(50), default='RSA-SHA256')
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    issuer = Column(String(255), nullable=False)  # Who issued the receipt
    verification_url = Column(String(500))  # URL for external verification
    
    # Legal compliance
    legal_framework = Column(String(100))  # 'eIDAS', 'ESIGN', 'UETA'
    retention_period = Column(Integer)  # Retention period in days


class ComplianceAuditLog(Base):
    """Comprehensive compliance audit log with regulatory reporting"""
    __tablename__ = 'compliance_audit_log'
    
    audit_id = Column(Integer, primary_key=True, autoincrement=True)
    audit_type = Column(String(100), nullable=False)  # 'gdpr_check', 'ai_act_compliance', 'internal_audit'
    
    # Audit scope
    target_type = Column(String(50), nullable=False)  # 'model', 'system', 'policy', 'consent_process'
    target_id = Column(String(255), nullable=False)
    auditor_id = Column(String(255), nullable=False)
    audit_framework = Column(String(100), nullable=False)  # 'GDPR', 'CCPA', 'AI_ACT', 'SOX', 'ISO27001'
    
    # Audit results
    compliance_status = Column(String(20), nullable=False)  # 'compliant', 'non_compliant', 'partial', 'under_review'
    risk_level = Column(String(20), nullable=False)  # 'low', 'medium', 'high', 'critical'
    compliance_score = Column(Float)  # 0-100 score
    
    # Findings and actions
    violations_found = Column(JSON)  # Detailed violation descriptions
    recommendations = Column(JSON)  # Remediation recommendations
    required_actions = Column(JSON)  # Mandatory actions with deadlines
    
    # Timeline and follow-up
    audit_date = Column(DateTime, default=datetime.utcnow)
    next_audit_date = Column(DateTime)
    remediation_deadline = Column(DateTime)
    follow_up_required = Column(Boolean, default=False)
    
    # Documentation
    audit_report_path = Column(String(500))
    evidence_links = Column(JSON)
    regulatory_filing_ref = Column(String(255))  # Reference for regulatory submissions


class DataLineage(Base):
    """Data lineage tracking for AI model inputs and decisions"""
    __tablename__ = 'data_lineage'
    
    lineage_id = Column(String(255), primary_key=True)
    model_id = Column(String(255), ForeignKey('models_registry.model_id'), nullable=False)
    prediction_id = Column(String(255), nullable=False)
    
    # Data source tracking
    data_sources = Column(JSON, nullable=False)  # List of source systems/databases
    data_processing_steps = Column(JSON, nullable=False)  # Transformation pipeline
    feature_derivation = Column(JSON)  # How features were calculated
    
    # Data quality metrics
    data_completeness = Column(Float)  # % of complete data
    data_accuracy_score = Column(Float)  # Data quality score
    outlier_flags = Column(JSON)  # Flagged outliers or anomalies
    
    # Privacy and consent tracking
    personal_data_used = Column(Boolean, default=False)
    consent_references = Column(JSON)  # References to applicable consent records
    data_retention_policy = Column(String(255))
    anonymization_applied = Column(Boolean, default=False)
    
    # Timestamps
    data_collection_time = Column(DateTime)
    processing_time = Column(DateTime, default=datetime.utcnow)
    retention_expires_at = Column(DateTime)


class RegulatoryReporting(Base):
    """Automated regulatory reporting and submission tracking"""
    __tablename__ = 'regulatory_reporting'
    
    report_id = Column(String(255), primary_key=True)
    regulation_type = Column(String(100), nullable=False)  # 'gdpr_art30', 'ccpa_annual', 'ai_act_submission'
    reporting_period_start = Column(DateTime, nullable=False)
    reporting_period_end = Column(DateTime, nullable=False)
    
    # Report contents
    report_data = Column(JSON, nullable=False)  # Aggregated compliance metrics
    models_covered = Column(JSON)  # List of models included in report
    incidents_reported = Column(JSON)  # Any incidents or violations
    
    # Submission tracking
    submission_status = Column(String(50), default='draft')  # 'draft', 'submitted', 'accepted', 'rejected'
    submission_date = Column(DateTime)
    regulatory_authority = Column(String(255))  # Which authority receives the report
    submission_reference = Column(String(255))  # Authority's reference number
    
    # Follow-up
    authority_response = Column(JSON)
    follow_up_required = Column(Boolean, default=False)
    next_submission_due = Column(DateTime)
    
    # Verification
    report_hash = Column(String(255))
    digital_signature = Column(String(500))
    compliance_officer_approval = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================================
# DATABASE CONNECTION & SETUP
# ============================================================

class DatabaseManager:
    """Centralized database management with thread safety"""
    
    def __init__(self, database_url=None):
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Database configuration with thread safety
        if database_url is None:
            db_path = os.path.join(data_dir, 'ai_governance.db')
            database_url = f"sqlite:///{db_path}"
        
        # Create engine with thread safety and connection pooling
        self.engine = create_engine(
            database_url, 
            echo=False,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={
                'check_same_thread': False,  # Allow SQLite to be used across threads
                'timeout': 30
            }
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False  # Prevent lazy loading issues in different threads
        )
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all tables"""
        Base.metadata.drop_all(bind=self.engine)


# Global database manager instance
db_manager = DatabaseManager()


# ============================================================
# DATABASE UTILITIES
# ============================================================

def get_db_session():
    """Get database session for dependency injection with proper error handling"""
    session = db_manager.get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def init_database():
    """Initialize database with tables"""
    db_manager.create_tables()
    print("✅ Database tables created successfully")


def reset_database():
    """Reset database (for development/testing)"""
    db_manager.drop_tables()
    db_manager.create_tables()
    print("🔄 Database reset successfully")


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()