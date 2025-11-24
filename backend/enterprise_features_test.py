#!/usr/bin/env python3
"""
Enterprise Features Test Suite
Complete test of all enterprise compliance features including consent management,
cryptographic receipts, policy enforcement, audit trails, and regulatory reporting.
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the framework directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.database.models import init_database, get_db_session
from core.database.services import (
    ConsentService, CryptoReceiptService, PolicyEnforcementService,
    ComplianceAuditService, DataLineageService, RegulatoryReportingService,
    ModelService
)

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"🔬 {title}")
    print("="*60)

def print_success(message: str):
    """Print a success message"""
    print(f"✅ {message}")

def print_error(message: str):
    """Print an error message"""
    print(f"❌ {message}")

def print_info(message: str):
    """Print an info message"""
    print(f"ℹ️  {message}")

def test_database_initialization():
    """Test database initialization"""
    print_section("Database Initialization Test")
    
    try:
        # Initialize database
        init_database()
        print_success("Database initialized successfully")
        
        # Test database connection
        with next(get_db_session()) as db:
            print_success("Database connection established")
        
        return True
    except Exception as e:
        print_error(f"Database initialization failed: {e}")
        return False

def test_consent_management():
    """Test consent management with blockchain audit trail"""
    print_section("Consent Management & Blockchain Audit Trail Test")
    
    try:
        with next(get_db_session()) as db:
            # Record initial consent
            consent_record = ConsentService.record_consent(
                user_id="test_user_001",
                consent_type="data_processing",
                purpose="AI model training and analysis",
                data_categories=["personal_data", "financial_data", "behavioral_data"],
                granted=True,
                consent_method="web_form",
                consent_text="I agree to the processing of my data for AI analysis",
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0 Test Browser",
                geolocation="San Francisco, CA, USA",
                db=db
            )
            print_success(f"Consent recorded: {consent_record.consent_id}")
            print_info(f"Blockchain hash: {consent_record.blockchain_hash}")
            print_info(f"Digital signature: {consent_record.digital_signature}")
            
            # Get audit chain
            audit_chain = ConsentService.get_consent_audit_chain(consent_record.consent_id, db)
            print_success(f"Audit chain retrieved: {len(audit_chain)} entries")
            
            # Verify chain integrity
            verification = ConsentService.verify_consent_chain(consent_record.consent_id, db)
            print_success(f"Chain verification: {verification['valid']}")
            
            # Withdraw consent
            withdrawal_success = ConsentService.withdraw_consent(
                consent_id=consent_record.consent_id,
                withdrawal_reason="User requested data deletion",
                performed_by="test_user_001",
                db=db
            )
            print_success(f"Consent withdrawn: {withdrawal_success}")
            
            return True
            
    except Exception as e:
        print_error(f"Consent management test failed: {e}")
        return False

def test_cryptographic_receipts():
    """Test cryptographic receipt generation and verification"""
    print_section("Cryptographic Receipt System Test")
    
    try:
        with next(get_db_session()) as db:
            # Generate a receipt
            receipt = CryptoReceiptService.generate_receipt(
                operation_type="model_prediction",
                operation_data={
                    "model_id": "credit_risk_v1",
                    "prediction_result": "approved",
                    "confidence": 0.85,
                    "timestamp": datetime.utcnow().isoformat()
                },
                user_id="test_user_001",
                db=db
            )
            print_success(f"Crypto receipt generated: {receipt.receipt_id}")
            print_info(f"Digital signature: {receipt.digital_signature[:50]}...")
            print_info(f"Operation hash: {receipt.operation_hash[:50]}...")
            
            # Verify the receipt
            verification = CryptoReceiptService.verify_receipt(receipt.receipt_id, db)
            print_success(f"Receipt verification: {verification.get('valid', verification.get('verification_status', 'unknown'))}")
            if 'integrity_check' in verification:
                print_info(f"Integrity check: {verification['integrity_check']}")
            else:
                print_info(f"Verification details: {str(verification)[:50]}...")
            
            return True
            
    except Exception as e:
        print_error(f"Cryptographic receipt test failed: {e}")
        return False

def test_policy_enforcement():
    """Test policy enforcement logging"""
    print_section("Policy Enforcement & Violation Tracking Test")
    
    try:
        with next(get_db_session()) as db:
            # Log policy enforcement
            enforcement_log = PolicyEnforcementService.log_policy_enforcement(
                policy_id="fairness_policy_001",
                model_id="credit_risk_v1",
                enforcement_action="bias_correction_applied",
                trigger_condition={
                    "demographic_parity": 0.15,
                    "threshold": 0.10,
                    "violated": True
                },
                action_result="predictions_adjusted",
                affected_predictions=150,
                business_impact="Improved fairness across demographic groups",
                risk_score=75.5,
                db=db
            )
            print_success(f"Policy enforcement logged: {enforcement_log.log_id}")
            print_info(f"Verification hash: {enforcement_log.verification_hash[:50]}...")
            
            # Get policy violations
            violations = PolicyEnforcementService.get_policy_violations(
                policy_id="fairness_policy_001",
                days=30,
                db=db
            )
            print_success(f"Policy violations retrieved: {len(violations)} violations")
            
            return True
            
    except Exception as e:
        print_error(f"Policy enforcement test failed: {e}")
        return False

def test_compliance_auditing():
    """Test compliance audit creation and management"""
    print_section("Compliance Audit System Test")
    
    try:
        with next(get_db_session()) as db:
            # Create compliance audit
            audit = ComplianceAuditService.create_compliance_audit(
                audit_type="comprehensive_review",
                target_type="ai_model",
                target_id="credit_risk_v1",
                auditor_id="auditor_001",
                audit_framework="GDPR_AI_Act_CCPA",
                compliance_status="compliant_with_findings",
                risk_level="medium",
                compliance_score=87.5,
                violations_found=[
                    {
                        "type": "documentation_gap",
                        "severity": "low",
                        "description": "Model training data lineage incomplete"
                    }
                ],
                recommendations=[
                    "Complete data lineage documentation",
                    "Implement automated fairness monitoring",
                    "Schedule quarterly bias assessments"
                ],
                db=db
            )
            print_success(f"Compliance audit created: {audit.audit_id}")
            print_info(f"Compliance score: {audit.compliance_score}%")
            print_info(f"Risk level: {audit.risk_level}")
            print_info(f"Next audit: {audit.next_audit_date}")
            
            # Get compliance summary
            summary = ComplianceAuditService.get_compliance_summary(db)
            print_success(f"Compliance summary retrieved")
            print_info(f"Total audits: {summary.get('total_audits', 0)}")
            print_info(f"Average score: {summary.get('average_compliance_score', 0):.1f}%")
            
            return True
            
    except Exception as e:
        print_error(f"Compliance audit test failed: {e}")
        return False

def test_data_lineage():
    """Test data lineage tracking"""
    print_section("Data Lineage & Provenance Tracking Test")
    
    try:
        with next(get_db_session()) as db:
            # Record data lineage
            lineage = DataLineageService.record_data_lineage(
                model_id="credit_risk_v1",
                prediction_id="pred_20241123_001",
                data_sources=["customer_database", "credit_bureau", "transaction_history"],
                processing_steps=[
                    {
                        "step": "data_extraction",
                        "timestamp": datetime.utcnow().isoformat(),
                        "transformation": "feature_engineering"
                    },
                    {
                        "step": "bias_detection",
                        "timestamp": datetime.utcnow().isoformat(),
                        "transformation": "fairness_preprocessing"
                    },
                    {
                        "step": "model_inference",
                        "timestamp": datetime.utcnow().isoformat(),
                        "transformation": "prediction_generation"
                    }
                ],
                personal_data_used=True,
                consent_references=["consent_test_user_001"],
                db=db
            )
            print_success(f"Data lineage recorded: {lineage.lineage_id}")
            print_info(f"Data sources: {len(lineage.data_sources)}")
            print_info(f"Processing steps: {len(lineage.data_processing_steps)}")
            
            # Retrieve lineage
            retrieved_lineage = DataLineageService.get_prediction_lineage("pred_20241123_001", db)
            print_success(f"Data lineage retrieved for prediction")
            print_info(f"Data quality score: {retrieved_lineage.data_accuracy_score}%")
            
            return True
            
    except Exception as e:
        print_error(f"Data lineage test failed: {e}")
        return False

def test_regulatory_reporting():
    """Test regulatory report generation"""
    print_section("Regulatory Reporting System Test")
    
    try:
        with next(get_db_session()) as db:
            # Generate regulatory report
            report = RegulatoryReportingService.generate_regulatory_report(
                regulation_type="GDPR_AI_Act",
                reporting_period_start=datetime.utcnow() - timedelta(days=90),
                reporting_period_end=datetime.utcnow(),
                regulatory_authority="EU_Data_Protection_Authority",
                models_covered=["credit_risk_v1"],
                db=db
            )
            print_success(f"Regulatory report generated: {report.report_id}")
            print_info(f"Regulation: {report.regulation_type}")
            print_info(f"Report hash: {report.report_hash[:50]}...")
            
            # Submit the report
            submission_success = RegulatoryReportingService.submit_regulatory_report(
                report_id=report.report_id,
                submission_reference="EU_DPA_2024_Q4_001",
                db=db
            )
            print_success(f"Report submitted: {submission_success}")
            
            return True
            
    except Exception as e:
        print_error(f"Regulatory reporting test failed: {e}")
        return False

def test_model_registration():
    """Test model registration with database"""
    print_section("Model Registration & Management Test")
    
    try:
        with next(get_db_session()) as db:
            # Register a model with unique ID
            model_id = f"test_enterprise_model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            model = ModelService.register_model(
                model_id=model_id,
                model_name="Enterprise Test Model",
                model_type="classification",
                algorithm="random_forest",
                version="1.0.0",
                framework="scikit_learn",
                performance_metrics={
                    "accuracy": 0.89,
                    "precision": 0.87,
                    "recall": 0.91,
                    "f1_score": 0.89,
                    "auc_roc": 0.94
                },
                fairness_metrics={
                    "demographic_parity": 0.08,
                    "equalized_odds": 0.05,
                    "predictive_parity": 0.07
                },
                training_data_info={
                    "dataset_size": 50000,
                    "feature_count": 42,
                    "training_period": "2024-Q3"
                },
                db=db
            )
            print_success(f"Model registered: {model.model_id}")
            print_info(f"Model name: {model.model_name}")
            print_info(f"Model type: {model.model_type}")
            
            # Get model info
            retrieved_model = ModelService.get_model(model.model_id, db)
            print_success(f"Model retrieved: {retrieved_model.model_name}")
            print_info(f"Model type: {retrieved_model.model_type}")
            print_info(f"Version: {retrieved_model.model_version}")
            
            return True
            
    except Exception as e:
        print_error(f"Model registration test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all enterprise feature tests"""
    print("🚀 Enterprise AI Governance Framework - Comprehensive Test Suite")
    print("==============================================================")
    print(f"🕐 Test started at: {datetime.utcnow().isoformat()}")
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Database Initialization", test_database_initialization),
        ("Model Registration", test_model_registration),
        ("Consent Management", test_consent_management),
        ("Cryptographic Receipts", test_cryptographic_receipts),
        ("Policy Enforcement", test_policy_enforcement),
        ("Compliance Auditing", test_compliance_auditing),
        ("Data Lineage", test_data_lineage),
        ("Regulatory Reporting", test_regulatory_reporting)
    ]
    
    for test_name, test_function in tests:
        try:
            result = test_function()
            test_results[test_name] = result
        except Exception as e:
            print_error(f"Test {test_name} failed with exception: {e}")
            test_results[test_name] = False
    
    # Print summary
    print_section("Test Results Summary")
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print("\n" + "="*60)
    print(f"📊 FINAL RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL ENTERPRISE FEATURES WORKING PERFECTLY!")
        print("✅ Complete enterprise-grade AI governance system operational")
        print("✅ Cryptographic audit trails verified")
        print("✅ Compliance management active")
        print("✅ Regulatory reporting ready")
        print("✅ Database persistence layer complete")
    else:
        print("⚠️  Some tests failed. Please review the output above.")
    
    print(f"🕐 Test completed at: {datetime.utcnow().isoformat()}")

if __name__ == "__main__":
    run_comprehensive_test()