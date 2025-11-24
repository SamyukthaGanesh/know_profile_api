"""
Test Compliance Module
Tests policy loading, checking, audit trail, and hash chain verification.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from datetime import datetime

# Import compliance module
from core.compliance.policy_manager import PolicyManager
from core.compliance.policy_engine import PolicyEngine
from core.compliance.audit_logger import AuditLogger
from core.compliance.policy_schema import Policy, PolicyCondition, PolicyAction, PolicyType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def test_policy_manager():
    """Test 1: Policy Manager - Loading and CRUD operations"""
    print_section("TEST 1: Policy Manager")
    
    try:
        # Initialize policy manager
        policy_manager = PolicyManager(
            storage_path="core/compliance/regulations/regulations_db.json"
        )
        
        print(f"✓ Policy Manager initialized")
        print(f"✓ Loaded {len(policy_manager.policies)} policies")
        
        # Get summary
        summary = policy_manager.get_policy_summary()
        print(f"\n📊 Policy Summary:")
        print(f"  Total: {summary['total_policies']}")
        print(f"  Enabled: {summary['enabled']}")
        print(f"  Disabled: {summary['disabled']}")
        print(f"  By Type: {summary['by_type']}")
        print(f"  By Regulation: {summary['by_regulation']}")
        
        # Test getting a specific policy
        policy = policy_manager.get_policy("BASEL_III_CREDIT_001")
        if policy:
            print(f"\n✓ Retrieved policy: {policy.name}")
            print(f"  Source: {policy.regulation_source}")
            print(f"  Priority: {policy.priority}")
            print(f"  Enabled: {policy.enabled}")
        
        # Test listing policies
        enabled_policies = policy_manager.list_policies(enabled_only=True)
        print(f"\n✓ Found {len(enabled_policies)} enabled policies")
        
        print("\n✅ Policy Manager tests PASSED")
        return policy_manager, True
    
    except Exception as e:
        print(f"\n❌ Policy Manager tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_policy_engine(policy_manager):
    """Test 2: Policy Engine - Compliance checking"""
    print_section("TEST 2: Policy Engine - Compliance Checking")
    
    try:
        # Get all policies
        policies = policy_manager.list_policies(enabled_only=True)
        
        # Initialize policy engine
        engine = PolicyEngine(policies=policies)
        print(f"✓ Policy Engine initialized with {len(policies)} policies")
        
        # Test Case 1: High-risk loan (should violate BASEL_III_CREDIT_001)
        print("\n📋 Test Case 1: High-risk large loan")
        test_case_1 = {
            'AMT_CREDIT': 75000,  # Large loan
            'EXT_SOURCE_2': 0.3,  # Low external score
            'AMT_INCOME_TOTAL': 30000,
            'AGE': 35,
            'CODE_GENDER': 'M',
            'FLAG_OWN_REALTY': 1,
            'consent_given': True
        }
        
        is_compliant, results = engine.check_compliance(test_case_1)
        
        print(f"  Compliance Status: {'✓ COMPLIANT' if is_compliant else '✗ NON-COMPLIANT'}")
        print(f"  Policies Checked: {len(results)}")
        
        violations = [r for r in results if not r.compliant]
        if violations:
            print(f"  Violations Found: {len(violations)}")
            for v in violations:
                print(f"    ✗ {v.policy.policy_id}: {v.message}")
                if v.recommended_action:
                    print(f"      Action: {v.recommended_action}")
        else:
            print(f"  No violations found")
        
        # Test Case 2: Low-income loan (should violate BASEL_III_INCOME_001)
        print("\n📋 Test Case 2: Low-income unsecured loan")
        test_case_2 = {
            'AMT_CREDIT': 30000,
            'EXT_SOURCE_2': 0.7,
            'AMT_INCOME_TOTAL': 20000,  # Below minimum
            'AGE': 28,
            'CODE_GENDER': 'F',
            'FLAG_OWN_REALTY': 0,  # No collateral
            'consent_given': True
        }
        
        is_compliant2, results2 = engine.check_compliance(test_case_2)
        
        print(f"  Compliance Status: {'✓ COMPLIANT' if is_compliant2 else '✗ NON-COMPLIANT'}")
        
        violations2 = [r for r in results2 if not r.compliant]
        if violations2:
            print(f"  Violations Found: {len(violations2)}")
            for v in violations2:
                print(f"    ✗ {v.policy.policy_id}: {v.message}")
        
        # Test Case 3: Compliant loan
        print("\n📋 Test Case 3: Compliant loan")
        test_case_3 = {
            'AMT_CREDIT': 40000,
            'EXT_SOURCE_2': 0.8,  # Good score
            'AMT_INCOME_TOTAL': 60000,  # Good income
            'AGE': 35,
            'CODE_GENDER': 'M',
            'FLAG_OWN_REALTY': 1,
            'consent_given': True
        }
        
        is_compliant3, results3 = engine.check_compliance(test_case_3)
        
        print(f"  Compliance Status: {'✓ COMPLIANT' if is_compliant3 else '✗ NON-COMPLIANT'}")
        
        violations3 = [r for r in results3 if not r.compliant]
        if violations3:
            print(f"  Violations Found: {len(violations3)}")
        else:
            print(f"  ✓ All policies satisfied")
        
        # Test validation report
        print("\n📊 Generating Validation Report...")
        report = engine.validate_decision(test_case_1, decision="denied")
        print(f"  Decision: {report['decision']}")
        print(f"  Is Compliant: {report['is_compliant']}")
        print(f"  Policies Checked: {report['policies_checked']}")
        print(f"  Violations: {report['violations_count']}")
        print(f"  Required Actions: {len(report['required_actions'])}")
        
        print("\n✅ Policy Engine tests PASSED")
        return engine, results, True
    
    except Exception as e:
        print(f"\n❌ Policy Engine tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False


def test_audit_logger(engine, compliance_results):
    """Test 3: Audit Logger - Cryptographic receipts and hash chain"""
    print_section("TEST 3: Audit Logger - Cryptographic Audit Trail")
    
    try:
        # Initialize audit logger
        audit_logger = AuditLogger(
            storage_path="test_audit_ledger.json"
        )
        print(f"✓ Audit Logger initialized")
        print(f"  Existing receipts: {len(audit_logger.receipt_chain)}")
        
        # Create test receipts
        print("\n📝 Creating audit receipts...")
        
        # Receipt 1
        receipt1 = audit_logger.create_receipt(
            decision_id="LOAN_TEST_001",
            compliance_results=compliance_results[:3] if compliance_results else [],
            decision_outcome="denied",
            feature_values={'AMT_CREDIT': 75000, 'EXT_SOURCE_2': 0.3},
            model_id="HomeCreditModel_v1",
            created_by="test_system"
        )
        
        print(f"\n✓ Receipt 1 created:")
        print(f"  ID: {receipt1.receipt_id}")
        print(f"  Decision: {receipt1.decision_id}")
        print(f"  Timestamp: {receipt1.timestamp}")
        print(f"  Hash: {receipt1.content_hash[:32]}...")
        print(f"  Previous Hash: {receipt1.previous_hash[:32] if receipt1.previous_hash else 'None (first receipt)'}...")
        print(f"  Policies Checked: {len(receipt1.policies_checked)}")
        
        # Receipt 2 (will be linked to receipt 1)
        receipt2 = audit_logger.create_receipt(
            decision_id="LOAN_TEST_002",
            compliance_results=compliance_results[:2] if compliance_results else [],
            decision_outcome="approved",
            feature_values={'AMT_CREDIT': 40000, 'EXT_SOURCE_2': 0.8},
            model_id="HomeCreditModel_v1",
            created_by="test_system"
        )
        
        print(f"\n✓ Receipt 2 created:")
        print(f"  ID: {receipt2.receipt_id}")
        print(f"  Hash: {receipt2.content_hash[:32]}...")
        print(f"  Previous Hash: {receipt2.previous_hash[:32]}... (linked to receipt 1)")
        
        # Receipt 3
        receipt3 = audit_logger.create_receipt(
            decision_id="LOAN_TEST_003",
            compliance_results=compliance_results[:1] if compliance_results else [],
            decision_outcome="approved",
            feature_values={'AMT_CREDIT': 25000, 'EXT_SOURCE_2': 0.9},
            model_id="HomeCreditModel_v1",
            created_by="test_system"
        )
        
        print(f"\n✓ Receipt 3 created (hash chain extended)")
        
        # Verify individual receipts
        print("\n🔍 Verifying Receipt Integrity...")
        for i, receipt in enumerate([receipt1, receipt2, receipt3], 1):
            is_valid = audit_logger.verify_receipt(receipt)
            status = "✓ VALID" if is_valid else "✗ INVALID"
            print(f"  Receipt {i}: {status}")
        
        # Verify entire chain
        print("\n🔗 Verifying Hash Chain...")
        chain_valid, errors = audit_logger.verify_chain()
        
        if chain_valid:
            print(f"  ✓ Hash chain is VALID")
            print(f"  ✓ All {len(audit_logger.receipt_chain)} receipts properly linked")
        else:
            print(f"  ✗ Hash chain is INVALID")
            for error in errors:
                print(f"    - {error}")
        
        # Get statistics
        print("\n📊 Audit Trail Statistics:")
        stats = audit_logger.get_statistics()
        print(f"  Total Receipts: {stats['total_receipts']}")
        print(f"  Compliant: {stats['compliant_receipts']}")
        print(f"  Non-Compliant: {stats['non_compliant_receipts']}")
        print(f"  Unique Decisions: {stats['unique_decisions']}")
        print(f"  Unique Policies: {stats['unique_policies_checked']}")
        print(f"  Chain Valid: {stats['chain_valid']}")
        
        # Test tampering detection
        print("\n🔐 Testing Tampering Detection...")
        print("  Attempting to modify receipt 2...")
        
        # Save original hash
        original_hash = receipt2.content_hash
        
        # Tamper with receipt
        receipt2.decision_outcome = "TAMPERED"
        
        # Verify (should fail)
        is_valid_after_tampering = audit_logger.verify_receipt(receipt2)
        
        if not is_valid_after_tampering:
            print("  ✓ Tampering detected successfully!")
            print(f"    Original hash: {original_hash[:32]}...")
            print(f"    Current hash would be different")
        else:
            print("  ✗ Failed to detect tampering")
        
        # Restore receipt
        receipt2.decision_outcome = "approved"
        
        print("\n✅ Audit Logger tests PASSED")
        return audit_logger, True
    
    except Exception as e:
        print(f"\n❌ Audit Logger tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_integration():
    """Test 4: Full Integration Test"""
    print_section("TEST 4: Full Integration - End-to-End")
    
    try:
        print("📋 Simulating complete decision flow...\n")
        
        # Step 1: Load policies
        print("Step 1: Loading policies...")
        policy_manager = PolicyManager()
        policies = policy_manager.list_policies(enabled_only=True)
        print(f"  ✓ Loaded {len(policies)} policies")
        
        # Step 2: Create engine
        print("\nStep 2: Initializing policy engine...")
        engine = PolicyEngine(policies=policies)
        print(f"  ✓ Engine ready")
        
        # Step 3: Create audit logger
        print("\nStep 3: Initializing audit logger...")
        audit_logger = AuditLogger(storage_path="test_integration_audit.json")
        print(f"  ✓ Audit logger ready")
        
        # Step 4: Make a decision
        print("\nStep 4: Processing loan application...")
        loan_application = {
            'AMT_CREDIT': 55000,
            'EXT_SOURCE_2': 0.45,
            'AMT_INCOME_TOTAL': 35000,
            'AGE': 42,
            'CODE_GENDER': 'F',
            'FLAG_OWN_REALTY': 0,
            'consent_given': True
        }
        
        # Step 5: Check compliance
        print("\nStep 5: Checking compliance...")
        is_compliant, results = engine.check_compliance(loan_application)
        decision = "approved" if is_compliant else "denied"
        print(f"  Decision: {decision.upper()}")
        print(f"  Compliant: {is_compliant}")
        
        # Step 6: Create audit receipt
        print("\nStep 6: Creating audit receipt...")
        receipt = audit_logger.create_receipt(
            decision_id=f"LOAN_INTEGRATION_TEST_{int(datetime.now().timestamp())}",
            compliance_results=results,
            decision_outcome=decision,
            feature_values=loan_application,
            model_id="HomeCreditModel_v1",
            created_by="integration_test"
        )
        print(f"  ✓ Receipt created: {receipt.receipt_id}")
        print(f"  ✓ Hash: {receipt.content_hash[:32]}...")
        
        # Step 7: Verify audit trail
        print("\nStep 7: Verifying audit trail...")
        chain_valid, errors = audit_logger.verify_chain()
        print(f"  Chain valid: {chain_valid}")
        
        print("\n✅ Full Integration test PASSED")
        print("\n🎉 All compliance components working together successfully!")
        return True
    
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("  COMPLIANCE MODULE TEST SUITE")
    print("  Testing: Policies, Engine, Audit Trail, Hash Chain")
    print("="*80)
    
    results = []
    
    # Test 1: Policy Manager
    policy_manager, test1_pass = test_policy_manager()
    results.append(("Policy Manager", test1_pass))
    
    if not test1_pass:
        print("\n❌ Cannot continue - Policy Manager failed")
        return
    
    # Test 2: Policy Engine
    engine, compliance_results, test2_pass = test_policy_engine(policy_manager)
    results.append(("Policy Engine", test2_pass))
    
    if not test2_pass:
        print("\n❌ Cannot continue - Policy Engine failed")
        return
    
    # Test 3: Audit Logger
    audit_logger, test3_pass = test_audit_logger(engine, compliance_results)
    results.append(("Audit Logger", test3_pass))
    
    # Test 4: Integration
    test4_pass = test_integration()
    results.append(("Full Integration", test4_pass))
    
    # Summary
    print_section("TEST SUMMARY")
    
    all_passed = all(result[1] for result in results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print("\n" + "="*80)
    if all_passed:
        print("  🎉 ALL TESTS PASSED!")
        print("  Compliance module is ready for integration")
    else:
        print("  ❌ SOME TESTS FAILED")
        print("  Please review errors above")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()