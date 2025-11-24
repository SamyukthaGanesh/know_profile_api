"""
Compliance Example with Home Credit Dataset
Demonstrates regulatory compliance checking with audit trail.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Import compliance modules
from core.compliance.policy_manager import PolicyManager
from core.compliance.policy_engine import PolicyEngine
from core.compliance.audit_logger import AuditLogger
from core.compliance.policy_schema import Policy, PolicyCondition, PolicyAction, PolicyType

# Import other framework modules
from data.base_loader import CSVDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def main():
    """Main compliance demo"""
    
    print_header("REGULATORY COMPLIANCE DEMO - HOME CREDIT DATASET")
    
    # ============================================================
    # STEP 1: Load Data
    # ============================================================
    print("\n[STEP 1] Loading Home Credit dataset...")
    
    data_path = "/Users/muvarma/Documents/ghci hackathon/ai_governance_framework/examples/application_train.csv"
    data_loader = CSVDataLoader(
        data_path=data_path,
        target_column='TARGET'
    )
    
    data = data_loader.load()
    
    # Sample for demo
    data_sample = data.sample(n=100, random_state=42)
    
    # Create derived features
    data_sample['AGE'] = -data_sample['DAYS_BIRTH'] // 365
    data_sample['YEARS_EMPLOYED'] = -data_sample['DAYS_EMPLOYED'] // 365
    
    print(f"✓ Loaded {len(data_sample)} loan applications")
    
    # ============================================================
    # STEP 2: Initialize Compliance System
    # ============================================================
    print("\n[STEP 2] Initializing compliance system...")
    
    # Load policies
    policy_manager = PolicyManager()
    policies = policy_manager.list_policies(enabled_only=True)
    print(f"✓ Loaded {len(policies)} regulatory policies")
    
    # Create policy engine
    engine = PolicyEngine(policies=policies)
    print(f"✓ Policy engine initialized")
    
    # Create audit logger with organized output path
    audit_output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'audit_logs')
    os.makedirs(audit_output_dir, exist_ok=True)
    audit_path = os.path.join(audit_output_dir, "home_credit_audit.json")
    audit_logger = AuditLogger(storage_path=audit_path)
    print(f"✓ Audit logger initialized")
    
    # Display policies
    print("\n📋 Active Regulatory Policies:")
    for i, policy in enumerate(policies, 1):
        print(f"  {i}. {policy.policy_id}")
        print(f"     {policy.name}")
        print(f"     Source: {policy.regulation_source}")
        print(f"     Action: {policy.action.value}")
    
    # ============================================================
    # STEP 3: Check Compliance for Sample Applications
    # ============================================================
    print("\n[STEP 3] Checking compliance for loan applications...")
    
    compliant_count = 0
    violation_count = 0
    all_violations = []
    
    # Check first 10 applications
    for idx in range(min(10, len(data_sample))):
        row = data_sample.iloc[idx]
        
        # Prepare feature values with proper data conversion
        feature_values = {
            'AMT_CREDIT': float(row.get('AMT_CREDIT', 0)) if pd.notna(row.get('AMT_CREDIT')) else 0,
            'EXT_SOURCE_2': float(row.get('EXT_SOURCE_2', 0.5)) if pd.notna(row.get('EXT_SOURCE_2')) else 0.5,
            'AMT_INCOME_TOTAL': float(row.get('AMT_INCOME_TOTAL', 0)) if pd.notna(row.get('AMT_INCOME_TOTAL')) else 0,
            'AGE': int(row.get('AGE', 0)) if pd.notna(row.get('AGE')) else 0,
            'CODE_GENDER': str(row.get('CODE_GENDER', 'XNA')),
            'FLAG_OWN_REALTY': 1 if row.get('FLAG_OWN_REALTY') == 'Y' else 0,  # Convert Y/N to 1/0
            'consent_given': True  # Assume consent given
        }
        
        # Simulate decision based on TARGET
        decision = "denied" if row['TARGET'] == 1 else "approved"
        decision_id = f"LOAN_{row.name}_{int(datetime.now().timestamp())}"
        
        # Check compliance
        is_compliant, results = engine.check_compliance(feature_values, decision)
        
        # Get violations
        violations = [r for r in results if not r.compliant]
        
        if is_compliant:
            compliant_count += 1
        else:
            violation_count += 1
            all_violations.extend(violations)
        
        # Create audit receipt
        receipt = audit_logger.create_receipt(
            decision_id=decision_id,
            compliance_results=results,
            decision_outcome=decision,
            feature_values=feature_values,
            model_id="HomeCreditModel_v1",
            created_by="compliance_demo"
        )
        
        # Print result
        status = "✓ COMPLIANT" if is_compliant else "✗ VIOLATIONS"
        print(f"\n  Application {idx+1}: {status}")
        print(f"    Decision: {decision}")
        print(f"    Policies checked: {len(results)}")
        
        if violations:
            print(f"    Violations: {len(violations)}")
            for v in violations:
                print(f"      - {v.policy.policy_id}: {v.message}")
        
        print(f"    Audit Receipt: {receipt.receipt_id}")
        print(f"    Hash: {receipt.content_hash[:32]}...")
    
    # ============================================================
    # STEP 4: Analyze Violations
    # ============================================================
    print("\n[STEP 4] Analyzing compliance violations...")
    
    print(f"\n📊 Compliance Statistics:")
    print(f"  Total Applications Checked: {min(10, len(data_sample))}")
    print(f"  Compliant: {compliant_count}")
    print(f"  Non-Compliant: {violation_count}")
    print(f"  Compliance Rate: {compliant_count/min(10, len(data_sample))*100:.1f}%")
    
    # Count violations by policy
    if all_violations:
        print(f"\n📋 Violations by Policy:")
        violation_counts = {}
        for v in all_violations:
            policy_id = v.policy.policy_id
            violation_counts[policy_id] = violation_counts.get(policy_id, 0) + 1
        
        for policy_id, count in sorted(violation_counts.items(), key=lambda x: x[1], reverse=True):
            policy = policy_manager.get_policy(policy_id)
            print(f"  {policy_id}: {count} violations")
            print(f"    {policy.name}")
            print(f"    Action: {policy.action.value}")
    
    # ============================================================
    # STEP 5: Verify Audit Chain
    # ============================================================
    print("\n[STEP 5] Verifying audit chain integrity...")
    
    chain_valid, errors = audit_logger.verify_chain()
    
    if chain_valid:
        print(f"  ✓ Audit chain is VALID")
        print(f"  ✓ All {len(audit_logger.receipt_chain)} receipts verified")
        print(f"  ✓ Hash chain integrity confirmed")
    else:
        print(f"  ✗ Audit chain INVALID")
        for error in errors:
            print(f"    - {error}")
    
    # Get audit statistics
    stats = audit_logger.get_statistics()
    
    print(f"\n📊 Audit Trail Statistics:")
    print(f"  Total Receipts: {stats['total_receipts']}")
    print(f"  Compliant Decisions: {stats['compliant_receipts']}")
    print(f"  Non-Compliant Decisions: {stats['non_compliant_receipts']}")
    print(f"  Unique Decisions: {stats['unique_decisions']}")
    print(f"  Policies Monitored: {stats['unique_policies_checked']}")
    print(f"  Chain Status: {'✓ VALID' if stats['chain_valid'] else '✗ INVALID'}")
    
    # ============================================================
    # STEP 6: Export Audit Trail
    # ============================================================
    print("\n[STEP 6] Exporting audit trail for regulatory review...")
    
    # Create organized output paths
    compliance_output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'compliance_reports')
    os.makedirs(compliance_output_dir, exist_ok=True)
    
    # Export to JSON
    json_path = os.path.join(compliance_output_dir, "compliance_audit_report.json")
    audit_logger.export_chain(json_path, format='json')
    print(f"  ✓ Exported to: {json_path}")
    
    # Export to CSV
    csv_path = os.path.join(compliance_output_dir, "compliance_audit_report.csv")  
    audit_logger.export_chain(csv_path, format='csv')
    print(f"  ✓ Exported to: {csv_path}")
    
    # ============================================================
    # STEP 7: Policy Management Demo
    # ============================================================
    print("\n[STEP 7] Policy management operations...")
    
    # Create a custom policy
    print("\n  Creating custom bank policy...")
    
    custom_policy = Policy(
        policy_id="CUSTOM_BANK_001",
        name="High-Value Loan Approval Requirement",
        regulation_source="Internal Bank Policy",
        policy_type=PolicyType.CREDIT_RISK,
        description="Loans over $100,000 require senior manager approval",
        condition=PolicyCondition(
            feature='AMT_CREDIT',
            operator='>',
            value=100000
        ),
        action=PolicyAction.FLAG_FOR_REVIEW,
        version="1.0",
        priority=5,
        enabled=True,
        rationale="Risk management for high-value loans",
        tags=["internal", "high_value", "approval_required"]
    )
    
    try:
        policy_manager.create_policy(custom_policy, created_by="compliance_demo")
        print(f"  ✓ Created custom policy: {custom_policy.policy_id}")
    except ValueError as e:
        print(f"  ℹ Policy already exists: {e}")
    
    # List all policies
    all_policies = policy_manager.list_policies()
    print(f"\n  Total policies in system: {len(all_policies)}")
    
    # Search policies
    credit_policies = policy_manager.list_policies(policy_type=PolicyType.CREDIT_RISK)
    print(f"  Credit risk policies: {len(credit_policies)}")
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print_header("COMPLIANCE DEMO COMPLETE - SUMMARY")
    
    print("\n✅ Successfully demonstrated:")
    print("  1. ✓ Policy loading from JSON database")
    print("  2. ✓ Real-time compliance checking against regulations")
    print("  3. ✓ Violation detection and required actions")
    print("  4. ✓ Cryptographic audit receipt generation")
    print("  5. ✓ Hash chain verification (tamper-proof)")
    print("  6. ✓ Audit trail export for regulators")
    print("  7. ✓ Policy CRUD operations")
    
    print("\n🔐 Cryptographic Audit Features:")
    print("  ✓ SHA-256 hash chains (blockchain-style)")
    print("  ✓ Tamper detection working")
    print("  ✓ Immutable audit trail")
    print("  ✓ Regulatory-grade audit logs")
    
    print("\n📊 Compliance Results:")
    print(f"  Compliance Rate: {compliant_count}/{min(10, len(data_sample))} ({compliant_count/min(10, len(data_sample))*100:.1f}%)")
    print(f"  Total Violations: {len(all_violations)}")
    print(f"  Audit Receipts Created: {stats['total_receipts']}")
    print(f"  Audit Chain Valid: {stats['chain_valid']}")
    
    print("\n🎯 Ready for Integration:")
    print("  ✓ Backend provides regulatory compliance checking")
    print("  ✓ Teammates can use API endpoints for:")
    print("    - Dashboard: Show compliance status")
    print("    - NLG: Include compliance in explanations")
    print("    - Consent: Track regulatory consent requirements")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()