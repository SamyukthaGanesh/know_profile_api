"""
Blockchain Visualization Example
Demonstrates blockchain-style visualization of audit and consent chains.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import logging
from datetime import datetime

# Import modules
from core.compliance.audit_logger import AuditLogger
from core.compliance.policy_manager import PolicyManager
from core.compliance.policy_engine import PolicyEngine
from core.consent.consent_manager import ConsentManager
from core.consent.consent_wallet import ConsentWallet
from core.consent.consent_schema import ConsentPurpose

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def visualize_compliance_chain():
    """Visualize compliance audit chain"""
    print_header("COMPLIANCE BLOCKCHAIN VISUALIZATION")
    
    # Initialize
    policy_manager = PolicyManager()
    policies = policy_manager.list_policies(enabled_only=True)
    engine = PolicyEngine(policies=policies)
    
    # Initialize audit system with organized output path
    audit_output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'audit_logs')
    os.makedirs(audit_output_dir, exist_ok=True)
    audit_path = os.path.join(audit_output_dir, "blockchain_demo_audit.json")
    audit_logger = AuditLogger(storage_path=audit_path)
    
    print(f"\n✓ Initialized with {len(policies)} policies")
    
    # Create sample decisions with audit trail
    print("\n📝 Creating sample decision chain...")
    
    sample_decisions = [
        {
            'id': 'LOAN_001',
            'features': {'AMT_CREDIT': 60000, 'EXT_SOURCE_2': 0.4, 'AGE': 35, 'CODE_GENDER': 'M', 'FLAG_OWN_REALTY': 1, 'AMT_INCOME_TOTAL': 50000, 'consent_given': True},
            'outcome': 'denied'
        },
        {
            'id': 'LOAN_002',
            'features': {'AMT_CREDIT': 30000, 'EXT_SOURCE_2': 0.8, 'AGE': 28, 'CODE_GENDER': 'F', 'FLAG_OWN_REALTY': 1, 'AMT_INCOME_TOTAL': 60000, 'consent_given': True},
            'outcome': 'approved'
        },
        {
            'id': 'LOAN_003',
            'features': {'AMT_CREDIT': 45000, 'EXT_SOURCE_2': 0.6, 'AGE': 42, 'CODE_GENDER': 'M', 'FLAG_OWN_REALTY': 0, 'AMT_INCOME_TOTAL': 40000, 'consent_given': True},
            'outcome': 'approved'
        },
        {
            'id': 'LOAN_004',
            'features': {'AMT_CREDIT': 80000, 'EXT_SOURCE_2': 0.3, 'AGE': 50, 'CODE_GENDER': 'F', 'FLAG_OWN_REALTY': 1, 'AMT_INCOME_TOTAL': 35000, 'consent_given': True},
            'outcome': 'denied'
        },
        {
            'id': 'LOAN_005',
            'features': {'AMT_CREDIT': 25000, 'EXT_SOURCE_2': 0.9, 'AGE': 31, 'CODE_GENDER': 'M', 'FLAG_OWN_REALTY': 1, 'AMT_INCOME_TOTAL': 70000, 'consent_given': True},
            'outcome': 'approved'
        }
    ]
    
    for decision in sample_decisions:
        # Check compliance
        is_compliant, results = engine.check_compliance(
            decision['features'],
            decision['outcome']
        )
        
        # Create audit receipt (blockchain block)
        receipt = audit_logger.create_receipt(
            decision_id=decision['id'],
            compliance_results=results,
            decision_outcome=decision['outcome'],
            feature_values=decision['features'],
            model_id="HomeCreditModel_v1",
            created_by="blockchain_demo"
        )
        
        status = "✓ COMPLIANT" if is_compliant else "✗ VIOLATIONS"
        violations_count = sum(1 for r in results if not r.compliant)
        
        print(f"  Block {len(audit_logger.receipt_chain)-1}: {decision['id']} - {status}")
        print(f"    Hash: {receipt.content_hash[:32]}...")
        print(f"    Previous: {receipt.previous_hash[:32] if receipt.previous_hash else 'Genesis'}...")
        print(f"    Violations: {violations_count}")
    
    # Verify chain
    print("\n🔗 Verifying blockchain...")
    chain_valid, errors = audit_logger.verify_chain()
    
    if chain_valid:
        print(f"  ✓ Blockchain is VALID")
        print(f"  ✓ All {len(audit_logger.receipt_chain)} blocks verified")
    else:
        print(f"  ✗ Blockchain is INVALID")
        for error in errors:
            print(f"    - {error}")
    
    # Generate graph data
    print("\n📊 Generating graph visualization data...")
    
    graph_data = {
        'nodes': [],
        'edges': []
    }
    
    for i, receipt in enumerate(audit_logger.receipt_chain):
        is_compliant = all(r.compliant for r in receipt.compliance_results)
        
        # Create node
        node = {
            'id': receipt.receipt_id,
            'block_number': i,
            'label': f"Block {i}",
            'timestamp': receipt.timestamp,
            'hash': receipt.content_hash,
            'decision_id': receipt.decision_id,
            'compliant': is_compliant,
            'color': '#4CAF50' if is_compliant else '#F44336'
        }
        graph_data['nodes'].append(node)
        
        # Create edge to previous
        if i > 0:
            edge = {
                'from': audit_logger.receipt_chain[i-1].receipt_id,
                'to': receipt.receipt_id,
                'hash_link': receipt.previous_hash
            }
            graph_data['edges'].append(edge)
    
    # Save graph data to organized output directory
    viz_output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'visualizations')
    os.makedirs(viz_output_dir, exist_ok=True)
    graph_path = os.path.join(viz_output_dir, 'compliance_blockchain_graph.json')
    
    with open(graph_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"  ✓ Graph data saved to: {graph_path}")
    print(f"  ✓ Nodes: {len(graph_data['nodes'])}")
    print(f"  ✓ Edges: {len(graph_data['edges'])}")
    
    # Print blockchain structure
    print("\n🔗 Blockchain Structure:")
    print("  " + "-"*76)
    for i, receipt in enumerate(audit_logger.receipt_chain[:5]):  # Show first 5
        is_compliant = all(r.compliant for r in receipt.compliance_results)
        symbol = "✓" if is_compliant else "✗"
        
        print(f"  | Block {i}: {receipt.receipt_id[:20]}... | {symbol} |")
        print(f"  | Hash: {receipt.content_hash[:32]}...                 |")
        if i < len(audit_logger.receipt_chain) - 1:
            print(f"  |         ↓ (hash chain link)                              |")
    
    if len(audit_logger.receipt_chain) > 5:
        print(f"  | ... {len(audit_logger.receipt_chain) - 5} more blocks ...                            |")
    print("  " + "-"*76)


def visualize_consent_chain():
    """Visualize user consent chain"""
    print_header("CONSENT BLOCKCHAIN VISUALIZATION")
    
    # Initialize consent system with organized output path
    consent_output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'audit_logs')
    os.makedirs(consent_output_dir, exist_ok=True) 
    consent_path = os.path.join(consent_output_dir, "blockchain_demo_consent.json")
    consent_manager = ConsentManager(storage_path=consent_path)
    
    user_id = "USER_BLOCKCHAIN_DEMO"
    
    print(f"\n📝 Creating consent chain for user: {user_id}")
    
    # Grant multiple consents
    consents = [
        ('AMT_INCOME_TOTAL', ConsentPurpose.CREDIT_DECISION),
        ('EXT_SOURCE_2', ConsentPurpose.CREDIT_DECISION),
        ('DAYS_EMPLOYED', ConsentPurpose.CREDIT_DECISION),
        ('AMT_INCOME_TOTAL', ConsentPurpose.FRAUD_DETECTION),
        ('AMT_INCOME_TOTAL', ConsentPurpose.MARKETING),
    ]
    
    # Create wallet with organized output path
    wallet_path = os.path.join(consent_output_dir, f"blockchain_demo_wallet_{user_id}.json")
    wallet = ConsentWallet(user_id=user_id, storage_path=wallet_path)
    
    for data_field, purpose in consents:
        consent = consent_manager.grant_consent(
            user_id=user_id,
            data_field=data_field,
            purpose=purpose,
            consent_channel="blockchain_demo"
        )
        
        # Add to wallet (creates receipt/block)
        receipt = wallet.add_consent(consent)
        
        print(f"  Block {len(wallet.receipt_chain)-1}: Granted {data_field} for {purpose.value}")
        print(f"    Hash: {receipt.content_hash[:32]}...")
    
    # Revoke one consent
    print(f"\n  Revoking marketing consent...")
    revoke_receipt = wallet.revoke_consent(
        consent_id=consents[4][0],  # Marketing consent
        metadata={'initiated_by': 'user'}
    )
    
    if revoke_receipt:
        print(f"  Block {len(wallet.receipt_chain)-1}: Revoked (Marketing)")
        print(f"    Hash: {revoke_receipt.content_hash[:32]}...")
    
    # Verify wallet chain
    print("\n🔗 Verifying consent blockchain...")
    chain_valid, errors = wallet.verify_wallet()
    
    if chain_valid:
        print(f"  ✓ Blockchain is VALID")
        print(f"  ✓ All {len(wallet.receipt_chain)} blocks verified")
    else:
        print(f"  ✗ Blockchain is INVALID")
    
    # Generate graph data
    print("\n📊 Generating consent graph data...")
    
    graph_data = {
        'nodes': [],
        'edges': []
    }
    
    for i, receipt in enumerate(wallet.receipt_chain):
        node = {
            'id': receipt.receipt_id,
            'block_number': i,
            'label': f"Block {i}\n{receipt.action.value}",
            'timestamp': receipt.timestamp,
            'hash': receipt.content_hash,
            'action': receipt.action.value,
            'color': '#2196F3' if receipt.action.value == 'grant' else '#F44336'
        }
        graph_data['nodes'].append(node)
        
        if i > 0:
            edge = {
                'from': wallet.receipt_chain[i-1].receipt_id,
                'to': receipt.receipt_id
            }
            graph_data['edges'].append(edge)
    
    # Save consent graph to organized output directory
    viz_output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'visualizations')
    os.makedirs(viz_output_dir, exist_ok=True)
    consent_graph_path = os.path.join(viz_output_dir, f'consent_blockchain_graph_{user_id}.json')
    
    with open(consent_graph_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"  ✓ Graph data saved to: {consent_graph_path}")
    
    # Print blockchain structure
    print("\n🔗 Consent Blockchain Structure:")
    print("  " + "-"*76)
    for i, receipt in enumerate(wallet.receipt_chain):
        action_symbol = "+" if receipt.action.value == 'grant' else "-"
        
        print(f"  | Block {i}: {receipt.receipt_id[:20]}... | {action_symbol} {receipt.action.value.upper()} |")
        print(f"  | Hash: {receipt.content_hash[:32]}...                 |")
        if i < len(wallet.receipt_chain) - 1:
            print(f"  |         ↓ (hash chain link)                              |")
    print("  " + "-"*76)
    
    # Summary
    summary = wallet.get_summary()
    print(f"\n📊 Wallet Summary:")
    print(f"  User: {summary.user_id}")
    print(f"  Total consents: {summary.total_consents}")
    print(f"  Active: {summary.active_consents}")
    print(f"  Revoked: {summary.revoked_consents}")
    print(f"  Blockchain blocks: {len(wallet.receipt_chain)}")


def main():
    """Main blockchain visualization demo"""
    
    print("\n" + "="*80)
    print("  BLOCKCHAIN VISUALIZATION DEMO")
    print("  Compliance Audit Chain + Consent Provenance Chain")
    print("="*80)
    
    # Visualize compliance blockchain
    visualize_compliance_chain()
    
    # Visualize consent blockchain
    visualize_consent_chain()
    
    # Summary
    print_header("BLOCKCHAIN VISUALIZATION COMPLETE")
    
    print("\n✅ Generated blockchain visualization data:")
    print("  1. ✓ Compliance audit blockchain")
    print("  2. ✓ Consent provenance blockchain")
    print("  3. ✓ Graph data (nodes + edges)")
    print("  4. ✓ Timeline data")
    print("  5. ✓ Block explorer data")
    
    print("\n📊 Generated Files in organized directories:")
    print("  📁 outputs/visualizations/:")
    print("    • compliance_blockchain_graph.json")
    print("    • consent_blockchain_graph_USER_BLOCKCHAIN_DEMO.json") 
    print("  📁 outputs/audit_logs/:")
    print("    • blockchain_demo_audit.json")
    print("    • blockchain_demo_consent.json")
    print("    • blockchain_demo_wallet_USER_BLOCKCHAIN_DEMO.json")
    
    print("\n🎨 Frontend Integration:")
    print("  Use graph data with:")
    print("  • D3.js for custom visualizations")
    print("  • vis.js for network graphs")
    print("  • Cytoscape.js for interactive graphs")
    print("  • React Flow for flow diagrams")
    
    print("\n🔗 Blockchain Features Available:")
    print("  ✓ Block-by-block view")
    print("  ✓ Hash chain visualization")
    print("  ✓ Integrity verification indicators")
    print("  ✓ Timeline view")
    print("  ✓ Block explorer (click to see details)")
    
    print("\n📡 API Endpoints for Dashboard:")
    print("  GET /blockchain/compliance/blocks")
    print("  GET /blockchain/consent/blocks/{user_id}")
    print("  GET /blockchain/verify/{block_id}")
    print("  GET /blockchain/graph/compliance")
    print("  GET /blockchain/graph/consent/{user_id}")
    print("  GET /blockchain/explorer/block/{block_id}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()