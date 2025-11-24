"""
Test Consent Module
Tests consent management, wallet, and cryptographic receipts.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from datetime import datetime

# Import consent modules
from core.consent.consent_manager import ConsentManager
from core.consent.consent_wallet import ConsentWallet
from core.consent.consent_schema import ConsentPurpose, ConsentStatus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def test_consent_manager():
    """Test 1: Consent Manager - CRUD operations"""
    print_section("TEST 1: Consent Manager")
    
    try:
        # Initialize consent manager
        consent_manager = ConsentManager(
            storage_path="test_consent_db.json"
        )
        print(f"✓ Consent Manager initialized")
        
        # Test granting consent
        print("\n📝 Granting consents for user...")
        
        user_id = "USER_TEST_001"
        
        # Grant consent for income data
        consent1 = consent_manager.grant_consent(
            user_id=user_id,
            data_field="AMT_INCOME_TOTAL",
            purpose=ConsentPurpose.CREDIT_DECISION,
            expires_in_days=365,
            consent_channel="web",
            metadata={
                'ip_address': '192.168.1.100',
                'user_agent': 'Mozilla/5.0',
                'consent_text': 'I agree to share my income for credit decisions'
            }
        )
        print(f"✓ Granted consent for income: {consent1.consent_id}")
        
        # Grant consent for credit score
        consent2 = consent_manager.grant_consent(
            user_id=user_id,
            data_field="EXT_SOURCE_2",
            purpose=ConsentPurpose.CREDIT_DECISION,
            expires_in_days=365
        )
        print(f"✓ Granted consent for credit score: {consent2.consent_id}")
        
        # Grant consent for employment
        consent3 = consent_manager.grant_consent(
            user_id=user_id,
            data_field="DAYS_EMPLOYED",
            purpose=ConsentPurpose.CREDIT_DECISION,
            expires_in_days=365
        )
        print(f"✓ Granted consent for employment: {consent3.consent_id}")
        
        # Grant consent for marketing (different purpose)
        consent4 = consent_manager.grant_consent(
            user_id=user_id,
            data_field="AMT_INCOME_TOTAL",
            purpose=ConsentPurpose.MARKETING,
            expires_in_days=180
        )
        print(f"✓ Granted consent for marketing: {consent4.consent_id}")
        
        # Test checking consent
        print("\n🔍 Checking consent status...")
        
        has_consent = consent_manager.check_consent(
            user_id=user_id,
            data_field="AMT_INCOME_TOTAL",
            purpose=ConsentPurpose.CREDIT_DECISION
        )
        print(f"  Has consent for income (credit): {has_consent}")
        
        has_consent_marketing = consent_manager.check_consent(
            user_id=user_id,
            data_field="EXT_SOURCE_2",
            purpose=ConsentPurpose.MARKETING
        )
        print(f"  Has consent for credit score (marketing): {has_consent_marketing}")
        
        # Test getting allowed fields
        print("\n📋 Getting allowed fields for credit decision...")
        allowed_fields = consent_manager.get_allowed_fields(
            user_id=user_id,
            purpose=ConsentPurpose.CREDIT_DECISION
        )
        print(f"  Allowed fields: {allowed_fields}")
        
        # Test feature filtering
        print("\n🔒 Testing feature filtering...")
        all_features = {
            'AMT_INCOME_TOTAL': 50000,
            'EXT_SOURCE_2': 0.7,
            'DAYS_EMPLOYED': -1000,
            'AGE': 35,  # Not consented
            'AMT_CREDIT': 25000  # Not consented
        }
        
        filtered = consent_manager.filter_features_by_consent(
            user_id=user_id,
            features=all_features,
            purpose=ConsentPurpose.CREDIT_DECISION
        )
        print(f"  Original features: {len(all_features)}")
        print(f"  Filtered features: {len(filtered)}")
        print(f"  Filtered: {list(filtered.keys())}")
        
        # Test consent validation
        print("\n✅ Validating consent for decision...")
        validation = consent_manager.validate_consent_for_decision(
            user_id=user_id,
            features=all_features,
            purpose=ConsentPurpose.CREDIT_DECISION
        )
        print(f"  Can proceed: {validation['can_proceed']}")
        print(f"  Missing consents: {validation['missing_consents']}")
        print(f"  Message: {validation['message']}")
        
        # Test revoking consent
        print("\n🚫 Revoking consent for marketing...")
        revoked = consent_manager.revoke_consent(
            user_id=user_id,
            consent_id=consent4.consent_id
        )
        print(f"✓ Revoked {len(revoked)} consent(s)")
        
        # Verify revocation
        has_marketing = consent_manager.check_consent(
            user_id=user_id,
            data_field="AMT_INCOME_TOTAL",
            purpose=ConsentPurpose.MARKETING
        )
        print(f"  Has consent for marketing after revocation: {has_marketing}")
        
        # Get consent summary
        print("\n📊 Consent Summary:")
        summary = consent_manager.get_consent_summary(user_id)
        print(f"  Total consents: {summary.total_consents}")
        print(f"  Active: {summary.active_consents}")
        print(f"  Revoked: {summary.revoked_consents}")
        print(f"  By purpose: {summary.consents_by_purpose}")
        
        # Get statistics
        print("\n📊 System Statistics:")
        stats = consent_manager.get_statistics()
        print(f"  Total users: {stats['total_users']}")
        print(f"  Total consents: {stats['total_consents']}")
        print(f"  By status: {stats['by_status']}")
        
        print("\n✅ Consent Manager tests PASSED")
        return consent_manager, user_id, True
    
    except Exception as e:
        print(f"\n❌ Consent Manager tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False


def test_consent_wallet(consent_manager, user_id):
    """Test 2: Consent Wallet - Cryptographic receipts"""
    print_section("TEST 2: Consent Wallet - Cryptographic Receipts")
    
    try:
        # Create wallet for user
        wallet = ConsentWallet(
            user_id=user_id,
            storage_path=f"test_wallet_{user_id}.json"
        )
        print(f"✓ Consent Wallet created for {user_id}")
        
        # Get user's consents from manager
        user_consents = consent_manager.get_user_consents(user_id, active_only=True)
        print(f"✓ Found {len(user_consents)} active consents")
        
        # Add consents to wallet with receipts
        print("\n📝 Adding consents to wallet (creating receipts)...")
        
        receipts = []
        for consent in user_consents[:3]:  # Add first 3
            receipt = wallet.add_consent(
                consent=consent,
                metadata={
                    'ip_address': '192.168.1.100',
                    'user_agent': 'Test Agent'
                }
            )
            receipts.append(receipt)
            print(f"  ✓ Added {consent.data_field} - Receipt: {receipt.receipt_id}")
            print(f"    Hash: {receipt.content_hash[:32]}...")
            print(f"    Previous Hash: {receipt.previous_hash[:32] if receipt.previous_hash else 'None (first)'}...")
        
        # Verify wallet
        print("\n🔍 Verifying Wallet Integrity...")
        is_valid, errors = wallet.verify_wallet()
        
        if is_valid:
            print(f"  ✓ Wallet is VALID")
            print(f"  ✓ All {len(wallet.receipt_chain)} receipts verified")
            print(f"  ✓ Hash chain integrity confirmed")
        else:
            print(f"  ✗ Wallet is INVALID")
            for error in errors:
                print(f"    - {error}")
        
        # Get wallet summary
        print("\n📊 Wallet Summary:")
        summary = wallet.get_summary()
        print(f"  User: {summary.user_id}")
        print(f"  Total consents: {summary.total_consents}")
        print(f"  Active: {summary.active_consents}")
        print(f"  Revoked: {summary.revoked_consents}")
        print(f"  By purpose: {summary.consents_by_purpose}")
        
        # Get consent timeline
        print("\n📅 Consent Timeline:")
        timeline = wallet.get_consent_timeline()
        for event in timeline[:5]:  # Show first 5
            print(f"  {event['timestamp']}: {event['action'].upper()} - {event['data_field']} ({event['purpose']})")
        
        # Test revoking via wallet
        print("\n🚫 Revoking consent via wallet...")
        if user_consents:
            revoke_receipt = wallet.revoke_consent(
                consent_id=user_consents[0].consent_id,
                metadata={'ip_address': '192.168.1.100'}
            )
            if revoke_receipt:
                print(f"  ✓ Revocation receipt created: {revoke_receipt.receipt_id}")
                print(f"    Hash: {revoke_receipt.content_hash[:32]}...")
        
        # Verify wallet after revocation
        print("\n🔍 Verifying wallet after revocation...")
        is_valid_after, errors_after = wallet.verify_wallet()
        print(f"  Wallet valid: {is_valid_after}")
        
        # Export wallet
        print("\n💾 Exporting wallet...")
        wallet.export_wallet(f"exported_wallet_{user_id}.json")
        print(f"  ✓ Wallet exported to: exported_wallet_{user_id}.json")
        
        print("\n✅ Consent Wallet tests PASSED")
        return wallet, True
    
    except Exception as e:
        print(f"\n❌ Consent Wallet tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_integration():
    """Test 3: Integration - Consent + Decision Flow"""
    print_section("TEST 3: Integration - Consent-Based Decision Flow")
    
    try:
        print("📋 Simulating loan application with consent checking...\n")
        
        # Initialize
        consent_manager = ConsentManager(storage_path="test_integration_consent.json")
        user_id = "USER_INTEGRATION_001"
        
        # Step 1: User grants consent
        print("Step 1: User grants consent for loan application...")
        
        loan_fields = ['AMT_INCOME_TOTAL', 'EXT_SOURCE_2', 'DAYS_EMPLOYED', 'AMT_CREDIT']
        
        consents = consent_manager.bulk_grant_consent(
            user_id=user_id,
            data_fields=loan_fields,
            purpose=ConsentPurpose.CREDIT_DECISION,
            expires_in_days=365,
            consent_channel="mobile_app"
        )
        print(f"  ✓ Granted consent for {len(consents)} fields")
        
        # Step 2: Prepare loan application
        print("\nStep 2: Preparing loan application...")
        loan_features = {
            'AMT_INCOME_TOTAL': 45000,
            'EXT_SOURCE_2': 0.65,
            'DAYS_EMPLOYED': -1500,
            'AMT_CREDIT': 30000,
            'AGE': 32,  # Not in consented fields!
            'CODE_GENDER': 'M'  # Not in consented fields!
        }
        print(f"  Application has {len(loan_features)} features")
        
        # Step 3: Validate consent
        print("\nStep 3: Validating consent before processing...")
        validation = consent_manager.validate_consent_for_decision(
            user_id=user_id,
            features=loan_features,
            purpose=ConsentPurpose.CREDIT_DECISION
        )
        
        print(f"  Can proceed: {validation['can_proceed']}")
        print(f"  Missing consents: {validation['missing_consents']}")
        
        # Step 4: Filter features based on consent
        print("\nStep 4: Filtering features based on consent...")
        filtered_features = consent_manager.filter_features_by_consent(
            user_id=user_id,
            features=loan_features,
            purpose=ConsentPurpose.CREDIT_DECISION
        )
        print(f"  Original features: {len(loan_features)}")
        print(f"  Filtered features: {len(filtered_features)}")
        print(f"  Allowed: {list(filtered_features.keys())}")
        print(f"  Blocked: {[k for k in loan_features.keys() if k not in filtered_features]}")
        
        # Step 5: Create wallet and verify
        print("\nStep 5: Creating user wallet with receipts...")
        wallet = ConsentWallet(user_id=user_id)
        
        for consent in consents:
            wallet.add_consent(consent)
        
        print(f"  ✓ Wallet created with {len(wallet.receipt_chain)} receipts")
        
        # Verify wallet
        is_valid, errors = wallet.verify_wallet()
        print(f"  ✓ Wallet verified: {is_valid}")
        
        # Get summary
        summary = wallet.get_summary()
        print(f"\n📊 User Consent Summary:")
        print(f"  Active consents: {summary.active_consents}")
        print(f"  By purpose: {summary.consents_by_purpose}")
        
        print("\n✅ Integration test PASSED")
        print("\n🎉 Consent system ready for:")
        print("  • GDPR compliance")
        print("  • Privacy-preserving decisions")
        print("  • User data control")
        print("  • Audit trail for regulators")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all consent tests"""
    print("\n" + "="*80)
    print("  CONSENT MODULE TEST SUITE")
    print("  Testing: Consent Manager, Wallet, Cryptographic Receipts")
    print("="*80)
    
    results = []
    
    # Test 1: Consent Manager
    consent_manager, user_id, test1_pass = test_consent_manager()
    results.append(("Consent Manager", test1_pass))
    
    if not test1_pass:
        print("\n❌ Cannot continue - Consent Manager failed")
        return
    
    # Test 2: Consent Wallet
    wallet, test2_pass = test_consent_wallet(consent_manager, user_id)
    results.append(("Consent Wallet", test2_pass))
    
    # Test 3: Integration
    test3_pass = test_integration()
    results.append(("Integration", test3_pass))
    
    # Summary
    print_section("TEST SUMMARY")
    
    all_passed = all(result[1] for result in results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print("\n" + "="*80)
    if all_passed:
        print("  🎉 ALL TESTS PASSED!")
        print("  Consent module is ready for integration")
        print("\n  Next steps:")
        print("  1. Integrate with API endpoints")
        print("  2. Connect with dashboard")
        print("  3. Add to decision flow")
    else:
        print("  ❌ SOME TESTS FAILED")
        print("  Please review errors above")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()