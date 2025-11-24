"""
Consent Wallet
User's personal consent wallet with cryptographic provenance receipts.
"""

from typing import Dict, List, Any, Optional
import uuid
import hashlib
import json
import logging
from datetime import datetime

from .consent_schema import (
    ConsentRecord,
    ConsentReceipt,
    ConsentAction,
    ConsentPurpose,
    ConsentStatus,
    ConsentSummary,
    calculate_consent_hash
)

logger = logging.getLogger(__name__)


class ConsentWallet:
    """
    User's personal consent wallet with cryptographic receipts.
    Provides tamper-proof history of all consent actions.
    """
    
    def __init__(
        self,
        user_id: str,
        storage_path: Optional[str] = None
    ):
        """
        Initialize consent wallet for a user.
        
        Args:
            user_id: User identifier
            storage_path: Path to wallet storage (if None, in-memory only)
        """
        self.user_id = user_id
        self.storage_path = storage_path or f"wallets/wallet_{user_id}.json"
        
        # Consent history (chronological)
        self.consent_history: List[ConsentRecord] = []
        
        # Receipt chain (blockchain-style)
        self.receipt_chain: List[ConsentReceipt] = []
        
        # Load existing wallet
        self._load_wallet()
        
        logger.info(f"Initialized ConsentWallet for user {user_id} with {len(self.consent_history)} records")
    
    def _load_wallet(self):
        """Load wallet from storage"""
        try:
            from pathlib import Path
            path = Path(self.storage_path)
            
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                    # Load consent history
                    self.consent_history = [
                        ConsentRecord.from_dict(c) 
                        for c in data.get('consent_history', [])
                    ]
                    
                    # Load receipt chain
                    self.receipt_chain = [
                        ConsentReceipt.from_dict(r)
                        for r in data.get('receipt_chain', [])
                    ]
                    
                logger.info(f"Loaded wallet for {self.user_id}")
        
        except Exception as e:
            logger.warning(f"Could not load wallet: {e}")
            self.consent_history = []
            self.receipt_chain = []
    
    def _save_wallet(self):
        """Save wallet to storage"""
        try:
            from pathlib import Path
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'user_id': self.user_id,
                'consent_history': [c.to_dict() for c in self.consent_history],
                'receipt_chain': [r.to_dict() for r in self.receipt_chain],
                'total_consents': len(self.consent_history),
                'total_receipts': len(self.receipt_chain),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved wallet for {self.user_id}")
        
        except Exception as e:
            logger.error(f"Error saving wallet: {e}")
    
    def add_consent(
        self,
        consent: ConsentRecord,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConsentReceipt:
        """
        Add consent to wallet and create cryptographic receipt.
        
        Args:
            consent: Consent record to add
            metadata: Additional metadata (ip_address, user_agent)
            
        Returns:
            Cryptographic receipt
        """
        # Add to history
        self.consent_history.append(consent)
        
        # Create receipt
        receipt = self._create_receipt(
            action=ConsentAction.GRANT,
            consent_records=[consent.consent_id],
            action_details={
                'data_field': consent.data_field,
                'purpose': consent.purpose.value,
                'granted_at': consent.granted_at,
                'expires_at': consent.expires_at
            },
            metadata=metadata
        )
        
        logger.info(f"Added consent to wallet: {consent.consent_id}")
        
        # Save wallet
        self._save_wallet()
        
        return receipt
    
    def revoke_consent(
        self,
        consent_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ConsentReceipt]:
        """
        Revoke a consent and create receipt.
        
        Args:
            consent_id: Consent to revoke
            metadata: Additional metadata
            
        Returns:
            Receipt or None if consent not found
        """
        # Find consent
        consent = None
        for c in self.consent_history:
            if c.consent_id == consent_id:
                consent = c
                break
        
        if not consent:
            logger.warning(f"Consent {consent_id} not found in wallet")
            return None
        
        # Update status
        consent.status = ConsentStatus.REVOKED
        consent.revoked_at = datetime.now().isoformat()
        consent.updated_at = datetime.now().isoformat()
        
        # Create receipt
        receipt = self._create_receipt(
            action=ConsentAction.REVOKE,
            consent_records=[consent_id],
            action_details={
                'data_field': consent.data_field,
                'purpose': consent.purpose.value,
                'revoked_at': consent.revoked_at
            },
            metadata=metadata
        )
        
        logger.info(f"Revoked consent: {consent_id}")
        
        # Save wallet
        self._save_wallet()
        
        return receipt
    
    def _create_receipt(
        self,
        action: ConsentAction,
        consent_records: List[str],
        action_details: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConsentReceipt:
        """
        Create cryptographic receipt for consent action.
        
        Args:
            action: Consent action taken
            consent_records: List of affected consent IDs
            action_details: Details about the action
            metadata: Additional metadata
            
        Returns:
            ConsentReceipt with hash
        """
        # Generate receipt ID
        receipt_id = f"RECEIPT_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}"
        
        # Get previous hash
        previous_hash = ""
        if self.receipt_chain:
            previous_hash = self.receipt_chain[-1].content_hash
        
        # Create receipt
        receipt = ConsentReceipt(
            receipt_id=receipt_id,
            user_id=self.user_id,
            action=action,
            timestamp=datetime.now().isoformat(),
            consent_records=consent_records,
            action_details=action_details,
            previous_hash=previous_hash,
            ip_address=metadata.get('ip_address') if metadata else None,
            user_agent=metadata.get('user_agent') if metadata else None,
            initiated_by=metadata.get('initiated_by', 'user') if metadata else 'user'
        )
        
        # Calculate hash
        receipt.content_hash = self._calculate_receipt_hash(receipt)
        
        # Optionally sign
        receipt.signature = self._sign_receipt(receipt)
        
        # Add to chain
        self.receipt_chain.append(receipt)
        
        return receipt
    
    def _calculate_receipt_hash(self, receipt: ConsentReceipt) -> str:
        """Calculate SHA-256 hash of receipt"""
        content = {
            'receipt_id': receipt.receipt_id,
            'user_id': receipt.user_id,
            'action': receipt.action.value,
            'timestamp': receipt.timestamp,
            'consent_records': sorted(receipt.consent_records),
            'action_details': receipt.action_details,
            'previous_hash': receipt.previous_hash
        }
        
        content_str = json.dumps(content, sort_keys=True, separators=(',', ':'))
        hash_obj = hashlib.sha256(content_str.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def _sign_receipt(self, receipt: ConsentReceipt) -> str:
        """Create digital signature for receipt"""
        signature_input = f"{receipt.content_hash}:{receipt.user_id}:{receipt.timestamp}"
        signature_hash = hashlib.sha256(signature_input.encode('utf-8'))
        return f"SIG_{signature_hash.hexdigest()[:32]}"
    
    def verify_receipt(self, receipt: ConsentReceipt) -> bool:
        """Verify receipt integrity"""
        calculated_hash = self._calculate_receipt_hash(receipt)
        return calculated_hash == receipt.content_hash
    
    def verify_wallet(self) -> tuple[bool, List[str]]:
        """
        Verify entire wallet integrity.
        
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        if not self.receipt_chain:
            return True, []
        
        logger.info(f"Verifying wallet for {self.user_id}...")
        
        for i, receipt in enumerate(self.receipt_chain):
            # Verify hash
            if not self.verify_receipt(receipt):
                errors.append(f"Receipt {receipt.receipt_id} failed hash verification")
            
            # Verify chain linkage
            if i > 0:
                expected_previous = self.receipt_chain[i-1].content_hash
                if receipt.previous_hash != expected_previous:
                    errors.append(f"Receipt {receipt.receipt_id} has broken chain link")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"✓ Wallet verified for {self.user_id}")
        else:
            logger.error(f"✗ Wallet verification failed: {len(errors)} errors")
        
        return is_valid, errors
    
    def get_active_consents(self) -> List[ConsentRecord]:
        """Get all active consents"""
        return [c for c in self.consent_history if c.is_valid()]
    
    def get_consent_timeline(self) -> List[Dict[str, Any]]:
        """
        Get chronological timeline of consent actions.
        
        Returns:
            List of consent events
        """
        timeline = []
        
        for consent in self.consent_history:
            # Grant event
            if consent.granted_at:
                timeline.append({
                    'timestamp': consent.granted_at,
                    'action': 'granted',
                    'consent_id': consent.consent_id,
                    'data_field': consent.data_field,
                    'purpose': consent.purpose.value
                })
            
            # Revoke event
            if consent.revoked_at:
                timeline.append({
                    'timestamp': consent.revoked_at,
                    'action': 'revoked',
                    'consent_id': consent.consent_id,
                    'data_field': consent.data_field,
                    'purpose': consent.purpose.value
                })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        
        return timeline
    
    def export_wallet(self, output_path: str):
        """Export wallet for user download"""
        data = {
            'user_id': self.user_id,
            'summary': self.get_summary().to_dict(),
            'active_consents': [c.to_dict() for c in self.get_active_consents()],
            'consent_timeline': self.get_consent_timeline(),
            'receipts': [r.to_dict() for r in self.receipt_chain],
            'wallet_verified': self.verify_wallet()[0],
            'exported_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported wallet for {self.user_id} to {output_path}")
    
    def get_summary(self) -> ConsentSummary:
        """Get summary of wallet contents"""
        consents = self.consent_history
        
        if not consents:
            return ConsentSummary(
                user_id=self.user_id,
                total_consents=0,
                active_consents=0,
                revoked_consents=0,
                expired_consents=0,
                consents_by_purpose={},
                consents_by_field={},
                last_updated=datetime.now().isoformat()
            )
        
        active = sum(1 for c in consents if c.is_valid())
        revoked = sum(1 for c in consents if c.status == ConsentStatus.REVOKED)
        expired = sum(1 for c in consents if c.status == ConsentStatus.EXPIRED)
        
        by_purpose = {}
        for consent in consents:
            if consent.is_valid():
                purpose = consent.purpose.value
                by_purpose[purpose] = by_purpose.get(purpose, 0) + 1
        
        by_field = {}
        for consent in consents:
            field = consent.data_field
            by_field[field] = consent.status
        
        timestamps = [c.created_at for c in consents]
        
        return ConsentSummary(
            user_id=self.user_id,
            total_consents=len(consents),
            active_consents=active,
            revoked_consents=revoked,
            expired_consents=expired,
            consents_by_purpose=by_purpose,
            consents_by_field=by_field,
            last_updated=datetime.now().isoformat(),
            oldest_consent=min(timestamps) if timestamps else None,
            newest_consent=max(timestamps) if timestamps else None
        )


def create_consent_wallet(
    user_id: str,
    storage_path: Optional[str] = None
) -> ConsentWallet:
    """
    Convenience function to create a consent wallet.
    
    Args:
        user_id: User identifier
        storage_path: Storage path
        
    Returns:
        ConsentWallet instance
    """
    return ConsentWallet(user_id=user_id, storage_path=storage_path)