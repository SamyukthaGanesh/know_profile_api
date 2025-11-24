"""
Audit Logger
Creates cryptographic audit trail with hash chains for immutable compliance records.
"""

from typing import Dict, List, Any, Optional, Tuple
import hashlib
import json
import uuid
from datetime import datetime
import logging

from .policy_schema import (
    AuditReceipt,
    ComplianceResult,
    Policy
)

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Creates tamper-proof audit logs using cryptographic hash chains.
    Each receipt is linked to previous receipt (blockchain-style).
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        enable_signature: bool = False
    ):
        """
        Initialize audit logger.
        
        Args:
            storage_path: Path to store audit receipts (JSON file)
            enable_signature: Whether to digitally sign receipts
        """
        self.storage_path = storage_path or "audit_ledger.json"
        self.enable_signature = enable_signature
        
        # In-memory chain
        self.receipt_chain: List[AuditReceipt] = []
        
        # Load existing chain if available
        self._load_chain()
        
        logger.info(f"Initialized AuditLogger with {len(self.receipt_chain)} existing receipts")
    
    def _load_chain(self):
        """Load existing audit chain from storage"""
        try:
            import os
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.receipt_chain = [
                        AuditReceipt.from_dict(r) for r in data.get('receipts', [])
                    ]
                logger.info(f"Loaded {len(self.receipt_chain)} receipts from {self.storage_path}")
        except Exception as e:
            logger.warning(f"Could not load existing audit chain: {e}")
            self.receipt_chain = []
    
    def _save_chain(self):
        """Save audit chain to storage"""
        try:
            import os
            os.makedirs(os.path.dirname(self.storage_path) or '.', exist_ok=True)
            
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'receipts': [r.to_dict() for r in self.receipt_chain],
                    'chain_length': len(self.receipt_chain),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
            
            logger.debug(f"Saved audit chain to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving audit chain: {e}")
    
    def create_receipt(
        self,
        decision_id: str,
        compliance_results: List[ComplianceResult],
        decision_outcome: Optional[str] = None,
        feature_values: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditReceipt:
        """
        Create a new audit receipt with cryptographic hash.
        
        Args:
            decision_id: Unique decision identifier
            compliance_results: Results of compliance checks
            decision_outcome: The decision made
            feature_values: Feature values used
            model_id: Model identifier
            created_by: User/system that created the decision
            metadata: Additional metadata (ip_address, user_agent, etc.)
            
        Returns:
            AuditReceipt with cryptographic hash
        """
        # Generate unique receipt ID
        receipt_id = f"AUDIT_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}"
        
        # Extract policy IDs
        policies_checked = [r.policy.policy_id for r in compliance_results]
        
        # Get previous hash (blockchain-style linking)
        previous_hash = ""
        if self.receipt_chain:
            previous_hash = self.receipt_chain[-1].content_hash
        
        # Create receipt (hash will be calculated after)
        receipt = AuditReceipt(
            receipt_id=receipt_id,
            decision_id=decision_id,
            timestamp=datetime.now().isoformat(),
            policies_checked=policies_checked,
            compliance_results=compliance_results,
            decision_outcome=decision_outcome,
            feature_values=feature_values,
            model_id=model_id,
            previous_hash=previous_hash,
            created_by=created_by,
            ip_address=metadata.get('ip_address') if metadata else None,
            user_agent=metadata.get('user_agent') if metadata else None
        )
        
        # Calculate content hash
        receipt.content_hash = self._calculate_hash(receipt)
        
        # Optionally sign the receipt
        if self.enable_signature:
            receipt.signature = self._sign_receipt(receipt)
        
        # Add to chain
        self.receipt_chain.append(receipt)
        
        # Persist to storage
        self._save_chain()
        
        logger.info(f"Created audit receipt: {receipt_id} for decision: {decision_id}")
        
        return receipt
    
    def _calculate_hash(self, receipt: AuditReceipt) -> str:
        """
        Calculate SHA-256 hash of receipt content.
        
        Args:
            receipt: Receipt to hash
            
        Returns:
            Hexadecimal hash string
        """
        # Create canonical representation
        content = {
            'receipt_id': receipt.receipt_id,
            'decision_id': receipt.decision_id,
            'timestamp': receipt.timestamp,
            'policies_checked': sorted(receipt.policies_checked),  # Sorted for determinism
            'compliance_results': [
                {
                    'policy_id': r.policy.policy_id,
                    'compliant': r.compliant,
                    'status': r.status.value
                }
                for r in receipt.compliance_results
            ],
            'decision_outcome': receipt.decision_outcome,
            'model_id': receipt.model_id,
            'previous_hash': receipt.previous_hash
        }
        
        # Convert to canonical JSON string
        content_str = json.dumps(content, sort_keys=True, separators=(',', ':'))
        
        # Calculate SHA-256 hash
        hash_obj = hashlib.sha256(content_str.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def _sign_receipt(self, receipt: AuditReceipt) -> str:
        """
        Digitally sign receipt (simplified version).
        In production, use proper digital signatures (RSA, ECDSA).
        
        Args:
            receipt: Receipt to sign
            
        Returns:
            Signature string
        """
        # Simplified signature: HMAC of hash
        # In production: Use private key to sign
        signature_input = f"{receipt.content_hash}:{receipt.timestamp}:{receipt.decision_id}"
        signature_hash = hashlib.sha256(signature_input.encode('utf-8'))
        return f"SIG_{signature_hash.hexdigest()[:32]}"
    
    def verify_receipt(self, receipt: AuditReceipt) -> bool:
        """
        Verify receipt integrity by recalculating hash.
        
        Args:
            receipt: Receipt to verify
            
        Returns:
            True if receipt is valid and unmodified
        """
        # Recalculate hash
        calculated_hash = self._calculate_hash(receipt)
        
        # Compare with stored hash
        if calculated_hash != receipt.content_hash:
            logger.warning(f"Receipt verification failed: {receipt.receipt_id}")
            return False
        
        return True
    
    def verify_chain(self) -> Tuple[bool, List[str]]:
        """
        Verify entire audit chain integrity.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if not self.receipt_chain:
            return True, []
        
        logger.info(f"Verifying audit chain with {len(self.receipt_chain)} receipts")
        
        for i, receipt in enumerate(self.receipt_chain):
            # Verify receipt hash
            if not self.verify_receipt(receipt):
                errors.append(f"Receipt {receipt.receipt_id} failed hash verification")
            
            # Verify chain linkage
            if i > 0:
                expected_previous_hash = self.receipt_chain[i-1].content_hash
                if receipt.previous_hash != expected_previous_hash:
                    errors.append(
                        f"Receipt {receipt.receipt_id} has broken chain link "
                        f"(expected: {expected_previous_hash[:16]}..., "
                        f"got: {receipt.previous_hash[:16]}...)"
                    )
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("✓ Audit chain verification passed")
        else:
            logger.error(f"✗ Audit chain verification failed with {len(errors)} errors")
        
        return is_valid, errors
    
    def get_receipt(self, receipt_id: str) -> Optional[AuditReceipt]:
        """
        Get receipt by ID.
        
        Args:
            receipt_id: Receipt identifier
            
        Returns:
            AuditReceipt or None if not found
        """
        for receipt in self.receipt_chain:
            if receipt.receipt_id == receipt_id:
                return receipt
        return None
    
    def get_receipts_by_decision(self, decision_id: str) -> List[AuditReceipt]:
        """
        Get all receipts for a specific decision.
        
        Args:
            decision_id: Decision identifier
            
        Returns:
            List of receipts
        """
        return [r for r in self.receipt_chain if r.decision_id == decision_id]
    
    def get_receipts_by_policy(self, policy_id: str) -> List[AuditReceipt]:
        """
        Get all receipts that checked a specific policy.
        
        Args:
            policy_id: Policy identifier
            
        Returns:
            List of receipts
        """
        return [
            r for r in self.receipt_chain 
            if policy_id in r.policies_checked
        ]
    
    def get_recent_receipts(self, n: int = 10) -> List[AuditReceipt]:
        """
        Get most recent receipts.
        
        Args:
            n: Number of receipts to return
            
        Returns:
            List of recent receipts
        """
        return self.receipt_chain[-n:]
    
    def search_receipts(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model_id: Optional[str] = None,
        created_by: Optional[str] = None,
        compliant_only: Optional[bool] = None
    ) -> List[AuditReceipt]:
        """
        Search receipts with filters.
        
        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            model_id: Filter by model
            created_by: Filter by creator
            compliant_only: Filter by compliance status
            
        Returns:
            Filtered list of receipts
        """
        results = self.receipt_chain.copy()
        
        # Filter by date range
        if start_date:
            results = [r for r in results if r.timestamp >= start_date]
        if end_date:
            results = [r for r in results if r.timestamp <= end_date]
        
        # Filter by model
        if model_id:
            results = [r for r in results if r.model_id == model_id]
        
        # Filter by creator
        if created_by:
            results = [r for r in results if r.created_by == created_by]
        
        # Filter by compliance
        if compliant_only is not None:
            results = [
                r for r in results 
                if all(cr.compliant for cr in r.compliance_results) == compliant_only
            ]
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get audit trail statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.receipt_chain:
            return {
                'total_receipts': 0,
                'chain_valid': True
            }
        
        # Count compliant vs non-compliant
        compliant_count = sum(
            1 for r in self.receipt_chain 
            if all(cr.compliant for cr in r.compliance_results)
        )
        
        # Unique decisions
        unique_decisions = len(set(r.decision_id for r in self.receipt_chain))
        
        # Unique policies checked
        all_policies = set()
        for receipt in self.receipt_chain:
            all_policies.update(receipt.policies_checked)
        
        # Date range
        timestamps = [r.timestamp for r in self.receipt_chain]
        
        # Verify chain
        is_valid, errors = self.verify_chain()
        
        return {
            'total_receipts': len(self.receipt_chain),
            'compliant_receipts': compliant_count,
            'non_compliant_receipts': len(self.receipt_chain) - compliant_count,
            'unique_decisions': unique_decisions,
            'unique_policies_checked': len(all_policies),
            'earliest_receipt': min(timestamps) if timestamps else None,
            'latest_receipt': max(timestamps) if timestamps else None,
            'chain_valid': is_valid,
            'chain_errors': errors if not is_valid else []
        }
    
    def export_chain(self, output_path: str, format: str = 'json'):
        """
        Export audit chain to file.
        
        Args:
            output_path: Path to output file
            format: Export format ('json' or 'csv')
        """
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump({
                    'receipts': [r.to_dict() for r in self.receipt_chain],
                    'statistics': self.get_statistics(),
                    'exported_at': datetime.now().isoformat()
                }, f, indent=2)
            logger.info(f"Exported audit chain to {output_path}")
        
        elif format == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'receipt_id', 'decision_id', 'timestamp', 
                    'policies_checked', 'is_compliant', 'hash'
                ])
                for receipt in self.receipt_chain:
                    writer.writerow([
                        receipt.receipt_id,
                        receipt.decision_id,
                        receipt.timestamp,
                        ','.join(receipt.policies_checked),
                        all(r.compliant for r in receipt.compliance_results),
                        receipt.content_hash
                    ])
            logger.info(f"Exported audit chain to CSV: {output_path}")


def create_audit_logger(
    storage_path: Optional[str] = None,
    **kwargs
) -> AuditLogger:
    """
    Convenience function to create an audit logger.
    
    Args:
        storage_path: Path to storage file
        **kwargs: Additional parameters
        
    Returns:
        AuditLogger instance
    """
    return AuditLogger(storage_path=storage_path, **kwargs)