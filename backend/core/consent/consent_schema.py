"""
Consent Schema
Defines data structures for granular consent tracking and provenance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from datetime import datetime
import hashlib
import json


class ConsentPurpose(Enum):
    """Purposes for which data can be used"""
    CREDIT_DECISION = "credit_decision"
    FRAUD_DETECTION = "fraud_detection"
    MARKETING = "marketing"
    RISK_ASSESSMENT = "risk_assessment"
    MODEL_TRAINING = "model_training"
    ANALYTICS = "analytics"
    THIRD_PARTY_SHARING = "third_party_sharing"
    PERSONALIZATION = "personalization"
    REGULATORY_REPORTING = "regulatory_reporting"


class ConsentStatus(Enum):
    """Status of consent"""
    GRANTED = "granted"
    DENIED = "denied"
    REVOKED = "revoked"
    EXPIRED = "expired"
    PENDING = "pending"


class ConsentAction(Enum):
    """Actions that can be taken on consent"""
    GRANT = "grant"
    REVOKE = "revoke"
    UPDATE = "update"
    EXPIRE = "expire"
    RENEW = "renew"


@dataclass
class ConsentRecord:
    """
    Individual consent record for a specific data field and purpose.
    """
    
    consent_id: str
    user_id: str
    
    # What data
    data_field: str  # e.g., "income", "credit_score", "employment_history"
    
    # For what purpose
    purpose: ConsentPurpose
    
    # Consent details
    status: ConsentStatus
    granted_at: Optional[str] = None
    revoked_at: Optional[str] = None
    expires_at: Optional[str] = None
    
    # Context
    consent_method: str = "explicit"  # "explicit", "implicit", "opt_out"
    consent_channel: str = "web"  # "web", "mobile", "email", "in_person"
    
    # Additional constraints
    data_retention_days: Optional[int] = None
    sharing_allowed: bool = False
    anonymization_required: bool = False
    
    # Metadata
    version: str = "1.0"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    consent_text_shown: Optional[str] = None  # Actual consent text user saw
    
    # Audit
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if consent is currently valid"""
        if self.status != ConsentStatus.GRANTED:
            return False
        
        # Check expiry
        if self.expires_at:
            expiry = datetime.fromisoformat(self.expires_at)
            if datetime.now() > expiry:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'consent_id': self.consent_id,
            'user_id': self.user_id,
            'data_field': self.data_field,
            'purpose': self.purpose.value,
            'status': self.status.value,
            'granted_at': self.granted_at,
            'revoked_at': self.revoked_at,
            'expires_at': self.expires_at,
            'consent_method': self.consent_method,
            'consent_channel': self.consent_channel,
            'data_retention_days': self.data_retention_days,
            'sharing_allowed': self.sharing_allowed,
            'anonymization_required': self.anonymization_required,
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'is_valid': self.is_valid()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsentRecord':
        """Create from dictionary"""
        return cls(
            consent_id=data['consent_id'],
            user_id=data['user_id'],
            data_field=data['data_field'],
            purpose=ConsentPurpose(data['purpose']),
            status=ConsentStatus(data['status']),
            granted_at=data.get('granted_at'),
            revoked_at=data.get('revoked_at'),
            expires_at=data.get('expires_at'),
            consent_method=data.get('consent_method', 'explicit'),
            consent_channel=data.get('consent_channel', 'web'),
            data_retention_days=data.get('data_retention_days'),
            sharing_allowed=data.get('sharing_allowed', False),
            anonymization_required=data.get('anonymization_required', False),
            version=data.get('version', '1.0'),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent'),
            consent_text_shown=data.get('consent_text_shown'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at')
        )


@dataclass
class ConsentReceipt:
    """
    Cryptographic receipt for consent action.
    Provides proof of consent with tamper-proof hash.
    """
    
    receipt_id: str
    user_id: str
    action: ConsentAction
    timestamp: str
    
    # What changed
    consent_records: List[str]  # List of consent_ids affected
    
    # Details
    action_details: Dict[str, Any]
    
    # Cryptographic verification
    content_hash: str = ""
    previous_hash: str = ""
    signature: str = ""
    
    # Context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    initiated_by: str = "user"  # "user", "system", "admin"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'receipt_id': self.receipt_id,
            'user_id': self.user_id,
            'action': self.action.value,
            'timestamp': self.timestamp,
            'consent_records': self.consent_records,
            'action_details': self.action_details,
            'content_hash': self.content_hash,
            'previous_hash': self.previous_hash,
            'signature': self.signature,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'initiated_by': self.initiated_by
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsentReceipt':
        """Create from dictionary"""
        return cls(
            receipt_id=data['receipt_id'],
            user_id=data['user_id'],
            action=ConsentAction(data['action']),
            timestamp=data['timestamp'],
            consent_records=data['consent_records'],
            action_details=data['action_details'],
            content_hash=data['content_hash'],
            previous_hash=data['previous_hash'],
            signature=data.get('signature', ''),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent'),
            initiated_by=data.get('initiated_by', 'user')
        )


@dataclass
class ConsentSummary:
    """
    Summary of user's consent status.
    """
    
    user_id: str
    total_consents: int
    active_consents: int
    revoked_consents: int
    expired_consents: int
    
    # By purpose
    consents_by_purpose: Dict[str, int]
    
    # By data field
    consents_by_field: Dict[str, ConsentStatus]
    
    # Timestamps
    last_updated: str
    oldest_consent: Optional[str] = None
    newest_consent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'user_id': self.user_id,
            'total_consents': self.total_consents,
            'active_consents': self.active_consents,
            'revoked_consents': self.revoked_consents,
            'expired_consents': self.expired_consents,
            'consents_by_purpose': self.consents_by_purpose,
            'consents_by_field': {k: v.value for k, v in self.consents_by_field.items()},
            'last_updated': self.last_updated,
            'oldest_consent': self.oldest_consent,
            'newest_consent': self.newest_consent
        }


def calculate_consent_hash(
    consent_data: Dict[str, Any],
    previous_hash: str = ""
) -> str:
    """
    Calculate SHA-256 hash for consent data.
    
    Args:
        consent_data: Consent data to hash
        previous_hash: Previous receipt hash for chaining
        
    Returns:
        Hexadecimal hash string
    """
    # Create canonical representation
    content = {
        'consent_id': consent_data.get('consent_id'),
        'user_id': consent_data.get('user_id'),
        'data_field': consent_data.get('data_field'),
        'purpose': consent_data.get('purpose'),
        'status': consent_data.get('status'),
        'timestamp': consent_data.get('timestamp'),
        'previous_hash': previous_hash
    }
    
    # Convert to canonical JSON
    content_str = json.dumps(content, sort_keys=True, separators=(',', ':'))
    
    # Calculate SHA-256
    hash_obj = hashlib.sha256(content_str.encode('utf-8'))
    return hash_obj.hexdigest()