"""
Consent Management Module
Manages user consent for data usage with cryptographic provenance.

Features:
- Granular consent tracking (per data field, per purpose)
- Cryptographic receipts for consent actions
- Consent wallet (user's consent history)
- GDPR-compliant consent management
- Immutable audit trail for consent changes
"""

from .consent_schema import (
    ConsentRecord,
    ConsentPurpose,
    ConsentStatus,
    ConsentAction,
    ConsentReceipt
)

from .consent_manager import (
    ConsentManager,
    create_consent_manager
)

from .consent_wallet import (
    ConsentWallet,
    create_consent_wallet
)

__all__ = [
    # Schema
    'ConsentRecord',
    'ConsentPurpose',
    'ConsentStatus',
    'ConsentAction',
    'ConsentReceipt',
    
    # Manager
    'ConsentManager',
    'create_consent_manager',
    
    # Wallet
    'ConsentWallet',
    'create_consent_wallet',
]

__version__ = "1.0.0"