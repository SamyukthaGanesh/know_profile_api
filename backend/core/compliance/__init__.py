"""
Compliance Module
Regulatory compliance checking with cryptographic audit trails.

This module provides:
- Machine-readable policy definitions
- Real-time compliance checking against regulations
- Cryptographic audit logging with hash chains
- Policy management (CRUD operations)
- Immutable audit trail for regulatory requirements
"""

from .policy_schema import (
    Policy,
    PolicyCondition,
    PolicyAction,
    PolicyType,
    ComplianceResult,
    AuditReceipt
)

from .policy_engine import (
    PolicyEngine,
    create_policy_engine
)

from .audit_logger import (
    AuditLogger,
    create_audit_logger
)

from .policy_manager import (
    PolicyManager,
    create_policy_manager
)

__all__ = [
    # Policy structures
    'Policy',
    'PolicyCondition',
    'PolicyAction',
    'PolicyType',
    'ComplianceResult',
    'AuditReceipt',
    
    # Engine
    'PolicyEngine',
    'create_policy_engine',
    
    # Audit
    'AuditLogger',
    'create_audit_logger',
    
    # Management
    'PolicyManager',
    'create_policy_manager',
]

__version__ = "1.0.0"