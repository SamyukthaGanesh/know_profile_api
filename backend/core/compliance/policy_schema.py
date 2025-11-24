"""
Policy Schema
Defines machine-readable policy structures for regulatory compliance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from datetime import datetime
import json


class PolicyType(Enum):
    """Types of policies"""
    CREDIT_RISK = "credit_risk"
    FAIRNESS = "fairness"
    DATA_PROTECTION = "data_protection"
    CONSUMER_PROTECTION = "consumer_protection"
    ANTI_DISCRIMINATION = "anti_discrimination"
    CAPITAL_REQUIREMENT = "capital_requirement"
    TRANSPARENCY = "transparency"
    CUSTOM = "custom"


class PolicyAction(Enum):
    """Actions when policy is violated"""
    DENY = "deny"
    FLAG_FOR_REVIEW = "flag_for_review"
    LOG_WARNING = "log_warning"
    REQUIRE_EXPLANATION = "require_explanation"
    BLOCK = "block"
    ALERT = "alert"


class ComplianceStatus(Enum):
    """Compliance check status"""
    COMPLIANT = "compliant"
    VIOLATION = "violation"
    WARNING = "warning"
    REQUIRES_REVIEW = "requires_review"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class PolicyCondition:
    """
    Represents a condition in a policy rule.
    Can be simple or compound (nested).
    """
    
    # Simple condition
    feature: Optional[str] = None
    operator: Optional[str] = None  # '>', '<', '>=', '<=', '==', '!=', 'in', 'not_in'
    value: Optional[Any] = None
    
    # Compound condition (AND/OR)
    logical_operator: Optional[str] = None  # 'AND', 'OR', 'NOT'
    sub_conditions: Optional[List['PolicyCondition']] = None
    
    def evaluate(self, feature_values: Dict[str, Any]) -> bool:
        """
        Evaluate if condition is met.
        
        Args:
            feature_values: Dictionary of feature values
            
        Returns:
            True if condition is satisfied
        """
        # Handle compound conditions
        if self.logical_operator:
            if not self.sub_conditions:
                return True
            
            results = [cond.evaluate(feature_values) for cond in self.sub_conditions]
            
            if self.logical_operator == 'AND':
                return all(results)
            elif self.logical_operator == 'OR':
                return any(results)
            elif self.logical_operator == 'NOT':
                return not results[0] if results else True
            else:
                raise ValueError(f"Unknown logical operator: {self.logical_operator}")
        
        # Handle simple condition
        if self.feature not in feature_values:
            return False  # Feature not present
        
        feature_value = feature_values[self.feature]
        
        # Evaluate based on operator
        if self.operator == '>':
            return feature_value > self.value
        elif self.operator == '<':
            return feature_value < self.value
        elif self.operator == '>=':
            return feature_value >= self.value
        elif self.operator == '<=':
            return feature_value <= self.value
        elif self.operator == '==':
            return feature_value == self.value
        elif self.operator == '!=':
            return feature_value != self.value
        elif self.operator == 'in':
            return feature_value in self.value
        elif self.operator == 'not_in':
            return feature_value not in self.value
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        if self.logical_operator:
            return {
                'logical_operator': self.logical_operator,
                'sub_conditions': [c.to_dict() for c in (self.sub_conditions or [])]
            }
        else:
            return {
                'feature': self.feature,
                'operator': self.operator,
                'value': self.value
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolicyCondition':
        """Create from dictionary"""
        if 'logical_operator' in data:
            return cls(
                logical_operator=data['logical_operator'],
                sub_conditions=[cls.from_dict(c) for c in data.get('sub_conditions', [])]
            )
        else:
            return cls(
                feature=data.get('feature'),
                operator=data.get('operator'),
                value=data.get('value')
            )


@dataclass
class Policy:
    """
    Represents a regulatory policy or business rule.
    """
    
    policy_id: str
    name: str
    regulation_source: str  # e.g., "Basel III", "GDPR", "Fair Lending Act"
    policy_type: PolicyType
    description: str
    
    # The actual rule
    condition: PolicyCondition
    action: PolicyAction
    
    # Metadata
    version: str = "1.0"
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    jurisdiction: Optional[str] = None
    priority: int = 0  # Higher priority checked first
    enabled: bool = True
    
    # Additional context
    rationale: Optional[str] = None
    references: Optional[List[str]] = None
    tags: List[str] = field(default_factory=list)
    
    # Audit trail
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: Optional[str] = None
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None
    
    def check_compliance(
        self,
        feature_values: Dict[str, Any],
        decision: Optional[Any] = None
    ) -> 'ComplianceResult':
        """
        Check if a decision complies with this policy.
        
        Args:
            feature_values: Feature values from the decision
            decision: The actual decision made
            
        Returns:
            ComplianceResult
        """
        if not self.enabled:
            return ComplianceResult(
                policy=self,
                status=ComplianceStatus.NOT_APPLICABLE,
                compliant=True,
                message="Policy is disabled"
            )
        
        # Evaluate condition
        try:
            condition_met = self.condition.evaluate(feature_values)
            
            # If condition is met, action should be taken
            if condition_met:
                return ComplianceResult(
                    policy=self,
                    status=ComplianceStatus.VIOLATION,
                    compliant=False,
                    message=f"Policy violated: {self.description}",
                    recommended_action=self.action.value,
                    feature_values=feature_values
                )
            else:
                return ComplianceResult(
                    policy=self,
                    status=ComplianceStatus.COMPLIANT,
                    compliant=True,
                    message="Policy satisfied",
                    feature_values=feature_values
                )
        
        except Exception as e:
            return ComplianceResult(
                policy=self,
                status=ComplianceStatus.REQUIRES_REVIEW,
                compliant=False,
                message=f"Error evaluating policy: {str(e)}",
                feature_values=feature_values
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'policy_id': self.policy_id,
            'name': self.name,
            'regulation_source': self.regulation_source,
            'policy_type': self.policy_type.value,
            'description': self.description,
            'condition': self.condition.to_dict(),
            'action': self.action.value,
            'version': self.version,
            'effective_date': self.effective_date,
            'expiry_date': self.expiry_date,
            'jurisdiction': self.jurisdiction,
            'priority': self.priority,
            'enabled': self.enabled,
            'rationale': self.rationale,
            'references': self.references,
            'tags': self.tags,
            'created_at': self.created_at,
            'created_by': self.created_by,
            'updated_at': self.updated_at,
            'updated_by': self.updated_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Policy':
        """Create from dictionary"""
        return cls(
            policy_id=data['policy_id'],
            name=data['name'],
            regulation_source=data['regulation_source'],
            policy_type=PolicyType(data['policy_type']),
            description=data['description'],
            condition=PolicyCondition.from_dict(data['condition']),
            action=PolicyAction(data['action']),
            version=data.get('version', '1.0'),
            effective_date=data.get('effective_date'),
            expiry_date=data.get('expiry_date'),
            jurisdiction=data.get('jurisdiction'),
            priority=data.get('priority', 0),
            enabled=data.get('enabled', True),
            rationale=data.get('rationale'),
            references=data.get('references'),
            tags=data.get('tags', []),
            created_at=data.get('created_at', datetime.now().isoformat()),
            created_by=data.get('created_by'),
            updated_at=data.get('updated_at'),
            updated_by=data.get('updated_by')
        )


@dataclass
class ComplianceResult:
    """
    Result of a compliance check.
    """
    
    policy: Policy
    status: ComplianceStatus
    compliant: bool
    message: str
    
    recommended_action: Optional[str] = None
    feature_values: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'policy_id': self.policy.policy_id,
            'policy_name': self.policy.name,
            'regulation_source': self.policy.regulation_source,
            'status': self.status.value,
            'compliant': self.compliant,
            'message': self.message,
            'recommended_action': self.recommended_action,
            'timestamp': self.timestamp
        }


@dataclass
class AuditReceipt:
    """
    Cryptographic receipt for audit trail.
    Immutable record of compliance check with hash chain.
    """
    
    receipt_id: str
    decision_id: str
    timestamp: str
    
    # What was checked
    policies_checked: List[str]
    compliance_results: List[ComplianceResult]
    
    # Decision details
    decision_outcome: Optional[str] = None
    feature_values: Optional[Dict[str, Any]] = None
    model_id: Optional[str] = None
    
    # Cryptographic verification
    content_hash: str = ""  # SHA256 of all content
    previous_hash: str = ""  # Hash of previous receipt (blockchain-style)
    signature: str = ""  # Digital signature (optional)
    
    # Metadata
    created_by: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'receipt_id': self.receipt_id,
            'decision_id': self.decision_id,
            'timestamp': self.timestamp,
            'policies_checked': self.policies_checked,
            'compliance_results': [r.to_dict() for r in self.compliance_results],
            'decision_outcome': self.decision_outcome,
            'feature_values': self.feature_values,
            'model_id': self.model_id,
            'content_hash': self.content_hash,
            'previous_hash': self.previous_hash,
            'signature': self.signature,
            'created_by': self.created_by,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditReceipt':
        """Create from dictionary"""
        # Note: compliance_results would need Policy objects to fully reconstruct
        # This is simplified for storage/retrieval
        return cls(
            receipt_id=data['receipt_id'],
            decision_id=data['decision_id'],
            timestamp=data['timestamp'],
            policies_checked=data['policies_checked'],
            compliance_results=[],  # Would need to reconstruct
            decision_outcome=data.get('decision_outcome'),
            feature_values=data.get('feature_values'),
            model_id=data.get('model_id'),
            content_hash=data['content_hash'],
            previous_hash=data['previous_hash'],
            signature=data.get('signature', ''),
            created_by=data.get('created_by'),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent')
        )


# Helper function to create example policies
def create_example_policy() -> Policy:
    """Create an example policy for demonstration"""
    return Policy(
        policy_id="BASEL_III_CREDIT_001",
        name="Minimum Credit Score for Large Loans",
        regulation_source="Basel III",
        policy_type=PolicyType.CREDIT_RISK,
        description="Credit score must be >= 650 for loans over $50,000",
        condition=PolicyCondition(
            logical_operator='AND',
            sub_conditions=[
                PolicyCondition(feature='AMT_CREDIT', operator='>', value=50000),
                PolicyCondition(feature='CREDIT_SCORE', operator='<', value=650)
            ]
        ),
        action=PolicyAction.DENY,
        version="1.0",
        jurisdiction="US",
        priority=10,
        rationale="Ensures adequate creditworthiness for large loans",
        references=["Basel III Framework", "Section 2.3.1"],
        tags=["credit_risk", "basel", "loan_approval"]
    )