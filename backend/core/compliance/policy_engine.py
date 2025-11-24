"""
Policy Engine
Evaluates decisions against regulatory policies and business rules.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

from .policy_schema import (
    Policy,
    ComplianceResult,
    ComplianceStatus,
    PolicyAction
)

logger = logging.getLogger(__name__)


class PolicyEngine:
    """
    Engine for evaluating compliance against multiple policies.
    Checks decisions in real-time and provides compliance reports.
    """
    
    def __init__(
        self,
        policies: Optional[List[Policy]] = None,
        strict_mode: bool = False
    ):
        """
        Initialize policy engine.
        
        Args:
            policies: List of policies to check
            strict_mode: If True, stop on first violation. If False, check all policies.
        """
        self.policies = policies or []
        self.strict_mode = strict_mode
        
        # Sort policies by priority (higher first)
        self._sort_policies()
        
        logger.info(f"Initialized PolicyEngine with {len(self.policies)} policies")
    
    def _sort_policies(self):
        """Sort policies by priority"""
        self.policies.sort(key=lambda p: p.priority, reverse=True)
    
    def add_policy(self, policy: Policy):
        """
        Add a policy to the engine.
        
        Args:
            policy: Policy to add
        """
        self.policies.append(policy)
        self._sort_policies()
        logger.info(f"Added policy: {policy.policy_id}")
    
    def remove_policy(self, policy_id: str) -> bool:
        """
        Remove a policy from the engine.
        
        Args:
            policy_id: ID of policy to remove
            
        Returns:
            True if removed, False if not found
        """
        initial_count = len(self.policies)
        self.policies = [p for p in self.policies if p.policy_id != policy_id]
        
        if len(self.policies) < initial_count:
            logger.info(f"Removed policy: {policy_id}")
            return True
        return False
    
    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get policy by ID"""
        for policy in self.policies:
            if policy.policy_id == policy_id:
                return policy
        return None
    
    def update_policy(self, policy: Policy) -> bool:
        """
        Update existing policy.
        
        Args:
            policy: Updated policy
            
        Returns:
            True if updated, False if not found
        """
        for i, p in enumerate(self.policies):
            if p.policy_id == policy.policy_id:
                self.policies[i] = policy
                self._sort_policies()
                logger.info(f"Updated policy: {policy.policy_id}")
                return True
        return False
    
    def check_compliance(
        self,
        feature_values: Dict[str, Any],
        decision: Optional[Any] = None,
        policy_ids: Optional[List[str]] = None
    ) -> Tuple[bool, List[ComplianceResult]]:
        """
        Check compliance against all policies.
        
        Args:
            feature_values: Feature values from the decision
            decision: The actual decision made (optional)
            policy_ids: Specific policy IDs to check (if None, check all)
            
        Returns:
            Tuple of (is_compliant, list of compliance results)
        """
        results = []
        is_compliant = True
        
        # Filter policies to check
        policies_to_check = self.policies
        if policy_ids:
            policies_to_check = [p for p in self.policies if p.policy_id in policy_ids]
        
        logger.info(f"Checking compliance against {len(policies_to_check)} policies")
        
        for policy in policies_to_check:
            # Skip disabled policies
            if not policy.enabled:
                continue
            
            # Check if policy is currently effective
            if not self._is_policy_effective(policy):
                continue
            
            # Run compliance check
            result = policy.check_compliance(feature_values, decision)
            results.append(result)
            
            # Track overall compliance
            if not result.compliant:
                is_compliant = False
                logger.warning(
                    f"Policy violation: {policy.policy_id} - {result.message}"
                )
                
                # In strict mode, stop on first violation
                if self.strict_mode:
                    break
        
        logger.info(
            f"Compliance check complete: {'COMPLIANT' if is_compliant else 'VIOLATIONS FOUND'}"
        )
        
        return is_compliant, results
    
    def _is_policy_effective(self, policy: Policy) -> bool:
        """
        Check if policy is currently effective based on dates.
        
        Args:
            policy: Policy to check
            
        Returns:
            True if policy is effective now
        """
        now = datetime.now()
        
        # Check effective date
        if policy.effective_date:
            effective_dt = datetime.fromisoformat(policy.effective_date)
            if now < effective_dt:
                return False
        
        # Check expiry date
        if policy.expiry_date:
            expiry_dt = datetime.fromisoformat(policy.expiry_date)
            if now > expiry_dt:
                return False
        
        return True
    
    def get_violations(
        self,
        feature_values: Dict[str, Any],
        decision: Optional[Any] = None
    ) -> List[ComplianceResult]:
        """
        Get only the violations (non-compliant results).
        
        Args:
            feature_values: Feature values
            decision: Decision made
            
        Returns:
            List of violation results
        """
        _, results = self.check_compliance(feature_values, decision)
        return [r for r in results if not r.compliant]
    
    def get_applicable_policies(
        self,
        feature_values: Dict[str, Any],
        policy_type: Optional[str] = None
    ) -> List[Policy]:
        """
        Get policies applicable to given features.
        
        Args:
            feature_values: Feature values
            policy_type: Filter by policy type (optional)
            
        Returns:
            List of applicable policies
        """
        applicable = []
        
        for policy in self.policies:
            if not policy.enabled:
                continue
            
            # Filter by type if specified
            if policy_type and policy.policy_type.value != policy_type:
                continue
            
            # Check if policy is effective
            if not self._is_policy_effective(policy):
                continue
            
            applicable.append(policy)
        
        return applicable
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """
        Get summary of all policies.
        
        Returns:
            Dictionary with policy statistics
        """
        enabled_count = sum(1 for p in self.policies if p.enabled)
        disabled_count = len(self.policies) - enabled_count
        
        # Count by type
        by_type = {}
        for policy in self.policies:
            ptype = policy.policy_type.value
            by_type[ptype] = by_type.get(ptype, 0) + 1
        
        # Count by regulation source
        by_source = {}
        for policy in self.policies:
            source = policy.regulation_source
            by_source[source] = by_source.get(source, 0) + 1
        
        return {
            'total_policies': len(self.policies),
            'enabled': enabled_count,
            'disabled': disabled_count,
            'by_type': by_type,
            'by_source': by_source,
            'strict_mode': self.strict_mode
        }
    
    def validate_decision(
        self,
        feature_values: Dict[str, Any],
        decision: Any,
        return_details: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of a decision.
        
        Args:
            feature_values: Feature values
            decision: Decision made
            return_details: Whether to return detailed results
            
        Returns:
            Validation report
        """
        is_compliant, results = self.check_compliance(feature_values, decision)
        
        violations = [r for r in results if not r.compliant]
        warnings = [r for r in results if r.status == ComplianceStatus.WARNING]
        
        # Determine required actions
        required_actions = []
        for violation in violations:
            if violation.recommended_action:
                required_actions.append({
                    'policy': violation.policy.policy_id,
                    'action': violation.recommended_action,
                    'reason': violation.message
                })
        
        report = {
            'is_compliant': is_compliant,
            'decision': decision,
            'timestamp': datetime.now().isoformat(),
            'policies_checked': len(results),
            'violations_count': len(violations),
            'warnings_count': len(warnings),
            'required_actions': required_actions
        }
        
        if return_details:
            report['detailed_results'] = [r.to_dict() for r in results]
        
        return report
    
    def bulk_check(
        self,
        decisions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check compliance for multiple decisions.
        
        Args:
            decisions: List of decision dictionaries with 'features' and 'decision' keys
            
        Returns:
            List of validation reports
        """
        reports = []
        
        logger.info(f"Running bulk compliance check for {len(decisions)} decisions")
        
        for i, dec in enumerate(decisions):
            features = dec.get('features', {})
            decision = dec.get('decision')
            
            report = self.validate_decision(features, decision, return_details=False)
            report['index'] = i
            reports.append(report)
        
        # Summary statistics
        total_violations = sum(r['violations_count'] for r in reports)
        compliant_count = sum(1 for r in reports if r['is_compliant'])
        
        logger.info(
            f"Bulk check complete: {compliant_count}/{len(decisions)} compliant, "
            f"{total_violations} total violations"
        )
        
        return reports
    
    def suggest_remediation(
        self,
        feature_values: Dict[str, Any],
        decision: Any
    ) -> List[Dict[str, Any]]:
        """
        Suggest remediations for policy violations.
        
        Args:
            feature_values: Current feature values
            decision: Current decision
            
        Returns:
            List of remediation suggestions
        """
        violations = self.get_violations(feature_values, decision)
        
        suggestions = []
        
        for violation in violations:
            suggestion = {
                'policy_id': violation.policy.policy_id,
                'policy_name': violation.policy.name,
                'violation': violation.message,
                'remediation': self._generate_remediation(violation, feature_values)
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_remediation(
        self,
        violation: ComplianceResult,
        feature_values: Dict[str, Any]
    ) -> str:
        """Generate remediation suggestion for a violation"""
        
        # Extract violated condition
        condition = violation.policy.condition
        
        # Simple remediation suggestions
        if condition.feature and condition.operator and condition.value:
            feature_name = condition.feature
            current_value = feature_values.get(feature_name, 'unknown')
            required_value = condition.value
            
            if condition.operator == '>=':
                return f"Increase {feature_name} from {current_value} to at least {required_value}"
            elif condition.operator == '<=':
                return f"Decrease {feature_name} from {current_value} to at most {required_value}"
            elif condition.operator == '>':
                return f"Increase {feature_name} from {current_value} to more than {required_value}"
            elif condition.operator == '<':
                return f"Decrease {feature_name} from {current_value} to less than {required_value}"
            elif condition.operator == '==':
                return f"Set {feature_name} to {required_value} (currently {current_value})"
            elif condition.operator == 'in':
                return f"Ensure {feature_name} is one of: {required_value}"
        
        return f"Review and address: {violation.message}"


def create_policy_engine(
    policies: Optional[List[Policy]] = None,
    **kwargs
) -> PolicyEngine:
    """
    Convenience function to create a policy engine.
    
    Args:
        policies: List of policies
        **kwargs: Additional parameters
        
    Returns:
        PolicyEngine instance
    """
    return PolicyEngine(policies=policies, **kwargs)