"""
Policy Manager
Manages regulatory policies with CRUD operations, versioning, and persistence.
"""

from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime
from pathlib import Path

from .policy_schema import (
    Policy,
    PolicyType,
    PolicyAction,
    PolicyCondition
)

logger = logging.getLogger(__name__)


class PolicyManager:
    """
    Manages the lifecycle of regulatory policies.
    Provides CRUD operations, versioning, and persistence.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        auto_save: bool = True
    ):
        """
        Initialize policy manager.
        
        Args:
            storage_path: Path to policy storage file (if None, uses default relative to framework)
            auto_save: Automatically save after modifications
        """
        # Use absolute path relative to this file's location
        if storage_path is None:
            framework_root = Path(__file__).parent.parent.parent
            storage_path = str(framework_root / "core" / "compliance" / "regulations" / "regulations_db.json")
        
        self.storage_path = storage_path
        self.auto_save = auto_save
        self.policies: Dict[str, Policy] = {}
        
        # Load existing policies
        self._load_policies()
        
        logger.info(f"Initialized PolicyManager with {len(self.policies)} policies")
    
    def _load_policies(self):
        """Load policies from storage"""
        try:
            path = Path(self.storage_path)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                    for policy_dict in data.get('policies', []):
                        policy = Policy.from_dict(policy_dict)
                        self.policies[policy.policy_id] = policy
                    
                logger.info(f"Loaded {len(self.policies)} policies from {self.storage_path}")
            else:
                logger.info(f"No existing policies found at {self.storage_path}")
                # Create default policies
                self._create_default_policies()
        
        except Exception as e:
            logger.error(f"Error loading policies: {e}")
            self.policies = {}
    
    def _save_policies(self):
        """Save policies to storage"""
        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'policies': [p.to_dict() for p in self.policies.values()],
                'total_count': len(self.policies),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.policies)} policies to {self.storage_path}")
        
        except Exception as e:
            logger.error(f"Error saving policies: {e}")
    
    def create_policy(
        self,
        policy: Policy,
        created_by: Optional[str] = None
    ) -> Policy:
        """
        Create a new policy.
        
        Args:
            policy: Policy to create
            created_by: User creating the policy
            
        Returns:
            Created policy
        """
        # Check if policy already exists
        if policy.policy_id in self.policies:
            raise ValueError(f"Policy with ID '{policy.policy_id}' already exists")
        
        # Set creation metadata
        policy.created_at = datetime.now().isoformat()
        policy.created_by = created_by
        
        # Add to collection
        self.policies[policy.policy_id] = policy
        
        logger.info(f"Created policy: {policy.policy_id}")
        
        # Auto-save if enabled
        if self.auto_save:
            self._save_policies()
        
        return policy
    
    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """
        Get policy by ID.
        
        Args:
            policy_id: Policy identifier
            
        Returns:
            Policy or None if not found
        """
        return self.policies.get(policy_id)
    
    def update_policy(
        self,
        policy_id: str,
        updates: Dict[str, Any],
        updated_by: Optional[str] = None,
        create_new_version: bool = True
    ) -> Policy:
        """
        Update an existing policy.
        
        Args:
            policy_id: Policy to update
            updates: Dictionary of fields to update
            updated_by: User updating the policy
            create_new_version: Create new version or overwrite
            
        Returns:
            Updated policy
        """
        policy = self.get_policy(policy_id)
        if not policy:
            raise ValueError(f"Policy '{policy_id}' not found")
        
        # Create new version if requested
        if create_new_version:
            # Increment version
            current_version = policy.version
            major, minor = current_version.split('.')
            new_version = f"{major}.{int(minor) + 1}"
            updates['version'] = new_version
        
        # Update fields
        for key, value in updates.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
        
        # Set update metadata
        policy.updated_at = datetime.now().isoformat()
        policy.updated_by = updated_by
        
        logger.info(f"Updated policy: {policy_id} (version: {policy.version})")
        
        # Auto-save if enabled
        if self.auto_save:
            self._save_policies()
        
        return policy
    
    def delete_policy(self, policy_id: str) -> bool:
        """
        Delete a policy.
        
        Args:
            policy_id: Policy to delete
            
        Returns:
            True if deleted, False if not found
        """
        if policy_id not in self.policies:
            logger.warning(f"Policy '{policy_id}' not found for deletion")
            return False
        
        del self.policies[policy_id]
        logger.info(f"Deleted policy: {policy_id}")
        
        # Auto-save if enabled
        if self.auto_save:
            self._save_policies()
        
        return True
    
    def enable_policy(self, policy_id: str) -> bool:
        """Enable a policy"""
        policy = self.get_policy(policy_id)
        if not policy:
            return False
        
        policy.enabled = True
        logger.info(f"Enabled policy: {policy_id}")
        
        if self.auto_save:
            self._save_policies()
        
        return True
    
    def disable_policy(self, policy_id: str) -> bool:
        """Disable a policy"""
        policy = self.get_policy(policy_id)
        if not policy:
            return False
        
        policy.enabled = False
        logger.info(f"Disabled policy: {policy_id}")
        
        if self.auto_save:
            self._save_policies()
        
        return True
    
    def list_policies(
        self,
        policy_type: Optional[PolicyType] = None,
        regulation_source: Optional[str] = None,
        enabled_only: bool = False,
        jurisdiction: Optional[str] = None
    ) -> List[Policy]:
        """
        List policies with optional filters.
        
        Args:
            policy_type: Filter by policy type
            regulation_source: Filter by regulation
            enabled_only: Only return enabled policies
            jurisdiction: Filter by jurisdiction
            
        Returns:
            List of policies
        """
        policies = list(self.policies.values())
        
        # Apply filters
        if policy_type:
            policies = [p for p in policies if p.policy_type == policy_type]
        
        if regulation_source:
            policies = [p for p in policies if p.regulation_source == regulation_source]
        
        if enabled_only:
            policies = [p for p in policies if p.enabled]
        
        if jurisdiction:
            policies = [p for p in policies if p.jurisdiction == jurisdiction]
        
        # Sort by priority (highest first)
        policies.sort(key=lambda p: p.priority, reverse=True)
        
        return policies
    
    def search_policies(self, query: str) -> List[Policy]:
        """
        Search policies by keyword.
        
        Args:
            query: Search query
            
        Returns:
            Matching policies
        """
        query_lower = query.lower()
        results = []
        
        for policy in self.policies.values():
            # Search in multiple fields
            if (query_lower in policy.name.lower() or
                query_lower in policy.description.lower() or
                query_lower in policy.regulation_source.lower() or
                any(query_lower in tag.lower() for tag in policy.tags)):
                results.append(policy)
        
        return results
    
    def get_policies_by_regulation(self, regulation_source: str) -> List[Policy]:
        """Get all policies from a specific regulation"""
        return [
            p for p in self.policies.values()
            if p.regulation_source == regulation_source
        ]
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about policies.
        
        Returns:
            Dictionary with statistics
        """
        total = len(self.policies)
        enabled = sum(1 for p in self.policies.values() if p.enabled)
        
        # Count by type
        by_type = {}
        for policy in self.policies.values():
            ptype = policy.policy_type.value
            by_type[ptype] = by_type.get(ptype, 0) + 1
        
        # Count by regulation
        by_regulation = {}
        for policy in self.policies.values():
            reg = policy.regulation_source
            by_regulation[reg] = by_regulation.get(reg, 0) + 1
        
        # Count by jurisdiction
        by_jurisdiction = {}
        for policy in self.policies.values():
            if policy.jurisdiction:
                by_jurisdiction[policy.jurisdiction] = by_jurisdiction.get(policy.jurisdiction, 0) + 1
        
        return {
            'total_policies': total,
            'enabled': enabled,
            'disabled': total - enabled,
            'by_type': by_type,
            'by_regulation': by_regulation,
            'by_jurisdiction': by_jurisdiction
        }
    
    def export_policies(
        self,
        output_path: str,
        format: str = 'json',
        include_disabled: bool = True
    ):
        """
        Export policies to file.
        
        Args:
            output_path: Output file path
            format: Export format ('json' or 'csv')
            include_disabled: Include disabled policies
        """
        policies = list(self.policies.values())
        
        if not include_disabled:
            policies = [p for p in policies if p.enabled]
        
        if format == 'json':
            data = {
                'policies': [p.to_dict() for p in policies],
                'exported_at': datetime.now().isoformat(),
                'count': len(policies)
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'policy_id', 'name', 'regulation_source', 'type',
                    'enabled', 'priority', 'version', 'created_at'
                ])
                
                for policy in policies:
                    writer.writerow([
                        policy.policy_id,
                        policy.name,
                        policy.regulation_source,
                        policy.policy_type.value,
                        policy.enabled,
                        policy.priority,
                        policy.version,
                        policy.created_at
                    ])
        
        logger.info(f"Exported {len(policies)} policies to {output_path}")
    
    def import_policies(
        self,
        file_path: str,
        overwrite_existing: bool = False
    ) -> int:
        """
        Import policies from file.
        
        Args:
            file_path: Path to import file
            overwrite_existing: Overwrite existing policies
            
        Returns:
            Number of policies imported
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            imported_count = 0
            
            for policy_dict in data.get('policies', []):
                policy = Policy.from_dict(policy_dict)
                
                # Check if exists
                if policy.policy_id in self.policies and not overwrite_existing:
                    logger.warning(f"Policy '{policy.policy_id}' already exists, skipping")
                    continue
                
                self.policies[policy.policy_id] = policy
                imported_count += 1
            
            logger.info(f"Imported {imported_count} policies from {file_path}")
            
            if self.auto_save:
                self._save_policies()
            
            return imported_count
        
        except Exception as e:
            logger.error(f"Error importing policies: {e}")
            return 0
    
    def _create_default_policies(self):
        """Create default example policies"""
        
        # Basel III - Minimum Credit Score
        policy1 = Policy(
            policy_id="BASEL_III_CREDIT_001",
            name="Minimum Credit Score for Large Loans",
            regulation_source="Basel III",
            policy_type=PolicyType.CREDIT_RISK,
            description="Credit score must be >= 650 for loans over $50,000",
            condition=PolicyCondition(
                logical_operator='AND',
                sub_conditions=[
                    PolicyCondition(feature='AMT_CREDIT', operator='>', value=50000),
                    PolicyCondition(feature='EXT_SOURCE_2', operator='<', value=0.5)
                ]
            ),
            action=PolicyAction.FLAG_FOR_REVIEW,
            version="1.0",
            jurisdiction="US",
            priority=10,
            rationale="Ensures adequate creditworthiness for large loans",
            tags=["credit_risk", "basel", "loan_approval"]
        )
        
        # Fair Lending - Age Discrimination
        policy2 = Policy(
            policy_id="FAIR_LENDING_AGE_001",
            name="Age-Based Lending Restriction",
            regulation_source="Equal Credit Opportunity Act",
            policy_type=PolicyType.ANTI_DISCRIMINATION,
            description="Cannot deny loans solely based on age for applicants under 18 or over 70",
            condition=PolicyCondition(
                logical_operator='OR',
                sub_conditions=[
                    PolicyCondition(feature='AGE', operator='<', value=18),
                    PolicyCondition(feature='AGE', operator='>', value=70)
                ]
            ),
            action=PolicyAction.REQUIRE_EXPLANATION,
            version="1.0",
            jurisdiction="US",
            priority=15,
            rationale="Prevents age discrimination in lending",
            tags=["fairness", "discrimination", "age"]
        )
        
        # GDPR - Data Protection
        policy3 = Policy(
            policy_id="GDPR_DATA_001",
            name="Consent Required for Data Processing",
            regulation_source="GDPR",
            policy_type=PolicyType.DATA_PROTECTION,
            description="Explicit consent required for processing personal data",
            condition=PolicyCondition(
                feature='consent_given',
                operator='==',
                value=False
            ),
            action=PolicyAction.BLOCK,
            version="1.0",
            jurisdiction="EU",
            priority=20,
            rationale="GDPR Article 6 - Lawfulness of processing",
            tags=["gdpr", "privacy", "consent"]
        )
        
        # Add default policies
        self.policies[policy1.policy_id] = policy1
        self.policies[policy2.policy_id] = policy2
        self.policies[policy3.policy_id] = policy3
        
        logger.info("Created 3 default policies")
        
        # Save
        if self.auto_save:
            self._save_policies()


def create_policy_manager(
    storage_path: Optional[str] = None,
    **kwargs
) -> PolicyManager:
    """
    Convenience function to create a policy manager.
    
    Args:
        storage_path: Path to storage file
        **kwargs: Additional parameters
        
    Returns:
        PolicyManager instance
    """
    if storage_path:
        return PolicyManager(storage_path=storage_path, **kwargs)
    return PolicyManager(**kwargs)