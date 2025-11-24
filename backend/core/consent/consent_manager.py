"""
Consent Manager
Manages consent records with CRUD operations and validation.
"""

from typing import Dict, List, Any, Optional, Set
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import uuid

from .consent_schema import (
    ConsentRecord,
    ConsentPurpose,
    ConsentStatus,
    ConsentAction,
    ConsentReceipt,
    ConsentSummary,
    calculate_consent_hash
)

logger = logging.getLogger(__name__)


class ConsentManager:
    """
    Manages consent records for all users.
    Provides CRUD operations, validation, and querying.
    """
    
    def __init__(
        self,
        storage_path: str = "consent_database.json",
        auto_save: bool = True,
        default_expiry_days: int = 365
    ):
        """
        Initialize consent manager.
        
        Args:
            storage_path: Path to consent storage file
            auto_save: Automatically save after modifications
            default_expiry_days: Default consent expiry period
        """
        self.storage_path = storage_path
        self.auto_save = auto_save
        self.default_expiry_days = default_expiry_days
        
        # Storage: user_id -> list of consent records
        self.consents: Dict[str, List[ConsentRecord]] = {}
        
        # Load existing consents
        self._load_consents()
        
        logger.info(f"Initialized ConsentManager with {self._count_total_consents()} consent records")
    
    def _load_consents(self):
        """Load consents from storage"""
        try:
            path = Path(self.storage_path)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                    for user_id, consent_list in data.get('consents', {}).items():
                        self.consents[user_id] = [
                            ConsentRecord.from_dict(c) for c in consent_list
                        ]
                    
                logger.info(f"Loaded consents for {len(self.consents)} users")
            else:
                logger.info(f"No existing consents found at {self.storage_path}")
        
        except Exception as e:
            logger.error(f"Error loading consents: {e}")
            self.consents = {}
    
    def _save_consents(self):
        """Save consents to storage"""
        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'consents': {
                    user_id: [c.to_dict() for c in consent_list]
                    for user_id, consent_list in self.consents.items()
                },
                'total_users': len(self.consents),
                'total_consents': self._count_total_consents(),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved consents to {self.storage_path}")
        
        except Exception as e:
            logger.error(f"Error saving consents: {e}")
    
    def _count_total_consents(self) -> int:
        """Count total consent records"""
        return sum(len(consents) for consents in self.consents.values())
    
    def grant_consent(
        self,
        user_id: str,
        data_field: str,
        purpose: ConsentPurpose,
        expires_in_days: Optional[int] = None,
        sharing_allowed: bool = False,
        anonymization_required: bool = False,
        consent_channel: str = "web",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConsentRecord:
        """
        Grant consent for a user.
        
        Args:
            user_id: User identifier
            data_field: Data field being consented to
            purpose: Purpose for data usage
            expires_in_days: Days until expiry (None = uses default)
            sharing_allowed: Allow third-party sharing
            anonymization_required: Require data anonymization
            consent_channel: Channel where consent was given
            metadata: Additional metadata (ip_address, user_agent, etc.)
            
        Returns:
            Created ConsentRecord
        """
        # Generate consent ID
        consent_id = f"CONSENT_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}"
        
        # Calculate expiry
        expiry_days = expires_in_days or self.default_expiry_days
        expires_at = (datetime.now() + timedelta(days=expiry_days)).isoformat()
        
        # Create consent record
        consent = ConsentRecord(
            consent_id=consent_id,
            user_id=user_id,
            data_field=data_field,
            purpose=purpose,
            status=ConsentStatus.GRANTED,
            granted_at=datetime.now().isoformat(),
            expires_at=expires_at,
            consent_channel=consent_channel,
            sharing_allowed=sharing_allowed,
            anonymization_required=anonymization_required,
            data_retention_days=expiry_days,
            ip_address=metadata.get('ip_address') if metadata else None,
            user_agent=metadata.get('user_agent') if metadata else None,
            consent_text_shown=metadata.get('consent_text') if metadata else None
        )
        
        # Add to user's consents
        if user_id not in self.consents:
            self.consents[user_id] = []
        
        self.consents[user_id].append(consent)
        
        logger.info(f"Granted consent: {consent_id} for user {user_id}")
        
        # Auto-save
        if self.auto_save:
            self._save_consents()
        
        return consent
    
    def revoke_consent(
        self,
        user_id: str,
        consent_id: Optional[str] = None,
        data_field: Optional[str] = None,
        purpose: Optional[ConsentPurpose] = None
    ) -> List[ConsentRecord]:
        """
        Revoke consent(s).
        
        Args:
            user_id: User identifier
            consent_id: Specific consent to revoke (or None)
            data_field: Revoke all consents for this field (or None)
            purpose: Revoke all consents for this purpose (or None)
            
        Returns:
            List of revoked consents
        """
        if user_id not in self.consents:
            logger.warning(f"No consents found for user: {user_id}")
            return []
        
        revoked = []
        
        for consent in self.consents[user_id]:
            # Match criteria
            should_revoke = False
            
            if consent_id and consent.consent_id == consent_id:
                should_revoke = True
            elif data_field and consent.data_field == data_field:
                if purpose is None or consent.purpose == purpose:
                    should_revoke = True
            elif purpose and consent.purpose == purpose:
                should_revoke = True
            
            # Revoke if matched and currently granted
            if should_revoke and consent.status == ConsentStatus.GRANTED:
                consent.status = ConsentStatus.REVOKED
                consent.revoked_at = datetime.now().isoformat()
                consent.updated_at = datetime.now().isoformat()
                revoked.append(consent)
        
        logger.info(f"Revoked {len(revoked)} consents for user {user_id}")
        
        # Auto-save
        if self.auto_save:
            self._save_consents()
        
        return revoked
    
    def check_consent(
        self,
        user_id: str,
        data_field: str,
        purpose: ConsentPurpose
    ) -> bool:
        """
        Check if user has valid consent for data field and purpose.
        
        Args:
            user_id: User identifier
            data_field: Data field to check
            purpose: Purpose to check
            
        Returns:
            True if valid consent exists
        """
        if user_id not in self.consents:
            return False
        
        for consent in self.consents[user_id]:
            if (consent.data_field == data_field and 
                consent.purpose == purpose and
                consent.is_valid()):
                return True
        
        return False
    
    def get_user_consents(
        self,
        user_id: str,
        active_only: bool = False
    ) -> List[ConsentRecord]:
        """
        Get all consents for a user.
        
        Args:
            user_id: User identifier
            active_only: Only return active consents
            
        Returns:
            List of consent records
        """
        if user_id not in self.consents:
            return []
        
        consents = self.consents[user_id]
        
        if active_only:
            consents = [c for c in consents if c.is_valid()]
        
        return consents
    
    def get_consent_summary(self, user_id: str) -> ConsentSummary:
        """
        Get summary of user's consents.
        
        Args:
            user_id: User identifier
            
        Returns:
            ConsentSummary object
        """
        consents = self.get_user_consents(user_id)
        
        if not consents:
            return ConsentSummary(
                user_id=user_id,
                total_consents=0,
                active_consents=0,
                revoked_consents=0,
                expired_consents=0,
                consents_by_purpose={},
                consents_by_field={},
                last_updated=datetime.now().isoformat()
            )
        
        # Count by status
        active = sum(1 for c in consents if c.is_valid())
        revoked = sum(1 for c in consents if c.status == ConsentStatus.REVOKED)
        expired = sum(1 for c in consents if c.status == ConsentStatus.EXPIRED)
        
        # Count by purpose
        by_purpose = {}
        for consent in consents:
            if consent.is_valid():
                purpose = consent.purpose.value
                by_purpose[purpose] = by_purpose.get(purpose, 0) + 1
        
        # Status by field (most recent)
        by_field = {}
        for consent in consents:
            field = consent.data_field
            # Keep most recent status for each field
            if field not in by_field:
                by_field[field] = consent.status
            else:
                # Update if this consent is more recent
                existing_consent = next(c for c in consents if c.data_field == field)
                if consent.created_at > existing_consent.created_at:
                    by_field[field] = consent.status
        
        # Timestamps
        timestamps = [c.created_at for c in consents]
        
        return ConsentSummary(
            user_id=user_id,
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
    
    def get_allowed_fields(
        self,
        user_id: str,
        purpose: ConsentPurpose
    ) -> Set[str]:
        """
        Get data fields user has consented to for a specific purpose.
        
        Args:
            user_id: User identifier
            purpose: Purpose to check
            
        Returns:
            Set of allowed data fields
        """
        if user_id not in self.consents:
            return set()
        
        allowed = set()
        
        for consent in self.consents[user_id]:
            if consent.purpose == purpose and consent.is_valid():
                allowed.add(consent.data_field)
        
        return allowed
    
    def filter_features_by_consent(
        self,
        user_id: str,
        features: Dict[str, Any],
        purpose: ConsentPurpose
    ) -> Dict[str, Any]:
        """
        Filter features based on user consent.
        Only include features user has consented to.
        
        Args:
            user_id: User identifier
            features: All available features
            purpose: Purpose for using features
            
        Returns:
            Filtered features dictionary
        """
        allowed_fields = self.get_allowed_fields(user_id, purpose)
        
        # Filter features
        filtered = {
            field: value 
            for field, value in features.items()
            if field in allowed_fields
        }
        
        logger.info(
            f"Filtered features for user {user_id}: "
            f"{len(features)} -> {len(filtered)} (purpose: {purpose.value})"
        )
        
        return filtered
    
    def bulk_grant_consent(
        self,
        user_id: str,
        data_fields: List[str],
        purpose: ConsentPurpose,
        **kwargs
    ) -> List[ConsentRecord]:
        """
        Grant consent for multiple data fields at once.
        
        Args:
            user_id: User identifier
            data_fields: List of data fields
            purpose: Purpose for all fields
            **kwargs: Additional parameters for grant_consent
            
        Returns:
            List of created consent records
        """
        consents = []
        
        for field in data_fields:
            consent = self.grant_consent(
                user_id=user_id,
                data_field=field,
                purpose=purpose,
                **kwargs
            )
            consents.append(consent)
        
        logger.info(f"Bulk granted {len(consents)} consents for user {user_id}")
        
        return consents
    
    def expire_old_consents(self) -> int:
        """
        Mark expired consents as expired.
        
        Returns:
            Number of consents expired
        """
        expired_count = 0
        now = datetime.now()
        
        for user_id, consent_list in self.consents.items():
            for consent in consent_list:
                if consent.status == ConsentStatus.GRANTED and consent.expires_at:
                    expiry = datetime.fromisoformat(consent.expires_at)
                    if now > expiry:
                        consent.status = ConsentStatus.EXPIRED
                        consent.updated_at = datetime.now().isoformat()
                        expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Expired {expired_count} old consents")
            
            if self.auto_save:
                self._save_consents()
        
        return expired_count
    
    def get_consent_gaps(
        self,
        user_id: str,
        required_fields: List[str],
        purpose: ConsentPurpose
    ) -> List[str]:
        """
        Identify data fields that need consent.
        
        Args:
            user_id: User identifier
            required_fields: Fields needed for operation
            purpose: Purpose for using fields
            
        Returns:
            List of fields without valid consent
        """
        allowed_fields = self.get_allowed_fields(user_id, purpose)
        
        gaps = [field for field in required_fields if field not in allowed_fields]
        
        return gaps
    
    def validate_consent_for_decision(
        self,
        user_id: str,
        features: Dict[str, Any],
        purpose: ConsentPurpose
    ) -> Dict[str, Any]:
        """
        Validate that user has consented to all features being used.
        
        Args:
            user_id: User identifier
            features: Features to be used in decision
            purpose: Purpose of the decision
            
        Returns:
            Validation result
        """
        required_fields = list(features.keys())
        allowed_fields = self.get_allowed_fields(user_id, purpose)
        gaps = self.get_consent_gaps(user_id, required_fields, purpose)
        
        is_valid = len(gaps) == 0
        
        return {
            'is_valid': is_valid,
            'user_id': user_id,
            'purpose': purpose.value,
            'required_fields': required_fields,
            'allowed_fields': list(allowed_fields),
            'missing_consents': gaps,
            'can_proceed': is_valid,
            'message': 'All required consents present' if is_valid else f'Missing consent for: {", ".join(gaps)}'
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get consent system statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_users = len(self.consents)
        total_consents = self._count_total_consents()
        
        # Count by status
        status_counts = {status.value: 0 for status in ConsentStatus}
        for consent_list in self.consents.values():
            for consent in consent_list:
                status_counts[consent.status.value] += 1
        
        # Count by purpose
        purpose_counts = {purpose.value: 0 for purpose in ConsentPurpose}
        for consent_list in self.consents.values():
            for consent in consent_list:
                if consent.is_valid():
                    purpose_counts[consent.purpose.value] += 1
        
        return {
            'total_users': total_users,
            'total_consents': total_consents,
            'by_status': status_counts,
            'by_purpose': purpose_counts,
            'active_consents': status_counts[ConsentStatus.GRANTED.value]
        }


def create_consent_manager(
    storage_path: Optional[str] = None,
    **kwargs
) -> ConsentManager:
    """
    Convenience function to create a consent manager.
    
    Args:
        storage_path: Path to storage file
        **kwargs: Additional parameters
        
    Returns:
        ConsentManager instance
    """
    if storage_path:
        return ConsentManager(storage_path=storage_path, **kwargs)
    return ConsentManager(**kwargs)