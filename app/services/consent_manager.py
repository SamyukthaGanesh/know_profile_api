"""
Consent Management Service
Stores user consents as boolean columns in users.csv
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import hashlib

class ConsentManager:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.users_file = self.data_dir / "users.csv"
        self.data_dir.mkdir(exist_ok=True)
        
        # Load users DataFrame
        if self.users_file.exists():
            self.users = pd.read_csv(self.users_file)
            self._ensure_consent_columns()
            print(f"✅ Loaded {len(self.users)} users with consent columns")
        else:
            raise FileNotFoundError("users.csv not found!")
    
    def _ensure_consent_columns(self):
        """Ensure consent boolean columns exist in users DataFrame"""
        consent_columns = {
            'consent_fraud_detection': 1,    # Default granted (1 = True)
            'consent_loan_approval': 1,      # Default granted
            'consent_personalization': 1,    # Default granted
            'consent_marketing': 0,          # Default revoked (0 = False)
            'consent_model_training': 0      # Default revoked
        }
        
        # Add columns if they don't exist
        modified = False
        for col, default_value in consent_columns.items():
            if col not in self.users.columns:
                self.users[col] = default_value
                modified = True
        
        if modified:
            self._save()
            print(f"✅ Added consent columns to users.csv")
    
    def _save(self):
        """Save users DataFrame to CSV"""
        self.users.to_csv(self.users_file, index=False)
    
    def _get_consent_mapping(self):
        """Map consent categories to column names"""
        return {
            'fraud_detection': 'consent_fraud_detection',
            'loan_approval': 'consent_loan_approval',
            'personalization': 'consent_personalization',
            'marketing': 'consent_marketing',
            'model_training': 'consent_model_training'
        }
    
    def _get_service_info(self):
        """Get service information for each consent type"""
        return {
            'fraud_detection': {
                'service_name': 'Fraud Detection AI',
                'service_description': 'Analyze transactions for suspicious activity'
            },
            'loan_approval': {
                'service_name': 'Loan Approval Model',
                'service_description': 'Use my data for credit decisions'
            },
            'personalization': {
                'service_name': 'Personalization Engine',
                'service_description': 'Customize offers based on my preferences'
            },
            'marketing': {
                'service_name': 'Marketing Analytics',
                'service_description': 'Include my anonymized data in marketing analysis'
            },
            'model_training': {
                'service_name': 'Model Training',
                'service_description': 'Use my data to improve AI models'
            }
        }
    
    def get_user_consents(self, user_id: str) -> List[Dict]:
        """Get all consents for a user"""
        user_row = self.users[self.users['user_id'] == user_id]
        
        if len(user_row) == 0:
            raise ValueError(f"User {user_id} not found")
        
        user_row = user_row.iloc[0]
        consent_mapping = self._get_consent_mapping()
        service_info = self._get_service_info()
        
        consents = []
        for category, col_name in consent_mapping.items():
            # Convert to int then to bool (handles both 0/1 and True/False)
            is_granted = bool(int(user_row[col_name]))
            
            # Generate consent ID
            consent_id = f"C-{user_id}-{category}"
            
            # Simulate usage count (based on hash of user_id + category)
            if is_granted:
                usage_hash = int(hashlib.md5(f"{user_id}{category}".encode()).hexdigest()[:8], 16)
                data_usage_count = (usage_hash % 200) + 50
                last_used = datetime.now().isoformat()
            else:
                data_usage_count = 0
                last_used = None
            
            info = service_info[category]
            consents.append({
                'consentId': consent_id,  # camelCase for frontend
                'serviceName': info['service_name'],
                'serviceDescription': info['service_description'],
                'category': category,
                'status': 'granted' if is_granted else 'revoked',
                'grantedAt': datetime.now().isoformat(),
                'revokedAt': None if is_granted else datetime.now().isoformat(),
                'dataUsageCount': data_usage_count,
                'lastUsed': last_used
            })
        
        return consents
    
    def get_consent_statistics(self, user_id: str) -> Dict:
        """Get consent statistics for a user"""
        consents = self.get_user_consents(user_id)
        
        total_active = sum(1 for c in consents if c['status'] == 'granted')
        total_revoked = sum(1 for c in consents if c['status'] == 'revoked')
        total_usage = sum(c['dataUsageCount'] for c in consents)  # camelCase
        
        return {
            'totalActive': total_active,
            'totalRevoked': total_revoked,
            'totalDataUsage': total_usage
        }
    
    def update_consent(self, consent_id: str, action: str) -> Dict:
        """Grant or revoke a consent"""
        # Parse consent_id to extract user_id and category
        # Format: C-U1000-fraud_detection
        parts = consent_id.split('-')
        if len(parts) != 3:
            raise ValueError(f"Invalid consent_id format: {consent_id}")
        
        user_id = parts[1]
        category = parts[2]
        
        # Get column name
        consent_mapping = self._get_consent_mapping()
        if category not in consent_mapping:
            raise ValueError(f"Unknown category: {category}")
        
        col_name = consent_mapping[category]
        
        # Find user
        user_idx = self.users[self.users['user_id'] == user_id].index
        if len(user_idx) == 0:
            raise ValueError(f"User {user_id} not found")
        
        user_idx = user_idx[0]
        
        # Update consent (store as 0 or 1, not True/False)
        new_value = 1 if (action == 'grant') else 0
        self.users.at[user_idx, col_name] = new_value
        self._save()
        
        # Get service info
        service_info = self._get_service_info()
        info = service_info[category]
        
        # Generate receipt
        timestamp = datetime.now().isoformat()
        receipt_id = f"CR-{int(datetime.now().timestamp())}"
        
        # Create hash chain
        prev_hash = self._get_last_hash(user_id)
        content = f"{receipt_id}{consent_id}{action}{timestamp}{prev_hash}"
        current_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        return {
            'receiptId': receipt_id,  # camelCase for frontend
            'consentId': consent_id,
            'userId': user_id,
            'serviceName': info['service_name'],
            'action': 'granted' if action == 'grant' else 'revoked',
            'timestamp': timestamp,
            'hash': f"0x{current_hash}",
            'previousHash': prev_hash
        }
    
    def _get_last_hash(self, user_id: str) -> str:
        """Get the last hash for a user (for blockchain-style chain)"""
        return f"0x{hashlib.sha256(user_id.encode()).hexdigest()[:16]}"

# Global instance
CONSENT_MANAGER = ConsentManager()
