"""
Configuration Settings
Centralized configuration for the AI Governance Framework.
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ExplainabilityConfig:
    """Configuration for explainability methods"""
    
    # SHAP settings
    shap_background_samples: int = 100
    shap_n_steps: int = 50
    
    # LIME settings
    lime_n_samples: int = 5000
    lime_n_features: int = 10
    
    # Integrated Gradients settings
    ig_n_steps: int = 50
    ig_baseline_type: str = 'zeros'  # 'zeros', 'mean', 'median'
    
    # Anchors settings
    anchors_threshold: float = 0.95
    anchors_beam_size: int = 4


@dataclass
class FairnessConfig:
    """Configuration for fairness metrics"""
    
    # Statistical Parity
    statistical_parity_threshold: float = 0.8  # 80% rule
    statistical_parity_absolute_threshold: float = 0.1
    
    # Equal Opportunity
    equal_opportunity_threshold: float = 0.8
    equal_opportunity_absolute_threshold: float = 0.1
    check_equalized_odds: bool = False
    
    # Calibration
    calibration_threshold: float = 0.1
    calibration_n_bins: int = 10
    calibration_strategy: str = 'uniform'  # 'uniform' or 'quantile'
    
    # Bias Detector
    confidence_level: float = 0.95


@dataclass
class ModelConfig:
    """Configuration for model management"""
    
    model_type: str = 'classification'  # 'classification' or 'regression'
    model_name: str = 'DefaultModel'
    model_version: str = '1.0'
    
    # Model storage - use outputs directory
    model_save_dir: str = 'outputs/saved_models'
    auto_save: bool = False  # Disabled by default - enable only when needed


@dataclass
class DataConfig:
    """Configuration for data processing"""
    
    # Data loading
    data_dir: str = 'data'
    cache_processed_data: bool = True
    
    # Preprocessing
    handle_missing: bool = True
    missing_strategy: str = 'median'  # 'mean', 'median', 'most_frequent'
    
    encode_categorical: bool = True
    encoding_method: str = 'label'  # 'label' or 'onehot'
    
    scale_features: bool = True
    scaling_method: str = 'standard'  # 'standard' or 'minmax'
    
    remove_outliers: bool = False
    outlier_method: str = 'iqr'  # 'iqr' or 'zscore'
    outlier_threshold: float = 1.5
    
    # Train/test split
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass
class LiteracyConfig:
    """Configuration for AI literacy prompts"""
    
    # Default audience settings
    default_user_literacy: str = 'intermediate'  # 'beginner', 'intermediate', 'advanced'
    default_banker_role: str = 'technical_analyst'
    
    # Prompt generation
    include_examples: bool = True
    max_prompt_length: int = 4000
    
    # Language settings
    default_language: str = 'en'
    supported_languages: list = field(default_factory=lambda: ['en'])

@dataclass
class ComplianceConfig:
    """Configuration for compliance module"""
    
    # Policy storage
    policy_storage_path: str = 'core/compliance/regulations/regulations_db.json'
    auto_load_policies: bool = True
    auto_save_policies: bool = True
    
    # Policy engine
    strict_mode: bool = False  # Stop on first violation or check all
    enable_policy_caching: bool = True
    
    # Audit logging
    audit_storage_path: str = 'audit_ledger.json'
    enable_audit_logging: bool = True
    enable_digital_signatures: bool = False
    audit_retention_days: int = 365  # Keep audit logs for 1 year
    
    # Hash chain settings
    hash_algorithm: str = 'sha256'
    verify_chain_on_startup: bool = True
    
    # Compliance thresholds
    max_violations_allowed: int = 0  # Zero tolerance
    require_explanation_threshold: float = 0.7  # Require explanation if confidence < 70%
    
    # Regulation sources
    default_regulations: list = field(default_factory=lambda: [
        'Basel III',
        'GDPR',
        'Equal Credit Opportunity Act',
        'Fair Credit Reporting Act'
    ])
    
    # Export settings
    enable_compliance_export: bool = True
    export_format: str = 'json'  # 'json' or 'csv'


@dataclass
class APIConfig:
    """Configuration for API server"""
    
    # Server settings
    host: str = '0.0.0.0'
    port: int = 8000
    debug: bool = False
    reload: bool = False
    
    # API settings
    api_title: str = 'AI Governance Framework API'
    api_version: str = '1.0.0'
    api_description: str = 'REST API for AI explainability, fairness, and literacy'
    
    # CORS settings
    allow_origins: list = field(default_factory=lambda: ['*'])
    allow_credentials: bool = True
    allow_methods: list = field(default_factory=lambda: ['*'])
    allow_headers: list = field(default_factory=lambda: ['*'])
    
    # Rate limiting
    enable_rate_limiting: bool = False
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    
    # Response settings
    max_response_size: int = 10000000  # 10MB


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    
    log_level: str = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_to_file: bool = False  # Disabled by default - enable only when needed
    log_file: str = 'outputs/logs/app.log'  # Put in outputs directory if used
    max_log_size: int = 10485760  # 10MB
    backup_count: int = 5


class Settings:
    """
    Main settings class that combines all configuration.
    Can load from environment variables or config files.
    """
    
    def __init__(
        self,
        env: str = 'development',
        config_file: Optional[str] = None
    ):
        """
        Initialize settings.
        
        Args:
            env: Environment ('development', 'production', 'testing')
            config_file: Path to config file (optional)
        """
        self.env = env
        
        # Initialize all config sections
        self.explainability = ExplainabilityConfig()
        self.fairness = FairnessConfig()
        self.model = ModelConfig()
        self.data = DataConfig()
        self.literacy = LiteracyConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.compliance = ComplianceConfig()  
        
        # Load from config file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Override with environment variables
        self.load_from_env()
        
        # Environment-specific overrides
        if env == 'production':
            self._set_production_defaults()
        elif env == 'testing':
            self._set_testing_defaults()
    
    def load_from_env(self):
        """Load settings from environment variables"""
        
        # API settings
        self.api.host = os.getenv('API_HOST', self.api.host)
        self.api.port = int(os.getenv('API_PORT', self.api.port))
        self.api.debug = os.getenv('API_DEBUG', str(self.api.debug)).lower() == 'true'
        
        # Model settings
        self.model.model_save_dir = os.getenv('MODEL_SAVE_DIR', self.model.model_save_dir)
        
        # Data settings
        self.data.data_dir = os.getenv('DATA_DIR', self.data.data_dir)
        
        # Logging
        self.logging.log_level = os.getenv('LOG_LEVEL', self.logging.log_level)
        self.logging.log_file = os.getenv('LOG_FILE', self.logging.log_file)
    
    def load_from_file(self, config_file: str):
        """Load settings from a config file (JSON or YAML)"""
        import json
        
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            if config_file.endswith('.json'):
                config_data = json.load(f)
            elif config_file.endswith(('.yaml', '.yml')):
                try:
                    import yaml
                    config_data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config files")
            else:
                raise ValueError("Config file must be JSON or YAML")
        
        # Update settings from config data
        self._update_from_dict(config_data)
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update settings from dictionary"""
        
        for section, values in config_data.items():
            if hasattr(self, section):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def _set_production_defaults(self):
        """Set production-specific defaults"""
        self.api.debug = False
        self.api.reload = False
        self.logging.log_level = 'WARNING'
        self.data.cache_processed_data = True
    
    def _set_testing_defaults(self):
        """Set testing-specific defaults"""
        self.api.debug = True
        self.logging.log_level = 'DEBUG'
        self.data.test_size = 0.3  # Larger test set for testing
        self.explainability.shap_background_samples = 50  # Faster for tests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'env': self.env,
            'explainability': self.explainability.__dict__,
            'fairness': self.fairness.__dict__,
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'literacy': self.literacy.__dict__,
            'api': self.api.__dict__,
            'logging': self.logging.__dict__,
            'compliance': self.compliance.__dict__,  
        }
    
    def save_to_file(self, config_file: str):
        """Save current settings to file"""
        import json
        
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_file.endswith('.json'):
                json.dump(self.to_dict(), f, indent=2)
            elif config_file.endswith(('.yaml', '.yml')):
                try:
                    import yaml
                    yaml.dump(self.to_dict(), f, default_flow_style=False)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config files")
    
    def create_directories(self):
        """Create necessary directories (only for actively used paths)"""
        directories = [
            # Only create data directory if it will be used
            self.data.data_dir,
            # Only create log directory if logging to file is enabled
            Path(self.logging.log_file).parent if self.logging.log_to_file else None
            # Skip model_save_dir - create on demand when actually saving models
        ]
        
        for directory in directories:
            if directory:  # Only create if not None
                Path(directory).mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings = None


def get_settings(
    env: Optional[str] = None,
    config_file: Optional[str] = None,
    force_reload: bool = False
) -> Settings:
    """
    Get or create global settings instance.
    
    Args:
        env: Environment ('development', 'production', 'testing')
        config_file: Path to config file
        force_reload: Force reload settings
        
    Returns:
        Settings instance
    """
    global _settings
    
    if _settings is None or force_reload:
        env = env or os.getenv('APP_ENV', 'development')
        _settings = Settings(env=env, config_file=config_file)
        _settings.create_directories()
    
    return _settings


# Example usage
if __name__ == "__main__":
    # Get settings
    settings = get_settings()
    
    # Print current settings
    print("Current Settings:")
    print(f"Environment: {settings.env}")
    print(f"API Host: {settings.api.host}:{settings.api.port}")
    print(f"Log Level: {settings.logging.log_level}")
    print(f"SHAP Background Samples: {settings.explainability.shap_background_samples}")
    print(f"Fairness Threshold: {settings.fairness.statistical_parity_threshold}")
    
    # Save to file
    settings.save_to_file('config/default_config.json')
    print("\n✓ Settings saved to config/default_config.json")