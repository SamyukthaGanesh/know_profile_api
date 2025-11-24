"""
Base Data Loader
Provides a data-agnostic interface for loading and processing datasets.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    Ensures consistent interface for all data sources.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to data file or directory
        """
        self.data_path = data_path
        self.data = None
        self.feature_names = None
        self.target_name = None
        self.sensitive_features = None
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load data from source.
        
        Returns:
            DataFrame with loaded data
        """
        pass
    
    @abstractmethod
    def get_features(self) -> pd.DataFrame:
        """
        Get feature columns.
        
        Returns:
            DataFrame with features only
        """
        pass
    
    @abstractmethod
    def get_target(self) -> pd.Series:
        """
        Get target column.
        
        Returns:
            Series with target values
        """
        pass
    
    def get_sensitive_features(self) -> Optional[pd.DataFrame]:
        """
        Get sensitive feature columns (for fairness analysis).
        
        Returns:
            DataFrame with sensitive features or None
        """
        if self.sensitive_features is None:
            return None
        
        if isinstance(self.sensitive_features, list):
            return self.data[self.sensitive_features]
        
        return self.sensitive_features
    
    def get_train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed
            stratify: Whether to stratify by target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        X = self.get_features()
        y = self.get_target()
        
        stratify_by = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_by
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary with dataset info
        """
        if self.data is None:
            return {'loaded': False}
        
        info = {
            'loaded': True,
            'n_samples': len(self.data),
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'has_missing': self.data.isnull().any().any(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Add target distribution
        if self.target_name and self.target_name in self.data.columns:
            target = self.data[self.target_name]
            info['target_distribution'] = target.value_counts().to_dict()
            info['target_type'] = str(target.dtype)
        
        # Add sensitive features info
        if self.sensitive_features:
            info['sensitive_features'] = (
                self.sensitive_features 
                if isinstance(self.sensitive_features, list) 
                else list(self.sensitive_features.columns)
            )
        
        return info
    
    def describe(self) -> pd.DataFrame:
        """Get descriptive statistics"""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        return self.data.describe()
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """Get first n rows"""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        return self.data.head(n)
    
    def sample(self, n: int = 5, random_state: int = 42) -> pd.DataFrame:
        """Get random sample"""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        return self.data.sample(n=min(n, len(self.data)), random_state=random_state)


class CSVDataLoader(BaseDataLoader):
    """
    Data loader for CSV files.
    """
    
    def __init__(
        self,
        data_path: str,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        sensitive_feature_columns: Optional[List[str]] = None,
        **csv_kwargs
    ):
        """
        Initialize CSV data loader.
        
        Args:
            data_path: Path to CSV file
            target_column: Name of target column
            feature_columns: Names of feature columns (if None, use all except target)
            sensitive_feature_columns: Names of sensitive feature columns
            **csv_kwargs: Additional arguments for pd.read_csv
        """
        super().__init__(data_path)
        self.target_name = target_column
        self.feature_columns = feature_columns
        self.sensitive_feature_columns = sensitive_feature_columns
        self.csv_kwargs = csv_kwargs
    
    def load(self) -> pd.DataFrame:
        """Load data from CSV file"""
        logger.info(f"Loading data from {self.data_path}")
        
        self.data = pd.read_csv(self.data_path, **self.csv_kwargs)
        
        # Set feature names
        if self.feature_columns is None:
            # Use all columns except target
            self.feature_names = [
                col for col in self.data.columns 
                if col != self.target_name
            ]
        else:
            self.feature_names = self.feature_columns
        
        # Set sensitive features
        if self.sensitive_feature_columns:
            self.sensitive_features = self.sensitive_feature_columns
        
        logger.info(f"Loaded {len(self.data)} rows with {len(self.feature_names)} features")
        
        return self.data
    
    def get_features(self) -> pd.DataFrame:
        """Get feature columns"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        return self.data[self.feature_names]
    
    def get_target(self) -> pd.Series:
        """Get target column"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        return self.data[self.target_name]


class DataFrameLoader(BaseDataLoader):
    """
    Data loader for existing pandas DataFrame.
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        sensitive_feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize DataFrame loader.
        
        Args:
            dataframe: Pandas DataFrame
            target_column: Name of target column
            feature_columns: Names of feature columns
            sensitive_feature_columns: Names of sensitive features
        """
        super().__init__(None)
        self.data = dataframe
        self.target_name = target_column
        self.feature_columns = feature_columns
        self.sensitive_feature_columns = sensitive_feature_columns
        
        # Set feature names
        if self.feature_columns is None:
            self.feature_names = [
                col for col in self.data.columns 
                if col != self.target_name
            ]
        else:
            self.feature_names = self.feature_columns
        
        # Set sensitive features
        if self.sensitive_feature_columns:
            self.sensitive_features = self.sensitive_feature_columns
    
    def load(self) -> pd.DataFrame:
        """Data already loaded"""
        logger.info(f"Using existing DataFrame with {len(self.data)} rows")
        return self.data
    
    def get_features(self) -> pd.DataFrame:
        """Get feature columns"""
        return self.data[self.feature_names]
    
    def get_target(self) -> pd.Series:
        """Get target column"""
        return self.data[self.target_name]


def create_data_loader(
    source: Union[str, pd.DataFrame],
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    sensitive_feature_columns: Optional[List[str]] = None,
    **kwargs
) -> BaseDataLoader:
    """
    Convenience function to create appropriate data loader.
    
    Args:
        source: Path to CSV file or DataFrame
        target_column: Name of target column
        feature_columns: Names of feature columns
        sensitive_feature_columns: Names of sensitive features
        **kwargs: Additional arguments
        
    Returns:
        BaseDataLoader instance
    """
    if isinstance(source, str):
        return CSVDataLoader(
            data_path=source,
            target_column=target_column,
            feature_columns=feature_columns,
            sensitive_feature_columns=sensitive_feature_columns,
            **kwargs
        )
    elif isinstance(source, pd.DataFrame):
        return DataFrameLoader(
            dataframe=source,
            target_column=target_column,
            feature_columns=feature_columns,
            sensitive_feature_columns=sensitive_feature_columns
        )
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")