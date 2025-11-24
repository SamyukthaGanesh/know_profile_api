"""
Data Processor
Handles data preprocessing, cleaning, and feature engineering.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Universal data processor for cleaning and transforming data.
    Works with any tabular dataset.
    """
    
    def __init__(self):
        """Initialize data processor"""
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = None
        self.is_fitted = False
    
    def handle_missing_values(
        self,
        data: pd.DataFrame,
        strategy: str = 'median',
        fill_value: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data: Input DataFrame
            strategy: 'mean', 'median', 'most_frequent', 'constant'
            fill_value: Value to use if strategy='constant'
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        data_copy = data.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
        categorical_cols = data_copy.select_dtypes(include=['object', 'category']).columns
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            if 'numeric_imputer' not in self.imputers or not self.is_fitted:
                numeric_strategy = strategy if strategy != 'most_frequent' else 'median'
                self.imputers['numeric_imputer'] = SimpleImputer(
                    strategy=numeric_strategy,
                    fill_value=fill_value
                )
                data_copy[numeric_cols] = self.imputers['numeric_imputer'].fit_transform(
                    data_copy[numeric_cols]
                )
            else:
                data_copy[numeric_cols] = self.imputers['numeric_imputer'].transform(
                    data_copy[numeric_cols]
                )
        
        # Handle categorical columns
        if len(categorical_cols) > 0:
            if 'categorical_imputer' not in self.imputers or not self.is_fitted:
                categorical_strategy = 'most_frequent' if strategy != 'constant' else 'constant'
                self.imputers['categorical_imputer'] = SimpleImputer(
                    strategy=categorical_strategy,
                    fill_value=fill_value
                )
                data_copy[categorical_cols] = self.imputers['categorical_imputer'].fit_transform(
                    data_copy[categorical_cols]
                )
            else:
                data_copy[categorical_cols] = self.imputers['categorical_imputer'].transform(
                    data_copy[categorical_cols]
                )
        
        missing_after = data_copy.isnull().sum().sum()
        logger.info(f"Missing values after imputation: {missing_after}")
        
        return data_copy
    
    def encode_categorical_features(
        self,
        data: pd.DataFrame,
        method: str = 'label',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            data: Input DataFrame
            method: 'label' or 'onehot'
            columns: Columns to encode (if None, encode all categorical)
            
        Returns:
            DataFrame with encoded features
        """
        logger.info(f"Encoding categorical features with method: {method}")
        
        data_copy = data.copy()
        
        # Identify categorical columns
        if columns is None:
            categorical_cols = data_copy.select_dtypes(include=['object', 'category']).columns
        else:
            categorical_cols = columns
        
        if len(categorical_cols) == 0:
            logger.info("No categorical columns to encode")
            return data_copy
        
        if method == 'label':
            # Label encoding
            for col in categorical_cols:
                if col not in self.encoders or not self.is_fitted:
                    self.encoders[col] = LabelEncoder()
                    data_copy[col] = self.encoders[col].fit_transform(
                        data_copy[col].astype(str)
                    )
                else:
                    # Handle unseen categories
                    try:
                        data_copy[col] = self.encoders[col].transform(
                            data_copy[col].astype(str)
                        )
                    except ValueError:
                        # For unseen categories, assign a new label
                        logger.warning(f"Unseen categories in {col}, assigning new labels")
                        data_copy[col] = data_copy[col].astype(str)
                        known_classes = set(self.encoders[col].classes_)
                        
                        def encode_with_unknown(x):
                            if x in known_classes:
                                return self.encoders[col].transform([x])[0]
                            else:
                                return -1  # Unknown category
                        
                        data_copy[col] = data_copy[col].apply(encode_with_unknown)
        
        elif method == 'onehot':
            # One-hot encoding
            data_copy = pd.get_dummies(
                data_copy,
                columns=categorical_cols,
                drop_first=True
            )
        
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        
        return data_copy
    
    def scale_features(
        self,
        data: pd.DataFrame,
        method: str = 'standard',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            data: Input DataFrame
            method: 'standard' (z-score) or 'minmax' (0-1)
            columns: Columns to scale (if None, scale all numeric)
            
        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Scaling features with method: {method}")
        
        data_copy = data.copy()
        
        # Identify numeric columns
        if columns is None:
            numeric_cols = data_copy.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = columns
        
        if len(numeric_cols) == 0:
            logger.info("No numeric columns to scale")
            return data_copy
        
        # Create or use existing scaler
        if method not in self.scalers or not self.is_fitted:
            if method == 'standard':
                self.scalers[method] = StandardScaler()
            elif method == 'minmax':
                self.scalers[method] = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            data_copy[numeric_cols] = self.scalers[method].fit_transform(
                data_copy[numeric_cols]
            )
        else:
            data_copy[numeric_cols] = self.scalers[method].transform(
                data_copy[numeric_cols]
            )
        
        logger.info(f"Scaled {len(numeric_cols)} numeric columns")
        
        return data_copy
    
    def remove_outliers(
        self,
        data: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 1.5,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Remove outliers from the dataset.
        
        Args:
            data: Input DataFrame
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or z-score threshold
            columns: Columns to check for outliers
            
        Returns:
            DataFrame with outliers removed
        """
        logger.info(f"Removing outliers with method: {method}, threshold: {threshold}")
        
        data_copy = data.copy()
        initial_rows = len(data_copy)
        
        # Identify numeric columns
        if columns is None:
            numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
        else:
            numeric_cols = columns
        
        if method == 'iqr':
            # IQR method
            for col in numeric_cols:
                Q1 = data_copy[col].quantile(0.25)
                Q3 = data_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                data_copy = data_copy[
                    (data_copy[col] >= lower_bound) & 
                    (data_copy[col] <= upper_bound)
                ]
        
        elif method == 'zscore':
            # Z-score method
            for col in numeric_cols:
                z_scores = np.abs((data_copy[col] - data_copy[col].mean()) / data_copy[col].std())
                data_copy = data_copy[z_scores < threshold]
        
        else:
            raise ValueError(f"Unknown outlier removal method: {method}")
        
        final_rows = len(data_copy)
        removed = initial_rows - final_rows
        logger.info(f"Removed {removed} outlier rows ({removed/initial_rows*100:.2f}%)")
        
        return data_copy
    
    def create_interaction_features(
        self,
        data: pd.DataFrame,
        feature_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create interaction features between pairs of features.
        
        Args:
            data: Input DataFrame
            feature_pairs: List of (feature1, feature2) tuples
            
        Returns:
            DataFrame with added interaction features
        """
        logger.info(f"Creating {len(feature_pairs)} interaction features")
        
        data_copy = data.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in data_copy.columns and feat2 in data_copy.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                data_copy[interaction_name] = data_copy[feat1] * data_copy[feat2]
        
        return data_copy
    
    def reduce_memory_usage(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce memory usage by downcasting numeric types.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with reduced memory usage
        """
        logger.info("Reducing memory usage...")
        
        initial_memory = data.memory_usage(deep=True).sum() / 1024**2
        
        data_copy = data.copy()
        
        # Downcast integers
        int_cols = data_copy.select_dtypes(include=['int']).columns
        for col in int_cols:
            data_copy[col] = pd.to_numeric(data_copy[col], downcast='integer')
        
        # Downcast floats
        float_cols = data_copy.select_dtypes(include=['float']).columns
        for col in float_cols:
            data_copy[col] = pd.to_numeric(data_copy[col], downcast='float')
        
        final_memory = data_copy.memory_usage(deep=True).sum() / 1024**2
        reduction = (initial_memory - final_memory) / initial_memory * 100
        
        logger.info(f"Memory reduced from {initial_memory:.2f}MB to {final_memory:.2f}MB ({reduction:.1f}% reduction)")
        
        return data_copy
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        handle_missing: bool = True,
        encode_categorical: bool = True,
        scale_features: bool = True,
        remove_outliers: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fit and transform data with all preprocessing steps.
        
        Args:
            data: Input DataFrame
            handle_missing: Whether to handle missing values
            encode_categorical: Whether to encode categorical features
            scale_features: Whether to scale numeric features
            remove_outliers: Whether to remove outliers
            **kwargs: Additional parameters for each step
            
        Returns:
            Processed DataFrame
        """
        logger.info("Fitting and transforming data...")
        
        self.feature_names = data.columns.tolist()
        processed_data = data.copy()
        
        # Remove outliers first (if enabled)
        if remove_outliers:
            processed_data = self.remove_outliers(
                processed_data,
                **kwargs.get('outlier_params', {})
            )
        
        # Handle missing values
        if handle_missing:
            processed_data = self.handle_missing_values(
                processed_data,
                **kwargs.get('missing_params', {})
            )
        
        # Encode categorical features
        if encode_categorical:
            processed_data = self.encode_categorical_features(
                processed_data,
                **kwargs.get('encoding_params', {})
            )
        
        # Scale features
        if scale_features:
            processed_data = self.scale_features(
                processed_data,
                **kwargs.get('scaling_params', {})
            )
        
        self.is_fitted = True
        logger.info("Data preprocessing complete")
        
        return processed_data
    
    def transform(
        self,
        data: pd.DataFrame,
        handle_missing: bool = True,
        encode_categorical: bool = True,
        scale_features: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Transform data using fitted processors (for test data).
        
        Args:
            data: Input DataFrame
            handle_missing: Whether to handle missing values
            encode_categorical: Whether to encode categorical features
            scale_features: Whether to scale numeric features
            **kwargs: Additional parameters
            
        Returns:
            Processed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Processor not fitted. Call fit_transform first.")
        
        logger.info("Transforming data...")
        
        processed_data = data.copy()
        
        # Handle missing values
        if handle_missing:
            processed_data = self.handle_missing_values(
                processed_data,
                **kwargs.get('missing_params', {})
            )
        
        # Encode categorical features
        if encode_categorical:
            processed_data = self.encode_categorical_features(
                processed_data,
                **kwargs.get('encoding_params', {})
            )
        
        # Scale features
        if scale_features:
            processed_data = self.scale_features(
                processed_data,
                **kwargs.get('scaling_params', {})
            )
        
        logger.info("Data transformation complete")
        
        return processed_data
    
    def get_feature_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with feature statistics
        """
        stats = {
            'n_features': len(data.columns),
            'n_samples': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.astype(str).to_dict(),
            'numeric_features': data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': data.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        return stats


def create_data_processor() -> DataProcessor:
    """
    Convenience function to create a data processor.
    
    Returns:
        DataProcessor instance
    """
    return DataProcessor()