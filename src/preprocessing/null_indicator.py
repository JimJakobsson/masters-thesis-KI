from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

class NullIndicator(BaseEstimator, TransformerMixin):
    """
    Transformer that creates binary indicators for null values in numeric features.
    For each numeric feature, creates a new column indicating whether the value was null (1) or not (0).
    
    The implementation is optimized to avoid DataFrame fragmentation by using pd.concat
    instead of multiple insert operations.
    
    Attributes:
        null_feature_names (List[str]): Names of the created null indicator features
        output_feature_names_ (List[str]): Names of all output features (original + null indicators)
        features_with_nulls_ (set): Set of feature names that contained null values during fit
    """
    
    def __init__(self):
        self.null_feature_names: List[str] = []
        self.output_feature_names_: List[str] = []
        self.features_with_nulls_: set = set()
        
    def fit(self, X: pd.DataFrame, y=None) -> 'NullIndicator':
        """
        Fit the transformer by identifying features that contain null values.
        
        Args:
            X: Input features to fit
            y: Ignored (included for sklearn compatibility)
            
        Returns:
            self: Returns the transformer instance
        """
        # Store original feature names
        self.output_feature_names_ = list(X.columns)
        
        # Identify features that actually contain nulls
        self.features_with_nulls_ = {
            col for col in X.columns 
            if X[col].isna().any()
        }
        
        # Generate null indicator feature names only for features with nulls
        self.null_feature_names = [
            f"{col}_nan" for col in self.features_with_nulls_
        ]
        
        # Add null indicator names to output features
        self.output_feature_names_.extend(self.null_feature_names)
        
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by adding null indicator columns.
        Uses pd.concat for better performance compared to multiple insert operations.
        
        Args:
            X: Input features to transform
            
        Returns:
            pd.DataFrame: Original dataframe with added null indicator columns
        """
        if not hasattr(self, 'features_with_nulls_'):
            raise NotFittedError(
                "This NullIndicator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        
        # Create DataFrame of null indicators only for features that had nulls during fit
        if self.features_with_nulls_:
            null_indicators = pd.DataFrame({
                f"{col}_nan": X[col].isna().astype(int)
                for col in self.features_with_nulls_
            }, index=X.index)
            
            # Combine original features with null indicators efficiently
            return pd.concat([X, null_indicators], axis=1)
        
        return X.copy()
        
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Get output feature names for the transformer.
        
        Args:
            input_features: Ignored (included for sklearn compatibility)
            
        Returns:
            List[str]: List of feature names that will be output by transform()
        """
        if not hasattr(self, 'output_feature_names_'):
            raise NotFittedError(
                "Transformer not fitted. Call fit before get_feature_names_out."
            )
        return self.output_feature_names_
# from typing import List, Tuple
# import pandas as pd
# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.exceptions import NotFittedError

# class NullIndicator(BaseEstimator, TransformerMixin):
#     """
#     Transformer that creates binary indicators for null values in numeric features.
#     For each numeric feature, creates a new column indicating whether the value was null (1) or not (0).
    
#     Attributes:
#         null_feature_names (List[str]): Names of the created null indicator features
#     """
    
#     def __init__(self):
#         self.null_feature_names: List[str] = []
#         self.output_feature_names_: List[str] = []  # Add this

        
#     def fit(self, X: pd.DataFrame, y=None) -> 'NullIndicator':
#         # Store original feature names plus null indicators
#         self.output_feature_names_ = list(X.columns)  # Original features
#         self.null_feature_names = [f"{col}_nan" for col in X.columns]
#         self.output_feature_names_.extend(self.null_feature_names)
#         return self
        
#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         """
#         Transform the data by adding null indicator columns.
        
#         Args:
#             X: Input features to transform
            
#         Returns:
#             pd.DataFrame: Original dataframe with added null indicator columns
#         """
#         X_copy = X.copy()
    
#         # Track which features actually have nulls
#         features_with_nulls = set()
        
#         for column in X.columns:
#             null_count = X[column].isna().sum()
#             indicator_name = f"{column}_nan"
#             X_copy[indicator_name] = X[column].isna().astype(int)
            
#             if null_count > 0:
#                 features_with_nulls.add(column)
        
      
        
#         return X_copy
#         # # Create null indicator columns
#         # for column in X.columns:
#         #     indicator_name = f"{column}_nan"
#         #     X_copy[indicator_name] = X[column].isna().astype(int)
#         #     if indicator_name not in self.null_feature_names:
#         #         self.null_feature_names.append(indicator_name)
           
#         #return X_copy
        
#     def get_feature_names_out(self, input_features=None) -> List[str]:
#         """sklearn compatible method to get output feature names"""
#         if not hasattr(self, 'output_feature_names_'):
#             raise NotFittedError("Transformer not fitted. Call fit before get_feature_names_out.")
#         return self.output_feature_names_