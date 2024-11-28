import pandas as pd
import numpy as np
from typing import Tuple

class DataCleaner:
    """Handles data cleaning operations"""

    def __init__(self, null_threshold: float):
        self.null_threshold = null_threshold

    def remove_high_null_features(self, X: pd.DataFrame, y: pd.Series, ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Remove features with high null ratios and return cleaned X and y.

        Args:
            X: Input features
            y: Target series
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Cleaned X and y
        """
        null_ratios = X.isnull().mean()
        high_null_features = null_ratios[null_ratios >= self.null_threshold].index

        if len(high_null_features) > 0:
            self._print_removed_features(high_null_features)
            X = X.copy()
            y = y.copy()
            X = X.loc[:, null_ratios < self.null_threshold]
            y = y.loc[X.index]

        return X, y

    @staticmethod
    def _print_removed_features(features: pd.Index) -> None:
        """Print removed features"""
        print(f"Removed features with high null values: {features}") 
        print(features.tolist())