import pandas as pd
import numpy as np
from typing import Tuple

class DataCleaner:
    """Handles data cleaning operations"""

    def __init__(self, null_threshold: float = 0.8):
        self.null_threshold = null_threshold

    def remove_high_null_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Remove features with high null ratios and return cleaned X and y.
        """
        null_ratios = X.isnull().mean()
        high_null_features = null_ratios[null_ratios >= self.null_threshold].index

        if len(high_null_features) > 0:
            print(f"Removing {len(high_null_features)} features with >{self.null_threshold*100}% null values:")
            print(high_null_features.tolist())

        X = X.copy()
        y = y.copy()
        X = X.loc[:, null_ratios < self.null_threshold]
        y = y.loc[X.index]

        return X, y
        