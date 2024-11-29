import pandas as pd
import numpy as np
from typing import Tuple

class DataCleaner:
    """Handles data cleaning operations"""

    def __init__(self, null_threshold: float):
        self.null_threshold = null_threshold

    def clean_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean features without modifying target"""
        X = X.copy()
        
        # Remove features with too many null values
        null_ratios = X.isnull().mean()
        high_null_features = null_ratios[null_ratios >= self.null_threshold].index
        
        if len(high_null_features) > 0:
            print(f"Removing {len(high_null_features)} features with >{self.null_threshold*100}% null values")
            X = X.drop(columns=high_null_features)
            
        return X