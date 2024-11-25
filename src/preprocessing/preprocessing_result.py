from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
@dataclass
class PreprocessingResult:
    """Container for preprocessing results"""
    X: pd.DataFrame
    y: pd.Series
    preprocessor: ColumnTransformer
    feature_names: list
