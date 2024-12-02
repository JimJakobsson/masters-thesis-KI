from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
@dataclass
class PreprocessingResult:
    """Container for preprocessing results"""
    X: pd.DataFrame
    preprocessor: ColumnTransformer
    feature_names: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
