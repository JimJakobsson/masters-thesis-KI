from typing import List, Tuple
import pandas as pd

class FeatureDetector:
    """Handles detection and classification of features"""
    @staticmethod
    def detect_feature_types(df: pd.DataFrame, 
                            max_unique: int = 10,
                            min_count: int = 10,
                            max_categories: int = 10) -> Tuple[List[str], List[str]]:
        """
        Detect categorical and numerical features in the dataset.
        
        Args:
            df: Input DataFrame
            max_unique: Maximum unique values for categorical features
            min_count: Minimum count for considering categorical features
        
        Returns:
            Tuple of (categorical_features, numeric_features)
        """
        categorical_features = []
        numeric_features = []
        high_cardinality_features = []  # Track features with too many categories
        for feature in df.columns:
            
            is_categorical = (
                df[feature].dtype == 'object' or
                df[feature].dtype == 'bool' or
                (df[feature].nunique() <= max_unique) and (df[feature].count() >= min_count)
                
            )
            if is_categorical:
                categorical_features.append(feature)
        numeric_features = [col for col in df.columns if col not in categorical_features]
        print("\nFeature Detection Results:")
        print(f"Categorical features ({len(categorical_features)}):")
        print(f"Numeric features ({len(numeric_features)}):")
                
        return categorical_features, numeric_features
    