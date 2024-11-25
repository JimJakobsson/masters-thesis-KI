from typing import List, Tuple
import pandas as pd

class FeatureDetector:
    """Handles detection and classification of features"""

    @staticmethod
    def detect_feature_types(df: pd.DataFrame, max_unique: int = 10, min_count: int = 10) -> Tuple[List[str], List[str]]:
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
        
        
        for feature in df.columns:
            if FeatureDetector._is_categorical(df[feature], max_unique, min_count):
                categorical_features.append(feature)

        numeric_features = [col for col in df.columns if col not in categorical_features]
        FeatureDetector._print_detection_results(categorical_features, numeric_features)
        return categorical_features, numeric_features
    
    @staticmethod       
    def _is_categorical(series: pd.Series, max_unique: int, min_count: int) -> bool:
        """Determine if a series is categorical"""
        return (
            series.dtype == 'object' or
            series.dtype == 'bool' or
            (series.nunique() <= max_unique and series.count() > min_count)
        )
    
    @staticmethod
    def _print_detection_results(categorical: List[str], numeric: List[str]) -> None:
        """Print feature detection results"""
        print("\nFeature Detection Results:")
        print(f"Categorical features ({len(categorical)}):")
        print(categorical)
        print(f"\nNumeric features ({len(numeric)}):")
        print(numeric)