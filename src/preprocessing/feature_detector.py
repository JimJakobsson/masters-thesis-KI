from typing import List, Tuple
import pandas as pd

class FeatureDetector:
    """Handles detection and classification of features"""
    # @staticmethod
    # def detect_feature_types(df: pd.DataFrame, 
    #                        max_unique: int = 10,
    #                        min_count: int = 10) -> Tuple[List[str], List[str]]:
    #     """
    #     Detect categorical and numerical features in the dataset.
        
    #     Args:
    #         df: Input DataFrame
    #         max_unique: Maximum unique values for categorical features
    #         min_count: Minimum count for categorical features
        
    #     Returns:
    #         Tuple of (categorical_features, numeric_features)
    #     """
    #     categorical_features = []
    #     numeric_features = []
        
    #     for feature in df.columns:
    #         # Get non-null values for analysis
    #         non_null_values = df[feature].dropna()
            
    #         if len(non_null_values) == 0:
    #             categorical_features.append(feature)
    #             continue
                
    #         try:
    #             # Try converting the values to float
    #             converted_values = pd.to_numeric(non_null_values)
                
    #             # Even if conversion succeeds, check if it should be categorical
    #             # based on number of unique values
    #             if converted_values.nunique() <= max_unique and df[feature].count() > min_count:
    #                 categorical_features.append(feature)
    #             else:
    #                 numeric_features.append(feature)
                    
    #         except ValueError:
    #             # If conversion fails, it contains non-numeric strings
    #             categorical_features.append(feature)
        
    #     # Print detection results
    #     print("\nFeature Detection Results:")
    #     print(f"\nCategorical Features ({len(categorical_features)}):")
    #     for feat in categorical_features:
    #         unique_vals = df[feat].dropna().unique()[:5]
    #         print(f"{feat}: {unique_vals}")
        
    #     print(f"\nNumeric Features ({len(numeric_features)}):")
    #     print(numeric_features)
        
    #     return categorical_features, numeric_features
    @staticmethod
    def detect_feature_types(df: pd.DataFrame, 
                            max_unique: int = 10,
                            min_count: int = 10) -> Tuple[List[str], List[str]]:
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
        
        for feature in df.columns:
            # Check if the column contains string values (excluding NaN/null)
            non_null_values = df[feature].dropna()
            
            # Convert to string and check for alphabetic characters
            contains_strings = (non_null_values
                .astype(str)
                .str.lower()
                .str.replace(r'[-+]?\d*\.?\d+', '', regex=True)  # Remove numeric patterns
                .str.replace(r'\s+', '', regex=True)  # Remove whitespace
                .str.replace('nan', '', regex=True)   # Remove nan strings
                .str.replace('null', '', regex=True)  # Remove null strings
                .str.contains('[a-zA-Z]')
                .any())
            
            # Count unique values excluding NaN
            n_unique = df[feature].nunique()
            
            if contains_strings:
                # If the column contains any non-numeric strings, it's categorical
                categorical_features.append(feature)
            else:
                try:
                    # Try converting non-null values to float
                    non_null_values.astype(float)
                    
                    # If successful and meets categorical criteria, mark as categorical
                    if n_unique <= max_unique and df[feature].count() >= min_count:
                        categorical_features.append(feature)
                    else:
                        numeric_features.append(feature)
                except (ValueError, TypeError):
                    # If conversion fails, treat as categorical
                    categorical_features.append(feature)
        
        # Print detection results
        print("\nFeature Detection Results:")
        print(f"Categorical features ({len(categorical_features)}):")
        print(categorical_features)
        print(f"\nNumeric features ({len(numeric_features)}):")
        print(numeric_features)
        
        return categorical_features, numeric_features