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
            # Check if the column contains any string values
            # contains_strings = (df[feature]
            #        .dropna()  # Remove NaN values
            #        .astype(str)
            #        .str.lower()  # Convert to lowercase to also catch 'NaN', 'nan', etc
            #        .str.replace('nan', '')  # Remove any remaining 'nan' strings
            #        .str.contains('[a-zA-Z]')
            #        .any())
            string_ratio = (df[feature]
                .dropna()  # Remove NaN values
                .astype(str)
                .str.lower()  # Convert to lowercase
                .str.replace('nan', '')  # Remove any remaining 'nan' strings
                .str.contains('[a-zA-Z]')  # Check for alphabetic characters
                .mean())
            contains_strings = string_ratio > 0.5 
            # Count unique values excluding NaN
            n_unique = df[feature].nunique()
            
            is_categorical = (
                df[feature].dtype == 'object' or  # Object type
                df[feature].dtype == 'bool' or    # Boolean type
                contains_strings or               # Contains any strings
                (n_unique <= max_unique and       # Few unique values
                 df[feature].count() > min_count)
            )
            
            if is_categorical:
                categorical_features.append(feature)
            else:
                try:
                    # Try converting to float to verify it's truly numeric
                    df[feature].astype(float)
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