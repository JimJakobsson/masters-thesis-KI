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
                print(f"categoricalfeature: {feature}")
                categorical_features.append(feature)
        numeric_features = [col for col in df.columns if col not in categorical_features]
        print("\nFeature Detection Results:")
        print(f"Categorical features ({len(categorical_features)}):")
        print(f"Numeric features ({len(numeric_features)}):")
        
        # for feature in df.columns:
        #     # Check if the column contains string values (excluding NaN/null)
        #     non_null_values = df[feature].dropna()
        #     n_unique = df[feature].nunique()
        #     n_count = df[feature].count()
            
        #     # Convert to string and check for alphabetic characters
        #     contains_strings = (non_null_values
        #         .astype(str)
        #         .str.lower()
        #         .str.replace(r'[-+]?\d*\.?\d+', '', regex=True)  # Remove numeric patterns
        #         .str.replace(r'\s+', '', regex=True)  # Remove whitespace
        #         .str.replace('nan', '', regex=True)   # Remove nan strings
        #         .str.replace('NaN', '', regex=True)   # Remove NaN strings
        #         .str.replace('null', '', regex=True)  # Remove null strings
        #         .str.replace('NULL', '', regex=True)  # Remove NULL strings
        #         .str.contains('[a-zA-Z]')
        #         .any())
            
        #     # print(f"\nAnalyzing feature: {feature}")
        #     # print(f"- Unique values: {n_unique}")
        #     # print(f"- Non-null count: {n_count}")
        #     # print(f"- Contains strings: {contains_strings}")
            
        #     if contains_strings:
        #         if n_unique <= max_unique and n_count >= min_count:
        #             categorical_features.append(feature)
        #             # print(f"=> Categorical (string-based)")
        #         else:
        #             high_cardinality_features.append((feature, n_unique))
        #             numeric_features.append(feature)  # Treat as numeric to avoid explosion
        #             # print(f"=> Numeric (high cardinality string)")
        #     else:
        #         try:
        #             non_null_values.astype(float)
        #             if n_unique <= max_unique and n_count >= min_count:
        #                 categorical_features.append(feature)
        #                 # print(f"=> Categorical (numeric)")
        #             else:
        #                 numeric_features.append(feature)
        #                 # print(f"=> Numeric")
        #         except (ValueError, TypeError):
        #             if n_unique <= max_unique and n_count >= min_count:
        #                 categorical_features.append(feature)
        #                 # print(f"=> Categorical (conversion failed)")
        #             else:
        #                 high_cardinality_features.append((feature, n_unique))
        #                 numeric_features.append(feature)  # Treat as numeric to avoid explosion
        #                 print(f"=> Numeric (high cardinality)")
        
        # # print("\nFeature Detection Summary:")
        # # print(f"Categorical features ({len(categorical_features)}):")
        # # for feat in categorical_features:
        # #     unique_vals = df[feat].dropna().unique()[:5]
        # #     print(f"- {feat}: {unique_vals}")
        
        # # print(f"\nNumeric features ({len(numeric_features)}):")
        # # print(numeric_features)
        
        # if high_cardinality_features:
        #     print("\nWarning: High Cardinality Features Detected:")
        #     for feat, n_unique in sorted(high_cardinality_features, key=lambda x: x[1], reverse=True):
        #         print(f"- {feat}: {n_unique} unique values")
        
        return categorical_features, numeric_features
        #     if contains_strings:
        #         # If the column contains any non-numeric strings, it's categorical
        #         categorical_features.append(feature)
        #     else:
        #         try:
        #             # Try converting non-null values to float
        #             non_null_values.astype(float)
                    
        #             # If successful and meets categorical criteria, mark as categorical
        #             if n_unique <= max_unique and df[feature].count() >= min_count:
        #                 categorical_features.append(feature)
        #             else:
        #                 numeric_features.append(feature)
        #         except (ValueError, TypeError):
        #             # If conversion fails, treat as categorical
        #             categorical_features.append(feature)
        
        # # Print detection results
        # print("\nFeature Detection Results:")
        # print(f"Categorical features ({len(categorical_features)}):")
        # print(categorical_features)
        # print(f"\nNumeric features ({len(numeric_features)}):")
        # print(numeric_features)
        
        # return categorical_features, numeric_features