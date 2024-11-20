from typing import Any, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import  SimpleImputer
from sklearn.pipeline import Pipeline
# from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class PreProcessing:
    def __init__(self, threshold: int = 2005):
        self._threshold = threshold
        self.categorical_features = None
        self.numeric_features = None

    def set_threshold(self, threshold: int) -> None:
        self._threshold = threshold
    
    def get_threshold(self) -> int:
        return self._threshold
    
    def set_labels(self, combined_tables) -> None:
        #Copy to avoid modifying the original data
        combined_tables = combined_tables.copy()
        #remove row with null death_yrmon
        combined_tables = combined_tables.dropna(subset=['death_yrmon'])

        #Convert death_yrmon to string, consistent format
        combined_tables['death_yrmon'] = combined_tables['death_yrmon'].apply(
            lambda x: str(int(x)) if pd.notnull(x) else None
        )

        #Create labels with validation. Handles cases where death_yrmon is not in the expected format
        def create_label(x: str) -> Optional[str]:
            try:
                if x and len(x) >= 4:
                    year = int(x[:4])
                    return '0' if year > self._threshold else '1'
                return None
            except (ValueError, TypeError):
                return None
        
        combined_tables['labels'] = combined_tables['death_yrmon'].apply(create_label)

        #Drop rows with null labels
        combined_tables = combined_tables.dropna(subset=['labels'])

        combined_tables['labels'] = combined_tables['labels'].astype(int)
        print(f"Labels created: {combined_tables['labels'].value_counts().to_dict()}")
        print("labels successfully set")
        #combined_tables = combined_tables.dropna(subset=['labels'])
        return combined_tables
    # def set_ages(self, combined_tables) -> None:
    def detect_categorical_features(self, X):
        """
        Detect categorical features in the dataset
        """
        def contains_strings(series):
            """
            Check if a series contains any actual string values,
            properly handling NULL/NaN values.
            """
            null_mask = (series.isna()) | (series.astype(str).str.upper() == 'NULL') | (series.astype(str).str == 'NaN')
            #Filter out NULL/NaN values
            non_null_series = series[~null_mask]
            print("series", series)
            #If all values are NULL/NaN, return False
            if non_null_series.empty:
                return False
            
            return series.astype(str).str.contains('[a-zA-Z]').any()
        if self.categorical_features is None:
            categorical_features = []

            for feature in X.columns:
                #Consider a feature categorical if:
                #1. It is an object 
                #2. It is a boolean
                #3. It is an integer with less than 10 unique values
                #4. It contains string values
                
                is_categorical = (
                    X[feature].dtype == 'object' or
                    X[feature].dtype == 'bool' or
                    (X[feature].nunique() <= 10) and (X[feature].count() > 10)  #or

                    #check to see if the column contains string values
                    #come back to this later
                    #contains_strings(X[feature])
                )

                if is_categorical:
                    categorical_features.append(feature)

            self.categorical_features = categorical_features
            self.numeric_features = [col for col in X.columns if col not in categorical_features]

            print("\nFeature Detection Results:")
            print(f"Categorical features ({len(self.categorical_features)}):")
            print(f"Numeric features ({len(self.numeric_features)}):")
        return self.categorical_features
        # if self.categorical_features is None:
        #     categorical_features = []
        #     for feature in X.columns:
        #         if (X[feature].dtype == 'object' or 
        #             X[feature].dtype == 'bool' or
        #             #check if the number of unique values is less than 10 and the type is not float
        #             (X[feature].nunique() <= 10)): 
        #             # (X[feature].nunique() <= 10 and X[feature].dtype != 'float64')):
        #             categorical_features.append(feature)
        #     self.categorical_features = categorical_features
        #     self.numeric_features = [col for col in X.columns if col not in categorical_features]
        #     print("\nFeature Detection Results:")
        #     print(f"Detected {len(self.categorical_features)} categorical features")
        #     print(f"Detected {len(self.numeric_features)} numeric features")
        # return self.categorical_features
    
    # def delete_null_features(self, X, y):
    #     threshold = 0.8
    #     X = X.loc[:, X.isnull().mean() < threshold]
    #     y = y.loc[X.index]
    #     return X, y
    def delete_null_features(self, X, y):
        """
        Remove features with too many null values and return cleaned X and y.
        """
        threshold = 0.8
        null_ratios = X.isnull().mean()
        high_null_features = null_ratios[null_ratios >= threshold].index
        
        if len(high_null_features) > 0:
            print(f"Removing {len(high_null_features)} features with >{threshold*100}% null values:")
            print(high_null_features.tolist())
        
        X = X.copy()
        y = y.copy()
        X = X.loc[:, null_ratios < threshold]
        y = y.loc[X.index]
        
        return X, y
    def get_feature_names(self, preprocessor, X):
        # Get feature names for numeric features
        numeric_features = X.columns.difference(self.categorical_features).tolist()
        
        # Get feature names for categorical features
        categorical_transformer = preprocessor.named_transformers_['cat']
        categorical_features = categorical_transformer.named_steps['onehot'].get_feature_names_out(self.categorical_features).tolist()
        
        # Combine feature names
        feature_names = numeric_features + categorical_features
        return feature_names
    
    def check_data_consistency(self, X_train, X_test):
        """Check for consistency between training and test data"""
        print("\nData Consistency Check:")
        #Check if the number of columns in training and test data are the same'
        print(f"Training transformed data shape: {X_train.shape}")
        print(f"Test transformed data shape: {X_test.shape}")
        if X_train.shape[1] != X_test.shape[1]:
            print(f"Training data has {X_train.shape[1]} columns while test data has {X_test.shape[1]} columns")
        else:
            print("Number of columns in training and test data are the same")
        
    def set_categorical_features(self, X):
        """
        Set the categorical features based on the input data. Numerical features are
        determined as the columns that are not categorical.
        """
        print("Setting categorical features")
        self.categorical_features = self.detect_categorical_features(X)
        print(f"Set {len(self.categorical_features)} categorical features")

    def create_pipeline(self):
        
        # categorical_features = self.detect_categorical_features(X)
        # numeric_features = X.columns.difference(categorical_features)

        #self.detect_categorical_features(X)

        numeric_transformer = Pipeline(steps=[
            #skip imputataion for now
            #skip all inputing for now
            # ('imputer', SimpleImputer(strategy='median')),
            # ('imputer', IterativeImputer(max_iter=10, random_state=0)),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder = 'drop', #Handle any columns that are not specified. Passthorugh means they are not transformed
            n_jobs=None 
        )
        
        return preprocessor
    