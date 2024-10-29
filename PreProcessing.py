from typing import Any, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import  SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class PreProcessing:
    def __init__(self, threshold: int = 2005):
        self._threshold = threshold

    def set_threshold(self, threshold: int) -> None:
        self._threshold = threshold
    
    def get_threshold(self) -> int:
        return self._threshold
    
    def set_labels(self, combined_tables) -> None:
        
        #remove row with null death_yrmon
        combined_tables = combined_tables.dropna(subset=['death_yrmon'])
        #print values that are floats
        print(combined_tables[combined_tables['death_yrmon'].str.contains('\.')]['death_yrmon'].unique())
        #combined_tables['death_yrmon'] = combined_tables['death_yrmon'].astype(str)
        #print all values for death_yrmon that are not a number
        # print(combined_tables[~combined_tables['death_yrmon'].str.isnumeric()]['death_yrmon'].unique())
        combined_tables['labels'] = combined_tables['death_yrmon'].apply(
            lambda x: '1' if x and int(x[:4]) > self._threshold
                 else '0' if x and int(x[:4]) <= self._threshold
                 else None
        )
        #combined_tables = combined_tables.dropna(subset=['labels'])
        return combined_tables
    
    # @staticmethod
    def detect_categorical_features(self, X):
        categorical_features = []
        for feature in X.columns:
            if X[feature].dtype == 'object' or (X[feature].nunique() <= 10 and X[feature].dtype != 'float64'):
                categorical_features.append(feature)
        return categorical_features
    
    def delete_null_features(self, X, y):
        threshold = 0.8
        X = X.loc[:, X.isnull().mean() < threshold]
        y = y.loc[X.index]
        return X, y
        #double check
    def create_pipeline(self, X):
        
        categorical_features = self.detect_categorical_features(X)
        numeric_features = X.columns.difference(categorical_features)

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
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    