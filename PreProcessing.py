from typing import Any, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class PreProcessing:
    def __init__(self, threshold: int = 2005):
        self._threshold = threshold

    def set_threshold(self, threshold: int) -> None:
        self._threshold = threshold
    
    def get_threshold(self) -> int:
        return self._threshold
    
    def set_labels(self, combined_tables) -> None:
        combined_tables['labels'] = combined_tables['death_yrmon'].apply(
            lambda x: '1' if x and int(x[:4]) > self._threshold
                 else '0' if x and int(x[:4]) <= self._threshold
                 else None
        )
        combined_tables = combined_tables.dropna(subset=['labels'])
        return combined_tables
    
    @staticmethod
    def detect_categorical_features(X):
        categorical_features = []
        for feature in X.columns:
            if X[feature].dtype == 'object' or (X[feature].min() >= 0 and X[feature].max() <= 10):
                categorical_features.append(feature)
        return categorical_features
    
    def create_pipeline(self, X):
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        categorical_features = self.detect_categorical_features(X)
        numeric_features = X.columns.difference(categorical_features)

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        return preprocessor