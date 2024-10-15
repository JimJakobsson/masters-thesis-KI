from typing import Any, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class PreProcessing:
    """A class for preprocessing combined tables with threshold-based labeling."""

    def __init__(self, combined_tables, threshold: int = 2005):
        """
        Initialize the PreProcessing object.

        :param combined_tables: The combined tables to process
        :param threshold: The threshold year for labeling, defaults to 2005
        """
        self._threshold = threshold
        self._combined_tables = combined_tables

    def set_threshold(self, threshold: int) -> None:
        """
        Set the threshold value.

        :param threshold: The new threshold value
        """
        self._threshold = threshold
    
    def get_threshold(self) -> int:
        """
        Get the current threshold value.

        :return: The current threshold value
        """
        return self._threshold
    
    def set_labels(self) -> None:
        """
        Set labels in the combined tables based on the death year and threshold.
        """
        self._combined_tables['labels'] = self._combined_tables['death_yrmon'].apply(
            lambda x: '1' if x and int(x[:4]) > self._threshold
                 else '0' if x and int(x[:4]) <= self._threshold
                 else None
        )
        self._combined_tables = self._combined_tables.dropna(subset=['labels'])

    def get_labeled_data(self) -> Any:
        """
        Get the labeled data.

        :return: The combined tables with labels
        """
        return self._combined_tables
    
    def detect_categorical_features(X):
        categorical_features = []
        for feature in X.columns:
            if X[feature].dtype == 'object' or (X[feature].min() >= 0 and X[feature].max() <= 10):
                categorical_features.append(feature)
        return categorical_features
    
    def create_pipeline(self, X):
        categorical_features = self.detect_categorical_features(X)
        numeric_features = X.columns.difference(categorical_features)

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', pd.get_dummies)
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', self.model)])
        return pipeline