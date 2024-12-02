from typing import Any, Dict, Tuple
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import pandas as pd

from experiment.experiment_config import ExperimentConfig

class ModelTrainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def create_pipeline(self, preprocessor, model) -> Pipeline:
        """Create a pipeline with the specified column transformer and model"""
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    
    def train_model(self, 
                   pipeline: Pipeline,
                   param_grid: Dict[str, Any],
                   X: pd.DataFrame,
                   y: pd.Series) -> GridSearchCV:
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=self.config.cv_folds,
            n_jobs=-1,
            verbose=1,
            scoring='accuracy',
            error_score='raise'
        )
        print("y shape", y.shape)
        print("x shape", X.shape)
        #print all features after encoding
        print("features", X.columns)
        grid_search.fit(X, y)
        return grid_search
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        return train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
    