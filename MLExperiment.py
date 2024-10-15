from Thesis import DatabaseReader, Evaluator, PreProcessing, ServerConnectionIPT1
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

class MLExperiment:
    def __init__(self, model: BaseEstimator, param_grid: dict, preprocessor: PreProcessing, evaluator: Evaluator, connection_class: DatabaseReader):
        self.start_time = datetime.now()
        self.server_connection = connection_class
        self.model = model
        self.param_grid = param_grid
        self.preprocessor = preprocessor
        self.evaluator = evaluator
        self.pipeline = None

    def load_and_preprocess_data(self):
        combined_tables = self.server_connection.read_IPT1_Table()
        return self.preprocessor.set_labels(combined_tables) 
    
    def prepare_features_and_labels(data):
        X = data.drop(['labels', 'twinnr', 'death_yrmon', 'birthdate1', 'age_death'], axis=1)
        y = data['labels']
        return X, y

    def create_pipeline(self, X):
        preprocessing_pipeline = self.preprocessor.create_pipeline(X)
        self.pipeline = Pipeline([
            ('preprocessor', preprocessing_pipeline),
            ('classifier', self.model)
        ])

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        grid_search = GridSearchCV(estimator=self.pipeline, param_grid=self.param_grid, 
                                   cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        return grid_search, X_test, y_test

    def run(self):
        data = self.load_and_preprocess_data()
        X, y = self.prepare_features_and_labels(data)
        self.create_pipeline(X)
        grid_search, X_test, y_test = self.train_model(X, y)
        self.evaluator.evaluate_model(grid_search, X_test, y_test)
        self.evaluator.plot_feature_importance(X, grid_search.best_estimator_)
        self.evaluator.plot_learning_curve(grid_search.best_estimator_, X, y)
        print("Total time:", datetime.now() - self.start_time)