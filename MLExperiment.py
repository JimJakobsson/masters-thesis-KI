import ServerConnectionIPT1
import PreProcessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report

class MLExperiment:
    def __init__(self, model: BaseEstimator, preprocessor: PreProcessing, param_grid: dict, connection_class=ServerConnectionIPT1):
        self.start_time = datetime.now()
        self.server_connection = connection_class()
        self.model = model
        self.param_grid = param_grid
        self.preprocessor = preprocessor

    def load_and_preprocess_data(self):
        combined_tables = self.server_connection.read_IPT1_Table()
        processed_tables = self.preprocessor(combined_tables, 1991)
        processed_tables.set_labels()
        return processed_tables.get_labeled_data()

    def prepare_features_and_labels(self, data):
        X = data.drop(['labels', 'twinnr', 'death_yrmon', 'birthdate1', 'age_death'], axis=1)
        y = data['labels']
        return X, y

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
        self.evaluate_model(grid_search, X_test, y_test)
        self.plot_feature_importance(X, grid_search.best_estimator_)
        self.plot_learning_curve(grid_search.best_estimator_, X, y)
        print("Total time:", datetime.now() - self.start_time)