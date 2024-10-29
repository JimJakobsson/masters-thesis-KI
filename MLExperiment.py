import DatabaseReader, Evaluator, PreProcessing, ServerConnectionIPT1
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

    def load_data(self):
        combined_tables = self.server_connection.read_table()
        return self.preprocessor.set_labels(combined_tables) 
    
    def prepare_features_and_labels(self, data):
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
        
        # Transform X_train and X_test using the fitted pipeline
        X_train_transformed = grid_search.best_estimator_.named_steps['preprocessor'].transform(X_train)
        X_test_transformed = grid_search.best_estimator_.named_steps['preprocessor'].transform(X_test)
        
        numeric_features = self.get_numeric_feature_names(X)
        categorical_features = self.get_categorical_feature_names(X, grid_search.best_estimator_.named_steps['preprocessor'])
        
        print(f"Number of numeric features: {len(numeric_features)}")
        print(f"Number of categorical features: {len(categorical_features)}")
        print(f"Total number of features: {len(numeric_features) + len(categorical_features)}")
        print(f"Shape of X_train_transformed: {X_train_transformed.shape}")
        
        feature_names = numeric_features + categorical_features.best_estimator_.named_steps['preprocessor']
        if X_train_transformed.shape[1] != len(feature_names):
            print("Mismatch between number of columns and feature names")
            print(f"Columns in transformed data: {X_train_transformed.shape[1]}")
            print(f"Number of feature names: {len(feature_names)}")
            # Adjust feature_names to match the number of columns
            feature_names = feature_names[:X_train_transformed.shape[1]]
        # Convert to DataFrame
        X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
        X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
        
        return grid_search, X_train_transformed, X_test_transformed, y_train, y_test

    def get_numeric_feature_names(self, X):
        return X.select_dtypes(include=[np.number]).columns.tolist()

    def get_categorical_feature_names(self, X, preprocessor):
        categorical_features = self.preprocessor.detect_categorical_features(X)
        onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        return onehot_encoder.get_feature_names_out(categorical_features).tolist()

    def run(self):
        """
        Runs the MLExperiment.

        Loads the data, prepares the features and labels, deletes null features,
        creates the pipeline, trains the model, evaluates the model, calculates
        SHAP feature importance, plots the learning curve, and prints the total time.

        Returns:
            None
        """
        data = self.load_data()
        X, y = self.prepare_features_and_labels(data)
        X, y = self.preprocessor.delete_null_features(X, y)
        self.create_pipeline(X)
        grid_search, X_train_transformed, X_test_transformed, y_train, y_test = self.train_model(X, y)
        print("Total time:", datetime.now() - self.start_time)
        self.evaluator.evaluate_model(grid_search, X_test_transformed, y_test)
        
        # Extract the model from pipeline
        best_model = grid_search.best_estimator_.named_steps['classifier']
        
        self.evaluator.shap_feature_importance(X_train_transformed, best_model)
        # self.evaluator.plot_learning_curve(grid_search.best_estimator_, X, y)