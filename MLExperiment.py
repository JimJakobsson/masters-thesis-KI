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
        
        print("Best parameters found")
        print(grid_search.best_params_)

        print("Best score found")
        print(grid_search.best_score_)
        
        return grid_search, X_test, y_test
      

    def get_numeric_feature_names(self, X):
        return X.select_dtypes(include=[np.number]).columns.tolist()

    def get_categorical_feature_names(self, X, preprocessor):
        categorical_features = self.preprocessor.detect_categorical_features(X)
        onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        return onehot_encoder.get_feature_names_out(categorical_features).tolist()

    def run(self):
        #Load the data from the server
        data = self.load_data()

        #Prepare the features and add labels
        X, y = self.prepare_features_and_labels(data)
        X, y = self.preprocessor.delete_null_features(X, y)
       
        #info about x and y before preprocessing
        print("Number of features:", X.shape[1])
        print("Number of samples:", X.shape[0])
        print("y shape:", y.shape)

        print("Number of samples:", X.shape[0])
        #Create pipeline to preprocess data
        self.create_pipeline(X)
        
        #Use the pipeline to preprocess data
        self.pipeline.fit(X, y)

        #info about x and y after preprocessing
        print("Number of features after preprocessing:", X.shape[1])
        print("Number of samples after preprocessing:", X.shape[0])
        print("y shape:", y.shape)

        print("Number of samples after preprocessing:", X.shape[0]) 
        #Train the model
        print("Training the model")
        grid_search, X_test, y_test = self.train_model(X, y)

        #Evaluate the model
        print("Evaluating the model")
        self.evaluator.evaluate_model(grid_search, X_test, y_test)

        # Evaluate the model using SHAP values
        print("Evaluating the model using SHAP values")

        pipeline = grid_search.best_estimator_ 
        self.evaluator.basic_shap_tree_evaluator(X_test, y_test, pipeline)


        # Plot feature importance
        # self.evaluator.plot_feature_importance(
        #     X,
        #     grid_search.best_estimator_,
        #     feature_names=X.columns
        # )
        
        # # Plot learning curve
        # self.evaluator.plot_learning_curve( 
        #     grid_search.best_estimator_,
        #     X,
        #     y
        # )
