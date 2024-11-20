import shap
from sklearn.pipeline import FunctionTransformer

import DatabaseReader, Evaluator, PreProcessing, ServerConnectionIPT1
from AgeExploration import AgeExploration
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
# from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
# from imblearn.over_sampling import SMOTE
from collections import Counter

class MLExperiment:
    def __init__(self, model: BaseEstimator, param_grid: dict, preprocessor: PreProcessing, evaluator: Evaluator, connection_class: DatabaseReader):
        self.start_time = datetime.now()
        self.server_connection = connection_class
        self.model = model
        self.param_grid = param_grid
        self.preprocessor = preprocessor
        self.evaluator = evaluator
        self.pipeline = None
        self.removed_features = []  # Track removed features


    def load_data(self):
        combined_tables = self.server_connection.read_table()
        return self.preprocessor.set_labels(combined_tables) 
    
    def prepare_features_and_labels(self, data):
        drop_columns = ['labels', 'twinnr', 'death_yrmon', 'age_death']

        # Add removed features to drop columns
        if hasattr(self, 'removed_features') and self.removed_features:
            drop_columns.extend(self.removed_features)
        
        X = data.drop(columns=[col for col in drop_columns if col in data.columns], axis=1)
        y = data['labels']
        return X, y
    # def smote_transform(self, X, y=None):
    #     if y is None:
    #         raise ValueError("y cannot be None")
    #     smote = SMOTE(random_state=42, sampling_strategy='minority')
    #     X_resampled, y_resampled = smote.fit_resample(X, y)
    #     print(f"Original class distribution: {Counter(y)}")
    #     print(f"Resampled class distribution: {Counter(y_resampled)}")
    #     return X_resampled, y_resampled
    def create_pipeline(self):
        # Update the preprocessor with removed features
        if hasattr(self, 'removed_features'):
            self.preprocessor.excluded_features = self.removed_features
        preprocessing_pipeline = self.preprocessor.create_pipeline()
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessing_pipeline),
            # ('smote', SMOTE(random_state=42, sampling_strategy='minority')),
            ('classifier', self.model)
        ])

    def get_feature_names_after_preprocessing(self, model):
        """Get feature names after preprocessing has been applied"""
        feature_names = []
        processor = model.named_steps['preprocessor']

        for name, _, columns in processor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                encoder = processor.named_transformers_['cat'].named_steps['onehot']
                cat_features = encoder.get_feature_names_out(columns)
                feature_names.extend(cat_features)
            else:
                raise ValueError(f'Invalid transformer name: {name}')
        return feature_names      
    
    def run(self):
        # Load the data from the server
        data_original = self.load_data()
        data = data_original.copy()
        # ages = AgeExploration()
        # ages.box_plot_age_combined(data)
        # ages.age_distribution_histogram(data)
        # data = self.preprocessor.set_ages(data)
        # Prepare the features and add labels
        X, y = self.prepare_features_and_labels(data)
        X, y = self.preprocessor.delete_null_features(X, y)
        
        #Boxplot of age distribution for the two classes
        # ages.box_plot_age_classes(X, y)
        #Detects whether the features are categorical or numeric and sets them as attributes in the preprocessor
        self.preprocessor.set_categorical_features(X)
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        # Info about X and y before preprocessing
        print("Number of features:", X.shape[1])
        print("Number of samples:", X.shape[0])
        print("y shape:", y.shape)
        #print values in X_train that is string
        # print(X_train.select_dtypes(include=['object']).head())

        # Create pipeline to preprocess data
        self.create_pipeline()

        grid_search = GridSearchCV(estimator=self.pipeline, param_grid=self.param_grid, cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)

        # Basic model evaluation
        self.evaluator.evaluate_model(grid_search, X_test, y_test)

        best_model = grid_search.best_estimator_
        print("Best parameters found:", grid_search.best_params_)
        print("Best score found:", grid_search.best_score_)

        # Feature importance analysis
        preprocessor = self.preprocessor
        aggregated_shap_values, feature_names = self.evaluator.calculate_feature_importance(best_model, X_test, preprocessor)

        # Plot feature importance
        self.evaluator.plot_feature_importance()
        
        self.evaluator.plot_shap_summary(aggregated_shap_values, X_test)
        # Plot SHAP summary
        
        # Create waterfall plot
        # Explain a strong class 1 prediction
        self.evaluator.plot_waterfall(best_model, X_test, 1, 20)

        #Define age groups
        age_groups = { 
            '50-59': {50, 59},
            '60-69': {60, 69},
            '70-79': {70, 79},
        }
        # set age in data as the difference between the first four digits of birthdate1 and 1985
        data_original['age'] = 1985 - data_original['birthdate1'].astype(str).str[:4].astype(int)
      
        for group_name, (age_min, age_max) in age_groups.items():
            # Filter data for the age group
            age_group_data = data_original[(data_original['age'] >= age_min) & (data_original['age'] <= age_max)]
            X_group, y_group = self.prepare_features_and_labels(age_group_data)
            # Use the same feature removal as the main dataset
            X_group = X_group.drop(columns=[col for col in self.removed_features if col in X_group.columns])
            

            # Split the data
            X_train_group, X_test_group, y_train_group, y_test_group = train_test_split(X_group, y_group, test_size=0.2, random_state=42)
            y_train_group = y_train_group.astype(int)
            y_test_group = y_test_group.astype(int)

            # Fit the model for the age group
            grid_search.fit(X_train_group, y_train_group)
            
            #Basic model evaluation for the age group
            self.evaluator.evaluate_model(grid_search, X_test_group, y_test_group)

            # Calculate SHAP values and feature importance for the age group
            aggregated_shap_values_group, feature_names_group = self.evaluator.calculate_feature_importance(best_model, X_test_group, preprocessor)

            # Plot feature importance for the age group
            self.evaluator.plot_feature_importance(feature_importance=self.evaluator.feature_importance, num_features=20, output_path=f'feature_importance_{group_name}.pdf')

            # Plot SHAP summary for the age group
            self.evaluator.plot_shap_summary(aggregated_shap_values_group, X_test_group, output_path=f'shap_summary_plot_{group_name}.pdf')

            # Plot waterfall plot for the age group
            self.evaluator.plot_waterfall(best_model, X_test_group, 1, 20, output_path=f'waterfall_plot_{group_name}.pdf')

                

     