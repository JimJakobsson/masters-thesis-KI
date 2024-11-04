import shap
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

    def create_pipeline(self):
        preprocessing_pipeline = self.preprocessor.create_pipeline()
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessing_pipeline),
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
        #Load the data from the server
        data = self.load_data()

        #Prepare the features and add labels
        X, y = self.prepare_features_and_labels(data)
        X, y = self.preprocessor.delete_null_features(X, y)

        self.preprocessor.set_categorical_features(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.2, 
                                                            random_state=42)
        

        #info about x and y before preprocessing
        print("Number of features:", X.shape[1])
        print("Number of samples:", X.shape[0])
        print("y shape:", y.shape)

        print("Number of samples:", X.shape[0])
        #Create pipeline to preprocess data
        self.create_pipeline()

        grid_search = GridSearchCV(estimator=self.pipeline, param_grid=self.param_grid, 
                                cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
        
        #Fit the grid search
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        print("Best parameters found")
        print(grid_search.best_params_)

        print("Best score found")
        print(grid_search.best_score_)

        y_pred = best_model.predict(X_test)
        print("Test accuracy")
        print(np.mean(y_pred == y_test))

        X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)

        #Create explainer
        explainer = shap.TreeExplainer(best_model.named_steps['classifier'])

        #Calculate SHAP values
        shap_values = explainer.shap_values(X_test_transformed)
        print(f"SHAP values shape: {np.shape(shap_values)}")
        
        
        #Get feature names after preprocessing
        feature_names = self.get_feature_names_after_preprocessing(best_model)

        import matplotlib.pyplot as plt

        # Create SHAP summary plot
        # shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)

        # Save the plot as a PDF
        # plt.savefig('shap_summary_plot.pdf')

        #Feature importance based on SHAP values
        # For binary classification, we'll take the mean absolute SHAP value across both classes
        mean_shap_values = np.abs(shap_values).mean(axis=2)  # Average across classes
        mean_importance = mean_shap_values.mean(axis=0)  # Average across samples

        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_importance
        })

        #Sort the features by importance. Include only the top 20 features
        feature_importance = feature_importance.sort_values(by='importance', ascending=False)
        feature_importance = feature_importance.head(20)
        
        # feature_importance.to_csv('feature_importance.csv', index=False)
        # print("Feature importance saved to feature_importance.csv")
        print("Feature importance:")
        print(feature_importance)


        
        

       