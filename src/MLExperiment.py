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

    def load_data(self):
        combined_tables = self.server_connection.read_table()
        return self.preprocessor.set_labels(combined_tables) 
    
    def prepare_features_and_labels(self, data):
        X = data.drop(['labels', 'twinnr', 'death_yrmon', 'age_death'], axis=1)
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

    def aggregate_shap_values(self, shap_values, feature_names):
        """
        Aggregate SHAP values for categorical features that were one-hot encoded.

        Args:
            shap_values (np.ndarray): SHAP values
            feature_names (list): Feature names
        Returns:
            Tuple containing:
            - Dictionary mapping original feature names to aggregated SHAP values
            - List of original feature names before one-hot encoding
        """
        # Get original feature names before one-hot encoding
        original_feature_names = self.preprocessor.numeric_features + self.preprocessor.categorical_features
        feature_names = np.array(feature_names)

        #initialize dictionary to store aggregated SHAP values
        aggregated_shap = {}
        processed_features = []

        # For binary classification, take absolute values and average across classes
        if len(shap_values.shape) == 3:  # Shape: (n_samples, n_features, n_classes)
            shap_values = np.abs(shap_values).mean(axis=2)

        #Iterate throguh each feature
        current_idx = 0
        for feature in original_feature_names:
            if feature in self.preprocessor.numeric_features:
                # If feature is numeric, add the SHAP values directly
                aggregated_shap[feature] = shap_values[:, current_idx]
                current_idx += 1
            else:
                #For categorical features, find all realted one-hot encoded columns
                feature_mask = np.array([col.startswith(f"{feature}_") for col in feature_names])
                if np.any(feature_mask):
                    #Sum SHAP values across all one-hot encoded columns
                    aggregated_values = np.abs(shap_values[:, feature_mask]).sum(axis=1)
                    aggregated_shap[feature] = aggregated_values
                    processed_features.append(feature)
                    current_idx += np.sum(feature_mask)
        # Calculate mean importance for each feature
        feature_importance = pd.DataFrame({
            'feature': list(aggregated_shap.keys()),
            'importance': [np.mean(np.abs(values)) for values in aggregated_shap.values()]
        })
        print(f"\nProcessed {len(processed_features)} categorical features out of {len(original_feature_names)} original features")
        print(f"Number of samples in SHAP values: {shap_values.shape[0]}")
        return aggregated_shap, original_feature_names, processed_features, feature_importance
    
    def run(self):
        # Load the data from the server
        data = self.load_data()
        ages = AgeExploration()
        ages.box_plot_age_combined(data)
        ages.age_distribution_histogram(data)
        # data = self.preprocessor.set_ages(data)
        # Prepare the features and add labels
        X, y = self.prepare_features_and_labels(data)
        X, y = self.preprocessor.delete_null_features(X, y)
        
        #Boxplot of age distribution for the two classes
        ages.box_plot_age_classes(X, y)
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
        print(X_train.select_dtypes(include=['object']).head())

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

        # Get ordered features for SHAP plots
        ordered_features = self.evaluator.feature_importance.sort_values(
            by='importance_abs_mean', ascending=False)['feature'].tolist()
        
        self.evaluator.plot_shap_summary(aggregated_shap_values, X_test, ordered_features)
        
        # Create waterfall plot
        X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
        # Explain a strong class 1 prediction
        self.evaluator.plot_waterfall(best_model, X_test, 1, 20)
        # self.evaluator.plot_waterfall(X_test_transformed, feature_names, X_test)


        #Old and working code below
        ############################################################
        # y_pred = best_model.predict(X_test)
        # print("Test accuracy:", np.mean(y_pred == y_test))

        # X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)

        # # Create explainer
        # explainer = shap.TreeExplainer(best_model.named_steps['classifier'])

        # # Calculate SHAP values
        # shap_values = explainer.shap_values(X_test_transformed)
        
        # # Get feature names after preprocessing
        # feature_names = self.get_feature_names_after_preprocessing(best_model)

        # print(f"SHAP values shape: {np.shape(shap_values)}")
        # print("Number of feature names:", len(feature_names))
        # print("First few feature names:", feature_names[:5])
        
        # # Aggregate SHAP values for categorical features
        # aggregated_shap_values, original_feature_names, processed_features, feature_importance = self.aggregate_shap_values(shap_values, feature_names)

        # # Sort the features by importance. Include only the top 20 features
        # feature_importance = feature_importance.sort_values(by='importance', ascending=False).head(20)
        
        # # Print the feature importance
        # print("Feature importance:")
        # print(feature_importance)

        

        # # Plot the feature importance
        # plt.figure(figsize=(10, 8))
        # plt.barh(feature_importance['feature'], feature_importance['importance'])
        # plt.xlabel('Mean |SHAP value|')
        # plt.ylabel('Feature')
        # plt.title('Feature Importance (Aggregated Categorical Features)')
        # plt.tight_layout()
        # plt.savefig('feature_importance_aggregated.pdf')
        # plt.close()

        # # Create SHAP summary plot with aggregated values
        # plt.figure(figsize=(10, 12))
      
        # # Get all processed features in order of importance
        # ordered_features = feature_importance['feature'].tolist()
        
        # # Create SHAP matrix and corresponding feature matrix
        # aggregated_shap_matrix = np.column_stack([aggregated_shap_values[feature] for feature in ordered_features])
        # X_test_features = X_test[ordered_features].copy()

        # print("\nShape of aggregated SHAP matrix:", aggregated_shap_matrix.shape)
        # print("Shape of X_test_features:", X_test_features.shape)

        # shap.summary_plot(aggregated_shap_matrix, X_test_features, feature_names=ordered_features, show=False)
        # plt.tight_layout()
        # plt.savefig('shap_summary_plot_aggregated.pdf')
        # plt.close()

        # # Create SHAP bar plot for absolute mean SHAP values
        # mean_shap_values = np.abs(aggregated_shap_matrix).mean(axis=0)
        # shap.plots.bar(shap.Explanation(values=mean_shap_values, feature_names=ordered_features))
        # plt.tight_layout()
        # plt.savefig('shap_bar_plot.pdf')
        # plt.close()

        # print("Shape of shap_values:", shap_values.shape)
        # print("Type of explainer.expected_value:", type(explainer.expected_value))
        # print("Value of explainer.expected_value:", explainer.expected_value)
       
        # # Get the first observation's SHAP values for class 1
        # observation_idx = 0
        # class_idx = 1
        # values = shap_values[observation_idx, :, class_idx]

        # # Get absolute SHAP values and sort by importance
        # feature_importance = np.abs(values)
        # top_n = 10  # Show top 10 most important features
        # top_indices = np.argsort(feature_importance)[-top_n:]

        # explanation = shap.Explanation(
        #     values=values[top_indices],
        #     base_values=float(explainer.expected_value[class_idx]),
        #     data=X_test_transformed[observation_idx][top_indices],
        #     feature_names=[feature_names[i] for i in top_indices]
        # )

        # # Create figure with larger size and better spacing
        # plt.figure(figsize=(12, 8))  # Increased figure size
        # shap.waterfall_plot(explanation, show=False)  # Don't show yet to allow modifications

        # # Get current axis
        # ax = plt.gca()

        # # Increase font sizes
        # plt.rcParams.update({'font.size': 12})  # Base font size
        # ax.set_xlabel('SHAP value', fontsize=14)
        # ax.set_title('Feature Contributions to Prediction', fontsize=16, pad=20)

        # # Adjust y-axis labels to be fully visible
        # ax.tick_params(axis='y', labelsize=12)  # Increase y-tick label size
        # plt.gcf().set_tight_layout(True)  # Ensure nothing is cut off

        # # Add more padding on the left for feature names
        # plt.subplots_adjust(left=0.3)  # Adjust this value if needed

        # # Increase spacing between elements
        # plt.margins(y=0.1)  # Add vertical margins

        # # Save with high resolution
        # plt.savefig('shap_waterfall_plot.pdf', bbox_inches='tight', dpi=300)
