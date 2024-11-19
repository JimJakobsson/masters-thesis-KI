from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
# from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.model_selection import learning_curve
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
# from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.kernel_ridge import KernelRidge


from sklearn.preprocessing import OrdinalEncoder
from scipy.sparse import issparse
import shap

class Evaluator:
    def __init__(self):
        self.results = {}
        self.output_dir = os.path.dirname(os.path.abspath(__file__))
        self.shap_values = None
        self.feature_importance = None
        self.explainer = None
        self.aggregated_shap = None

    def evaluate_model(self, grid_search, X_test, y_test):
        # print("\n Debug information vefore prediction:")
        # print("X_test columns:", X_test.columns.tolist())
        # print("Expected features values:", best_model.best_estimator_.named_steps['preprocessor'].get_feature_names_out())
        y_pred = (grid_search.predict_proba(X_test)[:,1] > 0.4).astype(int)
        # best_model = grid_search.best_estimator_
        # y_pred = best_model.predict(X_test)
        
        y_test=y_test.astype(int)
        self.results['best_params'] = grid_search.best_params_
        self.results['best_cv_score'] = grid_search.best_score_
        self.results['test_accuracy'] = accuracy_score(y_test, y_pred)
        self.results['classification_report'] = classification_report(y_test, y_pred)

        print(f"Best parameters: {self.results['best_params']}")
        print(f"Best cross-validation score: {self.results['best_cv_score']:.4f}")
        print(f"\nTest set accuracy: {self.results['test_accuracy']:.4f}")
        print("\nClassification Report:")
        print(self.results['classification_report'])

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

    def aggregate_shap_values(self, shap_values, feature_names, preprocessor):
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
        original_feature_names = preprocessor.numeric_features + preprocessor.categorical_features
        feature_names = np.array(feature_names)

        #initialize dictionary to store aggregated SHAP values
        aggregated_shap = {}
        processed_features = []

        #For binary classification, take absolute values and average across classes
        if len(shap_values.shape) == 3:  # Shape: (n_samples, n_features, n_classes)
            #Use .mean or .sum?
            # shap_values = np.abs(shap_values).mean(axis=2)
            shap_values = np.abs(shap_values).sum(axis=2)

        #Iterate throguh each feature
        current_idx = 0
        for feature in original_feature_names:
            if feature in preprocessor.numeric_features:
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
            'importance_abs_mean': [np.mean(np.abs(values)) for values in aggregated_shap.values()],
            'importance_mean': [np.mean(values) for values in aggregated_shap.values()]
        })
        print(f"\nProcessed {len(processed_features)} categorical features out of {len(original_feature_names)} original features")
        print(f"Number of samples in SHAP values: {shap_values.shape[0]}")

        return aggregated_shap, feature_importance
    def create_explainer(self,model, data=None):
        classifier = model.named_steps['classifier']

        #Tree based models
         # Tree-based models
        tree_based_models = (
            RandomForestClassifier, GradientBoostingClassifier, 
            DecisionTreeClassifier, XGBClassifier, XGBRegressor,
            # LGBMClassifier, LGBMRegressor, 
            CatBoostClassifier, CatBoostRegressor,
            AdaBoostClassifier
        )
        
        # Linear models
        linear_models = (
            LogisticRegression, LinearRegression
        )
        
        # Deep learning models
        deep_learning_models = (
            MLPClassifier,
        )

        try:
            if isinstance(classifier, tree_based_models):
                return shap.TreeExplainer(classifier)
            # Check for deep learning models
            elif isinstance(model, deep_learning_models):
                if data is None:
                    raise ValueError("Background data is required for DeepExplainer")
                return shap.DeepExplainer(model, data)
            
            # Check for linear models
            elif isinstance(model, linear_models):
                return shap.LinearExplainer(model, data)
           
            
        except Exception as e:
            raise Exception(f"Error creating SHAP explainer: {str(e)}")

    

    def calculate_feature_importance(self, best_model, X_test, preprocessor):
        """
        Calculate feature importance using SHAP values.
        """
        # Transform test data
        X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)

        # Create explainer
        # Check what instance the best model is of and choose corresponding explainer
        # Tree explainer for tree based models
        # Neural explainer for ANN 
        # background_data = X_test_transformed[:100] if len(X_test_transformed) > 100 else X_test_transformed
        background_data = X_test_transformed
        self.explainer = self.create_explainer(best_model, background_data)
        print("Explainer created: ", self.explainer)
        # self.explainer = shap.TreeExplainer(best_model.named_steps['classifier'])

        # Calculate SHAP values on test data
        # self.shap_values = self.
        shap_values = self.explainer.shap_values(X_test_transformed)
        # Only shap values for class 1
        self.shap_values = shap_values
        # Get feature names after preprocessing
        feature_names = self.get_feature_names_after_preprocessing(best_model)

        print(f"SHAP values shape: {np.shape(self.shap_values)}")
        print("Number of feature names:", len(feature_names))
        print("First few feature names:", feature_names[:5])

        # Aggregate SHAP values for one-hot encoded features.
        # Return feature importance and processed feature names
        self.aggregated_shap, self.feature_importance = self.aggregate_shap_values(
            self.shap_values, feature_names, preprocessor)

        return self.aggregated_shap, feature_names

    def plot_feature_importance(self, feature_importance=None, num_features=20):
        """Plot featrue importance"""
        if feature_importance is None:
            feature_importance = self.feature_importance
        
        # Sort features by importance. Get top 20 features
        feature_importance = feature_importance.sort_values('importance_abs_mean', ascending=False).head(num_features)

        #get count of features with importance = 0	
        zero_importance = feature_importance[feature_importance['importance_abs_mean'] == 0].shape[0]
        print(f"\nNumber of features with importance = 0: {zero_importance}")

        # Print the feature importance
        print("\nFeature Importance:")
        print(feature_importance)

        #Plot the feature importances
        plt.figure(figsize=(10, 8), dpi=300) # dpi=300 for high-quality plot
        sns.barplot(x='importance_abs_mean', y='feature', data=feature_importance, palette='rocket', hue = 'feature')
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Feature Importance', fontsize=14, pad=20)
        # plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'feature_importance.pdf')
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()

        print(f"Feature importance plot saved to: {output_path}")

        #Alternative plot
        plt.figure(figsize=(10, 8))
        feature_importance = feature_importance.sort_values('importance_abs_mean', ascending=True)
        plt.barh(feature_importance['feature'], feature_importance['importance_abs_mean'], color='royalblue')
        plt.xlabel('Mean |SHAP value|')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance_aggregated.pdf')
        plt.close()

    def plot_shap_summary(self, aggregated_shap, X_test, ordered_features):
        """Create SHAP summary plot"""
        
        # Create SHAP matrix and corresponding feature matrix
        aggregated_shap_matrix = np.column_stack([aggregated_shap[feature] for feature in ordered_features])
        X_test_features = X_test[ordered_features] #.copy() perhaps

        print("\nShape of aggregated SHAP matrix:", aggregated_shap_matrix.shape)
        print("Shape of X_test_features:", X_test_features.shape)

        #Summary plot
        plt.figure(figsize=(10, 12))
        shap.summary_plot(aggregated_shap_matrix, X_test_features, feature_names=ordered_features, show=False)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/shap_summary_plot_aggregated.pdf')
        plt.close()

        #SHAP bar plot for absolute mean SHAP values  
        plt.figure(figsize=(10, 12))
        shap.summary_plot(aggregated_shap_matrix, X_test_features, feature_names=ordered_features, show=False)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/shap_summary_plot_aggregated.pdf')
        plt.close()
    
    def plot_waterfall(self, best_model, X_test, class_to_explain, top_n=10):
        """
        Create SHAP waterfall plot for an observation that strongly predicts the class of interest
        
        Args:
            X_test_transformed: Transformed test data
            feature_names: Names of features after preprocessing
            X_test: Original test data
            class_to_explain: Which class prediction to explain (0 or 1)
            top_n: Number of top features to show
        """
        # Get model predictions
        probas = best_model.predict_proba(X_test)
        
        # Find an observation that strongly predicts the class we want to explain
        if class_to_explain == 1:
            # Find observation with highest probability of class 1
            observation_idx = np.argmax(probas[:, 1])
        else:
            # Find observation with highest probability of class 0
            observation_idx = np.argmax(probas[:, 0])
        
        pred_prob = probas[observation_idx, class_to_explain]
        
        print(f"\nAnalyzing observation {observation_idx}")
        print(f"Prediction probabilities: Class 0: {probas[observation_idx, 0]:.3f}, Class 1: {probas[observation_idx, 1]:.3f}")
        print(f"Model predicts class {class_to_explain} with {pred_prob:.3f} probability")
        
        # Rest of your waterfall plot code...
        # Get all features and their SHAP values for this observation
        all_features = self.feature_importance['feature'].tolist()
        all_values = np.array([self.aggregated_shap[feature][observation_idx] 
                            for feature in all_features])
        
        # Sort features by absolute SHAP value for this specific observation
        abs_values = np.abs(all_values)
        # top_indices = np.argsort(-abs_values)[:top_n]
        top_indices = np.argsort(-abs_values)
        
        top_features = [all_features[i] for i in top_indices]
        top_values = all_values[top_indices]
        top_data = X_test.iloc[observation_idx][top_features].values
        
        explanation = shap.Explanation(
            values=top_values,
            base_values=float(self.explainer.expected_value[class_to_explain]),
            data=top_data,
            feature_names=top_features
        )

        plt.figure(figsize=(20,15))
        shap.waterfall_plot(explanation, show=False)
        
        # Customize plot
        ax = plt.gca()
        plt.rcParams.update({'font.size': 12})
        ax.set_xlabel('SHAP value', fontsize=14)
        ax.set_title(f'How the model predicts Class {class_to_explain} (probability={pred_prob:.3f})', 
                    fontsize=16, pad=20)
        ax.tick_params(axis='y', labelsize=12)
        plt.gcf().set_tight_layout(True)
        plt.subplots_adjust(left=0.3)
        plt.margins(y=0.1)
        
        plt.savefig(f'{self.output_dir}/shap_waterfall_plot_class_{class_to_explain}.pdf', 
                    bbox_inches='tight', dpi=300)
        plt.close()


    def plot_learning_curve(self, model, X, y):
        """
        Plot and save learning curve showing training and validation scores.
        
        Args:
            model: Trained model
            X: Input features
            y: Target values
        """
        # Create figure with high-quality settings
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Calculate learning curve
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curve
        plt.plot(train_sizes, train_mean, 'o-', color='royalblue',
                label='Training score', linewidth=2, markersize=8)
        plt.plot(train_sizes, val_mean, 'o-', color='orangered',
                label='Cross-validation score', linewidth=2, markersize=8)
        
        # Add confidence intervals
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.15, color='royalblue')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.15, color='orangered')
        
        # Customize plot
        plt.xlabel('Training Examples', fontsize=12)
        plt.ylabel('Accuracy Score', fontsize=12)
        plt.title('Learning Curve', fontsize=14, pad=20)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis ranges
        plt.ylim(0, 1.1)
        
        # Save plot
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'learning_curve.pdf')
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Learning curve plot saved to: {output_path}")