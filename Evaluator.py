from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
# from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.model_selection import learning_curve
import os

from sklearn.preprocessing import OrdinalEncoder
from scipy.sparse import issparse
import shap

class Evaluator:
    def __init__(self):
        self.results = {}
        self.output_dir = os.path.dirname(os.path.abspath(__file__))

    def evaluate_model(self, grid_search, X_test, y_test):
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        self.results['best_params'] = grid_search.best_params_
        self.results['best_cv_score'] = grid_search.best_score_
        self.results['test_accuracy'] = accuracy_score(y_test, y_pred)
        self.results['classification_report'] = classification_report(y_test, y_pred)

        print(f"Best parameters: {self.results['best_params']}")
        print(f"Best cross-validation score: {self.results['best_cv_score']:.4f}")
        print(f"\nTest set accuracy: {self.results['test_accuracy']:.4f}")
        print("\nClassification Report:")
        print(self.results['classification_report'])

    # def shap_feature_importance(self, X_transformed, best_model, display=False):
    #     # Convert sparse matrix to dense if necessary
    #     if issparse(X_transformed):
    #         X_transformed = X_transformed.toarray()

    #     # Convert to DataFrame if it's not already
    #     if not isinstance(X_transformed, pd.DataFrame):
    #         X_transformed = pd.DataFrame(X_transformed)

    #     # Get feature names
    #     if hasattr(best_model.named_steps['preprocessor'], 'get_feature_names_out'):
    #         feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    #     else:
    #         feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]
        
    #     X_transformed.columns = feature_names

    #     # Create a SHAP explainer
    #     if hasattr(best_model.named_steps['classifier'], 'predict_proba'):
    #         explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
    #         shap_values = explainer.shap_values(X_transformed)
    #     else:
    #         explainer = shap.Explainer(best_model.named_steps['classifier'].predict, X_transformed)
    #         shap_values = explainer(X_transformed)

    #     # Debug information
    #     print(f"Number of feature names: {len(feature_names)}")
    #     print(f"Shape of shap_values: {np.array(shap_values).shape}")

    #     # Handle different shapes of SHAP values
    #     if isinstance(shap_values, list):
    #         # Multi-class case: sum absolute values across all classes
    #         mean_shap = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
    #     elif len(np.array(shap_values).shape) == 3:
    #         # Multi-class case for some explainers
    #         mean_shap = np.abs(shap_values).mean(axis=0).mean(axis=0)
    #     else:
    #         # Binary classification or regression case
    #         mean_shap = np.abs(shap_values).mean(axis=0)

    #     print(f"Shape of mean_shap after processing: {mean_shap.shape}")

    #     # Ensure feature_names and mean_shap have the same length
    #     if len(feature_names) != len(mean_shap):
    #         print("Warning: feature_names and mean_shap have different lengths. Adjusting...")
    #         min_length = min(len(feature_names), len(mean_shap))
    #         feature_names = feature_names[:min_length]
    #         mean_shap = mean_shap[:min_length]

    #     # Sort features by importance
    #     feature_importance = pd.DataFrame({'feature': feature_names, 'importance': mean_shap})
    #     feature_importance = feature_importance.sort_values('importance', ascending=False)

    #     # Select top 20 features
    #     top_features = feature_importance.head(20)

    #     # Create summary plot for selected features
    #     plt.figure(figsize=(12, 8))
    #     shap.summary_plot(
    #         shap_values,
    #         X_transformed,
    #         plot_type="bar",
    #         max_display=20,
    #         show=False
    #     )
    #     plt.title('SHAP Feature Importance')
    #     plt.tight_layout()
        
    #     # Save the plot
    #     plt.savefig(os.path.join(self.output_dir, 'shap_feature_importance.png'))
    #     print(f"SHAP feature importance plot saved to {os.path.join(self.output_dir, 'shap_feature_importance.png')}")
        
    #     if display:
    #         plt.show()
    #     else:
    #         plt.close()

    #     return top_features
    def shap_feature_importance(self, X, model):
        """
        Calculate SHAP feature importance, plot it, and save the plot as a file.

        Args:
            X (numpy.ndarray or pandas.DataFrame): The input features.
            model (sklearn.base.BaseEstimator): The trained model.

        Returns:
            str: The path to the saved plot file.
        """
        # Create a SHAP explainer
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        # Plot SHAP feature importance
        shap.summary_plot(shap_values, X, show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(self.output_dir, 'shap_feature_importance.png')
        plt.savefig(plot_path)
        print(f"SHAP feature importance plot saved to {plot_path}")

        return plot_path
    def plot_learning_curve(self, best_model, X, y, display=False):
        train_sizes, train_scores, test_scores = learning_curve(
            best_model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        plt.xlabel('Number of training examples')
        plt.ylabel('Accuracy score')
        plt.title(f'Learning Curve for {type(best_model.named_steps["classifier"]).__name__}')
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'learning_curve.png'))
        print(f"Learning curve plot saved to {os.path.join(self.output_dir, 'learning_curve.png')}")
        
        if display:
            plt.show()
        else:
            plt.close()
