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

from sklearn.preprocessing import OrdinalEncoder
from scipy.sparse import issparse
import shap

class Evaluator:
    def __init__(self):
        self.results = {}
        self.output_dir = os.path.dirname(os.path.abspath(__file__))

    def evaluate_model(self, grid_search, X_test, y_test):
        # print("\n Debug information vefore prediction:")
        # print("X_test columns:", X_test.columns.tolist())
        # print("Expected features values:", best_model.best_estimator_.named_steps['preprocessor'].get_feature_names_out())
        y_pred = grid_search.predict(X_test)
        # best_model = grid_search.best_estimator_
        # y_pred = best_model.predict(X_test)
        
        self.results['best_params'] = grid_search.best_params_
        self.results['best_cv_score'] = grid_search.best_score_
        self.results['test_accuracy'] = accuracy_score(y_test, y_pred)
        self.results['classification_report'] = classification_report(y_test, y_pred)

        print(f"Best parameters: {self.results['best_params']}")
        print(f"Best cross-validation score: {self.results['best_cv_score']:.4f}")
        print(f"\nTest set accuracy: {self.results['test_accuracy']:.4f}")
        print("\nClassification Report:")
        print(self.results['classification_report'])

    # def basic_shap_tree_evaluator(self, X, y, pipeline):
    #     # Get the RandomForestClassifier from the pipeline
    #     model = pipeline.named_steps['classifier']
        
    #     # Create a TreeExplainer specifically for the Random Forest
    #     explainer = shap.TreeExplainer(model)
        
    #     # Transform the data using the preprocessor first
    #     X_transformed = pipeline.named_steps['preprocessor'].transform(X)
        
    #     # Calculate SHAP values
    #     shap_values = explainer.shap_values(X_transformed)
    #     print(f"SHAP values shape: {np.shape(shap_values)}")
       
    #     #waterfall plot for the first observation
    #     shap.initjs()
    #     shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X.iloc[0,:], matplotlib=True)
    #     #shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X.iloc[0,:])
    #     #shap.force_plot(explainer.expected_value[1], shap_values[1], X)

    #     #save the plot as pdf
    #     plt.savefig(os.path.join(self.output_dir, 'shap_plot.pdf'), format='pdf', bbox_inches='tight')
    
    def basic_shap_tree_evaluator(self, X_test, y_test, pipeline, sample_index=0, top_n=10):
        model = pipeline.named_steps['classifier']
        explainer = shap.TreeExplainer(model)
        
        # Transform the data using the preprocessor first
        X_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
        
        # Calculate SHAP values for both classes
        shap_values = explainer.shap_values(X_transformed)
        
        # Get original feature names and values
        feature_names = X_test.columns.tolist()
        original_values = X_test.iloc[sample_index]
        
        # Calculate net SHAP values (difference between classes)
        net_shap = shap_values[1][sample_index] - shap_values[0][sample_index]
        
        # Get absolute net SHAP values for feature importance ranking
        abs_shap = np.abs(net_shap)
        
        # Get indices of top N most important features
        top_feature_idx = np.argsort(abs_shap)[-top_n:][::-1]
        
        # Filter out features with zero contribution
        significant_idx = [idx for idx in top_feature_idx if abs_shap[idx] > 0]
        
        if len(significant_idx) == 0:
            print("No significant feature contributions found!")
            return
            
        plt.figure(figsize=(12, max(8, len(significant_idx) * 0.5)))
        
        # Get prediction details
        actual_class = y_test.iloc[sample_index]
        pred_proba = pipeline.predict_proba(X_test.iloc[[sample_index]])[0]
        pred_class = pipeline.predict(X_test.iloc[[sample_index]])[0]
        
        # Create explanation with significant features only
        explanation = shap.Explanation(
            values=net_shap[significant_idx],
            base_values=0.5,  # Start from 0.5 for binary classification
            data=original_values[significant_idx],
            feature_names=[f"{feature_names[i]} ({original_values[feature_names[i]]:.1f})" 
                        for i in significant_idx]
        )
        
        shap.waterfall_plot(explanation, show=False)
        
        # Add informative title
        plt.title(f'Most Influential Features for Sample {sample_index}\n'
                f'Actual Class: {actual_class}, Predicted Class: {pred_class}\n'
                f'Base Probability: 0.500 → Final Probability: {pred_proba[1]:.3f}')
        
        # Adjust y-axis to show full probability range
        plt.ylim(min(0.2, pred_proba[1] - 0.05), 0.55)
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, f'shap_waterfall_plot_sample_{sample_index}.pdf'), 
                    format='pdf', 
                    bbox_inches='tight',
                    dpi=300)
        plt.close()

        # Print detailed information
        print(f"\nSample {sample_index} details:")
        print(f"Actual class: {actual_class}")
        print(f"Predicted class: {pred_class}")
        print(f"Base probability: 0.500")
        print(f"Final probability: {pred_proba[1]:.3f}")
        print("\nTop feature contributions:")
        for idx in significant_idx:
            name = feature_names[idx]
            value = original_values[name]
            net_contribution = net_shap[idx]
            print(f"{name}: {value:.1f} (net contribution: {net_contribution:.4f})")
            print(f"  Class 0 contribution: {shap_values[0][sample_index][idx]:.4f}")
            print(f"  Class 1 contribution: {shap_values[1][sample_index][idx]:.4f}")

    def plot_feature_importance(self, X, pipeline, feature_names=None):
        """
        Plot and save feature importance for top 10 features using SHAP values.
        
        Args:
            X: Input features (DataFrame or array)
            pipeline: Trained scikit-learn pipeline
            feature_names: List of feature names (optional)
        """
        # Transform the input data using the pipeline's preprocessor
        X_transformed = pipeline.named_steps['preprocessor'].transform(X)
        
        # Get feature names from preprocessor if not provided
        if feature_names is None:
            try:
                feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            except:
                feature_names = [f'Feature {i}' for i in range(X_transformed.shape[1])]
        
        # Create figure with high-quality settings
        plt.figure(figsize=(12, 8), dpi=300)
        
        # Get the classifier from the pipeline
        model = pipeline.named_steps['classifier']
        
        try:
            # Try using TreeExplainer first
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)
            
            # Handle different shapes of SHAP values
            if isinstance(shap_values, list):
                mean_shap = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)
                
        except Exception as e:
            print(f"TreeExplainer failed, falling back to KernelExplainer: {str(e)}")
            # Fall back to KernelExplainer if TreeExplainer fails
            explainer = shap.KernelExplainer(model.predict_proba, X_transformed)
            shap_values = explainer.shap_values(X_transformed)
            mean_shap = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_shap
        }).sort_values('Importance', ascending=False)
        
        # Plot top 10 features
        top_10 = importance_df.head(10)
        bars = plt.barh(np.arange(len(top_10)), top_10['Importance'], 
                       color='royalblue', alpha=0.8)
        
        # Customize plot
        plt.yticks(np.arange(len(top_10)), top_10['Feature'], fontsize=10)
        plt.xlabel('Mean |SHAP value|', fontsize=12)
        plt.title('Top 10 Most Important Features', fontsize=14, pad=20)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}',
                    ha='left', va='center', fontsize=10)
        
        # Add grid
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save plot
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'feature_importance.pdf')
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved to: {output_path}")
        return importance_df

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