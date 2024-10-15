from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.model_selection import learning_curve


class Evaluator:
    def __init__(self):
        self.results = {}

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

    def plot_feature_importance(self, X, best_model):
        if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
            feature_importance = best_model.named_steps['classifier'].feature_importances_
            feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
            plt.title('Top 10 Most Important Features')
            plt.tight_layout()
            plt.show()
        else:
            print("Feature importance plot is not available for this model type.")

    def plot_learning_curve(self, best_model, X, y):
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
        plt.show()