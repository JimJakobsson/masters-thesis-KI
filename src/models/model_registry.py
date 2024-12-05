from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from models.model_config import ModelConfig

class ModelRegistry:
    """Registry for machine learning models"""

    @staticmethod
    def get_random_forest_config() -> ModelConfig:
        """Get the configuration for a random forest model"""
        return ModelConfig(
            name='Random Forest',
            model=RandomForestClassifier(),
             param_grid={
                'classifier__bootstrap': [False],  # Try both bootstrapping options
                'classifier__ccp_alpha': [0.006, 0.008],  # Add pruning options
                'classifier__class_weight': [
                    {0: 1, 1: 2.7},
                    # {0: 1, 1: 2.5},
                    
                ],  # More class weight ratios
                'classifier__criterion': ['entropy'],  # Try both split criteria
                'classifier__max_depth': [30],  # Search around successful depth
                'classifier__max_features': ['sqrt'],  # Both feature selection methods
                'classifier__min_samples_leaf': [1],  # Vary leaf size requirements
                'classifier__min_samples_split': [6],  # Vary split requirements
                'classifier__n_estimators': [93,94,95,96,97,98],  # Search around successful number
                'classifier__random_state': [42]  # Keep for reproducibility
                
            },
            description='A random forest model'
        )
    @staticmethod
    def get_logistic_regression_config() -> ModelConfig:
        """Get the configuration for a logistic regression model"""
        return ModelConfig(
            name='Logistic Regression',
            model=LogisticRegression(),
            param_grid={
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga'],
                'classifier__max_iter': [100, 500, 1000],
                'classifier__class_weight': [None, 'balanced'],
                'classifier__random_state': [42]
            },
            description='A logistic regression model'
        )
    @staticmethod
    def get_hist_gradient_boosting_config() -> ModelConfig:
        """Get the configuration for a histogram gradient boosting model"""
        return ModelConfig(
            name='Histogram Gradient Boosting',
            model=HistGradientBoostingClassifier(),
             param_grid={
                'classifier__learning_rate': [0.3, 0.4],  # Search around 0.3
                'classifier__max_depth': [5, 6],  # Search around 5
                'classifier__max_iter': [100, 125],  # Search around 100
                'classifier__min_samples_leaf': [20, 25],  # Search around 20
                'classifier__l2_regularization': [10.0, 15.0],  # Search around 10.0
                'classifier__max_bins': [225, 255],  # Search around 255
                'classifier__class_weight': [
                    # {0: 1, 1: 1.75},
                    {0: 1, 1: 2},
                    # {0: 1, 1: 2.25}
                ],  # Fine-tune class weights around 2
                'classifier__early_stopping': [True],  # Keep this as it worked well
                'classifier__validation_fraction': [0.1, 0.15],  # Try slightly larger validation set
                'classifier__n_iter_no_change': [10, 12],  # Search around 10
                'classifier__random_state': [42]  # Keep for reproducibility
            },
            description='A histogram gradient boosting model'
        )
    
    def get_bagging_config() -> ModelConfig:
        """Get the configuration for a bagging model"""
        return ModelConfig(
            name='Bagging',
            model=BaggingClassifier(),
            param_grid={
                'classifier__n_estimators': [10, 50, 100],
                'classifier__max_samples': [0.5, 0.7, 1.0],
                'classifier__max_features': [0.5, 0.7, 1.0],
                'classifier__random_state': [42]
            },
            description='A bagging model'
        )
    

    # @staticmethod
    # def get_gradient_boosting_config() -> ModelConfig:
    #     """Get the configuration for a gradient boosting model"""
    #     return ModelConfig(
    #         name='Gradient Boosting',
    #         model=GradientBoostingClassifier(),
    #         param_grid={
    #             'classifier__n_estimators': [100, 200, 300],
    #             'classifier__max_depth': [3, 5, 7]
    #         },
    #         description='A gradient boosting model'
    #     )
    