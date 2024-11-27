from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
            
             'classifier__bootstrap': [False], 
             'classifier__ccp_alpha': [0.0], 
            #  'classifier__class_weight': [0: 1, 1:], 
             'classifier__criterion': ['entropy'], 
             'classifier__max_depth': [20], 
             'classifier__max_features': ['sqrt'], 
             'classifier__min_samples_leaf': [2], 
             'classifier__min_samples_split': [5], 
             'classifier__n_estimators': [200]
            },
            description='A random forest model'
        )
    # @staticmethod
    # def get_logistic_regression_config() -> ModelConfig:
    #     """Get the configuration for a logistic regression model"""
    #     return ModelConfig(
    #         name='Logistic Regression',
    #         model=LogisticRegression(),
    #         param_grid={
    #             'classifier__C': [0.1, 1.0, 10.0]
    #         },
    #         description='A logistic regression model'
    #     )
    

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
    