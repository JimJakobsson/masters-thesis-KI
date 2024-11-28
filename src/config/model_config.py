
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Confiuguratoin for model types and their corresponding SHAP explainers"""
    TREE_BASED_MODELS = (
        'RandomForestClassifier', 'GradientBoostingClassifier', 
        'DecisionTreeClassifier', 'XGBClassifier', 'XGBRegressor',
        'CatBoostClassifier', 'CatBoostRegressor', 'AdaBoostClassifier'
    )

    LINEAR_MODELS = (
        'LogisticRegression', 'LinearSVC', 'RidgeClassifier', 'RidgeClassifierCV',
        'SGDClassifier', 'SGDRegressor', 'Ridge', 'Lasso', 'ElasticNet', 'LassoCV'
    )

    DEEP_LEARNING_MODELS = (
        'MLPClassifier', 'MLPRegressor'
    )