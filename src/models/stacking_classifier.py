from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from .base_estimators import get_optimized_hgb, get_optimized_rf
from .model_config import ModelConfig

def get_stacking_config() -> ModelConfig:
    """Get the configuration for a stacking classifier"""
    base_estimators = [
        ('hgb', get_optimized_hgb()),
        ('rf', get_optimized_rf())
    ]

    base_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=RandomForestClassifier(random_state=42),
        stack_method='predict_proba'
    )
    
    param_grid = {
        'final_estimator__bootstrap': [True],
        'final_estimator__ccp_alpha': (0.0001, 0.1),
        'final_estimator__criterion': ['entropy', 'gini'],
        'final_estimator__max_depth': (15, 60),
        'final_estimator__max_features': ['sqrt', 'log2'],
        'final_estimator__max_leaf_nodes': (30, 120),
        'final_estimator__min_impurity_decrease': (0.0, 0.1),
        'final_estimator__min_samples_leaf': (20, 70),
        'final_estimator__min_samples_split': (5, 80),
        'final_estimator__n_estimators': (60, 200),
        'final_estimator__random_state': [42]
    }

    def param_suggest(trial):
        params = {
            'final_estimator__bootstrap': trial.suggest_categorical('final_estimator__bootstrap', param_grid['final_estimator__bootstrap']),
            'final_estimator__ccp_alpha': trial.suggest_float('final_estimator__ccp_alpha', param_grid['final_estimator__ccp_alpha'][0], param_grid['final_estimator__ccp_alpha'][1]),
            'final_estimator__criterion': trial.suggest_categorical('final_estimator__criterion', param_grid['final_estimator__criterion']),
            'final_estimator__max_depth': trial.suggest_int('final_estimator__max_depth', param_grid['final_estimator__max_depth'][0], param_grid['final_estimator__max_depth'][1]),
            'final_estimator__max_features': trial.suggest_categorical('final_estimator__max_features', param_grid['final_estimator__max_features']),
            'final_estimator__max_leaf_nodes': trial.suggest_int('final_estimator__max_leaf_nodes', param_grid['final_estimator__max_leaf_nodes'][0], param_grid['final_estimator__max_leaf_nodes'][1]),
            'final_estimator__min_impurity_decrease': trial.suggest_float('final_estimator__min_impurity_decrease', param_grid['final_estimator__min_impurity_decrease'][0], param_grid['final_estimator__min_impurity_decrease'][1]),
            'final_estimator__min_samples_leaf': trial.suggest_int('final_estimator__min_samples_leaf', param_grid['final_estimator__min_samples_leaf'][0], param_grid['final_estimator__min_samples_leaf'][1]),
            'final_estimator__min_samples_split': trial.suggest_int('final_estimator__min_samples_split', param_grid['final_estimator__min_samples_split'][0], param_grid['final_estimator__min_samples_split'][1]),
            'final_estimator__n_estimators': trial.suggest_int('final_estimator__n_estimators', param_grid['final_estimator__n_estimators'][0], param_grid['final_estimator__n_estimators'][1]),
            'final_estimator__random_state': param_grid['final_estimator__random_state'][0],
        }
        return params
    
    return ModelConfig(
        name='Stacking Classifier',
        model=base_model,
        param_grid=param_grid,
        param_suggest=param_suggest,
        description='Stacking classifier with Optuna parameter suggestions for final estimator'
    )