from sklearn.ensemble import BaggingClassifier
from .base_estimators import get_optimized_hgb, get_optimized_rf
from .model_config import ModelConfig

from sklearn.ensemble import BaggingClassifier
from .base_estimators import get_optimized_hgb, get_optimized_rf
from .model_config import ModelConfig

def get_bagging_config() -> ModelConfig:
    """Get the configuration for a bagging classifier"""
    base_estimators = [
        ('hgb', get_optimized_hgb()),
        ('rf', get_optimized_rf())
    ]

    base_model = BaggingClassifier(
        base_estimator=base_estimators,
        n_estimators=10,
        n_jobs=-1,
        random_state=42
    )
    
    param_grid = {
        'n_estimators': (10, 100),
        'max_samples': (0.1, 1.0),
        'max_features': (0.1, 1.0),
        'bootstrap': [True, False],
        'bootstrap_features': [True, False],
        'oob_score': [True, False],
        'warm_start': [True, False],
        'n_jobs': [-1],
        'random_state': [42]
    }

    def param_suggest(trial):
        """Suggest parameters for bagging classifier"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *param_grid['n_estimators']),
            'max_samples': trial.suggest_float('max_samples', *param_grid['max_samples']),
            'max_features': trial.suggest_float('max_features', *param_grid['max_features']),
            'bootstrap': trial.suggest_categorical('bootstrap', param_grid['bootstrap']),
            'bootstrap_features': trial.suggest_categorical('bootstrap_features', param_grid['bootstrap_features']),
            'oob_score': trial.suggest_categorical('oob_score', param_grid['oob_score']),
            'warm_start': trial.suggest_categorical('warm_start', param_grid['warm_start']),
            'n_jobs': param_grid['n_jobs'][0],
            'random_state': param_grid['random_state'][0]
        }
        return params
    
    return ModelConfig(
        name='Bagging Classifier',
        model=base_model,
        param_grid=param_grid,
        param_suggest=param_suggest,
        description='Bagging classifier with Optuna parameter suggestions'
    )