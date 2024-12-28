from sklearn.ensemble import HistGradientBoostingClassifier
from .model_config import ModelConfig

def get_hist_gradient_boosting_config() -> ModelConfig:
    """Get the configuration for a histogram gradient boosting model"""
    param_grid = {
        'l2_regularization': (0.0, 10),
        'learning_rate': (0.001, 0.5),
        'max_bins': (70, 120),
        'max_depth': (2, 20),
        'max_iter': (100, 1500),
        'max_leaf_nodes': (15, 120),
        'min_samples_leaf': (1, 150),
        'n_iter_no_change': (5, 35),
        'random_state': [42],
        'tol': (1e-5, 1e-1),
        'validation_fraction': (0.1, 0.9)
    }
    
    def param_suggest(trial):
        params = {
            'early_stopping': True,
            'l2_regularization': trial.suggest_float('l2_regularization', *param_grid['l2_regularization'], log=True),
            'learning_rate': trial.suggest_float('learning_rate', *param_grid['learning_rate'], log=True),
            'max_bins': trial.suggest_int('max_bins', *param_grid['max_bins']),
            'max_depth': trial.suggest_int('max_depth', *param_grid['max_depth']),
            'max_iter': trial.suggest_int('max_iter', *param_grid['max_iter']),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', *param_grid['max_leaf_nodes']),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', *param_grid['min_samples_leaf']),
            'n_iter_no_change': trial.suggest_int('n_iter_no_change', *param_grid['n_iter_no_change']),
            'random_state': 42,
            'tol': trial.suggest_float('tol', *param_grid['tol'], log=True),
            'validation_fraction': trial.suggest_float('validation_fraction', *param_grid['validation_fraction']),
        }
        return params

    return ModelConfig(
        name='Histogram Gradient Boosting',
        model=HistGradientBoostingClassifier(random_state=param_grid['random_state'], class_weight={0: 1.0, 1: 1.0}),
        param_grid=param_grid,
        param_suggest=param_suggest,
        description='Histogram gradient boosting model with Optuna parameter suggestions'
    )