from sklearn.ensemble import BaggingClassifier
from .base_estimators import get_optimized_hgb, get_optimized_rf
from .model_config import ModelConfig

def get_bagging_config(classifier: str) -> ModelConfig:
    """Get the configuration for a bagging classifier with improved hyperparameter ranges"""
    # Get the base estimator
    if classifier == 'hist_gradient_boosting':
        estimator = get_optimized_hgb()
    elif classifier == 'random_forest':
        estimator = get_optimized_rf()
    else:
        raise ValueError(f"Unknown classifier: {classifier}. Must be 'hist_gradient_boosting' or 'random_forest'")

    base_model = BaggingClassifier(
        estimator=estimator,
        n_estimators=50,  # Start with a reasonable number of estimators
        max_samples=0.8,  # Use a good portion of samples
        max_features=0.8,  # Use a good portion of features
        bootstrap=True,   # Enable bootstrapping
        n_jobs=-1,
        random_state=42
    )
    
    # Improved parameter ranges based on typical best practices
    param_grid = {
        # More estimators often leads to better performance
        'n_estimators': (50, 200),
        
        # Sample size range that ensures enough data for each estimator
        'max_samples': (0.7, 1.0),
        
        # Feature range that ensures enough features while allowing diversity
        'max_features': (0.6, 1.0),
        
        # Only use bootstrap=True to be safe
        'bootstrap': [True],
        'bootstrap_features': [True, False],
        
        # Disable out-of-bag score for now
        'oob_score': [False],
        
        # Fixed parameters
        'n_jobs': [-1],
        'random_state': [42],
        'verbose': [0]
    }

    def param_suggest(trial):
        """Suggest parameters with intelligent combinations"""
        n_estimators = trial.suggest_int('n_estimators', *param_grid['n_estimators'])
        bootstrap = trial.suggest_categorical('bootstrap', param_grid['bootstrap'])
        
        # OOB score must be False when bootstrap is False
        oob_score = False
        if bootstrap:
            # Only suggest oob_score if bootstrap is True
            oob_score = trial.suggest_categorical('oob_score', [False])  # For now, keep it False always to avoid issues
        
        params = {
            'n_estimators': n_estimators,
            'max_samples': trial.suggest_float('max_samples', *param_grid['max_samples']),
            'max_features': trial.suggest_float('max_features', *param_grid['max_features']),
            'bootstrap': bootstrap,
            'bootstrap_features': trial.suggest_categorical('bootstrap_features', param_grid['bootstrap_features']),
            'oob_score': oob_score,
            'n_jobs': param_grid['n_jobs'][0],
            'random_state': param_grid['random_state'][0],
            'verbose': param_grid['verbose'][0]
        }
        
        return params
    
    return ModelConfig(
        name=f'Bagging Classifier with {classifier}',
        model=base_model,
        param_grid=param_grid,
        param_suggest=param_suggest,
        description=f'Bagging classifier using {classifier} as base estimator with improved parameter ranges'
    )