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
        n_jobs=1,        # Avoid nested parallelization
        random_state=42
    )
    
    # Modified parameter ranges with controlled parallelization
    param_grid = {
        # Reduced maximum number of estimators to manage memory better
        'n_estimators': (30, 100),
        
        # Sample size range that ensures enough data for each estimator
        'max_samples': (0.7, 0.9),
        
        # Feature range that ensures enough features while allowing diversity
        'max_features': (0.6, 0.9),
        
        # Only use bootstrap=True to be safe
        'bootstrap': [True],
        'bootstrap_features': [True, False],
        
        # Disable out-of-bag score
        'oob_score': [False],
        
        # Fixed parameters - using single-threaded processing
        'n_jobs': [1],   # Prevent nested parallelization
        'random_state': [42],
        'verbose': [0]
    }

    def param_suggest(trial):
        """Suggest parameters with intelligent combinations"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *param_grid['n_estimators']),
            'max_samples': trial.suggest_float('max_samples', *param_grid['max_samples']),
            'max_features': trial.suggest_float('max_features', *param_grid['max_features']),
            'bootstrap': True,  # Always True for stability
            'bootstrap_features': trial.suggest_categorical('bootstrap_features', param_grid['bootstrap_features']),
            'oob_score': False,  # Always False for stability
            'n_jobs': 1,        # Prevent nested parallelization
            'random_state': param_grid['random_state'][0],
            'verbose': param_grid['verbose'][0]
        }
        
        return params
    
    return ModelConfig(
        name=f'Bagging Classifier with {classifier}',
        model=base_model,
        param_grid=param_grid,
        param_suggest=param_suggest,
        description=f'Bagging classifier using {classifier} as base estimator with memory-efficient parameter ranges'
    )