from sklearn.ensemble import VotingClassifier
from .base_estimators import get_optimized_hgb, get_optimized_rf
from .model_config import ModelConfig

def get_voting_config() -> ModelConfig:
    """Get the configuration for a voting classifier"""
    base_estimators = [
        ('hgb', get_optimized_hgb()),
        ('rf', get_optimized_rf())
    ]

    base_model = VotingClassifier(
        estimators=base_estimators,
        voting='soft',
        n_jobs=-1
    )
    
    WEIGHT_CONFIGS = {
        'equal': [1.0, 1.0],
        'more_rf': [1.0, 2.0],
        'more_hgb': [2.0, 1.0],
        'slight_more_rf': [1.0, 1.5],
        'slight_more_hgb': [1.5, 1.0],
        'more_rf_2': [1.0, 3.0],
        'more_hgb_2': [3.0, 1.0],
    }
    
    param_grid = {
        'voting': ['soft'],
        'weight_config': list(WEIGHT_CONFIGS.keys())
    }

    def param_suggest(trial):
        weight_config = trial.suggest_categorical('weight_config', param_grid['weight_config'])
        params = {
            'voting': 'soft',
            'weights': WEIGHT_CONFIGS[weight_config]
        }
        return params

    return ModelConfig(
        name='Voting Classifier',
        model=base_model,
        param_grid=param_grid,
        param_suggest=param_suggest,
        description='Voting classifier with optimized weights and soft voting'
    )