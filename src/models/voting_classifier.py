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
   
    param_grid = {
        'voting': ['soft'], # Soft votiong. Hard voting not used
    }


    return ModelConfig(
        name='Voting Classifier',
        model=base_model,
        param_grid=param_grid,
        param_suggest=param_grid,
        description='Voting classifier with optimized weights and soft voting'
    )