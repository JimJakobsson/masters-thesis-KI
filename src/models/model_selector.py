from typing import List
from models.model_config import ModelConfig
from models.model_registry import ModelRegistry


class ModelSelector:
    """Model selector class to select the model based on the model name"""

    def __init__(self):
        self.registry = {
            'random_forest': ModelRegistry.get_random_forest_config,
            # 'gradient_boosting': ModelRegistry.get_gradient_boosting_config,
            # 'logistic_regression': ModelRegistry.get_logistic_regression_config,
            # 'hist_gradient_boosting': ModelRegistry.get_hist_gradient_boosting_config,
            # 'bagging': ModelRegistry.get_bagging_config,
            # 'svm': ModelRegistry.get_svm_config,
            # 'xgboost': ModelRegistry.get_xgboost_config,
            # 'catboost': ModelRegistry.get_catboost_config
        }

    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.registry.keys())
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for specified model"""
        if model_name not in self.registry:
            raise ValueError(f"Unknown model: {model_name}. Available models: {self.get_available_models()}")
        return self.registry[model_name]()
    
    def get_model_description(self, model_name: str) -> str:
        """Get description of specified model"""
        return self.get_model_config(model_name).description