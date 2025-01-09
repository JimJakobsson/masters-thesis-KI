from sklearn.ensemble import RandomForestClassifier
from .model_config import ModelConfig

def get_random_forest_config() -> ModelConfig:
    """Get the configuration for a random forest model"""
   
    param_grid = {
        
        'bootstrap': [True],
        'ccp_alpha': (0.001, 0.01),
        'criterion': ['entropy'],
        'max_depth': (20, 60),
        'max_features': ['sqrt'],
        'max_leaf_nodes': (15, 60),
        'max_samples': (0.5, 1.0),
        'min_impurity_decrease': (0.0, 0.1),
        'min_samples_leaf': (1, 20),
        'min_samples_split': (5, 30),
        'min_weight_fraction_leaf': (0.0, 0.1),
        'n_estimators': (80, 160),
        'oob_score': [False],
        'random_state': 42,
    }
    
    def param_suggest(trial):
        

        base_params = {

            'bootstrap': trial.suggest_categorical('bootstrap', param_grid['bootstrap']),
            'ccp_alpha': trial.suggest_float('ccp_alpha', param_grid['ccp_alpha'][0], param_grid['ccp_alpha'][1]),
            'criterion': trial.suggest_categorical('criterion', param_grid['criterion']),
            'max_depth': trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
            'max_features': trial.suggest_categorical('max_features', param_grid['max_features']),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', param_grid['max_leaf_nodes'][0], param_grid['max_leaf_nodes'][1]),
            'max_samples': trial.suggest_float('max_samples', param_grid['max_samples'][0], param_grid['max_samples'][1]),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', param_grid['min_impurity_decrease'][0], param_grid['min_impurity_decrease'][1]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0], param_grid['min_samples_leaf'][1]),
            'min_samples_split': trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0], param_grid['min_samples_split'][1]),
            'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', param_grid['min_weight_fraction_leaf'][0], param_grid['min_weight_fraction_leaf'][1]),
            'n_estimators': trial.suggest_int('n_estimators', param_grid['n_estimators'][0], param_grid['n_estimators'][1]),
            'oob_score': trial.suggest_categorical('oob_score', param_grid['oob_score']),
            'random_state': param_grid['random_state'],
        }
        pipeline_params = {f'classifier__{k}': v for k, v in base_params.items()}
        print(f"Trial parameters: {pipeline_params}")
        return pipeline_params
        
    return ModelConfig(
        name='Random Forest',
        model=RandomForestClassifier(random_state=param_grid['random_state']),
        param_grid=param_grid,
        param_suggest=param_suggest,
        description='Random forest model with Optuna parameter suggestions'
    )
