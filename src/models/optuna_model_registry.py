from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    BaggingClassifier, StackingClassifier, VotingClassifier
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from models.model_config import ModelConfig

class OptunaModelRegistry:
    """Registry for machine learning models with Optuna-compatible parameter spaces"""
    
    @staticmethod
    def get_random_forest_config() -> ModelConfig:
        """Get the configuration for a random forest model"""
        param_grid = {
            'bootstrap': [True],
            'ccp_alpha': (0.001, 0.01),
            'criterion': ['entropy'],
            'max_depth': (20, 60),
            'max_features': ['sqrt'],
            'max_leaf_nodes': (15, 60),
            'max_samples': (0.1, 1.0),
            'min_impurity_decrease': (0.0, 0.1),
            'min_samples_leaf': (1, 20),
            'min_samples_split': (5, 30),
            'min_weight_fraction_leaf': (0.0, 0.5),
            'n_estimators': (80, 160),
            'oob_score': [True, False],
            'random_state': 42,
        }
        
        def param_suggest(trial):
            # Create base model parameters without pipeline prefix
           
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

            # Convert to pipeline parameters by adding prefix

            pipeline_params = {f'classifier__{k}': v for k, v in base_params.items()}
            print(f"Trial parameters: {pipeline_params}")
            return pipeline_params
            
        return ModelConfig(
            name='Random Forest',
            model=RandomForestClassifier(random_state=param_grid['random_state'], class_weight={0: 1.0, 1: 1.0}),
            param_grid=param_grid,
            param_suggest=param_suggest,
            description='Random forest model with Optuna parameter suggestions'
        )
    
    @staticmethod
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
    
    @staticmethod
    def get_stacking_config() -> ModelConfig:
        """Get the configuration for a stacking classifier"""
        # Base estimators are fixed with optimal configurations
        base_estimators = [
            ('hgb', HistGradientBoostingClassifier(
                random_state=42, class_weight={0: 1, 1: 2},
                early_stopping=True, l2_regularization=25.065867997544807,
                learning_rate=0.4666024344469928, max_bins=56, max_depth=28,
                max_iter=2908, min_samples_leaf=27,
                n_iter_no_change=48, validation_fraction=0.13172302905106784
            )),
            ('rf', RandomForestClassifier(
                random_state=42, bootstrap=True,
                ccp_alpha=0.004986320557962982, class_weight={0: 1, 1: 2.5},
                criterion='entropy', max_depth=15,
                max_features='sqrt', min_samples_leaf=6,
                min_samples_split=14, n_estimators=98
            ))
        ]

        # Create initial stacking classifier
        base_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=RandomForestClassifier(random_state=42),
            stack_method='predict_proba'
        )
        
        # Define parameter ranges for final estimator
        param_grid = {
            'final_estimator__n_estimators': (60, 110),
            'final_estimator__max_depth': (5, 40),
            'final_estimator__min_samples_split': (5, 30),
            'final_estimator__min_samples_leaf': (3, 15),
            'final_estimator__ccp_alpha': (0.0001, 0.01),
            'final_estimator__bootstrap': [True, False],
            'final_estimator__max_features': ['sqrt', 'log2'],
            'final_estimator__criterion': ['entropy', 'gini'],
            'final_estimator__random_state': [42]
        }
        
        return ModelConfig(
            name='StackingClassifier',
            model=base_model,
            param_grid=param_grid,
            param_suggest=None,  # We don't need param_suggest anymore
            description='Stacking classifier with Optuna parameter suggestions for final estimator'
        )
