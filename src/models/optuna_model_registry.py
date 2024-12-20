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
            'n_estimators': (60, 110),
            'max_depth': (5, 40),
            'min_samples_split': (5, 30),
            'min_samples_leaf': (3, 15),
            'ccp_alpha': (0.0001, 0.01),
            'bootstrap': [True, False],
            'max_features': ['sqrt', 'log2'],
            'criterion': ['entropy', 'gini'],
            'random_state': 42
        }
        
        def param_suggest(trial):
            # Create base model parameters without pipeline prefix
            base_params = {
                'n_estimators': trial.suggest_int('n_estimators', param_grid['n_estimators'][0], param_grid['n_estimators'][1]),
                'max_depth': trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'min_samples_split': trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0], param_grid['min_samples_split'][1]),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0], param_grid['min_samples_leaf'][1]),
                'ccp_alpha': trial.suggest_float('ccp_alpha', param_grid['ccp_alpha'][0], param_grid['ccp_alpha'][1]),
                'bootstrap': trial.suggest_categorical('bootstrap', param_grid['bootstrap']),
                'max_features': trial.suggest_categorical('max_features', param_grid['max_features']),
                'criterion': trial.suggest_categorical('criterion', param_grid['criterion']),
                'random_state': param_grid['random_state'],
                'class_weight': {0: 1, 1: trial.suggest_float('class_weight_ratio', 1.0, 3.0)}
            }

            # Convert to pipeline parameters by adding prefix

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
    @staticmethod
    def get_hist_gradient_boosting_config() -> ModelConfig:
        """Get the configuration for a histogram gradient boosting model"""
        param_grid = {
            'learning_rate': (0.0001, 1.0),  
            'max_depth': (2, 30),       
            'max_iter': (100, 3000),  
            'min_samples_leaf': (1, 100),  
            'l2_regularization': (0.0, 50.0),  
            'max_bins': (2, 255),  
            # 'class_weight_ratio': (1.0, 10.0),  
            'validation_fraction': (0.05, 0.35),  
            'n_iter_no_change': (5, 50),  
            'early_stopping': [True],
            'random_state': [42]
        }
        
        def param_suggest(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', *param_grid['learning_rate'], log=True),
                'max_depth': trial.suggest_int('max_depth', *param_grid['max_depth']),
                'max_iter': trial.suggest_int('max_iter', *param_grid['max_iter']),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', *param_grid['min_samples_leaf']),
                'l2_regularization': trial.suggest_float('l2_regularization', *param_grid['l2_regularization'], log=True),
                'max_bins': trial.suggest_int('max_bins', *param_grid['max_bins']),
                'validation_fraction': trial.suggest_float('validation_fraction', *param_grid['validation_fraction']),
                'n_iter_no_change': trial.suggest_int('n_iter_no_change', *param_grid['n_iter_no_change']),
                'early_stopping': True,
                'random_state': 42,
                'class_weight': {0: 1, 1: trial.suggest_float('class_weight_ratio', 1.0, 3.0)}

            }
            
            ratio = trial.suggest_float('class_weight_ratio', *param_grid['class_weight_ratio'])
            params['class_weight'] = {0: 1, 1: ratio}
            
            return params
            
        return ModelConfig(
            name='Histogram Gradient Boosting',
            model=HistGradientBoostingClassifier(),
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






    # @staticmethod
    # def get_stacking_config() -> ModelConfig:
    #     """Get the configuration for a stacking classifier"""
    #     # Base estimators are fixed with optimal configurations
    #     base_estimators = [
    #         ('hgb', HistGradientBoostingClassifier(
    #             random_state=42, class_weight={0: 1, 1: 2},
    #             early_stopping=True, l2_regularization=25.065867997544807,
    #             learning_rate=0.4666024344469928, max_bins=56, max_depth=28,
    #             max_iter=2908, min_samples_leaf=27,
    #             n_iter_no_change=48, validation_fraction=0.13172302905106784
    #         )),
    #         ('rf', RandomForestClassifier(
    #             random_state=42, bootstrap=True,
    #             ccp_alpha=0.004986320557962982, class_weight={0: 1, 1: 2.5},
    #             criterion='entropy', max_depth=15,
    #             max_features='sqrt', min_samples_leaf=6,
    #             min_samples_split=14, n_estimators=98
    #         ))
    #     ]

    #     # Define base model (will be modified during optimization)
    #     base_model = StackingClassifier(
    #         estimators=base_estimators,
    #         final_estimator=RandomForestClassifier(random_state=42),
    #         stack_method='predict_proba'
    #     )
        
    #     # Define parameter ranges for final estimator only
    #     param_grid = {
    #         'n_estimators': (60, 110),
    #         'max_depth': (5, 40),
    #         'min_samples_split': (5, 30),
    #         'min_samples_leaf': (3, 15),
    #         'ccp_alpha': (0.0001, 0.01),
    #         'bootstrap': [True, False],
    #         'max_features': ['sqrt', 'log2'],
    #         'criterion': ['entropy', 'gini'],
    #         'random_state': [42]
    #     }
        
    #     def param_suggest(trial):
    #         # Create parameters for the final estimator
    #         params = {
    #             'n_estimators': trial.suggest_int('n_estimators', *param_grid['n_estimators']),
    #             'max_depth': trial.suggest_int('max_depth', *param_grid['max_depth']),
    #             'min_samples_split': trial.suggest_int('min_samples_split', *param_grid['min_samples_split']),
    #             'min_samples_leaf': trial.suggest_int('min_samples_leaf', *param_grid['min_samples_leaf']),
    #             'ccp_alpha': trial.suggest_float('ccp_alpha', *param_grid['ccp_alpha']),
    #             'bootstrap': trial.suggest_categorical('bootstrap', param_grid['bootstrap']),
    #             'max_features': trial.suggest_categorical('max_features', param_grid['max_features']),
    #             'criterion': trial.suggest_categorical('criterion', param_grid['criterion']),
    #             'random_state': param_grid['random_state'],
    #             'class_weight': {0: 1, 1: trial.suggest_float('class_weight_ratio', 1.0, 3.0)}
    #         }
            
    #         # Create a new StackingClassifier with the suggested parameters for final_estimator
    #         return {
    #             'estimators': base_estimators,
    #             'final_estimator': RandomForestClassifier(**params),
    #             'stack_method': 'predict_proba'
    #         }

    #     return ModelConfig(
    #         name='StackingClassifier',
    #         model=base_model,
    #         param_grid=param_grid,
    #         param_suggest=param_suggest,
    #         description='Stacking classifier with Optuna parameter suggestions'
    #     )