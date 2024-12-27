import numpy as np
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
            class_weight={0: 1, 1: 1},
            early_stopping=True,
            l2_regularization=2.294280717958801,
            learning_rate=0.23040698858775321,
            max_bins=115,
            max_depth=12,
            max_iter=11,
            max_leaf_nodes=33,
            min_samples_leaf=81,
            n_iter_no_change=17,
            tol=0.030672753967226428,
            validation_fraction=0.8821528541761128,
            random_state=42,
            )),
            ('rf', RandomForestClassifier(
            bootstrap=True,
            ccp_alpha= 0.004681320432681322,
            class_weight={0: 1, 1: 1},
            criterion='entropy',
            max_depth=46,
            max_features='sqrt',
            max_leaf_nodes=30,
            max_samples=0.4343415014797324,
            min_impurity_decrease= 0.022289899203164085,
            min_samples_leaf=1,
            min_samples_split=25,
            min_weight_fraction_leaf=0.001594042722389311,
            n_estimators=84,
            oob_score=False,
            random_state=42
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
            'final_estimator__bootstrap': [True],
            'final_estimator__ccp_alpha': (0.0001, 0.1),
            'final_estimator__criterion': ['entropy', 'gini'],
            'final_estimator__max_depth': (15, 60),
            'final_estimator__max_features': ['sqrt', 'log2'],
            'final_estimator__max_leaf_nodes': (30, 120),
            'final_estimator__min_impurity_decrease': (0.0, 0.1),
            'final_estimator__min_samples_leaf': (20, 70),
            'final_estimator__min_samples_split': (5, 80),
            'final_estimator__n_estimators': (60, 200),
            'final_estimator__random_state': [42]
        }

        def param_suggest(trial):
            params = {
            'final_estimator__bootstrap': trial.suggest_categorical('final_estimator__bootstrap', param_grid['final_estimator__bootstrap']),
            'final_estimator__ccp_alpha': trial.suggest_float('final_estimator__ccp_alpha', param_grid['final_estimator__ccp_alpha'][0], param_grid['final_estimator__ccp_alpha'][1]),
            'final_estimator__criterion': trial.suggest_categorical('final_estimator__criterion', param_grid['final_estimator__criterion']),
            'final_estimator__max_depth': trial.suggest_int('final_estimator__max_depth', param_grid['final_estimator__max_depth'][0], param_grid['final_estimator__max_depth'][1]),
            'final_estimator__max_features': trial.suggest_categorical('final_estimator__max_features', param_grid['final_estimator__max_features']),
            'final_estimator__max_leaf_nodes': trial.suggest_int('final_estimator__max_leaf_nodes', param_grid['final_estimator__max_leaf_nodes'][0], param_grid['final_estimator__max_leaf_nodes'][1]),
            'final_estimator__min_impurity_decrease': trial.suggest_float('final_estimator__min_impurity_decrease', param_grid['final_estimator__min_impurity_decrease'][0], param_grid['final_estimator__min_impurity_decrease'][1]),
            'final_estimator__min_samples_leaf': trial.suggest_int('final_estimator__min_samples_leaf', param_grid['final_estimator__min_samples_leaf'][0], param_grid['final_estimator__min_samples_leaf'][1]),
            'final_estimator__min_samples_split': trial.suggest_int('final_estimator__min_samples_split', param_grid['final_estimator__min_samples_split'][0], param_grid['final_estimator__min_samples_split'][1]),
            'final_estimator__n_estimators': trial.suggest_int('final_estimator__n_estimators', param_grid['final_estimator__n_estimators'][0], param_grid['final_estimator__n_estimators'][1]),
            'final_estimator__random_state': param_grid['final_estimator__random_state'][0],
            }
            return params
        
        return ModelConfig(
            name='Stacking Classifier',
            model=base_model,
            param_grid=param_grid,
            param_suggest=param_suggest,
            description='Stacking classifier with Optuna parameter suggestions for final estimator'
        )

    @staticmethod
    def get_voting_config() -> ModelConfig:
        """Get the configuration for a voting classifier"""
        # Base estimators are fixed with optimal configurations
        base_estimators = [
            ('hgb', HistGradientBoostingClassifier(
                class_weight={0: 1, 1: 1},
                early_stopping=True,
                l2_regularization=2.294280717958801,
                learning_rate=0.23040698858775321,
                max_bins=115,
                max_depth=12,
                max_iter=11,
                max_leaf_nodes=33,
                min_samples_leaf=81,
                n_iter_no_change=17,
                tol=0.030672753967226428,
                validation_fraction=0.8821528541761128,
                random_state=42,
            )),
            ('rf', RandomForestClassifier(
                bootstrap=True,
                ccp_alpha=0.004681320432681322,
                class_weight={0: 1, 1: 1},
                criterion='entropy',
                max_depth=46,
                max_features='sqrt',
                max_leaf_nodes=30,
                max_samples=0.4343415014797324,
                min_impurity_decrease=0.022289899203164085,
                min_samples_leaf=1,
                min_samples_split=25,
                min_weight_fraction_leaf=0.001594042722389311,
                n_estimators=84,
                oob_score=False,
                random_state=42
            ))
        ]

        # Create initial voting classifier
        base_model = VotingClassifier(
            estimators=base_estimators,
            voting='soft',
            n_jobs=-1
        )
        
            # Define weight configurations
        WEIGHT_CONFIGS = {
            'equal': [1.0, 1.0],          # equal weights
            'more_rf': [1.0, 2.0],        # more weight on RF
            'more_hgb': [2.0, 1.0],       # more weight on HGB
            'slight_more_rf': [1.0, 1.5],  # slightly more weight on RF
            'slight_more_hgb': [1.5, 1.0], # slightly more weight on HGB
            'more_rf_2': [1.0, 3.0],      # more weight on RF
            'more_hgb_2': [3.0, 1.0],     # more weight on HGB

        }
        
        # Define parameter ranges
        param_grid = {
            'voting': ['soft'],
            'weight_config': list(WEIGHT_CONFIGS.keys())  # Use string identifiers for storage
        }

        def param_suggest(trial):
            """Suggest parameters for voting classifier"""
            # Get the weight configuration identifier
            weight_config = trial.suggest_categorical('weight_config', param_grid['weight_config'])
            
            params = {
                'voting': 'soft',
                'weights': WEIGHT_CONFIGS[weight_config]  # Convert to actual weights
            }
            return params
        param_grid['param_suggest'] = param_suggest
        return ModelConfig(
            name='Voting Classifier',
            model=base_model,
            param_grid=param_grid,
            param_suggest=param_suggest,
            description='Voting classifier with optimized weights and soft voting'
        )