import optuna
import time
from typing import Any, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
import pandas as pd
from experiment.experiment_config import ExperimentConfig
from experiment.optuna_wrapper import OptunaWrapper

class ModelTrainerOptuna:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def create_pipeline(self, preprocessor, model) -> Pipeline:
        """Create a pipeline with the specified column transformer and model"""
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

    def suggest_parameter(self, trial, param_name: str, param_range: Any) -> Any:
        """Suggest a parameter value based on its type and range"""
        # If it's a tuple of two values, treat it as a range
        if isinstance(param_range, tuple) and len(param_range) == 2:
            if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                return trial.suggest_int(param_name, param_range[0], param_range[1])
            elif isinstance(param_range[0], float) or isinstance(param_range[1], float):
                return trial.suggest_float(param_name, param_range[0], param_range[1])
                
        # Handle lists as categorical choices
        elif isinstance(param_range, list):
            if len(param_range) == 0:
                raise ValueError(f"Empty parameter range for {param_name}")
            if len(param_range) == 1:
                return param_range[0]
            return trial.suggest_categorical(param_name, param_range)
            
        # If it's a single value, return it directly
        return param_range

    def train_model(self,
                   pipeline: Pipeline,
                   param_grid: Dict[str, Any],
                   X: pd.DataFrame,
                   y: pd.Series,
                   n_trials: int = 10) -> Any:
        """Train model using Optuna for hyperparameter optimization"""

        #Number of trials from the confguration
        n_trials = self.config.n_trials_optuna

        def objective(trial):
            try:
                # Get base model class
                model_class = pipeline.named_steps['classifier'].__class__
                # Special handling for StackingClassifier
                if isinstance(pipeline.named_steps['classifier'], StackingClassifier):
                    # Get base configuration
                    base_config = {
                        'estimators': pipeline.named_steps['classifier'].estimators,
                        'stack_method': pipeline.named_steps['classifier'].stack_method,
                    }
                    
                    # Create parameters for final estimator
                    final_estimator_params = {}
                    for param_name, param_range in param_grid.items():
                        # Remove 'classifier__' prefix if it exists
                        clean_name = param_name.replace('classifier__', '')
                        # Remove 'final_estimator__' prefix and use as parameter name
                        param_key = clean_name.replace('final_estimator__', '')
                        final_estimator_params[param_key] = self.suggest_parameter(
                            trial, param_key, param_range
                        )
                    
                    # Create new final estimator
                    final_estimator = RandomForestClassifier(**final_estimator_params)
                    base_config['final_estimator'] = final_estimator
                    
                    # Create new model instance
                    model = model_class(**base_config)
                else:
                    # Suggest parameters
                    params = {}
                    for param_name, param_range in param_grid.items():
                        clean_name = param_name.replace('classifier__', '')
                        params[clean_name] = self.suggest_parameter(trial, clean_name, param_range)
                     # Separately handle class weight optimization

                #  # Suggest just the weight value for class 1
                # class_1_weight = self.suggest_parameter(
                #     trial,
                #     'class_weight_ratio',  # Using a different name to avoid confusion
                #     [1.0, 2.0, 2.5, 3.0, 3.5, 4.0]  # Possible weight values for class 1
                # )
                
                # # Construct the class_weight dictionary after getting the suggested value
                # params['class_weight'] = {
                #     0: 1.0,  # Fixed weight for class 0
                #     1: class_1_weight  # Suggested weight for class 1
                # }
                # Create new model instance with suggested parameters
                model = model_class(**params)
                
                # Create and evaluate pipeline
                trial_pipeline = Pipeline([
                    ('preprocessor', pipeline.named_steps['preprocessor']),
                    ('classifier', model)
                ])
                
                scores = cross_val_score(
                    trial_pipeline,
                    X, y,
                    cv=self.config.cv_folds,
                    scoring='accuracy',
                    n_jobs=-1
                )
                return scores.mean()
            
            except Exception as e:
                print(f"Trial failed with parameters {params} due to error: {str(e)}")
                raise  # Re-raise the exception for Optuna to handle

        # Create study with pruner
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        print(f"\nStarting optimization with {n_trials} trials...")
        time_start = time.time()
        
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            catch=(ValueError, TypeError)
        )
        
        time_end = time.time()

        if len(study.trials) == 0:
            raise ValueError("No successful trials completed. Check parameter ranges and model configuration.")

        

        print(f"\nOptimization completed in {time_end - time_start:.2f} seconds")
        print(f"Best parameters: {study.best_params}")
        #print all parameters and their values, not just from the best trial

        print(f"Best score: {study.best_value:.4f}")
        
        #Create final model with the best parameters
        if isinstance(pipeline.named_steps['classifier'], StackingClassifier):
            # Get base configuration for stacking classifier
            base_config = {
                'estimators': pipeline.named_steps['classifier'].estimators,
                'stack_method': pipeline.named_steps['classifier'].stack_method,
            }
            
            # Create parameters for final estimator
            final_estimator_params = {}
            for param_name, value in study.best_params.items():

                # Remove 'final_estimator__' prefix
                param_key = param_name.replace('final_estimator__', '')
                final_estimator_params[param_key] = value
            
            # Create final estimator with best parameters
            final_estimator = RandomForestClassifier(**final_estimator_params)
            base_config['final_estimator'] = final_estimator
            
            # Create final stacking classifier
            final_model = pipeline.named_steps['classifier'].__class__(**base_config)
        else:
            best_params = {k.replace('classifier__', ''): v for k, v in study.best_params.items()}
            print(f"new_best_params", best_params)
            final_model = pipeline.named_steps['classifier'].__class__(**best_params)
        
        final_pipeline = Pipeline([
            ('preprocessor', pipeline.named_steps['preprocessor']),
            ('classifier', final_model)
        ])
        
        final_pipeline.fit(X, y)

        

        return OptunaWrapper(final_pipeline, study)

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        return train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )