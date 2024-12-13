import optuna
import time
from typing import Any, Dict, Tuple
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
        
        def objective(trial):
            try:
                # Get base model class
                model_class = pipeline.named_steps['classifier'].__class__
                
                # Suggest parameters
                params = {}
                for param_name, param_range in param_grid.items():
                    clean_name = param_name.replace('classifier__', '')
                    params[clean_name] = self.suggest_parameter(trial, clean_name, param_range)
                
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
        print(f"Best score: {study.best_value:.4f}")

        # Create final model with best parameters
        best_params = {k.replace('classifier__', ''): v for k, v in study.best_params.items()}
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