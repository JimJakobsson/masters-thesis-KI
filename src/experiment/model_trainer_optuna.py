import optuna
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
import time
from typing import Any, Counter, Dict, Tuple
from sklearn.base import clone

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from experiment.experiment_config import ExperimentConfig
from experiment.optuna_wrapper import OptunaWrapper
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

class ModelTrainerOptuna:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._verbose_smote = True

    def print_distribution(self, X, y, stage):
        """Helper method to print class distribution"""
        if y is not None and stage in ["before SMOTE", "after SMOTE"] and self._verbose_smote:
            from collections import Counter
            counts = Counter(y)
            print(f"\nSamples {stage}: {dict(counts)} (total={len(y)})")

    class SMOTEWithPrinting(SMOTE):
        """Custom SMOTE class that prints distribution information"""
        def __init__(self, outer_instance, cv_fold=None, **kwargs):
            super().__init__(**kwargs)
            self.outer_instance = outer_instance
            self.cv_fold = cv_fold
            
        def fit_resample(self, X, y):
            self.outer_instance.print_distribution(X, y, "before SMOTE")
            X_resampled, y_resampled = super().fit_resample(X, y)
            self.outer_instance.print_distribution(X_resampled, y_resampled, "after SMOTE")
            return X_resampled, y_resampled

    def set_smote_verbose(self, verbose: bool):
        """Set whether SMOTE should print distribution information"""
        self._verbose_smote = verbose

    def create_pipeline(self, preprocessor, model, cv_fold=None) -> Pipeline:
        """Create a pipeline with preprocessing, SMOTE, and classification steps"""
        return Pipeline([
            ('preprocessor', preprocessor),
            ('smote', self.SMOTEWithPrinting(self, cv_fold=cv_fold, 
                                           random_state=42, 
                                           sampling_strategy='minority')),
            ('classifier', model)
        ])

    def suggest_parameter(self, trial, param_name: str, param_range: Any) -> Any:
        """Suggest a parameter value based on its type and range"""
        if isinstance(param_range, tuple) and len(param_range) == 2:
            if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                return trial.suggest_int(param_name, param_range[0], param_range[1])
            elif isinstance(param_range[0], float) or isinstance(param_range[1], float):
                return trial.suggest_float(param_name, param_range[0], param_range[1])
        elif isinstance(param_range, list):
            if not param_range:
                raise ValueError(f"Empty parameter range for {param_name}")
            if len(param_range) == 1:
                return param_range[0]
            return trial.suggest_categorical(param_name, param_range)
        return param_range

    def train_model(self,
                   pipeline: Pipeline,
                   param_grid: Dict[str, Any],
                   X: pd.DataFrame,
                   y: pd.Series,
                   n_trials: int = 10) -> Any:
        """Train model using Optuna for hyperparameter optimization"""
        
        n_trials = self.config.n_trials_optuna
        model_class = pipeline.named_steps['classifier'].__class__

        # Use StratifiedKFold for better fold balance
        cv = StratifiedKFold(n_splits=self.config.cv_folds, 
                            shuffle=True, 
                            random_state=self.config.random_state)
        
        # Apply preprocessing once before optimization
        preprocessor = clone(pipeline.named_steps['preprocessor'])
        X_preprocessed = preprocessor.fit_transform(X)
        
        # Store feature names if available
        feature_names = None
        if hasattr(preprocessor, 'get_feature_names_out'):
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                pass

        def objective(trial):
            try:
                print(f"\nTrial {trial.number}")
                
                # Handle different classifier types
                if isinstance(pipeline.named_steps['classifier'], BaggingClassifier):
                    if 'param_suggest' in param_grid:
                        params = param_grid['param_suggest'](trial)
                    else:
                        params = {k.replace('classifier__', ''): self.suggest_parameter(trial, k, v)
                                for k, v in param_grid.items()}
                    model = model_class(
                        estimator=pipeline.named_steps['classifier'].estimator,
                        **params
                    )
                
                elif isinstance(pipeline.named_steps['classifier'], StackingClassifier):
                    base_config = {
                        'estimators': pipeline.named_steps['classifier'].estimators,
                        'stack_method': pipeline.named_steps['classifier'].stack_method,
                    }
                    final_estimator_params = {k.replace('final_estimator__', ''): 
                                            self.suggest_parameter(trial, k, v)
                                            for k, v in param_grid.items()}
                    base_config['final_estimator'] = RandomForestClassifier(**final_estimator_params)
                    model = model_class(**base_config)
                
                elif isinstance(pipeline.named_steps['classifier'], VotingClassifier):
                    suggested_params = param_grid['param_suggest'](trial)
                    base_config = {
                        'estimators': pipeline.named_steps['classifier'].estimators,
                        'n_jobs': -1,
                        'voting': suggested_params['voting'],
                        'weights': suggested_params['weights']
                    }
                    model = model_class(**base_config)
                
                else:
                    params = {k.replace('classifier__', ''): self.suggest_parameter(trial, k, v)
                            for k, v in param_grid.items()
                            if k not in ['param_suggest', 'weight_config']}
                    
                    # Remove warm_start parameter as we'll handle incremental fitting differently
                    params.pop('warm_start', None)
                    
                    model = model_class(**params)

                print(f"Trial {trial.number} parameters:", params)

                # Initialize metric storage
                fold_scores = []
                fold_feature_importances = []
                fold_precision = []
                fold_recall = []
                fold_f1 = []
                
                # Cross-validation with multiple metrics
                for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_preprocessed, y)):
                    # Split data for this fold
                    if isinstance(X_preprocessed, np.ndarray):
                        X_train_fold = X_preprocessed[train_idx]
                        X_val_fold = X_preprocessed[val_idx]
                    else:
                        X_train_fold = X_preprocessed[X.index[train_idx]]
                        X_val_fold = X_preprocessed[X.index[val_idx]]
                    
                    y_train_fold = y.iloc[train_idx]
                    y_val_fold = y.iloc[val_idx]
                    
                    # Apply SMOTE only to training fold
                    smote = SMOTE(random_state=42 + fold_idx, 
                                sampling_strategy='minority',
                                k_neighbors=min(5, sum(y_train_fold == 1) - 1))
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)
                    
                    # Train model - handle incremental fitting if needed
                    if hasattr(model, 'n_estimators'):
                        n_estimators = model.n_estimators
                        model.n_estimators = 1  # Start with 1 estimator
                        model.fit(X_train_resampled, y_train_resampled)
                        
                        # Incrementally add estimators
                        for _ in range(2, n_estimators + 1):
                            model.n_estimators += 1
                            model.fit(X_train_resampled, y_train_resampled)
                    else:
                        model.fit(X_train_resampled, y_train_resampled)
                    
                    # Calculate metrics
                    y_pred = model.predict(X_val_fold)
                    fold_scores.append(accuracy_score(y_val_fold, y_pred))
                    fold_precision.append(precision_score(y_val_fold, y_pred))
                    fold_recall.append(recall_score(y_val_fold, y_pred))
                    fold_f1.append(f1_score(y_val_fold, y_pred))
                    
                    if hasattr(model, 'feature_importances_'):
                        fold_feature_importances.append(model.feature_importances_)
                
                # Calculate and print metrics
                mean_metrics = {
                    'Accuracy': (np.mean(fold_scores), np.std(fold_scores)),
                    'Precision': (np.mean(fold_precision), np.std(fold_precision)),
                    'Recall': (np.mean(fold_recall), np.std(fold_recall)),
                    'F1-score': (np.mean(fold_f1), np.std(fold_f1))
                }
                
                print("\nCross-validation metrics:")
                for metric_name, (mean_val, std_val) in mean_metrics.items():
                    print(f"{metric_name}: {mean_val:.4f} (±{std_val:.4f})")
                
                # Handle feature importances
                if fold_feature_importances:
                    importances_array = np.array(fold_feature_importances)
                    mean_importance = np.mean(importances_array, axis=0)
                    std_importance = np.std(importances_array, axis=0)
                    
                    top_indices = np.argsort(mean_importance)[-5:][::-1]
                    print("\nTop 5 feature importances:")
                    for idx in top_indices:
                        feature_name = feature_names[idx] if feature_names is not None else f"Feature {idx}"
                        print(f"{feature_name}: {mean_importance[idx]:.6f} (±{std_importance[idx]:.6f})")
                
                return mean_metrics['F1-score'][0]  # Optimize for F1-score
                
            except Exception as e:
                print(f"Trial failed due to error: {str(e)}")
                raise

        # Time start
        time_start = time.time()
        
        # Create and run study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(
                seed=self.config.random_state,
                n_startup_trials=10,
                multivariate=True
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )

        print(f"\nStarting optimization with {n_trials} trials...")
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[lambda study, trial: print(f"\nBest value so far: {study.best_value:.4f}")]
        )

        print(f"\nOptimization completed in {time.time() - time_start:.2f} seconds")
        print(f"Best parameters: {study.best_params}")
        print(f"Best score: {study.best_value:.4f}")

        # Create and train final model
        if isinstance(pipeline.named_steps['classifier'], BaggingClassifier):
            if 'param_suggest' in param_grid:
                best_params = param_grid['param_suggest'](study.best_trial)
            else:
                best_params = {k.replace('classifier__', ''): v 
                            for k, v in study.best_params.items()}
            final_model = model_class(
                estimator=pipeline.named_steps['classifier'].estimator,
                **best_params
            )
        elif isinstance(pipeline.named_steps['classifier'], StackingClassifier):
            base_config = {
                'estimators': pipeline.named_steps['classifier'].estimators,
                'stack_method': pipeline.named_steps['classifier'].stack_method,
                'final_estimator': RandomForestClassifier(**{
                    k.replace('final_estimator__', ''): v 
                    for k, v in study.best_params.items()
                })
            }
            final_model = model_class(**base_config)
        elif isinstance(pipeline.named_steps['classifier'], VotingClassifier):
            final_model = model_class(
                estimators=pipeline.named_steps['classifier'].estimators,
                n_jobs=-1,
                voting=study.best_params.get('voting', 'soft'),
                weights=study.best_params.get('weights', None)
            )
        else:
            final_model = model_class(**{
                k.replace('classifier__', ''): v 
                for k, v in study.best_params.items()
                if k != 'warm_start'  # Exclude warm_start from final model
            })

        # Train final pipeline
        final_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, sampling_strategy='minority')),
            ('classifier', final_model)
        ])
        
        print("\nTraining final model on full dataset...")
        final_pipeline.fit(X, y)

        return OptunaWrapper(final_pipeline, study)

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data into train and test sets"""
        return train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )