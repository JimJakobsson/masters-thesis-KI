from pathlib import Path
from datetime import datetime
from typing import Counter, Dict, Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

import DatabaseReader
from evaluation.model_evaluator import ModelEvaluator
from experiment.datahandler import DataHandler
from experiment.experiment_config import AgeGroup, ExperimentConfig
from experiment.model_trainer_grid_search import ModelTrainerGridSearch
from experiment.model_trainer_optuna import ModelTrainerOptuna
from preprocessing import preprocessing_result
from preprocessing.data_preprocessor import DataPreprocessor
from preprocessing.preprocessing_result import PreprocessingResult
from imblearn.over_sampling import SMOTE


class Experiment: 
    def __init__(self,
                 model: BaseEstimator,
                 param_grid: Dict,
                 db_reader: DatabaseReader,
                 experiment_config: ExperimentConfig,
                 output_dir: Optional[Path] = None):
        
        self.config = experiment_config
        self.output_dir = Path(output_dir) if output_dir else Path('outputs')
        self.start_time = datetime.now()
        
        # Initialize components
        self.data_handler = DataHandler(self.config)
        # self.trainer = ModelTrainerGridSearch(self.config)
        self.trainer = ModelTrainerOptuna(self.config)

        self.preprocessor = DataPreprocessor()
        self.evaluator = ModelEvaluator(self.output_dir)
        
        # Store inputs
        self.model = model
        self.param_grid = param_grid
        self.db_reader = db_reader

    def _save_classification_report(self, report: str, best_params: str, optimal_threshold) -> None:
        """Saves classification report to a file"""
        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
            f.write("\n\nBest parameters found during grid search:\n")
            f.write(str(best_params))
            f.write("\n\nOptimal threshold for classification:\n")
            f.write(str(optimal_threshold))
        print("\nClassification report and best params saved to 'classification_report.txt'")

    def _evaluate_age_group(self,
                          data: pd.DataFrame,
                          age_group: AgeGroup,
                          grid_search: GridSearchCV,
                          preprocessor: ColumnTransformer,
                          suffix: str) -> None:
        """
        Evaluates model performance for a specific age group following the same procedure
        as the main run() method.
        """
        try:
            # Filter age group data
            age_data = self.data_handler.filter_age_group(data, age_group)
            
            # Create labels same way as run()
            labeled_data = self.preprocessor.create_labels(
                data=age_data,
                base_year=self.config.base_year,
                death_threshold=self.config.death_threshold
            )
            
            # Drop the same columns as in run()
            labeled_data = labeled_data.drop(columns=['age'])
            
            # Prepare data for training
            X, y = self.preprocessor.get_features_and_target(labeled_data)
            
            # Split before preprocessing (same as run())
            X_train, X_test, y_train, y_test = self.trainer.split_data(X, y)
            
            # Create preprocessor using training data
            preprocessor = self.preprocessor.create_preprocessor(X_train)
            
            # Create and train the complete pipeline
            pipeline = self.trainer.create_pipeline(preprocessor, self.model)
            age_grid_search = self.trainer.train_model(pipeline, self.param_grid, X_train, y_train)
            
            # Evaluate the model with optimal threshold
            result = self.evaluator.evaluate_model(
                age_grid_search, X_test, y_test, self.config.threshold
            )
            
            # Save classification report with age group suffix
            classification_report = result['classification_report']
            best_params = result['best_params']
            optimal_threshold = result['optimal_threshold']
            
            report_path = self.output_dir / f'classification_report{suffix}.txt'
            with open(report_path, 'w') as f:
                f.write(classification_report)
                f.write("\n\nBest parameters found during grid search:\n")
                f.write(str(best_params))
                f.write("\n\nOptimal threshold for classification:\n")
                f.write(str(optimal_threshold))
                
            print(f"\nClassification report and best params saved for age group {age_group.name}")
            
            # Calculate feature importance
            aggregated_shap, feature_importance_dataframe, feature_importance_abs_mean = self.evaluator.calculate_feature_importance(
                best_model=age_grid_search.best_estimator_,
                X_test=X_test,
            )
            
            # Plot all visualizations
            self.evaluator.plot_all(
                model=age_grid_search.best_estimator_,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                feature_importance_dataframe=feature_importance_dataframe,
                feature_importance_abs_mean=feature_importance_abs_mean,
                aggregated_shap=aggregated_shap,
                class_to_explain=1,
                output_suffix=suffix
            )
            
            print(f"\nCompleted evaluation for age group: {age_group.name}")
            
        except Exception as e:
            print(f"Error evaluating age group {age_group.name}: {str(e)}")
            raise
    
    def run(self) -> None:
        """
            Runs the complete experiment pipeline with proper preprocessing handling.
        """
        # Load and process data
        cache_path = Path("misc/combined_tables_ipt1.csv") 
        
        raw_data = self.db_reader.read_ipt_data(self.config.data_table, use_cache = False, cache_path=cache_path)
        raw_data = self.data_handler.calculate_ages(raw_data)
        labeled_data = self.preprocessor.create_labels(
            data=raw_data, 
            base_year=self.config.base_year, 
            death_threshold = self.config.death_threshold)
        #Drop features age_death and age
        #REMOVE LATER
        # labeled_data = labeled_data.drop(columns=['age_death', 'age'])
        labeled_data = labeled_data.drop(columns=['age'])

        #Prepare data for training
        X, y = self.preprocessor.get_features_and_target(labeled_data)

        #Split before preprocessing
        X_train, X_test, y_train, y_test = self.trainer.split_data(X, y)

        #Create the preprocessor using training data
        preprocessor = self.preprocessor.create_preprocessor(X_train)
        

        #Create and train the complete pipeline
        pipeline = self.trainer.create_pipeline(preprocessor, self.model)
        grid_search = self.trainer.train_model(pipeline, self.param_grid, X_train, y_train)

        #Evaluate the model with optimal threshold
        result = self.evaluator.evaluate_model(
            grid_search, X_test, y_test, self.config.threshold)
        
        classification_report = result['classification_report']
        best_params = result['best_params']
        optimal_threshold = result['optimal_threshold']
        
        self._save_classification_report(classification_report, best_params, optimal_threshold)

        aggregated_shap, feature_importance_dataframe, feature_importance_abs_mean = self.evaluator.calculate_feature_importance(
            best_model=grid_search.best_estimator_,
            X_test=X_test,
        )
        
        self.evaluator.plot_all(
        model=grid_search.best_estimator_,
        X_train=X_train,  # Pass raw training data for learning curves
        X_test=X_test,    # Pass raw test data for SHAP analysis
        y_train=y_train,
        y_test=y_test,
        feature_importance_dataframe=feature_importance_dataframe,
        feature_importance_abs_mean=feature_importance_abs_mean,
        aggregated_shap=aggregated_shap,
        class_to_explain=1,  # Typically 1 for binary classification
        output_suffix=''     # Empty for base model
    )

        if self.config.evaluate_age_groups:
            age_groups = [
                AgeGroup('50-59', 50, 59),
                AgeGroup('60-69', 60, 69),
                AgeGroup('70-79', 70, 79),
            ]

            for age_group in age_groups:
                self._evaluate_age_group(
                    data=raw_data,
                    age_group=age_group,
                    grid_search=grid_search,
                    preprocessor=preprocessor,
                    suffix=f"_{age_group.name}"
                )