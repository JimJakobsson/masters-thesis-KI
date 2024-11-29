from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

import DatabaseReader
from evaluation.model_evaluator import ModelEvaluator
from experiment.datahandler import DataHandler
from experiment.experiment_config import AgeGroup, ExperimentConfig
from experiment.model_trainer import ModelTrainer
from preprocessing.data_preprocessor import DataPreprocessor
from preprocessing.preprocessing_result import PreprocessingResult

class Experiment: 
    def __init__(self,
                 model: BaseEstimator,
                 param_grid: Dict,
                 db_reader: DatabaseReader,
                 output_dir: Optional[Path] = None,
                 experiment_config: Optional[ExperimentConfig] = None):
        
        self.config = experiment_config
        self.output_dir = Path(output_dir) if output_dir else Path('outputs')
        self.start_time = datetime.now()
        
        # Initialize components
        self.data_handler = DataHandler(self.config)
        self.trainer = ModelTrainer(self.config)
        self.preprocessor = DataPreprocessor()
        self.evaluator = ModelEvaluator(self.output_dir)
        
        # Store inputs
        self.model = model
        self.param_grid = param_grid
        self.db_reader = db_reader
    
    def evaluate_age_group(self,
                      data: pd.DataFrame,
                      age_group: AgeGroup,
                      grid_search: GridSearchCV,
                      prep_result: PreprocessingResult) -> None:
        """
        Evaluate model performance for specific age group.
        
        Args:
            data: Raw input data
            age_group: Age group configuration
            grid_search: Fitted grid search object
            prep_result: Original preprocessing result containing fitted preprocessor
        """
        try:
            # Filter age group data
            age_data = self.data_handler.filter_age_group(data, age_group)
            
            # Create labels
            labeled_age_data = self.preprocessor.create_labels(age_data)
            
            # Get features and target using the same preprocessing steps
            X_group, y_group = self.preprocessor.get_features_and_target(labeled_age_data)
            
            # Split the age group data
            X_train_group, X_test_group, y_train_group, y_test_group = self.trainer.split_data(
                X_group, y_group
            )
            
            # Process training and test data using the original fitted preprocessor
            train_result = self.preprocessor.process(
                X_train_group,
                fit=False  # Use already fitted preprocessor
            )
            
            test_result = self.preprocessor.process(
                X_test_group,
                fit=False
            )
            
            # Fit grid search on age group data
            grid_search.fit(train_result.X, y_train_group)
            
            # Evaluate and plot results
            output_suffix = f"_{age_group.name}"
            self.evaluator.evaluate_model(
                grid_search,
                test_result.X,
                y_test_group
            )
            
            #calculate feature importance
            self.evaluator.calculate_feature_importance(
                grid_search.best_estimator_,
                test_result.X,
                prep_result.preprocessor
            )

            #plot all
            self.evaluator.plot_all(
                grid_search,
                test_result.X,
                y_test_group,
                prep_result.preprocessor,
                output_suffix
            )
            
            print(f"\nCompleted evaluation for age group: {age_group.name}")
            
        except Exception as e:
            print(f"Error evaluating age group {age_group.name}: {str(e)}")
            raise
    
    def run(self) -> None:
        # Load and process data
        raw_data = self.db_reader.read_ipt1_data()
        raw_data = self.data_handler.calculate_ages(raw_data)
        labeled_data = self.preprocessor.create_labels(raw_data)
        
        X, y = self.preprocessor.get_features_and_target(labeled_data)

        #Split before preprocessing
        X_train, X_test, y_train, y_test = self.trainer.split_data(X, y)

        #Process training data. Fit and transform
        train_result = self.preprocessor.process(X_train, fit=True)

        #Process test data. Only transform
        test_result = self.preprocessor.process(X_test, fit=False)

        pipeline = self.trainer.create_pipeline(train_result.preprocessor, self.model)
        grid_search = self.trainer.train_model(pipeline, self.param_grid, train_result.X, y_train)

        #Evaluate base model
        self.evaluator.evaluate_model(grid_search, test_result.X, y_test)
        self.evaluator.calculate_feature_importance(
            grid_search.best_estimator_, test_result.X, train_result.preprocessor)
        self.evaluator.plot_all(grid_search, test_result.X, y_test, train_result.preprocessor)

        age_groups = [
            AgeGroup('50-59', 50, 59),
            AgeGroup('60-69', 60, 69),
            AgeGroup('70-79', 70, 79),
        ]

        for age_group in age_groups:
            self.evaluate_age_group(
                data=raw_data,
                age_group=age_group,
                grid_search=grid_search,
                prep_result=train_result
            )