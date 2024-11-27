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
                 config: Optional[ExperimentConfig] = None):
        
        self.config = ExperimentConfig()
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
        
        age_data = self.data_handler.filter_age_group(data, age_group)
        X_group, y_group = self.data_handler.prepare_data(age_data)
        X_train, X_test, y_train, y_test = self.trainer.split_data(X_group, y_group)
        
        grid_search.fit(X_train, y_train)
        
        output_suffix = f"_{age_group.name}"
        self.evaluator.evaluate_model(grid_search, X_test, y_test)
        self.evaluator.calculate_and_plot_all(
            grid_search.best_estimator_,
            X_test,
            X_group,
            y_group,
            prep_result.preprocessor,
            output_suffix
        )
    
    def run(self) -> None:
        # Load and process data
        raw_data = self.db_reader.read_ipt1_data()
        raw_data = self.data_handler.calculate_ages(raw_data)
        prep_result = self.preprocessor.process(raw_data)
        
        # Train base model
        X, y = prep_result.X, prep_result.y
        X_train, X_test, y_train, y_test = self.trainer.split_data(X, y)

        pipeline = self.trainer.create_pipeline(prep_result.preprocessor, self.model)
        grid_search = self.trainer.train_model(pipeline, self.param_grid, X_train, y_train)

        #Evaluate base model
        self.evaluator.evaluate_model(grid_search, X_test, y_test)
        self.evaluator.calculate_and_plot_all(
            grid_search.best_estimator_,
            X_test,
            X,
            y,
            prep_result.preprocessor
        )

        # Evaluate age groups
        age_groups = [
            AgeGroup('50-59', 50, 59),
            AgeGroup('60-69', 60, 69),
            AgeGroup('70-79', 70, 79),
        ]

        for age_group in age_groups:
            self.evaluate_age_group(raw_data, age_group, grid_search, prep_result)