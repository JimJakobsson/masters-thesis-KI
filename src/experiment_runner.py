from datetime import datetime
import os
import logging
import sys
from typing import List, Optional

import DatabaseReader
from database.IPT1_reader import IPT1Reader
from database.database_config import DatabaseConfig
from experiment.experiment_config import ExperimentConfig
from models.experiment_suite import ExperimentSuite
from models.model_selector import ModelSelector

class ExperimentRunner:
    """Manages the execution of machine learning experiments"""
    
    def __init__(self):
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.output_dir = os.path.join('outputs', self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True) # Create output directory, if it does not exist
        
        # Initialize components
        self.setup_logging()
        self.model_selector = ModelSelector()
    
    def setup_logging(self) -> None:
        """Configure logging with both file and console output"""
        logging_path = os.path.join(self.output_dir, 'experiment.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logging_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def setup_database(self) -> DatabaseReader:
        """Configure and return database reader"""
        try:
            db_config = DatabaseConfig(
                server='kosmos',
                database='SATSA_ARKIV'
            )
            return IPT1Reader(db_config)
        except Exception as e:
            logging.error(f"Database setup failed: {str(e)}")
            raise
    
    def create_experiment_config(self) -> ExperimentConfig:
        """Create configuration for experiments"""
        return ExperimentConfig(
            # excluded_columns=['labels', 'twinnr', 'death_yrmon', 'age_death'],
            # base_year=1985,
            # test_size=0.2,
            # random_state=42,
            # cv_folds=3
        )
    def list_available_models(self) -> None:
        """Print information about available models"""
        logging.info("\nAvailable Models:")
        for model_name in self.model_selector.get_available_models():
            description = self.model_selector.get_model_description(model_name)
            logging.info(f"- {model_name}: {description}")
            
    def run_experiments(self, selected_models: Optional[List[str]] = None) -> None:
        """Run the experiment suite with selected models"""
        # Setup database connection
        db_reader = self.setup_database()
        
        # Create experiment configuration
        experiment_config = self.create_experiment_config()
        
        # Initialize and run experiment suite
        suite = ExperimentSuite(
            db_reader=db_reader,
            output_dir=self.output_dir,
            config=experiment_config,
            models=selected_models
        )
        
        # Display available models
        self.list_available_models()
        
        # Run experiments
        logging.info("\nStarting experiments...")
        try:
            suite.run_experiments()
            logging.info("All experiments completed successfully")
        except Exception as e:
            logging.error(f"Experiment execution failed: {str(e)}", exc_info=True)
        finally:
            logging.info(f"\nResults and log file saved in: {self.output_dir}")
    