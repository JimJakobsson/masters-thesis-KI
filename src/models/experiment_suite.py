import os
from typing import List, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging

from experiment.experiment_config import ExperimentConfig

from .model_selector import ModelSelector
from experiment.experiment import Experiment
from database.reader import DatabaseReader

from pathlib import Path

class ExperimentSuite:
    """Runs experiments with multiple models"""
    
    def __init__(self, 
                 db_reader: DatabaseReader,
                 output_dir: Path,
                 config: Optional[ExperimentConfig] = None,
                 models: Optional[List[str]] = None):
        self.db_reader = db_reader
        self.output_dir = output_dir
        self.confg = config
        self.model_selector = ModelSelector()
        self.models = models or self.model_selector.get_available_models()
    
    def run_experiments(self) -> None:
        """Run experiments for all specified models"""
        for model_name in self.models:
            try:
                logging.info(f"\nStarting experiment with {model_name}")
                
                # Get model configuration
                model_config = self.model_selector.get_model_config(model_name)
                
                # Create model-specific output directory
                model_output_dir = Path(os.path.join(self.output_dir, model_name))
                model_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Run experiment
                experiment = Experiment(
                    model=model_config.model,
                    param_grid=model_config.param_grid,
                    db_reader=self.db_reader,
                    output_dir=model_output_dir
                )
                
                experiment.run()
                logging.info(f"Completed experiment with {model_name}")
                
            except Exception as e:
                logging.error(f"Error in {model_name} experiment: {str(e)}", exc_info=True)