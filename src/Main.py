from pathlib import Path
from datetime import datetime
import logging
import subprocess
import sys
from typing import Dict, Any

# ML models and utilities
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from experiment_runner import ExperimentRunner

# Custom modules
# from src.database.config import DatabaseConfig
from database.IPT_reader import IPTReader
# from src.preprocessing.config import PreprocessingConfig 
# from src.ml.config import ExperimentConfig
# from src.ml.experiment import MLExperiment

from database.database_config import DatabaseConfig

class Main:
    def main(ipt_table: str = "IPT1"):
        """Entry point for the experiment runner"""

        try:
            # Load the mebauth module
            subprocess.run('module load mebauth', shell=True, capture_output=True, text=True)

            runner = ExperimentRunner()

            runner.run_experiments()
        except Exception as e:
            logging.error(f"Experiment runner failed: {str(e)}")
            sys.exit(1)
    
    if __name__ == "__main__":
        main()

        
# class Main:
#     def main():
#         """Runs the experiemnt"""
#         config = DatabaseConfig()
#         reader = IPT1Reader(config)

#         try: 
#             df = reader.read_ipt1_data()
#             print("Data loaded sucessfully")
#             print(f"DataFrame shape: {df.shape}")
#         except Exception as e:
#             print(f"Failed to read data: {str(e)}")

            
#     if __name__ == "__main__":
#         main()