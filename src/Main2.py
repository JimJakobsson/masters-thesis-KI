from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any

# ML models and utilities
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Custom modules
from src.database.config import DatabaseConfig
from src.database.readers import IPT1Reader
from src.preprocessing.config import PreprocessingConfig 
from src.ml.config import ExperimentConfig
from src.ml.experiment import MLExperiment

from database.database_config import DatabaseConfig


class Main:
    def main():
        """Runs the experiemnt"""
        config = DatabaseConfig()
        reader = IPT1Reader(config)

        try: 
            df = reader.read_ipt1_data()
            print("Data loaded sucessfully")
            print(f"DataFrame shape: {df.shape}")
        except Exception as e:
            print(f"Failed to read data: {str(e)}")

            
    if __name__ == "__main__":
        main()