import pandas as pd
from typing import Tuple

from experiment.experiment_config import ExperimentConfig, AgeGroup
from ..preprocessing.preprocessing_result import PreprocessingResult

class DataHandler:
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        drop_cols = [col for col in self.config.excluded_columns if col in df.columns]
        X = df.drop(columns=drop_cols)
        y = df['labels']
        return X, y
    
    def calculate_ages(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['age'] = self.config.base_year - pd.to_numeric(
            df['birthdate1'].astype(str).str[:4],
            errors='coerce'
        )
        return df
    
    def filter_age_group(self, df: pd.DataFrame, age_group: AgeGroup) -> pd.DataFrame:
        return df[
            (df['age'] >= age_group.min_age) & 
            (df['age'] <= age_group.max_age)
        ]