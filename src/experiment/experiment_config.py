from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

@dataclass
class AgeGroup:
    name: str
    min_age: int
    max_age: int

@dataclass
class ExperimentConfig:
    excluded_columns: List[str] = ('labels', 'twinnr', 'death_yrmon', 'age_death', 'punching')
    base_year: int = 1987
    # base_year: int = 1990
    death_threshold: int = 10
    data_table: str = "IPT1"
    # data_table: str = "IPT2"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_trials_optuna: int = 10
    threshold: float = 0.5 # Threshold for classification. Higher values benefits class 0, lower values benefits class 1
    # TODO: Determine if imputation and SMOTE should be used
    # imputation_and_SMOTE: bool = True #Whether to use imputation and SMOTE