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

    # Base year is the year when the data was collected
    base_year: int = 1987 #IPT1 

    # base_year: int = 1990 #IPT2
    death_threshold: int = 20 # Threshold for death, in number of years after base year

    data_table: str = "IPT1"
    # data_table: str = "IPT2"

    test_size: float = 0.2 # Test size for train-test split

    random_state: int = 42 # Random state for reproducibility

    cv_folds: int = 5 # Number of cross-validation folds

    n_trials_optuna: int = 300 # Number of trials for Optuna

    evaluate_age_groups: bool = False # Whether to evaluate age groups

    threshold: float = None # Threshold for classification. Higher values benefits class 0, lower values benefits class 1
    
    # TODO: Determine if imputation and SMOTE should be used
    # imputation_and_SMOTE: bool = True #Whether to use imputation and SMOTE
    # This variable will engage the imputation and SMOTE in the pipeline