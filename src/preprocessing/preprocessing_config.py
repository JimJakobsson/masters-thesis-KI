from dataclasses import dataclass
from typing import Dict

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing parameters"""
    NULL_THRESHOLD: float = 0.8
    MAX_UNIQUE_VALUES_FOR_CATEGORICAL: int = 10
    MIN_COUNT_FOR_CATEGORICAL: int = 10
    MAX_CATEGORIES_PER_FEATURE: int = 10