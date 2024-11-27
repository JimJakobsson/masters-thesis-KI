from dataclasses import dataclass
from typing import Any, Dict
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass

class ModelConfig:
    """Configuration for a machine learning model"""
    name: str
    model: BaseEstimator
    param_grid: Dict[str, Any]
    description: str

