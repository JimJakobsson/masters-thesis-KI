from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass

class ModelConfig:
    """Configuration for a machine learning model"""
    name: str
    model: BaseEstimator
    description: str
    param_grid: Optional[Dict[str, Any]] = None
    param_ranges: Optional[Dict[str, Any]] = None
    param_suggest: Optional[Callable[[Any], Dict[str, Any]]] = None

