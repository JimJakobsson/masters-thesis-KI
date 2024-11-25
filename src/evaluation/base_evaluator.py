from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate_model(self, model: Any, X_test: Any, y_test: Any) -> Dict:
        """Evaluate model performance."""
        pass

    @abstractmethod
    def calculate_feature_importance(self, model: Any, X_test: Any) -> Tuple[Dict, Any]:
        """Calculate feature importance."""
        pass