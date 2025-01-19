from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class ReportClassification:
    """Generates classification report"""

    @staticmethod
    def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
     
        """
        Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary containing accuracy and classification report
        """
        
        
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'classification_report': classification_report(y_true, y_pred, zero_division=1)
            }
            
            return metrics
            
        except Exception as e:
            raise ValueError(f"Error calculating metrics: {str(e)}")