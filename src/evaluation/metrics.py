from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    average_precision_score
)
from dataclasses import dataclass

@dataclass
class ClassificationMetrics:
    """Container for classification metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    avg_precision: float
    confusion_matrix: np.ndarray
    classification_report: str
    probability_stats: Dict[str, float]

class Metrics:
    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        average: str = 'binary'
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities for the positive class
            average: Averaging strategy for multi-class metrics
            
        Returns:
            Dictionary containing various classification metrics
        """
        try:
            # Basic classification metrics
            metrics = ClassificationMetrics(
                accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, average=average),
                recall=recall_score(y_true, y_pred, average=average),
                f1=f1_score(y_true, y_pred, average=average),
                roc_auc=roc_auc_score(y_true, y_prob),
                avg_precision=average_precision_score(y_true, y_prob),
                confusion_matrix=confusion_matrix(y_true, y_pred),
                classification_report=classification_report(y_true, y_pred),
                probability_stats={
                    'mean': float(np.mean(y_prob)),
                    'std': float(np.std(y_prob)),
                    'min': float(np.min(y_prob)),
                    'max': float(np.max(y_prob)),
                    'median': float(np.median(y_prob))
                }
            )
            
            # Convert to dictionary
            metrics_dict = {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1,
                'roc_auc': metrics.roc_auc,
                'average_precision': metrics.avg_precision,
                'confusion_matrix': metrics.confusion_matrix.tolist(),
                'classification_report': metrics.classification_report,
                'probability_stats': metrics.probability_stats
            }
            
            # Calculate additional metrics from confusion matrix
            tn, fp, fn, tp = metrics.confusion_matrix.ravel()
            additional_metrics = {
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
                'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,  # False Negative Rate
                'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False Positive Rate
            }
            metrics_dict.update(additional_metrics)
            
            return metrics_dict
        
        except Exception as e:
            raise ValueError(f"Error calculating metrics: {str(e)}")

    def format_metrics_for_display(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics dictionary into a readable string.
        
        Args:
            metrics: Dictionary of classification metrics
            
        Returns:
            Formatted string for display
        """
        display_order = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'roc_auc', 'average_precision', 'specificity', 'npv'
        ]
        
        formatted_lines = ["Classification Metrics:"]
        formatted_lines.append("-" * 50)
        
        # Add main metrics
        for metric in display_order:
            if metric in metrics:
                formatted_lines.append(f"{metric:20}: {metrics[metric]:.4f}")
        
        # Add probability statistics
        if 'probability_stats' in metrics:
            formatted_lines.append("\nProbability Statistics:")
            formatted_lines.append("-" * 50)
            for stat, value in metrics['probability_stats'].items():
                formatted_lines.append(f"{stat:20}: {value:.4f}")
        
        # Add confusion matrix
        if 'confusion_matrix' in metrics:
            formatted_lines.append("\nConfusion Matrix:")
            formatted_lines.append("-" * 50)
            formatted_lines.append("[[TN, FP],")
            formatted_lines.append(" [FN, TP]]")
            formatted_lines.append(str(metrics['confusion_matrix']))
        
        # Add classification report
        if 'classification_report' in metrics:
            formatted_lines.append("\nDetailed Classification Report:")
            formatted_lines.append("-" * 50)
            formatted_lines.append(metrics['classification_report'])
        
        return "\n".join(formatted_lines)