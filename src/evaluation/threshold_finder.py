import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

class ThresholdFinder:
    def __init__(self):
        pass

    # def find_optimal_threshold(self, y_true: pd.Series, y_prob: np.ndarray) -> float:
    #     """
    #     Find the optimal classification threshold for binary classification
    #     using F1 score as the metric to optimize.

    #     Args:
    #         y_true: True labels
    #         y_prob: Predicted probabilities for class 1

    #     Returns:
    #         float: Optimal threshold value
    #     """
    #     thresholds = np.linspace(0.1, 0.9, 100)  # Test thresholds from 0.1 to 0.9
    #     f1_scores = []

    #     for threshold in thresholds:
    #         y_pred = (y_prob > threshold).astype(int)
    #         f1 = f1_score(y_true, y_pred)
    #         f1_scores.append(f1)

    #     optimal_idx = np.argmax(f1_scores)
    #     optimal_threshold = thresholds[optimal_idx]

    #     # Print results
    #     print(f"\nOptimal threshold found: {optimal_threshold:.3f}")
    #     print(f"Best F1 score: {f1_scores[optimal_idx]:.3f}")

    #     return optimal_threshold
    
    def find_optimal_threshold(self, y_true: pd.Series, y_prob: np.ndarray) -> float:
        """
        Find the optimal classification threshold for binary classification
        using accuracy as the metric to optimize.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities for class 1

        Returns:
            float: Optimal threshold value
        """
        thresholds = np.linspace(0.1, 0.9, 1000)  # Test thresholds from 0.1 to 0.9
        accuracies = []

        for threshold in thresholds:
            y_pred = (y_prob > threshold).astype(int)
            accuracy = np.mean(y_pred == y_true)
            accuracies.append(accuracy)

        optimal_idx = np.argmax(accuracies)
        optimal_threshold = thresholds[optimal_idx]

        # Print results
        print(f"\nOptimal threshold found: {optimal_threshold:.3f}")
        print(f"Best accuracy: {accuracies[optimal_idx]:.3f}")

        return optimal_threshold