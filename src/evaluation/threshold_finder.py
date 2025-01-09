import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

class ThresholdFinder:
    def __init__(self):
        pass
    
    def find_optimal_threshold(self, y_true: pd.Series, y_prob: np.ndarray) -> float:
        """
        Find the optimal classification threshold for binary classification
        using geometric mean of TPR and TNR (equivalent to geometric mean method in ROC plotter).

        Args:
            y_true: True labels
            y_prob: Predicted probabilities for class 1

        Returns:
            float: Optimal threshold value
        """
        # Calculate ROC curve points
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Calculate geometric mean (equivalent to sqrt(TPR * (1-FPR)))
        geometric_mean = np.sqrt(tpr * (1-fpr))
        
        # Find optimal threshold
        optimal_idx = np.argmax(geometric_mean)
        optimal_threshold = thresholds[optimal_idx]

        # Calculate final scores at optimal threshold
        y_pred = (y_prob > optimal_threshold).astype(int)
        final_accuracy = np.mean(y_pred == y_true)
        
        # Print results
        print(f"\nOptimal threshold found: {optimal_threshold:.3f}")
        print(f"Accuracy at optimal threshold: {final_accuracy:.3f}")
        # print(f"Geometric mean at optimal threshold: {geometric_mean[optimal_idx]:.3f}")

        return optimal_threshold