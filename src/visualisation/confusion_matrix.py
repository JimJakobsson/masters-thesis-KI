import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Optional, Union
from visualisation.baseplotter import BasePlotter

class ConfusionMatrixPlotter(BasePlotter):
    def __init__(self, output_dir: str, title: str = 'Confusion Matrix'):
        """
        Initialize the ConfusionMatrixPlotter.

        Parameters:
        output_dir (str): Directory to save the plots
        title (str): Title of the plot
        """
        self.title = title
        self.output_dir = output_dir

    def plot(self, 
            y_true: Union[List, np.ndarray],
            y_pred: Union[List, np.ndarray],
            labels: Optional[List[str]] = None,
            cmap: str = 'Blues',
            figsize: tuple = (10, 8)):
        """
        Plot confusion matrix with both counts and percentages.

        Parameters:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names (if None, will use numerical labels)
        cmap: Color map for the plot
        figsize: Figure size (width, height) in inches
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create figure
        plt.figure(figsize=figsize)
        
        # Create annotated heatmap with both counts and percentages
        sns.heatmap(cm, annot=np.asarray([
            [f'{count}\n({percentage:.1%})' 
             for count, percentage in zip(row_counts, row_percentages)]
            for row_counts, row_percentages in zip(cm, cm_norm)
        ]), fmt='', cmap=cmap, square=True,
        xticklabels=labels, yticklabels=labels, cbar=True)

        # Customize plot
        plt.title(self.title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Rotate x-labels if they are strings
        if labels is not None and any(isinstance(label, str) for label in labels):
            plt.xticks(rotation=45, ha='right')

        # Add value counts to title
        value_counts = np.bincount(y_true)
        class_distribution = [f"Class {i}: {count}" for i, count in enumerate(value_counts)]
        plt.title(f"{self.title}\n({', '.join(class_distribution)})")

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save plot
        self.save_plot('confusion_matrix.pdf')
        plt.close()

    def plot_with_metrics(self,
                         y_true: Union[List, np.ndarray],
                         y_pred: Union[List, np.ndarray],
                         labels: Optional[List[str]] = None):
        """
        Plot confusion matrix with both counts, percentages and additional metrics.

        Parameters:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Compute metrics
        tn, fp, fn, tp = cm.ravel()
        metrics = {
            'Accuracy': (tp + tn) / (tp + tn + fp + fn),
            'Precision': tp / (tp + fp),
            'Recall': tp / (tp + fn),
            'F1 Score': 2 * tp / (2 * tp + fp + fn),
            'Specificity': tn / (tn + fp)
        }

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), 
                                      gridspec_kw={'width_ratios': [2, 1]})

        # Plot confusion matrix with counts and percentages
        sns.heatmap(cm, annot=np.asarray([
            [f'{count}\n({percentage:.1%})' 
             for count, percentage in zip(row_counts, row_percentages)]
            for row_counts, row_percentages in zip(cm, cm_norm)
        ]), fmt='', cmap='Blues', square=True,
        xticklabels=labels, yticklabels=labels, ax=ax1)

        ax1.set_title(self.title)
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')

        # Plot metrics
        metrics_colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#ff99cc']
        y_pos = np.arange(len(metrics))
        
        ax2.barh(y_pos, list(metrics.values()), color=metrics_colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(list(metrics.keys()))
        ax2.set_xlim(0, 1)
        ax2.set_title('Performance Metrics')
        
        # Add value labels on bars
        for i, v in enumerate(metrics.values()):
            ax2.text(v, i, f'{v:.3f}', va='center')

        plt.tight_layout()
        self.save_plot('confusion_matrix_with_metrics.pdf')
        plt.close()