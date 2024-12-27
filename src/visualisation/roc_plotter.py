from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from visualisation.baseplotter import BasePlotter

class ROCPlotter(BasePlotter):
    def __init__(self, output_dir, title='ROC Curve'):
        """
        Initializes the ROCPlotter with a title.

        Parameters:
        title (str): Title of the plot.
        """
        self.title = title
        self.output_dir = output_dir

    def plot(self, y_true, y_scores):
        """
        Plots the ROC curve.

        Parameters:
        y_true (array-like): True binary labels.
        y_scores (array-like): Target scores, can either be probability estimates of the positive class, confidence values, or binary decisions.
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.title)
        plt.legend(loc="lower right")
        plt.show()

        # Save plot
        self.save_plot('roc_curve.pdf')