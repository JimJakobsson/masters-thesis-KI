from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from visualisation.baseplotter import BasePlotter

class AUCPlotter(BasePlotter):
    def __init__(self, output_dir: str, title: str = 'AUC Plot'):
        """
        Initializes the AUCPlotter.

        Parameters:
        output_dir (str): Directory to save the plots
        title (str): Title of the plot
        """
        self.title = title
        self.output_dir = output_dir

    def plot(self, y_true, y_scores, n_bootstraps=1000):
        """
        Plots the AUC curve with confidence intervals.

        Parameters:
        y_true: True binary labels
        y_scores: Target scores or probability estimates
        n_bootstraps: Number of bootstrap samples for confidence intervals
        """
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # Calculate the main ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        main_auc = auc(fpr, tpr)

        # Initialize arrays for bootstrap calculations
        tprs = []
        aucs = []
        base_fpr = np.linspace(0, 1, 101)  # Common x-axis for all curves

        # Bootstrap to calculate confidence intervals
        rng = np.random.RandomState(42)
        for _ in range(n_bootstraps):
            # Sample with replacement
            indices = rng.randint(0, len(y_true), len(y_true))
            
            if len(np.unique(y_true[indices])) < 2:
                continue
                
            # Calculate ROC curve for bootstrap sample
            fpr_boot, tpr_boot, _ = roc_curve(y_true[indices], y_scores[indices])
            
            # Store AUC value
            aucs.append(auc(fpr_boot, tpr_boot))
            
            # Interpolate TPR values to common FPR axis
            tpr_interp = np.interp(base_fpr, fpr_boot, tpr_boot)
            tpr_interp[0] = 0.0  # Force start at 0
            tprs.append(tpr_interp)

        # Calculate confidence intervals
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std_tprs = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + 2*std_tprs, 1)
        tprs_lower = np.maximum(mean_tprs - 2*std_tprs, 0)

        # Calculate 95% CI for AUC
        auc_ci = np.percentile(aucs, [2.5, 97.5])

        # Create the plot
        plt.figure(figsize=(8, 6))
        
        # Plot main ROC curve
        plt.plot(fpr, tpr, 'b-', label=f'AUC = {main_auc:.3f}')
        
        # Plot confidence interval
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, 
                        color='grey', alpha=0.3, 
                        label=f'95% CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]')
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', label='Random')

        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        # Save plot
        self.save_plot('auc_plot.pdf')
        plt.close()