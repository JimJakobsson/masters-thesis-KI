from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from visualisation.baseplotter import BasePlotter

class ROCPlotter(BasePlotter):
    

    def find_optimal_threshold(self, fpr, tpr, thresholds, method='youden'):
        """
        Find the optimal threshold using different methods.
        """
        if method == 'youden':
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
        elif method == 'geometric':
            geometric_mean = np.sqrt(tpr * (1-fpr))
            optimal_idx = np.argmax(geometric_mean)
        elif method == 'closest_to_perfect':
            distances = np.sqrt(fpr**2 + (1-tpr)**2)
            optimal_idx = np.argmin(distances)
        else:
            raise ValueError("Method must be one of ['youden', 'geometric', 'closest_to_perfect']")
        
        return thresholds[optimal_idx], optimal_idx

    def plot(self, y_true, y_scores, show_thresholds=True):
        """
        Plots the ROC curve with optimal thresholds.
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=self.config.FIGURE_SIZES['roc'],)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        
        if show_thresholds:
            # Define methods with proper color codes
            methods = {
                # 'youden': ('r', 'Youden'),  # 'r' for red
                'geometric': ('g', 'Geometric Mean'),  # 'g' for green
                # 'closest_to_perfect': ('b', 'Closest to Perfect')  # 'b' for blue
            }
            
            for method, (color, name) in methods.items():
                opt_threshold, opt_idx = self.find_optimal_threshold(
                    fpr, tpr, thresholds, method)
                
                # Plot optimal point using proper matplotlib format
                plt.plot(fpr[opt_idx], tpr[opt_idx], f'{color}o',  # Use single character color code
                        label=f'{name} (threshold={opt_threshold:.2f})')
                
                # Add annotation
                plt.annotate(f'{opt_threshold:.2f}',
                           (fpr[opt_idx], tpr[opt_idx]),
                           xytext=(10, 10),
                           textcoords='offset points',
                           color=color)

        # Plot random classifier line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Save plot
        self.save_plot('roc_curve.pdf')
        plt.close()