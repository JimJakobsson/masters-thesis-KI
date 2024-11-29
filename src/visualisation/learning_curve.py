import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import learning_curve
from .baseplotter import BasePlotter

class LearningCurvePlotter(BasePlotter):
    """Handles learning curve visualization"""
    
    def plot(self, 
            model: BaseEstimator,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
            output_suffix: str = '') -> None:
        """
        Plot learning curves showing training and validation performance.
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            output_suffix: Suffix for output filename
        """
        plt.figure(figsize=self.config.FIGURE_SIZES['learning'], 
                  dpi=self.config.DPI)
        # Combine data for cross-validation
        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])
        # Calculate curves
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )
        
        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot curves
        self._plot_curve(train_sizes, train_mean, train_std, 'Training')
        self._plot_curve(train_sizes, val_mean, val_std, 'Validation')
        
        # Customize plot
        self._customize_plot()
        
        # Save plot
        self.save_plot('learning_curve.pdf', output_suffix)
    
    def _plot_curve(self, sizes: np.ndarray, mean: np.ndarray, 
                    std: np.ndarray, label: str) -> None:
        """Helper method to plot a single curve with confidence interval"""
        color = (self.config.COLORS['primary'] if label == 'Training' 
                else self.config.COLORS['secondary'])
        
        plt.plot(sizes, mean, 'o-', label=label, 
                color=color, linewidth=2, markersize=8)
        plt.fill_between(
            sizes,
            mean - std,
            mean + std,
            alpha=0.15,
            color=color
        )
    
    def _customize_plot(self) -> None:
        """Customize plot appearance"""
        plt.xlabel('Training Examples', 
                  fontsize=self.config.FONT_SIZES['label'])
        plt.ylabel('Accuracy Score', 
                  fontsize=self.config.FONT_SIZES['label'])
        plt.title('Learning Curve', 
                 fontsize=self.config.FONT_SIZES['title'],
                 pad=20)
        plt.legend(loc='lower right', 
                  fontsize=self.config.FONT_SIZES['legend'])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1.1)