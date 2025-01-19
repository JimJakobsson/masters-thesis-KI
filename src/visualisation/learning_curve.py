# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.base import BaseEstimator
# from sklearn.model_selection import learning_curve
# from .baseplotter import BasePlotter

# class LearningCurvePlotter(BasePlotter):
#     """Handles learning curve visualization"""
    
#     def plot(self, 
#             model: BaseEstimator,
#             X_train: pd.DataFrame,
#             X_test: pd.DataFrame,
#             y_train: pd.Series,
#             y_test: pd.Series,
#             output_suffix: str = '') -> None:
#         """
#         Plot learning curves showing training and validation performance.
        
#         Args:
#             model: Trained model
#             X_train: Training features
#             X_test: Test features
#             y_train: Training labels
#             y_test: Test labels
#             output_suffix: Suffix for output filename
#         """
#         plt.figure(figsize=self.config.FIGURE_SIZES['learning'], 
#                   dpi=self.config.DPI)
#         # Combine data for cross-validation
#         X = pd.concat([X_train, X_test])
#         y = pd.concat([y_train, y_test])
#         # Calculate curves
#         train_sizes = np.linspace(0.1, 1.0, 10)
#         train_sizes, train_scores, val_scores = learning_curve(
#             model, X, y,
#             train_sizes=train_sizes,
#             cv=5,
#             n_jobs=-1,
#             scoring='accuracy'
#         )
        
#         # Calculate statistics
#         train_mean = np.mean(train_scores, axis=1)
#         train_std = np.std(train_scores, axis=1)
#         val_mean = np.mean(val_scores, axis=1)
#         val_std = np.std(val_scores, axis=1)
        
#         # Plot curves
#         self._plot_curve(train_sizes, train_mean, train_std, 'Training')
#         self._plot_curve(train_sizes, val_mean, val_std, 'Validation')
        
#         # Customize plot
#         self._customize_plot()
        
#         # Save plot
#         self.save_plot('learning_curve.pdf', output_suffix)
    
#     def _plot_curve(self, sizes: np.ndarray, mean: np.ndarray, 
#                     std: np.ndarray, label: str) -> None:
#         """Helper method to plot a single curve with confidence interval"""
#         color = (self.config.COLORS['primary'] if label == 'Training' 
#                 else self.config.COLORS['secondary'])
        
#         plt.plot(sizes, mean, 'o-', label=label, 
#                 color=color, linewidth=2, markersize=8)
#         plt.fill_between(
#             sizes,
#             mean - std,
#             mean + std,
#             alpha=0.15,
#             color=color
#         )
    
#     def _customize_plot(self) -> None:
#         """Customize plot appearance"""
#         plt.xlabel('Training Examples', 
#                   fontsize=self.config.FONT_SIZES['label'])
#         plt.ylabel('Accuracy Score', 
#                   fontsize=self.config.FONT_SIZES['label'])
#         plt.title('Learning Curve', 
#                  fontsize=self.config.FONT_SIZES['title'],
#                  pad=20)
#         plt.legend(loc='lower right', 
#                   fontsize=self.config.FONT_SIZES['legend'])
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.ylim(0, 1.1)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import learning_curve
from .baseplotter import BasePlotter
from typing import Tuple

class LearningCurvePlotter(BasePlotter):
    """Handles learning curve visualization with enhanced confidence intervals"""
    
    def plot(self, 
            model: BaseEstimator,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
            output_suffix: str = '',
            n_points: int = 20,
            confidence_level: float = 0.95) -> None:
        """
        Plot learning curves with confidence intervals.
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            output_suffix: Suffix for output filename
            n_points: Number of points for learning curve (default: 20)
            confidence_level: Confidence level for intervals (default: 0.95)
        """
        plt.figure(figsize=self.config.FIGURE_SIZES['learning'], 
                  dpi=self.config.DPI)
        
        # Combine data for cross-validation
        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])
        
        # Calculate curves with more points
        train_sizes = np.linspace(0.1, 1.0, n_points)
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=5,
            n_jobs=1,
            scoring='accuracy',
            shuffle=True,
            random_state=42
        )
        
        # Calculate statistics with confidence intervals
        train_stats = self._calculate_stats(train_scores, confidence_level)
        val_stats = self._calculate_stats(val_scores, confidence_level)
        
        # Plot curves
        self._plot_curve(train_sizes, train_stats, 'Training')
        self._plot_curve(train_sizes, val_stats, 'Validation')
        
        # Add confidence interval legend
        self._add_confidence_legend(confidence_level)
        
        # Customize plot
        self._customize_plot()
        
        # Save plot
        self.save_plot('learning_curve.pdf', output_suffix)
    
    def _calculate_stats(self, scores: np.ndarray, 
                        confidence_level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate mean and confidence intervals"""
        mean = np.mean(scores, axis=1)
        std = np.std(scores, axis=1)
        z_value = abs(np.percentile(np.random.standard_normal(10000), 
                                  (1 - confidence_level) * 100 / 2))
        ci = z_value * std / np.sqrt(scores.shape[1])
        return mean, mean - ci, mean + ci
    
    def _plot_curve(self, sizes: np.ndarray, 
                    stats: Tuple[np.ndarray, np.ndarray, np.ndarray], 
                    label: str) -> None:
        """Plot curve with enhanced confidence interval visualization"""
        mean, ci_lower, ci_upper = stats
        color = (self.config.COLORS['primary'] if label == 'Training' 
                else self.config.COLORS['secondary'])
        
        # Plot mean line with markers
        plt.plot(sizes, mean, 'o-', label=label, 
                color=color, linewidth=2, markersize=6)
        
        # Plot confidence interval
        plt.fill_between(
            sizes,
            ci_lower,
            ci_upper,
            alpha=0.2,
            color=color,
            label=f'{label} CI'
        )
        
        # Add error bars at sample points
        plt.errorbar(sizes, mean, 
                    yerr=np.vstack((mean - ci_lower, ci_upper - mean)),
                    fmt='none', color=color, capsize=3, alpha=0.5)
    
    def _add_confidence_legend(self, confidence_level: float) -> None:
        """Add confidence interval information to legend"""
        plt.plot([], [], color='gray', alpha=0.2, linewidth=10,
                label=f'{confidence_level*100:.0f}% Confidence Interval')
    
    def _customize_plot(self) -> None:
        """Customize plot appearance"""
        plt.xlabel('Training Examples', 
                  fontsize=self.config.FONT_SIZES['label'])
        plt.ylabel('Accuracy Score', 
                  fontsize=self.config.FONT_SIZES['label'])
        plt.title('Learning Curve with Confidence Intervals', 
                 fontsize=self.config.FONT_SIZES['title'],
                 pad=20)
        plt.legend(loc='lower right', 
                  fontsize=self.config.FONT_SIZES['legend'],
                  framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1.1)
        plt.xlim(plt.xlim()[0] * 0.9, plt.xlim()[1] * 1.05)