from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .baseplotter import BasePlotter
from config.plot_config import PlotConfig


class FeatureImportancePlotter(BasePlotter):
    """Handles feature importance visualization"""
    
    def plot(self, feature_importance: pd.DataFrame, 
             num_features: int = 20, 
             output_suffix: str = '') -> None:
        """Plot feature importance"""
     
        feature_importance = feature_importance.head(num_features)
        plt.figure(figsize=self.config.FIGURE_SIZES['feature'], 
                  dpi=self.config.DPI)
       
        plt.barh(feature_importance['feature'], 
                 feature_importance['importance_abs_mean'], 
                 color='b')
        # Customize plot
        plt.xlabel('Mean |SHAP Value|', 
                  fontsize=self.config.FONT_SIZES['label'])
        plt.ylabel('Feature', 
                  fontsize=self.config.FONT_SIZES['label'])
        plt.title('Feature Importance', 
                 fontsize=self.config.FONT_SIZES['title'], 
                 pad=20)
        
        # Save plot
        self.save_plot('feature_importance.pdf', output_suffix)

        self.plot_relative_feature_importance(feature_importance,
                                           num_features, 
                                           output_suffix)

    def plot_relative_feature_importance(self, feature_importance: pd.DataFrame, 
                                           num_features: int = 20, 
                                           output_suffix: str = '') -> None:
        """Plot Min-max normalization to scale values between 0 and 1 """
        min_val = feature_importance['importance_abs_mean'].min()
        max_val = feature_importance['importance_abs_mean'].max()
        normalized_values = (feature_importance['importance_abs_mean'] - min_val) / (max_val - min_val)

        plt.figure(figsize=self.config.FIGURE_SIZES['feature'],
                     dpi=self.config.DPI)
        plt.barh(feature_importance['feature'],
                    normalized_values,
                    color='b')
        # Customize plot
        plt.xlabel('Relative Importance',
                    fontsize=self.config.FONT_SIZES['label'])
        plt.ylabel('Feature',
                    fontsize=self.config.FONT_SIZES['label'])
        plt.title('Relative Feature Importance',
                    fontsize=self.config.FONT_SIZES['title'],
                    pad=20)
        
        # Save plot
        self.save_plot('relative_feature_importance.pdf', output_suffix)