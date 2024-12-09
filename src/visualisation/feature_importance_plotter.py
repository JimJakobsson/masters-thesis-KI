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
        # Sort and filter features
        # feature_importance = (feature_importance
        #                     .sort_values('importance_abs_mean', ascending=False)
        #                     .head(num_features))
        feature_importance = feature_importance.head(num_features)
        plt.figure(figsize=self.config.FIGURE_SIZES['feature'], 
                  dpi=self.config.DPI)
        
        # Create plot
        # sns.barplot(
        #     x='importance_abs_mean',
        #     y='feature',
        #     data=feature_importance,
        #     palette='rocket'
        # )
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