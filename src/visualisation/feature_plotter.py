from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ..config.plot_config import PlotConfig

class FeaturePlotter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.plot_config = PlotConfig()

    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                              num_features: int = 20) -> None:
        """Plot feature importance"""
        feature_importance = feature_importance.sort_values(
            'importance_abs_mean', ascending=False).head(num_features)

        plt.figure(figsize=self.plot_config.FIGURE_SIZE, 
                  dpi=self.plot_config.DPI)
        sns.barplot(x='importance_abs_mean', y='feature', 
                   data=feature_importance, 
                   palette='rocket')
        
        plt.xlabel('Mean |SHAP Value|', 
                  fontsize=self.plot_config.FONT_SIZES['label'])
        plt.ylabel('Feature', 
                  fontsize=self.plot_config.FONT_SIZES['label'])
        plt.title('Feature Importance', 
                 fontsize=self.plot_config.FONT_SIZES['title'], 
                 pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.pdf', 
                   bbox_inches='tight')
        plt.close()