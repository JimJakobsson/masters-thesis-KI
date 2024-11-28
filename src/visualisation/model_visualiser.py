from pathlib import Path
from typing import Any, Dict
import pandas as pd

from visualisation.feature_importance_plotter import FeatureImportancePlotter
from visualisation.learning_curve import LearningCurvePlotter
from visualisation.shap_plots import ShapPlotter

class ModelVisualiser:
    """Coordinates all visualization tasks"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.feature_plotter = FeatureImportancePlotter(output_dir)
        self.shap_plotter = ShapPlotter(output_dir)
        self.learning_plotter = LearningCurvePlotter(output_dir)
    
    def create_all_plots(self, 
                        model: Any,
                        X_test: pd.DataFrame,
                        X: pd.DataFrame,
                        y: pd.Series,
                        feature_importance: pd.DataFrame,
                        aggregated_shap: Dict,
                        explainer: Any,
                        output_suffix: str = '') -> None:
        """Create all visualization plots"""
        # Feature importance plot
        self.feature_plotter.plot(
            feature_importance,
            output_suffix=output_suffix
        )
        
        # SHAP plots
        self.shap_plotter.plot_summary(
            aggregated_shap,
            X_test,
            output_suffix=output_suffix
        )
        self.shap_plotter.plot_waterfall(
            model,
            X_test,
            class_to_explain=1,
            explainer=explainer,
            aggregated_shap=aggregated_shap,
            output_suffix=output_suffix
        )
        
        # Learning curve plot
        self.learning_plotter.plot(
            model,
            X,
            y,
            output_suffix=output_suffix
        )