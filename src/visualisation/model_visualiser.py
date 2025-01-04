from pathlib import Path
from typing import Any, Dict
import pandas as pd

from visualisation.auc_plotter import AUCPlotter
from visualisation.confusion_matrix import ConfusionMatrixPlotter
from visualisation.feature_importance_plotter import FeatureImportancePlotter
from visualisation.learning_curve import LearningCurvePlotter
from visualisation.roc_plotter import ROCPlotter
from visualisation.shap_plots import ShapPlotter

class ModelVisualiser:
    """Coordinates all visualization tasks"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.feature_plotter = FeatureImportancePlotter(output_dir)
        self.shap_plotter = ShapPlotter(output_dir)
        self.learning_plotter = LearningCurvePlotter(output_dir)
        self.roc_plotter = ROCPlotter(output_dir)
        self.auc_plotter = AUCPlotter(output_dir)
        self.confusion_matrix_plotter = ConfusionMatrixPlotter(output_dir)
    
    def create_all_plots(self, 
                        model: Any,
                        X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series,
                        feature_importance_dataframe: pd.DataFrame,
                        feature_importance_abs_mean: pd.DataFrame,
                        aggregated_shap: Dict,
                        explainer: Any,
                        class_to_explain: int = 1,
                        output_suffix: str = '') -> None:
        """Create all visualization plots"""
        # Feature importance plot
        self.feature_plotter.plot(
            feature_importance_abs_mean,
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
            class_to_explain=class_to_explain,
            explainer=explainer,
            feature_importance_abs_mean=feature_importance_abs_mean,
            aggregated_shap=aggregated_shap,
            output_suffix=output_suffix
        )
        
        # Learning curve plot
        self.learning_plotter.plot(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            output_suffix=output_suffix
        )

        # ROC curve plot
        y_scores = model.predict_proba(X_test)[:, 1]
        self.roc_plotter.plot(y_test, y_scores, show_thresholds=True, output_suffix=output_suffix)

        #Confusion matrix plot
        y_pred = model.predict(X_test)
        self.confusion_matrix_plotter.plot(
            y_true=y_test,
            y_pred=y_pred,
            labels=['Negative', 'Positive'],  # Add custom labels
            cmap='Blues',    # Use default color scheme
            figsize=(8, 8),   # Default size
            output_suffix=output_suffix
        )
                                           

        # AUC plot
        # self.auc_plotter.plot(y_test, y_scores, show_t)
