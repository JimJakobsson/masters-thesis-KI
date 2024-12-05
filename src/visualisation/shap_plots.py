from typing import Any
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
from .baseplotter import BasePlotter

class ShapPlotter(BasePlotter):
    """Handles SHAP-based visualizations"""
    def plot(self, *args, **kwargs) -> None:
        pass
    def plot_summary(self, aggregated_shap: dict, 
                    X_test: pd.DataFrame,
                    output_suffix: str = '') -> None:
        """Create SHAP summary plot"""
        plt.figure(figsize=self.config.FIGURE_SIZES['shap'])
        
        # Prepare data
        shap_matrix = np.column_stack(
            [aggregated_shap[feature] for feature in aggregated_shap.keys()]
        )
        features = X_test[list(aggregated_shap.keys())]
        
        # Create plot
        shap.summary_plot(
            shap_matrix,
            features,
            feature_names=list(aggregated_shap.keys()),
            plot_type="dot",
            max_display=20,
            show=False,
            
        )
        
        # Save plot
        self.save_plot('shap_summary.pdf', output_suffix)
    
    def plot_waterfall(self, model, X_test: pd.DataFrame, 
                      class_to_explain: int,
                      explainer: Any,
                      feature_importance_abs_mean: pd.DataFrame,
                      aggregated_shap: dict,
                      output_suffix: str = '') -> None:
        plt.figure(figsize=self.config.FIGURE_SIZES['waterfall'])
        
        probas = model.predict_proba(X_test)
        observation_idx = (np.argmax(probas[:, 1]) 
                         if class_to_explain == 1 
                         else np.argmax(probas[:, 0]))
        
        features = list(feature_importance_abs_mean['feature'])  # Use your predefined order
        values = np.array([aggregated_shap[feature][observation_idx] for feature in features])
        data = X_test.iloc[observation_idx][features].values
        # Check if expected_value is a list or a single value
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = float(explainer.expected_value[class_to_explain])
        else:
            base_value = float(explainer.expected_value)

        explanation = shap.Explanation(
            values=values,
            base_values=base_value,
            data=data,
            feature_names=features
        )
        
        shap.waterfall_plot(explanation, show=False)
        plt.title(f'Prediction Explanation for Class {class_to_explain}',
                 fontsize=self.config.FONT_SIZES['title'])
        self.save_plot(f'waterfall_class_{class_to_explain}.pdf', output_suffix)