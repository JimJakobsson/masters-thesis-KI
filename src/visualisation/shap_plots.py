from typing import Any
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
from .baseplotter import BasePlotter

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
        
        # Separate original features and null indicators
        original_features = {k: v for k, v in aggregated_shap.items() 
                           if not k.endswith('_nan')}
        null_indicators = {k: v for k, v in aggregated_shap.items() 
                         if k.endswith('_nan')}
        
        # Create feature matrix for original features
        original_shap_matrix = np.column_stack(
            [original_features[feature] for feature in original_features.keys()]
        )
        original_feature_data = X_test[list(original_features.keys())]
        
        # Create feature matrix for null indicators
        if null_indicators:
            # Create binary indicators based on original features
            null_feature_data = pd.DataFrame({
                name: X_test[name.replace('_nan', '')].isna().astype(int)
                for name in null_indicators.keys()
            })
            
            null_shap_matrix = np.column_stack(
                [null_indicators[feature] for feature in null_indicators.keys()]
            )
            
            # Combine original and null indicator features
            feature_data = pd.concat([original_feature_data, null_feature_data], axis=1)
            shap_matrix = np.column_stack([original_shap_matrix, null_shap_matrix])
            feature_names = list(original_features.keys()) + list(null_indicators.keys())
        else:
            feature_data = original_feature_data
            shap_matrix = original_shap_matrix
            feature_names = list(original_features.keys())
        
        # Create plot
        shap.summary_plot(
            shap_matrix,
            feature_data,
            feature_names=feature_names,
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
        
        # Handle null indicator features in data preparation
        data = []
        for feature in features:
            if feature.endswith('_nan'):
                # For null indicators, use the isna() status of the original feature
                orig_feature = feature.replace('_nan', '')
                value = float(np.isnan(X_test.iloc[observation_idx][orig_feature]))
            else:
                value = X_test.iloc[observation_idx][feature]
            data.append(value)
        data = np.array(data)
        
        # Check if expected_value is a list or a single value
        if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > class_to_explain:
            base_value = float(explainer.expected_value[class_to_explain])
        else:
            base_value = float(explainer.expected_value)

        #if 'birthdate1' in features, display only the first four digits of the birthdate
        if 'birthdate1' in features:
            idx = features.index('birthdate1')
            data[idx] = str(data[idx])[:4]

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


# class ShapPlotter(BasePlotter):
#     """Handles SHAP-based visualizations"""
#     def plot(self, *args, **kwargs) -> None:
#         pass
#     def plot_summary(self, aggregated_shap: dict, 
#                     X_test: pd.DataFrame,
#                     output_suffix: str = '') -> None:
#         """Create SHAP summary plot"""
#         plt.figure(figsize=self.config.FIGURE_SIZES['shap'])
        
#         # Prepare data
#         shap_matrix = np.column_stack(
#             [aggregated_shap[feature] for feature in aggregated_shap.keys()]
#         )
#         features = X_test[list(aggregated_shap.keys())]
        
#         # Create plot
#         shap.summary_plot(
#             shap_matrix,
#             features,
#             feature_names=list(aggregated_shap.keys()),
#             plot_type="dot",
#             max_display=20,
#             show=False,
            
#         )
        
#         # Save plot
#         self.save_plot('shap_summary.pdf', output_suffix)
    
#     def plot_waterfall(self, model, X_test: pd.DataFrame, 
#                       class_to_explain: int,
#                       explainer: Any,
#                       feature_importance_abs_mean: pd.DataFrame,
#                       aggregated_shap: dict,
#                       output_suffix: str = '') -> None:
#         plt.figure(figsize=self.config.FIGURE_SIZES['waterfall'])
        
#         probas = model.predict_proba(X_test)
#         observation_idx = (np.argmax(probas[:, 1]) 
#                          if class_to_explain == 1 
#                          else np.argmax(probas[:, 0]))
        
#         features = list(feature_importance_abs_mean['feature'])  # Use your predefined order
#         values = np.array([aggregated_shap[feature][observation_idx] for feature in features])
#         data = X_test.iloc[observation_idx][features].values
#         # Check if expected_value is a list or a single value
#         if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > class_to_explain:
#             base_value = float(explainer.expected_value[class_to_explain])
#         else:
#             # Fall back to using the single value
#             base_value = float(explainer.expected_value)

#         #if 'birthdate1' in features, display only the first four digits of the birthdate
#         #to avoid cluttering the plot with the full date and to protect privacy
#         if 'birthdate1' in features:
#             data[features.index('birthdate1')] = str(data[features.index('birthdate1')])[:4]

#         explanation = shap.Explanation(
#             values=values,
#             base_values=base_value,
#             data=data,
#             feature_names=features
#         )
        
#         shap.waterfall_plot(explanation, show=False)
#         plt.title(f'Prediction Explanation for Class {class_to_explain}',
#                  fontsize=self.config.FONT_SIZES['title'])
#         self.save_plot(f'waterfall_class_{class_to_explain}.pdf', output_suffix)