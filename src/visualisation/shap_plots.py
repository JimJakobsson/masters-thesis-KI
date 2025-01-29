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
from experiment.experiment_config import ExperimentConfig

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
    
        probas = model.predict_proba(X_test)
        
        # Get indices of 5 random people ages 60-70 years old
        
        # Get indices of 5 random people ages 60-70 years old
        age_filter = (1987 - X_test['birthdate1'].astype(str).str[:4].astype(int) >= 60) & (1987 - X_test['birthdate1'].astype(str).str[:4].astype(int) <= 70)

        eligible_indices = np.where(age_filter)[0]

        print(f"Found {len(eligible_indices)} samples in the specified age range.")
        if len(eligible_indices) < 5:
            raise ValueError("Not enough samples in the specified age range.")
        
        # Get probabilities for eligible indices
        eligible_probas = probas[eligible_indices, class_to_explain]
        # Get indices of top 5 highest risk individuals
        top_5_local_indices = np.argsort(eligible_probas)[-5:][::-1]  # Sort descending
        people_indices = eligible_indices[top_5_local_indices]  # Convert to global indices
        
        features = list(feature_importance_abs_mean['feature'])  # Use predefined order of features
        
        for i, observation_idx in enumerate(people_indices):
            plt.figure(figsize=self.config.FIGURE_SIZES['waterfall'])
            
            probability = probas[observation_idx, class_to_explain]
            print(f"Sample {i+1} - Probability of class {class_to_explain}: {probability:.4f}")
            
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

            # Handle birthdate1 if present
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
            plt.title(f'Feature Impacts For a Random Person Aged {1987 - int(data[features.index("birthdate1")]):.0f}',
                    fontsize=self.config.FONT_SIZES['title'],
                    ha='center')
            self.save_plot(f'waterfall_class_{class_to_explain}_sample_{i+1}.pdf', output_suffix)
