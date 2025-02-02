from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from config.model_config import ModelConfig
from config.path_config import PathConfig
from .base_evaluator import BaseEvaluator
from .report_classification import ReportClassification
from visualisation.model_visualiser import ModelVisualiser
from visualisation.feature_importance_plotter import FeatureImportancePlotter
from visualisation.shap_plots import ShapPlotter
from visualisation.learning_curve import LearningCurvePlotter
from .metrics import Metrics
from sklearn.model_selection import StratifiedShuffleSplit
from .threshold_finder import ThresholdFinder
# from ..utils.validation import validate_shap_calculation

class ModelEvaluator(BaseEvaluator):
    """Handles model evaluation and SHAP explanationss"""

    def __init__(self, output_dir: Optional[str] = None):
        self.results: Dict = {}
        self.output_dir = Path(output_dir) if output_dir else PathConfig.OUTPUT_DIR
        self.shap_values: Optional[np.ndarray] = None
        self.feature_importance_dataframe: Optional[pd.DataFrame] = None
        self.explainer: Optional[Any] = None
        self.aggregated_shap: Optional[Dict] = None
        
        # Initialize plotters
        self.model_visualiser = ModelVisualiser(self.output_dir)
        self.feature_plotter = FeatureImportancePlotter(self.output_dir)
        self.shap_plotter = ShapPlotter(self.output_dir)
        self.learning_curve_plotter = LearningCurvePlotter(self.output_dir)
        self.threshold_finder = ThresholdFinder()

        #Initialize report
        self.report = ReportClassification()
        self.metrics = Metrics()   
    
    def evaluate_model(self, grid_search: BaseEstimator, X_test: pd.DataFrame, 
                  y_test: pd.Series, threshold: float = None) -> Dict[str, Any]:
        """Evaluate model performance using classification metrics"""
        
        y_prob = grid_search.predict_proba(X_test)[:, 1]
        print(f"Predicted probabilities: {y_prob}")
        
        # Find optimal threshold if none provided
        if threshold is None:
            threshold = self.threshold_finder.find_optimal_threshold(y_test, y_prob)
        
        print(f"Using threshold: {threshold}")
        y_pred = (y_prob > threshold).astype(int)
        y_test = y_test.astype(int)
        
        # Print a classification report
        report = self.report.print_classification_report(y_test, y_pred)
        self.results.update({'classification_report': report})
        
        # Calculate and store metrics
        metrics_result = self.metrics.calculate_classification_metrics(y_test, y_pred, y_prob)
        self.results.update({
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'optimal_threshold': threshold,
            **metrics_result
        })
        
        print(self.metrics.format_metrics_for_display(metrics_result))
        return self.results
    
    def get_feature_names_after_preprocessing(self, model):
        """Get feature names after preprocessing has been applied"""
        feature_names = []
        processor = model.named_steps['preprocessor']

        for name, transformer, columns in processor.transformers_:
            if name == 'num':
                # Get names from the numeric pipeline
                numeric_pipeline = processor.named_transformers_['num']
                
                # First get null indicator names
                null_indicator = numeric_pipeline.named_steps['null_indicator']
                if hasattr(null_indicator, 'null_feature_names'):
                    feature_names.extend([f"{col}_nan" for col in columns])

                feature_names.extend(columns)
            elif name == 'cat':
                encoder = processor.named_transformers_['cat'].named_steps['onehot']
                cat_features = encoder.get_feature_names_out(columns)
                feature_names.extend(cat_features)
            elif name == 'remainder':
                # Handle remainder columns based on drop_remainder parameter
                if transformer == 'drop':
                    print(f"Remainder columns being dropped: {columns}")
                else:
                    # If remainder columns are passed through, add them to feature names
                    if isinstance(columns, slice):
                        remainder_cols = range(columns.start or 0, columns.stop, columns.step or 1)
                        feature_names.extend([f"feature_{i}" for i in remainder_cols])
                    else:
                        feature_names.extend(columns)
            else:
                raise ValueError(f'Invalid transformer name: {name}')
            
        print(f"Total features after preprocessing: {len(feature_names)}")
        
        return feature_names
    
    def calculate_feature_importance(self, best_model: BaseEstimator, 
                               X_test: pd.DataFrame) -> Tuple[Dict, List[str]]:
        """
        Calculate feature importance using SHAP values, properly handling feature extraction
        from the ColumnTransformer.
        
        Args:
            best_model: The trained model pipeline
            X_test: Test features
            
        Returns:
            Tuple containing:
            - Dictionary with aggregated SHAP values
            - List of feature names
            
        The method handles the extraction of numeric and categorical features from the
        ColumnTransformer's configuration, then uses these to properly aggregate SHAP
        values for categorical features that were one-hot encoded.
        """
        # Get the column transformer from the pipeline
        column_transformer = best_model.named_steps['preprocessor']
        # Transform the test data using the pipeline's preprocessor
        X_test_transformed = column_transformer.transform(X_test)

        #Create explainer using just the classifier
        self.explainer = self._create_explainer(best_model, X_test_transformed)

        #check if explainer is KernelExplainer

        if isinstance(self.explainer, shap.KernelExplainer):
            self.shap_values = self.explainer.shap_values(X_test_transformed, silent=True)
        else:
            self.shap_values = self.explainer.shap_values(X_test_transformed)

        # if isinstance(best_model, ModelConfig.KERNEL_EXPLAINER_MODELS):
        #     self.shap_values = self.explainer.shap_values(X_test_transformed, silent=True)
        # else:
        #     self.shap_values = self.explainer.shap_values(X_test_transformed)
        
        # Get feature names after preprocessing
        preprocessed_feature_names = self.get_feature_names_after_preprocessing(best_model)

        # Ensure feature names match the SHAP values shape
        if len(preprocessed_feature_names) != self.shap_values.shape[1]:
            print(f"Warning: Feature name count ({len(preprocessed_feature_names)}) doesn't match SHAP values shape ({self.shap_values.shape[1]})")
            # Trim feature names to match SHAP values
            preprocessed_feature_names = preprocessed_feature_names[:self.shap_values.shape[1]]

        print("SHAP values shape:", self.shap_values.shape)
        print("Feature names:", len(preprocessed_feature_names))
        # Extract numeric and categorical features from transformer configuration
        numeric_features = []
        categorical_features = []
        
        # Iterate through transformers to get feature lists
        for name, _, columns in column_transformer.transformers_:
            if name == 'num':
                numeric_features = columns
            elif name == 'cat':
                categorical_features = columns
        
        # Calculate aggregated SHAP values
        self.aggregated_shap, feature_importance_dataframe, feature_importance_abs_mean = self._aggregate_shap_values(
            self.shap_values,
            preprocessed_feature_names,
            numeric_features,
            categorical_features
        )
        
        return self.aggregated_shap, feature_importance_dataframe, feature_importance_abs_mean
    
    def _aggregate_shap_values(self, 
                             shap_values: np.ndarray,
                             processed_feature_names: List[str],
                             numeric_features: List[str],
                             categorical_features: List[str]) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """
        Aggregate SHAP values for categorical features that were one-hot encoded.
        
        Args:
            shap_values: SHAP values from explainer
            feature_names: Names of features after preprocessing
            preprocessing_result: PreprocessingResult containing feature information
            
        Returns:
            Tuple containing:
            - Dictionary mapping original feature names to aggregated SHAP values
            - DataFrame with feature importance statistics
        """           
        # Print feature counts by type
        null_indicators = [f for f in processed_feature_names if f.endswith('_nan')]
        print(f"Number of null indicators: {len(null_indicators)}")
        # If we have binary classification, use values for positive class
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        
        # Get original feature names (before one-hot encoding)
        original_features = (numeric_features + categorical_features)
        processed_feature_names = np.array(processed_feature_names)
        
        # Initialize dictionary for aggregated SHAP values
        aggregated_shap = {}
        processed_indices = set()
        current_idx = 0
        
        # Process each original feature
        for feature in original_features:
            if feature in numeric_features:
                if current_idx not in processed_indices:
                    # Add the original feature first
                    if current_idx < len(processed_feature_names):
                        feature_name = processed_feature_names[current_idx]
                        # print(f"Processing {feature_name} at index {current_idx}")
                        aggregated_shap[feature] = shap_values[:, current_idx]
                        processed_indices.add(current_idx)
                        current_idx += 1
                    
                    # Check for null indicator
                    null_feature_name = f"{feature}_nan"
                    if current_idx < len(processed_feature_names) and processed_feature_names[current_idx] == null_feature_name:
                        # Only include null indicator if there are actual null values
                        null_values = shap_values[:, current_idx]
                        if np.any(null_values != 0):  # Only include if it has non-zero SHAP values
                            # print(f"Processing {null_feature_name} at index {current_idx}")
                            aggregated_shap[null_feature_name] = null_values
                        processed_indices.add(current_idx)
                        current_idx += 1
                # if current_idx not in processed_indices:
                #     # # For numeric features, just copy the SHAP values directly
                #     # if current_idx < len(processed_feature_names):
                #     #     aggregated_shap[feature] = shap_values[:, current_idx]
                #     #     current_idx += 1
                #     aggregated_shap[feature] = shap_values[:, current_idx]
                #     processed_indices.add(current_idx)
                #     current_idx += 1
            else:
                # For categorical features, find all related one-hot encoded columns
                feature_mask = np.array([col.startswith(f"{feature}_") for col in processed_feature_names])
                if np.any(feature_mask):
                    indices = np.where(feature_mask)[0]
                    if not any(idx in processed_indices for idx in indices):
                    # Sum SHAP values across all one-hot encoded columns
                    # Don't take absolute values to preserve direction of impact
                        aggregated_values = shap_values[:, feature_mask].sum(axis=1)
                        aggregated_shap[feature] = aggregated_values
                        processed_indices.update(indices)
                    current_idx += np.sum(feature_mask)
        
        # Create feature importance DataFrame
        feature_importance_dataframe = pd.DataFrame({
            'feature': list(aggregated_shap.keys()),
            'importance_abs_mean': [np.mean(np.abs(values)) for values in aggregated_shap.values()],
            'importance_mean': [np.mean(values) for values in aggregated_shap.values()],
            'importance_std': [np.std(values) for values in aggregated_shap.values()] 
        })
        
        # Sort by absolute importance
        #feature_importance = feature_importance.sort_values('importance_abs_mean', ascending=False).head(num_features)

        feature_importance_abs_mean = feature_importance_dataframe.sort_values(
            'importance_abs_mean', ascending=False)
        print("features importance abs mean", feature_importance_abs_mean.head(10))
         # Print null indicator statistics
        null_indicators = feature_importance_abs_mean[
            feature_importance_abs_mean['feature'].str.endswith('_nan')]
        if not null_indicators.empty:
            print("\nNull indicator importance:")
            print(null_indicators[['feature', 'importance_abs_mean']].head(10))

        return aggregated_shap, feature_importance_dataframe, feature_importance_abs_mean 
     


    def _create_explainer(self, model: BaseEstimator,
                    data: Optional[np.ndarray] = None) -> Any:
        """Create appropriate SHAP explainer based on model type
        
        This function creates SHAP explainers optimized for maximum data usage.
        Different explainer types handle background data differently:
        - TreeExplainer: Can efficiently handle all data points
        - DeepExplainer: Needs to compute background values, so we use chunking
        - LinearExplainer: Can handle large datasets efficiently
        - KernelExplainer: Most computationally intensive, requires careful data handling
        
        Args:
            model: Trained sklearn pipeline containing a classifier
            data: Background data for explainer initialization
            
        Returns:
            SHAP explainer instance appropriate for the model type
        """
        classifier = model.named_steps['classifier']
        model_name = classifier.__class__.__name__

        def prepare_background_data(data: np.ndarray, 
                                explainer_type: str) -> np.ndarray:
            """Prepare background data optimized for each explainer type
            
            Different explainers have different computational characteristics:
            - TreeExplainer: O(n_samples)
            - LinearExplainer: O(n_samples)
            - DeepExplainer: O(n_samples * n_features)
            - KernelExplainer: O(n_samples^2 * n_features)
            """
            if data is None:
                raise ValueError("Background data required for explainer")
                
            # For tree and linear models, we can use all data
            if explainer_type in ['tree', 'linear']:
                return data
                
            # For deep learning models, chunk data if it's too large
            elif explainer_type == 'deep':
                max_deep_samples = 200  # Adjust based on available memory
                if len(data) > max_deep_samples:
                    # Use systematic sampling to maintain distribution
                    step = len(data) // max_deep_samples
                    return data[::step]
                return data
                
            # For kernel explainer, we need to be more selective
            elif explainer_type == 'kernel':
                # Use progressive sampling to find optimal sample size
                from sklearn.model_selection import train_test_split
                
                max_kernel_samples = 200 # Adjust based on available memory and time
                if len(data) <= max_kernel_samples:
                    return data
                    
                # Take a larger sample first, then subsample if needed
                background, _ = train_test_split(
                    data,
                    train_size=max_kernel_samples,
                    stratify=classifier.predict(data) if hasattr(classifier, 'predict') else None,
                    random_state=42
                )
                return background

        try:
            if model_name in ModelConfig.TREE_BASED_MODELS:
                print("Creating Tree explainer with full dataset")
                # Tree-based models can efficiently handle the full dataset
                return shap.TreeExplainer(classifier)
                
            elif model_name in ModelConfig.DEEP_LEARNING_MODELS:
                background = prepare_background_data(data, 'deep')
                print(f"Creating Deep explainer with {len(background)} background samples")
                return shap.DeepExplainer(model, background)
                
            elif model_name in ModelConfig.LINEAR_MODELS:
                background = prepare_background_data(data, 'linear')
                print(f"Creating Linear explainer with {len(background)} background samples")
                return shap.LinearExplainer(model, background)
                
            elif model_name in ModelConfig.KERNEL_EXPLAINER_MODELS:
                background = prepare_background_data(data, 'kernel')
                print(f"Creating Kernel explainer with {len(background)} background samples")
                return shap.KernelExplainer(
                    classifier.predict_proba,
                    background,
                    silent=False
                )
                
            else:
                raise ValueError(f"Unsupported model type: {model_name}")
                
        except Exception as e:
            raise Exception(f"Error creating SHAP explainer: {str(e)}")
        
    def plot_all(self, model: BaseEstimator, 
                 X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series,
                 feature_importance_dataframe: pd.DataFrame,
                 feature_importance_abs_mean: pd.DataFrame,
                 aggregated_shap: Dict,
                 class_to_explain: int = 1,
                 output_suffix: str = '') -> None:
        """
        Generate all plots using the appropriate data for each visualization.
        
        Args:
            model: Trained model
            X_train: Training features for learning curves
            X_test: Test features for SHAP analysis
            y_train: Training labels for learning curves
            y_test: Test labels for evaluation
            class_to_explain: Class to explain in SHAP waterfall plot
            output_suffix: Suffix for output files
        """
        if not all([feature_importance_dataframe is not None,
                   aggregated_shap is not None,
                   self.explainer is not None]):
            raise ValueError("Must run calculate_feature_importance before plotting")

        self.model_visualiser.create_all_plots(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_importance_dataframe=feature_importance_dataframe,
            feature_importance_abs_mean=feature_importance_abs_mean,
            aggregated_shap=aggregated_shap,
            explainer=self.explainer,
            class_to_explain=class_to_explain,
            output_suffix=output_suffix
        )