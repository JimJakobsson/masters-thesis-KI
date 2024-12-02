from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from config.model_config import ModelConfig
from config.path_config import PathConfig
from preprocessing.preprocessing_result import PreprocessingResult
from .base_evaluator import BaseEvaluator
from .report_classification import ReportClassification
from visualisation.model_visualiser import ModelVisualiser
from visualisation.feature_importance_plotter import FeatureImportancePlotter
from visualisation.shap_plots import ShapPlotter
from visualisation.learning_curve import LearningCurvePlotter
from preprocessing.pipeline_creator import PipelineCreator
from .metrics import Metrics
from preprocessing.preprocessing_result import PreprocessingResult  # Import the class

# from ..utils.validation import validate_shap_calculation

class ModelEvaluator(BaseEvaluator):
    """Handles model evaluation and SHAP explanationss"""

    def __init__(self, output_dir: Optional[str] = None):
        self.results: Dict = {}
        self.output_dir = Path(output_dir) if output_dir else PathConfig.OUTPUT_DIR
        self.shap_values: Optional[np.ndarray] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.explainer: Optional[Any] = None
        self.aggregated_shap: Optional[Dict] = None
        
        # Initialize plotters
        self.model_visualiser = ModelVisualiser(self.output_dir)
        self.feature_plotter = FeatureImportancePlotter(self.output_dir)
        self.shap_plotter = ShapPlotter(self.output_dir)
        self.learning_curve_plotter = LearningCurvePlotter(self.output_dir)

        #Initialize report
        self.report = ReportClassification()
        self.metrics = Metrics()   
    
    def evaluate_model(self, grid_search: BaseEstimator, X_test: pd.DataFrame, 
                      y_test: pd.Series, threshold: float = 0.4) -> Dict[str, Any]:
        """Evaluate model performance using classification metrics"""
        y_prob = grid_search.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > threshold).astype(int)
        y_test = y_test.astype(int)
        
        #Prints a classification report
        report = self.report.print_classification_report(y_test, y_pred)
        metrics_result = self.metrics.calculate_classification_metrics(y_test, y_pred, y_prob)
        
        self.results.update({
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            **metrics_result
        })
        
        #print metrics using format_metrics_for_display
        print(self.metrics.format_metrics_for_display(metrics_result))
        return self.results
    
    def get_feature_names_after_preprocessing(self, model):
        """Get feature names after preprocessing has been applied"""
        feature_names = []
        processor = model.named_steps['preprocessor']

        for name, transformer, columns in processor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                encoder = processor.named_transformers_['cat'].named_steps['onehot']
                cat_features = encoder.get_feature_names_out(columns)
                feature_names.extend(cat_features)
            elif name == 'remainder' :
                continue
            else:
                raise ValueError(f'Invalid transformer name: {name}')
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
        
        # Create SHAP explainer and calculate values
        self.explainer = self._create_explainer(best_model, X_test_transformed)
        self.shap_values = self.explainer.shap_values(X_test_transformed)
        
        # Get feature names after preprocessing
        feature_names = self.get_feature_names_after_preprocessing(best_model)
        
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
        self.aggregated_shap, self.feature_importance = self._aggregate_shap_values(
            self.shap_values,
            feature_names,
            numeric_features,
            categorical_features
        )
        
        return self.aggregated_shap, feature_names
    
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
        # If we have binary classification, use values for positive class
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        
        # Get original feature names (before one-hot encoding)
        original_features = (numeric_features + categorical_features)
        processed_feature_names = np.array(processed_feature_names)
        
        # Initialize dictionary for aggregated SHAP values
        aggregated_shap = {}
        current_idx = 0
        
        # Process each original feature
        for feature in original_features:
            if feature in numeric_features:
                # For numeric features, just copy the SHAP values directly
                if current_idx < len(processed_feature_names):
                    aggregated_shap[feature] = shap_values[:, current_idx]
                    current_idx += 1
            else:
                # For categorical features, find all related one-hot encoded columns
                feature_mask = np.array([col.startswith(f"{feature}_") for col in processed_feature_names])
                
                if np.any(feature_mask):
                    # Sum SHAP values across all one-hot encoded columns
                    # Don't take absolute values to preserve direction of impact
                    aggregated_values = shap_values[:, feature_mask].sum(axis=1)
                    aggregated_shap[feature] = aggregated_values
                    current_idx += np.sum(feature_mask)
        
        # Create feature importance DataFrame
        self.feature_importance = pd.DataFrame({
            'feature': list(aggregated_shap.keys()),
            'importance_abs_mean': [np.mean(np.abs(values)) for values in aggregated_shap.values()],
            'importance_mean': [np.mean(values) for values in aggregated_shap.values()],
            'importance_std': [np.std(values) for values in aggregated_shap.values()]
        })
        
        # Sort by absolute importance
        feature_importance_abs_mean = self.feature_importance.sort_values(
            'importance_abs_mean', 
            ascending=False
        ).reset_index(drop=True)
        
        
        return aggregated_shap, feature_importance_abs_mean, 
     
    def _create_explainer(self, model: BaseEstimator, 
                         data: Optional[np.ndarray] = None) -> Any:
        """Create appropriate SHAP explainer based on model type"""
        classifier = model.named_steps['classifier']
        model_name = classifier.__class__.__name__

        try:
            if model_name in ModelConfig.TREE_BASED_MODELS:
                return shap.TreeExplainer(classifier)
            elif model_name in ModelConfig.DEEP_LEARNING_MODELS:
                if data is None:
                    raise ValueError("Background data required for DeepExplainer")
                return shap.DeepExplainer(model, data)
            elif model_name in ModelConfig.LINEAR_MODELS:
                return shap.LinearExplainer(model, data)
            else:
                raise ValueError(f"Unsupported model type: {model_name}")
        except Exception as e:
            raise Exception(f"Error creating SHAP explainer: {str(e)}")
        
    def plot_all(self, model: BaseEstimator, 
                 X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series,
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
        if not all([self.feature_importance is not None,
                   self.aggregated_shap is not None,
                   self.explainer is not None]):
            raise ValueError("Must run calculate_feature_importance before plotting")

        self.model_visualiser.create_all_plots(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_importance=self.feature_importance,
            aggregated_shap=self.aggregated_shap,
            explainer=self.explainer,
            class_to_explain=class_to_explain,
            output_suffix=output_suffix
        )

        # self.feature_plotter.plot_feature_importance(self.feature_importance)
        # self.shap_plotter.plot_shap_summary(self.aggregated_shap, X_test)
        # self.shap_plotter.plot_waterfall(best_model, X_test, class_to_explain,
        #                                self.explainer, self.aggregated_shap)
        # self.learning_curve_plotter.plot(best_model, X, y)