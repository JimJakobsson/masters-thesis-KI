# from ..config.model_config import ModelConfig
# from ..config.path_config import PathConfig
# from .base_evaluator import BaseEvaluator
# from .metrics import calculate_classification_metrics
# from ..visualization.feature_plots import FeaturePlotter
# from ..visualization.shap_plots import ShapPlotter
# from ..visualization.learning_curves import LearningCurvePlotter
# from ..utils.preprocessing import get_feature_names_after_preprocessing
# from ..utils.validation import validate_shap_calculation

# class ModelEvaluator(BaseEvaluator):
#     """Handles model evaluation and SHAP explanationss"""

#     def __init__(self, output_dir: Optional[str] = None):
#         self.results: Dict = {}
#         self.output_dir = Path(output_dir) if output_dir else PathConfig.OUTPUT_DIR
#         self.shap_values: Optional[np.ndarray] = None
#         self.feature_importance: Optional[pd.DataFrame] = None
#         self.explainer: Optional[Any] = None
#         self.aggregated_shap: Optional[Dict] = None
        
#         # Initialize plotters
#         self.feature_plotter = FeaturePlotter(self.output_dir)
#         self.shap_plotter = ShapPlotter(self.output_dir)
#         self.learning_curve_plotter = LearningCurvePlotter(self.output_dir)

    