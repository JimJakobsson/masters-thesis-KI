from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer

from preprocessing.preprocessing_result import PreprocessingResult
from config.preprocessing_config import PreprocessingConfig
from .feature_detector import FeatureDetector
from .label_processor import LabelProcessor
from .data_cleaner import DataCleaner
from .pipeline_creator import PipelineCreator

class DataPreprocessor:
    """Main preprocessing class that orchestrates the preprocessing pipeline"""
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = PreprocessingConfig()
        # self.config = config
        self.feature_detector = FeatureDetector()
        self.label_processor = LabelProcessor()
        self.data_cleaner = DataCleaner(self.config.NULL_THRESHOLD)
        self.feature_names: Optional[list] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.categorical_features: Optional[List[str]] = None
        self.numeric_features: Optional[List[str]] = None
    
    def create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Creates and configures the preprocessor without applying it.
        
        Args:
            X: Input features to detect types from
            
        Returns:
            Configured ColumnTransformer ready to be used in a pipeline
        """
        # Clean data first since we need it for feature detection
        X_cleaned = self.data_cleaner.clean_features(X)
     
        # Detect features types
        self.categorical_features, self.numeric_features = self.feature_detector.detect_feature_types(
            X_cleaned,
            max_unique=self.config.MAX_UNIQUE_VALUES_FOR_CATEGORICAL,
            min_count=self.config.MIN_COUNT_FOR_CATEGORICAL,
            max_categories=self.config.MAX_CATEGORIES_PER_FEATURE
        )
        # preproc_result = PreprocessingResult(
        #     X=X_cleaned,
        #     preprocessor=self.preprocessor,
        #     feature_names=self.feature_names,
        #     numeric_features=self.numeric_features,
        #     categorical_features=self.categorical_features,

        # )
        # Validate numeric columns
        self._validate_numeric_columns(X_cleaned)
        
        # Create preprocessor
        pipeline_creator = PipelineCreator(
            categorical_features = self.categorical_features,
            numeric_features = self.numeric_features,
        )
        
        pipeline = pipeline_creator.create_columntransformer() 
    
        return pipeline
      
    def _validate_numeric_columns(self, X: pd.DataFrame) -> None:
        """Validate numeric columns and raise informative errors if issues are found"""
        if self.numeric_features is None:
            return
            
        for col in self.numeric_features:
            non_numeric_mask = pd.to_numeric(X[col], errors='coerce').isna() & X[col].notna()
            if non_numeric_mask.any():
                problematic_values = X.loc[non_numeric_mask, col].unique()
                raise ValueError(
                    f"Column '{col}' contains non-numeric values: {problematic_values}. "
                    "Please check your data or consider treating this column as categorical."
                )
    def create_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create labels from raw data"""
        return self.label_processor.create_labels(data)

    def get_features_and_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target from labeled data"""
        X = data.drop(['labels', 'death_yrmon', 'twinnr', 'punching'], axis=1)
        y = data['labels']
        return X, y

    # def process(self, X: pd.DataFrame, fit: bool = True) -> PreprocessingResult:
    #     """
    #     Process data with option to fit or just transform.
        
    #     Args:
    #         X: Input features
    #         fit: Whether to fit the preprocessor (True for training, False for test)
            
    #     Returns:
    #         PreprocessingResult containing processed data and metadata
    #     """
    #     try:
    #         # Clean data (this doesn't require fitting)
    #         X_cleaned = self.data_cleaner.clean_features(X)
            
    #         if fit:
    #             # Only detect features and create pipeline when fitting
    #             self.categorical_features, self.numeric_features = self.feature_detector.detect_feature_types(
    #                 X_cleaned,
    #                 max_unique=self.config.MAX_UNIQUE_VALUES_FOR_CATEGORICAL,
    #                 min_count=self.config.MIN_COUNT_FOR_CATEGORICAL
    #             )
    #             # Validate numeric columns before processing
    #             self._validate_numeric_columns(X_cleaned)
    #             # Create and fit preprocessor
    #             pipeline_creator = PipelineCreator(
    #                 self.numeric_features, 
    #                 self.categorical_features
    #             )
    #             self.preprocessor = pipeline_creator.create_columntransformer()

    #             print("Original feature names:", list(X_cleaned.columns))
    #             X_transformed = self.preprocessor.fit_transform(X_cleaned)
    #             self.feature_names = pipeline_creator.get_feature_names(self.preprocessor)
    #             # print("Transformed feature names:", self.feature_names)
   
    #             # Get feature names after fitting
    #             # self.feature_names = pipeline_creator.get_feature_names(self.preprocessor)
                
    #         else:
    #             # Verify preprocessor exists
    #             if self.preprocessor is None:
    #                 raise ValueError("Preprocessor must be fitted before transform")
                
    #             # Only transform using existing preprocessor
    #             X_transformed = self.preprocessor.transform(X_cleaned)
            
    #         # Create DataFrame with feature names
    #         X_processed = pd.DataFrame(
    #             X_transformed,
    #             columns=self.feature_names,
    #             index=X.index
    #         )
            
    #         return PreprocessingResult(
    #             X=X_processed,
    #             preprocessor=self.preprocessor,
    #             feature_names=self.feature_names,
    #             numeric_features=self.numeric_features,
    #             categorical_features=self.categorical_features,
    #         )
            
    #     except Exception as e:
    #         print(f"\nError processing features. Data sample:")
    #         print(X.head())
    #         print("\nFeature types:")
    #         print(X.dtypes)
    #         raise RuntimeError(f"Error in preprocessing: {str(e)}") from e

    def get_feature_names(self) -> List[str]:
        """Get current feature names"""
        if self.feature_names is None:
            raise ValueError("No feature names available. Preprocessor must be fitted first.")
        return self.feature_names
    
    def get_numeric_features(self) -> List[str]:
        """Get current numeric features"""
        if self.numeric_features is None:
            raise ValueError("No numeric features available. Must fit data first.")
        return self.numeric_features
    def get_categorical_features(self) -> List[str]:
        """Get current categorical features"""
        if self.categorical_features is None:
            raise ValueError("No categorical features available. Must fit data first.")
        return self.categorical_features

    def get_preprocessor(self) -> ColumnTransformer:
        """Get current preprocessor"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not available. Must fit data first.")
        return self.preprocessor