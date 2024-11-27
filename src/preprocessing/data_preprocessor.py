from typing import Optional

import pandas as pd

from preprocessing.preprocessing_result import PreprocessingResult
from ..config import PreprocessingConfig
from .feature_detector import FeatureDetector
from .label_processor import LabelProcessor
from .data_cleaner import DataCleaner
from .pipeline_creator import PreprocessingPipeline

class DataPreprocessor:
    """Main preprocessing class that orchestrates the preprocessing pipeline"""
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.feature_detector = FeatureDetector()
        self.label_processor = LabelProcessor()
        self.data_cleaner = DataCleaner(self.config.null_threshold)
        self.featuer_names: Optional[list] = None

    def process(self, df: pd.DataFrame) -> PreprocessingPipeline:
        """Process the input data.
            Complete preprocessing pipeline that detects features, creates labels, removes high null features,
        """
        # Create labels
        df = self.label_processor.create_labels(df)
        
        # Separate features and target
        X = df.drop(['labels', 'death_yrmon'], axis=1)
        y = df['labels']
        
        # Clean data
        X, y = self.data_cleaner.remove_high_null_features(X, y)
        
        # Detect features
        categorical_features, numeric_features = FeatureDetector.detect_feature_types(
            X,
            max_unique=self.config.MAX_UNIQUE_VALUES_FOR_CATEGORICAL,
            min_count=self.config.MIN_COUNT_FOR_CATEGORICAL
        )
        
        # Create preprocessing pipeline
        pipeline = PreprocessingPipeline(numeric_features, categorical_features)
        preprocessor = pipeline.create()
        
        # Get feature names
        feature_names = pipeline.get_feature_names(preprocessor, X)
        
        return PreprocessingResult(X, y, preprocessor, feature_names)