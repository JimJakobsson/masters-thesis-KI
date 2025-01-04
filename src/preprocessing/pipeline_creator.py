from typing import Counter, List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as Pipeline
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder

from preprocessing.null_indicator import NullIndicator

class PipelineCreator:
    """Handles creation of preprocessing pipeline"""
    
    def __init__(self, categorical_features: List[str], numeric_features: List[str]):
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
    
    def create_columntransformer(self) -> ColumnTransformer:
        """
        Create a preprocessing pipeline with numeric and categorical transformers.

        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        print("\nCreating ColumnTransformer:")
        print(f"Numeric features ({len(self.numeric_features)}):", self.numeric_features[:5], "...")
        print(f"Categorical features ({len(self.categorical_features)}):", self.categorical_features[:5], "...")
        
        return ColumnTransformer(
            transformers=[
                ('num', self._create_numeric_pipeline(), self.numeric_features),
                ('cat', self._create_categorical_pipeline(), self.categorical_features)
            ],
            remainder='drop',
            n_jobs=None
        )
    
    @staticmethod
    def _create_numeric_pipeline() -> Pipeline:
        """Create a pipeline for numeric features"""
        return Pipeline(steps=[
            # imputer possible but ignored for now
            ('null_indicator', NullIndicator()),
            # ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
    
    @staticmethod
    def _create_categorical_pipeline() -> Pipeline:
        """Create a pipeline for categorical features"""
        return Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', 
                                     sparse_output=False, 
                                     
                                     ))	
        ])
    
    # @staticmethod
    # def create_pipeline(column_transformer: ColumnTransformer, model) -> Pipeline:
    #     """Create a pipeline with the specified column transformer and model"""
    #     def print_data_distribution(X, y, stage):
    #         print(f"\nData distribution {stage} SMOTE:")
    #         print(Counter(y))

    #     def smote_transform(X, y):
    #         print_data_distribution(X, y, "before")
    #         smote = SMOTE(random_state=42, sampling_strategy='minority')
    #         X_resampled, y_resampled = smote.fit_resample(X, y)
    #         print_data_distribution(X_resampled, y_resampled, "after")
    #         return X_resampled, y_resampled

    #     return Pipeline(steps=[
    #         ('preprocessor', column_transformer),
    #         ('smote', FunctionTransformer(smote_transform, validate=False)),
    #         ('classifier', model)
    #     ])
     
    # def check_data_consistency(self, X_train, X_test):
    #     """Check for consistency between training and test data"""
    #     print("\nData Consistency Check:")
    #     print(f"Training transformed data shape: {X_train.shape}")
    #     print(f"Test transformed data shape: {X_test.shape}")
    #     if X_train.shape[1] != X_test.shape[1]:
    #         print(f"Warning: training data has {X_train.shape[1]} columns while test data has {X_test.shape[1]} columns")
    #     else:
    #         print("Number of columns in training and test data are the same")