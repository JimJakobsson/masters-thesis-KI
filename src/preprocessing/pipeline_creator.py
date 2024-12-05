from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
    
    @staticmethod
    def create_pipeline(column_transformer: ColumnTransformer, model) -> Pipeline:
        """Create a pipeline with the specified column transformer and model"""
        return Pipeline(steps=[
            ('preprocessor', column_transformer),
            ('classifier', model)
        ])
    
    # def get_feature_names(self, column_transformer: ColumnTransformer) -> List[str]:
    #     """Get feature names after preprocessing"""
    #     if not hasattr(column_transformer, 'transformers_'):
    #         raise ValueError("ColumnTransformer is not fitted yet. Please fit the transformer before getting feature names.")
        
    #     # Get numeric feature names
    #     numeric_transformer = column_transformer.named_transformers_['num']
    #     numeric_features = numeric_transformer.get_feature_names_out(self.numeric_features).tolist()

    #     # Get categorical feature names from the categorical transformer
    #     categorical_transformer = column_transformer.named_transformers_['cat']
    #     categorical_features = categorical_transformer.named_steps['onehot'].get_feature_names_out(
    #         self.categorical_features
    #     ).tolist()

    #     return numeric_features + categorical_features
    # @staticmethod
    # def get_feature_names(column_transformer: ColumnTransformer, input_features: Optional[List[str]] = None) -> List[str]:
    #     """Get feature names after preprocessing
        
    #     Args:
    #         column_transformer: Fitted ColumnTransformer
    #         input_features: Optional list of input feature names
            
    #     Returns:
    #         List of feature names after transformation
    #     """
    #     if not hasattr(column_transformer, 'transformers_'):
    #         raise ValueError("ColumnTransformer is not fitted yet. Please fit the transformer before getting feature names.")
        
    #     # Get the numeric transformer and its features
    #     numeric_transformer = column_transformer.named_transformers_['num']
    #     numeric_features = column_transformer.feature_names_in_[
    #         column_transformer.transformers_[0][2]  # Index for numeric features
    #     ].tolist()
        
    #     # Get the categorical transformer and its features
    #     categorical_transformer = column_transformer.named_transformers_['cat']
    #     categorical_features = column_transformer.feature_names_in_[
    #         column_transformer.transformers_[1][2]  # Index for categorical features
    #     ].tolist()
        
    #     # Get transformed feature names
    #     numeric_names = numeric_transformer.get_feature_names_out(numeric_features).tolist()
    #     categorical_names = categorical_transformer.named_steps['onehot'].get_feature_names_out(
    #         categorical_features
    #     ).tolist()
        
    #     return numeric_names + categorical_names
    
    def check_data_consistency(self, X_train, X_test):
        """Check for consistency between training and test data"""
        print("\nData Consistency Check:")
        print(f"Training transformed data shape: {X_train.shape}")
        print(f"Test transformed data shape: {X_test.shape}")
        if X_train.shape[1] != X_test.shape[1]:
            print(f"Warning: training data has {X_train.shape[1]} columns while test data has {X_test.shape[1]} columns")
        else:
            print("Number of columns in training and test data are the same")