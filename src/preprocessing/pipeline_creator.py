from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class PipelineCreator:
    """Handles creation of preprocessing pipeline"""
    
    def __init__(self, categorical_features: List[str], numeric_features: List[str]):
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
    
    def create_column_transformer(self) -> ColumnTransformer:
        """Create a preprocessing pipeline with numeric and categorical transformers"""
        return ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(), self.categorical_features)
            ],
            remainder='drop',
            n_jobs=None
        )
    
    @staticmethod
    def create_pipeline(column_transformer: ColumnTransformer, model) -> Pipeline:
        """Create a pipeline with the specified column transformer and model"""
        return Pipeline(steps=[
            ('preprocessor', column_transformer),
            ('classifier', model)
        ])
    
    def get_feature_names(self, column_transformer: ColumnTransformer) -> List[str]:
        """Get feature names after preprocessing"""
        numeric_features = self.numeric_features
        categorical_transformer = column_transformer.named_transformers_['cat']
        categorical_features = categorical_transformer.named_steps['onehot'].get_feature_names_out(
            self.categorical_features
        ).tolist()

        return numeric_features + categorical_features
    
    def check_data_consistency(self, X_train, X_test):
        """Check for consistency between training and test data"""
        print("\nData Consistency Check:")
        print(f"Training transformed data shape: {X_train.shape}")
        print(f"Test transformed data shape: {X_test.shape}")
        if X_train.shape[1] != X_test.shape[1]:
            print(f"Training data has {X_train.shape[1]} columns while test data has {X_test.shape[1]} columns")
        else:
            print("Number of columns in training and test data are the same")