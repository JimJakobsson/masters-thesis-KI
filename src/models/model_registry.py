from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from models.model_config import ModelConfig

class ModelRegistry:
    """Registry for machine learning models"""

    @staticmethod
    def get_random_forest_config() -> ModelConfig:
        """Get the configuration for a random forest model"""
        return ModelConfig(
            name='Random Forest',
            model=RandomForestClassifier(),
             param_grid={
                'classifier__bootstrap': [False],  # Try both bootstrapping options
                'classifier__ccp_alpha': [0.1, 0.001, 0.0001],  # Add pruning options
                'classifier__class_weight': [
                    {0: 1, 1: 3},
                    {0: 1, 1: 2.5},
                    {0: 1, 1: 2.}
                    
                ],  # More class weight ratios
                'classifier__criterion': ['entropy'],  # Try both split criteria
                'classifier__max_depth': [20, 30, None],  # Search around successful depth
                'classifier__max_features': ['sqrt'],  # Both feature selection methods
                'classifier__min_samples_leaf': [1],  # Vary leaf size requirements
                'classifier__min_samples_split': [3, 6, 9, 12],  # Vary split requirements
                'classifier__n_estimators': [80, 90, 95, 100],  # Search around successful number
                'classifier__random_state': [42]  # Keep for reproducibility
                
            },
            description='A random forest model'
        )
    @staticmethod
    def get_decision_tree_config() -> ModelConfig:
        """Get configuration for DecisionTreeClassifier optimized for medical data with 400 features
        Returns:
            ModelConfig: Configuration with model, parameters, and metadata
        """
        return ModelConfig(
            name='DecisionTree',
            model=DecisionTreeClassifier(random_state=42),
            param_grid={
                # Control tree depth
                'classifier__max_depth': [10, 20,30],  # Relatively shallow for interpretability
                
                # Minimum samples parameters
                'classifier__min_samples_split': [10, 20, 50],  # Larger values for stability
                'classifier__min_samples_leaf': [5, 10, 20],    # Prevent very small leaf nodes
                
                # Feature selection parameters
                'classifier__max_features': ['sqrt', 'log2'],  # Reduce feature space
                
                # Handle imbalanced classes
                'classifier__class_weight': ['balanced', None],
                
                # Splitting criteria
                'classifier__criterion': ['gini', 'entropy'],
                
                # Prevent perfect splits that might be noise
                'classifier__min_impurity_decrease': [0.0001, 0.001, 0.01]
            },
            description='Decision tree classifier optimized for medical data analysis'
        )

    @staticmethod
    def get_hist_gradient_boosting_config() -> ModelConfig:
        """Get the configuration for a histogram gradient boosting model"""
        return ModelConfig(
            name='Histogram Gradient Boosting',
            model=HistGradientBoostingClassifier(),
             param_grid={
                'classifier__learning_rate': [0.3, 0.4],  # Search around 0.3
                'classifier__max_depth': [5, 6],  # Search around 5
                'classifier__max_iter': [100, 125],  # Search around 100
                'classifier__min_samples_leaf': [20, 25],  # Search around 20
                'classifier__l2_regularization': [10.0, 15.0],  # Search around 10.0
                'classifier__max_bins': [225, 255],  # Search around 255
                'classifier__class_weight': [
                    # {0: 1, 1: 1.75},
                    {0: 1, 1: 2},
                    # {0: 1, 1: 2.25}
                ],  # Fine-tune class weights around 2
                'classifier__early_stopping': [True],  # Keep this as it worked well
                'classifier__validation_fraction': [0.1, 0.15],  # Try slightly larger validation set
                'classifier__n_iter_no_change': [10, 12],  # Search around 10
                'classifier__random_state': [42]  # Keep for reproducibility
            },
            description='A histogram gradient boosting model'
        )
    
    def get_bagging_config() -> ModelConfig:
        """Get the configuration for a bagging model"""
        return ModelConfig(
            name='Bagging',
            model=BaggingClassifier(),
            param_grid={
            'classifier__n_estimators': [10, 50, 100, 200],  # Number of base estimators
            'classifier__max_samples': [0.5, 0.7, 0.8, 1.0],  # Fraction of samples to draw
            'classifier__max_features': [0.5, 0.7, 0.8, 1.0],  # Fraction of features to draw
            'classifier__bootstrap': [True],  # Whether to sample with replacement
            'classifier__bootstrap_features': [False],  # Whether to sample features with replacement
            'classifier__estimator': [
                # DecisionTreeClassifier(max_depth=10),
                DecisionTreeClassifier(max_depth=20),
                DecisionTreeClassifier(max_depth=30),
                DecisionTreeClassifier()  # Unlimited depth
            ],
            'classifier__random_state': [42]  # For reproducibility
        },
            description='A bagging model'
        )
    
    @staticmethod
    def get_stacking_config() -> ModelConfig:
        """Get the configuration for a stacking model
        Returns:
        ModelConfig: Configuration with model, parameters, and metadata
        """
        # Define base estimators with preprocessing
        base_estimators = [
            ('hgb', HistGradientBoostingClassifier(random_state=42)),  # Best native null handling
            ('rf', RandomForestClassifier(random_state=42)),           # Handles nulls via surrogate splits
            ('et', ExtraTreesClassifier(random_state=42))             # Different tree-building strategy
        ]
        
        return ModelConfig(
        name='StackingClassifier',
        model=StackingClassifier(
            estimators=base_estimators,
            final_estimator=HistGradientBoostingClassifier(random_state=42),
            stack_method='predict_proba'
        ),
        param_grid={
            # HistGradientBoosting parameters (best with nulls)
            'classifier__estimators__hgb__class_weight': [{0: 1, 1: 2}], 
            'classifier__estimators__hgb__early_stopping': [True], 
            'classifier__estimators__hgb__l2_regularization': [10.0],
            'classifier__estimators__hgb__learning_rate': [0.3],
            'classifier__estimators__hgb__max_bins': [225],
            'classifier__estimators__hgb__max_depth': [6], 
            'classifier__estimators__hgb__max_iter': [100],
            'classifier__estimators__hgb__min_samples_leaf':[ 20], 
            'classifier__estimators__hgb__n_iter_no_change': [10],
            'classifier__estimators__hgb__random_state': [42],
            'classifier__estimators__hgb__validation_fraction': [0.1],
            
            # RandomForest parameters           
            'classifier__estimators__rf__bootstrap': [False],
            'classifier__estimators__rf__ccp_alpha': [0.001], 
            'classifier__estimators__rf__class_weight': [{0: 1, 1: 2.5}],
            'classifier__estimators__rf__criterion': ['entropy'],
            'classifier__estimators__rf__max_depth': [20],
            'classifier__estimators__rf__max_features': ['sqrt'], 
            'classifier__estimators__rf__min_samples_leaf':[ 1],
            'classifier__estimators__rf__min_samples_split': [12], 
            'classifier__estimators__rf__n_estimators': [100],
            'classifier__estimators__rf__random_state': [42],

            # ExtraTrees parameters
            'classifier__estimators__et__n_estimators': [100, 200],
            'classifier__estimators__et__max_features': ['sqrt', 'log2'],
            'classifier__estimators__et__min_samples_leaf': [10, 20],
            'classifier__estimators__et__max_depth': [10, 20, None],
            'classifier__estimators__et__class_weight': ['balanced', 'balanced_subsample'],
            
            # Meta-classifier parameters (HistGradientBoosting)
            'classifier__final_estimator__max_iter': [100, 200],
            'classifier__final_estimator__learning_rate': [0.01, 0.1],
            'classifier__final_estimator__max_depth': [3, 5],
            'classifier__final_estimator__l2_regularization': [1.0, 10.0],
            
            
        },
        description='Stacking classifier for medical data that handles NULL values natively'
        )

    @staticmethod
    def get_voting_config() -> ModelConfig:
        """Get configuration for VotingClassifier with diverse base estimators
        
        Returns:
            ModelConfig: Configuration with model, parameters, and metadata
        """
        # Define estimators with preprocessing where needed
        estimators = [
            ('hgb', HistGradientBoostingClassifier(random_state=42)),  # Best native NaN handling
            ('rf', RandomForestClassifier(random_state=42)),           # Handles NaN via surrogate splits
            ('et', ExtraTreesClassifier(random_state=42))             # Another approach to tree building
        ]
        
        return ModelConfig(
            name='VotingClassifier',
            model=VotingClassifier(
                estimators=estimators,
                voting='soft'  # Use probability predictions
            ),
            param_grid={
                    # HistGradientBoosting parameters (best with nulls)
                'classifier__class_weight': [{0: 1, 1: 2}], 
                'classifier__early_stopping': [True], 
                'classifier__l2_regularization': 10.0,
                'classifier__learning_rate': 0.3,
                'classifier__max_bins': 225,
                'classifier__max_depth': 6, 
                'classifier__max_iter': 100,
                'classifier__min_samples_leaf': 20, 
                'classifier__n_iter_no_change': 10,
                'classifier__random_state': 42,
                'classifier__validation_fraction': 0.1,
                
                # RandomForest parameters           
                'classifier__bootstrap': False,
                'classifier__ccp_alpha': 0.001, 
                'classifier__class_weight': [{0: 1, 1: 2.5}],
                'classifier__criterion': ['entropy'],
                'classifier__max_depth': 20,
                'classifier__max_features': ['sqrt'], 
                'classifier__min_samples_leaf': 1,
                'classifier__min_samples_split': 12, 
                'classifier__n_estimators': 100,
                'classifier__random_state': 42,
                
                # ExtraTrees parameters
                'et__n_estimators': [100, 200],
                'et__max_depth': [10, 20, None],
                'et__min_samples_leaf': [10, 20],
                'et__max_features': ['sqrt', 'log2'],
                'et__class_weight': ['balanced', 'balanced_subsample'],
                
                # Voting parameters
                'voting': ['soft'],  # Hard voting still works but soft usually better
                
                # Weights for each classifier
                'weights': [[1, 1, 1],    # Equal weights
                        [2, 1, 1],     # Emphasis on HistGradientBoosting
                        [1, 2, 1],     # Emphasis on RandomForest
                        [1, 1, 2]]     # Emphasis on ExtraTrees
            },
            description='Voting classifier with NaN-handling estimators'
    )
    @staticmethod
    def get_logistic_regression_config() -> ModelConfig:
        """Get the configuration for a logistic regression model"""
        return ModelConfig(
            name='Logistic Regression',
            model=LogisticRegression(),
            param_grid={
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga'],
                'classifier__max_iter': [100, 500, 1000],
                'classifier__class_weight': [None, 'balanced'],
                'classifier__random_state': [42]
            },
            description='A logistic regression model'
        )

    # @staticmethod
    # def get_gradient_boosting_config() -> ModelConfig:
    #     """Get the configuration for a gradient boosting model"""
    #     return ModelConfig(
    #         name='Gradient Boosting',
    #         model=GradientBoostingClassifier(),
    #         param_grid={
    #             'classifier__n_estimators': [100, 200, 300],
    #             'classifier__max_depth': [3, 5, 7]
    #         },
    #         description='A gradient boosting model'
    #     )
    