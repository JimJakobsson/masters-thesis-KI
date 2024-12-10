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
                'classifier__bootstrap': [True, False],  # Try both bootstrapping options
                'classifier__ccp_alpha': [0.1, 0.001, 0.0001],  # Add pruning options
                'classifier__class_weight': [
                    {0: 1, 1: 3},
                    {0: 1, 1: 2.5},
                    {0: 1, 1: 2.}
                    
                ],  # More class weight ratios
                'classifier__criterion': ['entropy'],  # Try both split criteria
                'classifier__max_depth': [20, 30],  # Search around successful depth
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
            param_grid={'classifier__class_weight':[ {0: 1, 1: 2}], 
                        'classifier__criterion': ['gini'], 
                        'classifier__max_depth': [15], 
                        'classifier__max_features': [None], 
                        'classifier__min_impurity_decrease':[ 0.0001], 
                        'classifier__min_samples_leaf':[ 1], 
                        'classifier__min_samples_split':[ 5]},
            description='Decision tree classifier optimized for medical data analysis with enhanced flexibility'
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
        """Get the configuration for an enhanced bagging model suitable for medical data
        
        Returns:
            ModelConfig: Configuration with diverse estimators and parameters
        """
        # Define base estimators with different configurations
        base_estimators = [
            # HistGradientBoostingClassifier for native NULL handling
            HistGradientBoostingClassifier(
                random_state=42,
                max_depth=10,
                learning_rate=0.1,
                early_stopping=True
            ),
            
            
            DecisionTreeClassifier(
                max_depth=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=43
            ),
            
            # More flexible decision tree
            DecisionTreeClassifier(
                max_depth=None,
                min_samples_leaf=20,
                min_samples_split=10,
                random_state=44
            )
        ]
        
        return ModelConfig(
            name='Bagging',
            model=BaggingClassifier(random_state=42),
            param_grid={
                # Core bagging parameters
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_samples': [0.5, 0.7, 1.0],
                'classifier__max_features': [0.6, 0.8, 1.0],
                
                # Sampling strategy
                'classifier__bootstrap': [True],  # Bootstrap sampling for robustness
                'classifier__bootstrap_features': [False, True],  # Try both feature sampling approaches
                
                # Try different base estimators
                'classifier__estimator': base_estimators,
                
                # Fixed parameters for reproducibility
                'classifier__random_state': [42],
                
                # Parallel processing
                'classifier__n_jobs': [-1]  # Use all available cores
            },
            description='Enhanced bagging model with diverse estimators for medical data'
        )

    
    def get_stacking_config() -> ModelConfig:
        
        """Get the configuration for a stacking model with properly structured parameter grid
        
        This code defines a stacking classifier, which is an ensemble learning method that 
        combines multiple base models.

        The stacking approach:

        Takes predictions from both base models using probability estimates ('predict_proba')
        Uses another Gradient Boosting Classifier as the final meta-learner
        Can weight the importance of each base model's predictions (controlled by 'classifier__weights' parameter)

        Returns:
        ModelConfig: Configuration with model, parameters, and metadata
        """
        base_estimators = [
            ('hgb', HistGradientBoostingClassifier(
                random_state=42,
                class_weight={0: 1, 1: 2},
                early_stopping=True,
                l2_regularization=10.0,
                learning_rate=0.3,
                max_bins=225,
                max_depth=6,
                max_iter=100,
                min_samples_leaf=20,
                n_iter_no_change=10,
                validation_fraction=0.1
            )),
            ('rf', RandomForestClassifier(
                random_state=42,
                bootstrap=False,
                ccp_alpha=0.001,
                class_weight={0: 1, 1: 2.5},
                criterion='entropy',
                max_depth=20,
                max_features='sqrt',
                min_samples_leaf=1,
                min_samples_split=12,
                n_estimators=100
            )),
            
        ]
        
        # Create the StackingClassifier
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            # final_estimator=HistGradientBoostingClassifier(random_state=42),
            final_estimator=RandomForestClassifier(random_state=42),
            stack_method='predict_proba'
        )
        
        # Only tune the final estimator parameters
        param_grid = {
            # Final estimator parameters
            'classifier__final_estimator__n_estimators': [100, 200, 300],
            'classifier__final_estimator__max_depth': [20, 30, 40],
            'classifier__final_estimator__min_samples_leaf': [5, 10, 15],
            'classifier__final_estimator__max_features': ['sqrt'],
            'classifier__final_estimator__bootstrap': [False],
            
            'classifier__final_estimator__min_samples_split': [5, 10, 15],
            
            # Stack weights for base estimators
            'classifier__final_estimator__class_weight': [
                {0: 1, 1: 1},  # Equal weights
                {0: 1, 1: 2},  # Emphasis on HistGradientBoosting
                {0: 2, 1: 1},  # Emphasis on RandomForest
               ]  
        }
       
        return ModelConfig(
            name='StackingClassifier',
            model=stacking_clf,
            param_grid=param_grid,
            description='Stacking classifier for medical data that handles NULL values natively'
        )

    @staticmethod
    def get_voting_config() -> ModelConfig:
        """Get configuration for VotingClassifier with diverse base estimators
        
        Returns:
            ModelConfig: Configuration with model, parameters, and metadata
        """
        # Define estimators with preprocessing where needed
        # estimators = [
        #     ('hgb', HistGradientBoostingClassifier(random_state=42)),  # Best native NaN handling
        #     ('rf', RandomForestClassifier(random_state=42)),           # Handles NaN via surrogate splits
        #     ('et', ExtraTreesClassifier(random_state=42))             # Another approach to tree building
        # ]
        base_estimators = [
            ('hgb', HistGradientBoostingClassifier(
                random_state=42,
                class_weight={0: 1, 1: 2},
                early_stopping=True,
                l2_regularization=10.0,
                learning_rate=0.3,
                max_bins=225,
                max_depth=6,
                max_iter=100,
                min_samples_leaf=20,
                n_iter_no_change=10,
                validation_fraction=0.1
            )),
            ('rf', RandomForestClassifier(
                random_state=42,
                bootstrap=False,
                ccp_alpha=0.001,
                class_weight={0: 1, 1: 2.5},
                criterion='entropy',
                max_depth=20,
                max_features='sqrt',
                min_samples_leaf=1,
                min_samples_split=12,
                n_estimators=100
            )),
            
        ]

        voting_clf = VotingClassifier(
            estimators=base_estimators,
            voting='soft'  # Use probability predictions
        )

        return ModelConfig(
            name='VotingClassifier',
            model=voting_clf,
            param_grid={
                
                # Weights for each classifier
                'classifier__weights': [[1, 1],    # Equal weights
                        [2, 1],     # Emphasis on HistGradientBoosting
                        [1, 2]]     # Emphasis on RandomForest
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

    