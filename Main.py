from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from PreProcessing import PreProcessing  # Import the class
from Evaluator import Evaluator  # Import the class
from ServerConnectionIPT1 import ServerConnectionIPT1  # Import the class
from MLExperiment import MLExperiment  # Import the class


class Main:
    def main():
        """
        This function represents the main entry point of the program.

        It performs the following steps:
        1. Initializes a RandomForestClassifier model with a random state of 42.
        2. Defines a dictionary of hyperparameters for the random forest model.
           - 'n_estimators': a list of the number of trees in the forest to consider.
           - 'max_depth': a list of the maximum depth of the tree.
           - 'min_samples_split': a list of the minimum number of samples required to split an internal node.
           - 'min_samples_leaf': a list of the minimum number of samples required to be at a leaf node.
           - 'max_features': a list of the number of features to consider when looking for the best split.
           - 'bootstrap': a list of boolean values indicating whether bootstrap samples are used when building trees.
           - 'criterion': a list of the function to measure the quality of a split.
           - 'class_weight': a list of the weights associated with classes in the target variable.
           - 'ccp_alpha': a list of non-negative values for Minimal Cost-Complexity Pruning.

        3. Initializes a PreProcessing object for data preprocessing.
        4. Initializes an Evaluator object for model evaluation.
        5. Initializes a ServerConnectionIPT1 object for server connection.
        6. Initializes an MLExperiment object with the random forest model, hyperparameters, preprocessor, evaluator, and connection.
        7. Runs the experiment.

        This function does not return any value.
        """
        #  rf_params = {
        #     'classifier__n_estimators': [50, 100, 200],
        #     'classifier__max_depth': [10, 20, None],
        #     'classifier__min_samples_split': [2, 5, 10],
        #     'classifier__min_samples_leaf': [1, 2, 4],
        #     'classifier__max_features': ['sqrt', 'log2'],
        #     'classifier__bootstrap': [True, False],
        #     'classifier__criterion': ['gini', 'entropy'],
        #     'classifier__class_weight': [None, 'balanced'],
        #     'classifier__ccp_alpha': [0.0, 0.1]
        # }

        rf_model = RandomForestClassifier(random_state=42)
        rf_params = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [10, None],
            'classifier__min_samples_split': [2, 10],
            'classifier__min_samples_leaf': [1, 4],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__bootstrap': [True, False],
            'classifier__criterion': ['gini', 'entropy']
        }
        preprocessor = PreProcessing()
        evaluator = Evaluator()
        connection = ServerConnectionIPT1()
        # connection.read_table()
        experiment = MLExperiment(rf_model, rf_params, preprocessor, evaluator, connection)

        experiment.run()

    if __name__ == "__main__":
        main()
