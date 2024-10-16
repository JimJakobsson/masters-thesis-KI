from sklearn.ensemble import RandomForestClassifier

from PreProcessing import PreProcessing  # Import the class
from Evaluator import Evaluator  # Import the class
from ServerConnectionIPT1 import ServerConnectionIPT1  # Import the class
from MLExperiment import MLExperiment  # Import the class


class Main:
    def main():
        rf_model = RandomForestClassifier(random_state=42)
        rf_params = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2'],
        }
        preprocessor = PreProcessing()
        evaluator = Evaluator()
        connection = ServerConnectionIPT1()

        experiment = MLExperiment(rf_model, rf_params, preprocessor, evaluator, connection)

        experiment.run()
    if __name__ == "__main__":
        main()
