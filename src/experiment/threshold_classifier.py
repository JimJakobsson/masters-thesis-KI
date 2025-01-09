# class ThresholdClassifier:
#     """Wrapper for classifier that applies a custom threshold for binary classification"""
#     def __init__(self, classifier, threshold=0.5):
#         self.classifier = classifier
#         self.threshold = threshold

#     def fit(self, X, y):
#         self.classifier.fit(X, y)
#         return self

#     def predict(self, X):
#         proba = self.classifier.predict_proba(X)
#         return (proba[:, 1] >= self.threshold).astype(int)

#     def predict_proba(self, X):
#         return self.classifier.predict_proba(X)

#     def get_params(self, deep=True):
#         return {
#             "classifier": self.classifier,
#             "threshold": self.threshold
#         }

#     def set_params(self, **params):
#         if "threshold" in params:
#             self.threshold = params["threshold"]
#         if "classifier" in params:
#             self.classifier = params["classifier"]
#         return self