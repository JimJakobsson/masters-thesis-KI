class OptunaWrapper:
    def __init__(self, pipeline, study):
        self.best_estimator_ = pipeline
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        self.study = study
        self.threshold = 0.5  # Default threshold value

    def predict(self, X):
        """Predict using the optimal threshold"""
        probas = self.best_estimator_.predict_proba(X)
        return (probas[:, 1] >= self.threshold).astype(int)
    
    def predict_proba(self, X):
        """Delegate predict_proba to the best estimator"""
        return self.best_estimator_.predict_proba(X)
    
    def fit(self, X, y):
        """Delegate fit to the best estimator"""
        return self.best_estimator_.fit(X, y)
    
    def score(self, X, y):
        """Delegate score to the best estimator"""
        return self.best_estimator_.score(X, y)
    
    @property
    def named_steps(self):
        """Access named steps of the pipeline"""
        return self.best_estimator_.named_steps
#Old wrapper prior to thresholding 2025-01-04
# class OptunaWrapper:
#     def __init__(self, pipeline, study):
#         self.best_estimator_ = pipeline
#         self.best_params_ = study.best_params
#         self.best_score_ = study.best_value
#         self.study = study

#     def predict(self, X):
#         """Delegate predict to the best estimator"""
#         return self.best_estimator_.predict(X)
    
#     def predict_proba(self, X):
#         """Delegate predict_proba to the best estimator"""
#         return self.best_estimator_.predict_proba(X)
    
#     def fit(self, X, y):
#         """Delegate fit to the best estimator"""
#         return self.best_estimator_.fit(X, y)
    
#     def score(self, X, y):
#         """Delegate score to the best estimator"""
#         return self.best_estimator_.score(X, y)