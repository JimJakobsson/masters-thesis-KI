class OptunaWrapper:
    def __init__(self, pipeline, study):
        self.best_estimator_ = pipeline
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        self.study = study

    def predict(self, X):
        """Delegate predict to the best estimator"""
        return self.best_estimator_.predict(X)
    
    def predict_proba(self, X):
        """Delegate predict_proba to the best estimator"""
        return self.best_estimator_.predict_proba(X)
    
    def fit(self, X, y):
        """Delegate fit to the best estimator"""
        return self.best_estimator_.fit(X, y)
    
    def score(self, X, y):
        """Delegate score to the best estimator"""
        return self.best_estimator_.score(X, y)