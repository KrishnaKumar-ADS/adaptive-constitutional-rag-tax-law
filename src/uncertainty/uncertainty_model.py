from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

class UncertaintyModel:
    def __init__(self):
        # We use a small, simple classifier to avoid overfitting 
        # on the ~4 features (retrieval_confidence, evidence_agreement, coverage, entropy)
        self.model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the classifier.
        X: array of shape (n_samples, 4) containing the features.
        y: array of shape (n_samples,) containing binary labels (1=hallucination/invalid, 0=valid)
        """
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the raw probability of class 1 (hallucination).
        """
        return self.model.predict_proba(X)[:, 1]
