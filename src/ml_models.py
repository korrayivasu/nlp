"""
Classical Machine Learning Models for Academic Paper Classification.
Includes Calibrated Linear SVM, Complement Naive Bayes, Logistic Regression,
and Ensemble Voting Classifiers.
"""

import os
import joblib
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


class ClassicalMLClassifier:
    """
    Wrapper for Scikit-Learn text classification models with calibrated probability outputs.
    """

    MODEL_TYPES = {
        "calibrated_svm": "Calibrated Linear SVM",
        "complement_nb": "Complement Naive Bayes",
        "logistic_regression": "Logistic Regression (L2)",
        "random_forest": "Random Forest Classifier",
        "ensemble_voting": "Soft Voting Ensemble"
    }

    def __init__(self, model_type: str = "calibrated_svm", **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.classes_ = None
        self.model = self._initialize_model(model_type, **kwargs)

    def _initialize_model(self, model_type: str, **kwargs) -> Any:
        if model_type == "calibrated_svm":
            # LinearSVC wrapped in CalibratedClassifierCV for calibrated class probabilities
            base_svc = LinearSVC(C=kwargs.get("C", 1.0), max_iter=2000, random_state=42)
            return CalibratedClassifierCV(estimator=base_svc, cv=3)

        elif model_type == "complement_nb":
            alpha = kwargs.get("alpha", 0.5)
            norm = kwargs.get("norm", False)
            return ComplementNB(alpha=alpha, norm=norm)

        elif model_type == "multinomial_nb":
            alpha = kwargs.get("alpha", 0.5)
            return MultinomialNB(alpha=alpha)

        elif model_type == "logistic_regression":
            C = kwargs.get("C", 2.0)
            max_iter = kwargs.get("max_iter", 1000)
            return LogisticRegression(C=C, max_iter=max_iter, random_state=42, solver="lbfgs")

        elif model_type == "random_forest":
            n_estimators = kwargs.get("n_estimators", 150)
            max_depth = kwargs.get("max_depth", None)
            return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)

        elif model_type == "ensemble_voting":
            svc = CalibratedClassifierCV(estimator=LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=3)
            cnb = ComplementNB(alpha=0.5)
            lr = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
            return VotingClassifier(
                estimators=[
                    ("svm", svc),
                    ("cnb", cnb),
                    ("lr", lr)
                ],
                voting="soft"
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(self.MODEL_TYPES.keys())}")

    def fit(self, X, y):
        """Fit model on feature matrix X and target labels y."""
        self.model.fit(X, y)
        self.classes_ = getattr(self.model, "classes_", np.unique(y))
        return self

    def predict(self, X) -> np.ndarray:
        """Predict top class labels."""
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probability distribution."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X)
            # Softmax approximation
            exp_d = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_d / exp_d.sum(axis=1, keepdims=True)
        else:
            raise AttributeError(f"Model {self.model_type} does not support probability estimation.")

    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """Evaluate model performance and return metric dictionary."""
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "classification_report": report,
            "y_true": list(y_test),
            "y_pred": list(y_pred)
        }

    def save(self, filepath: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "classes": self.classes_, "type": self.model_type}, filepath)

    def load(self, filepath: str):
        """Load model from disk."""
        data = joblib.load(filepath)
        self.model = data["model"]
        self.classes_ = data["classes"]
        self.model_type = data.get("type", "unknown")
        return self


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=20, n_classes=5, n_informative=10, random_state=42)
    clf = ClassicalMLClassifier("calibrated_svm")
    clf.fit(X, y)
    probs = clf.predict_proba(X[:2])
    print("Probabilities shape:", probs.shape)
    print("Classes:", clf.classes_)
