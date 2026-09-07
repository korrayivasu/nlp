import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
import numpy as np
from src.ml_models import ClassicalMLClassifier
from src.deep_models import PyTorchAbstractClassifier
from src.feature_extractor import FeatureExtractor


def test_feature_extractor():
    corpus = [
        "quantum entanglement and superconductivity in physics",
        "deep neural networks and transformer models in computer science",
        "elliptic curves and algebraic varieties in mathematics"
    ]
    fe = FeatureExtractor(max_features=50)
    X = fe.fit_transform(corpus)
    assert X.shape[0] == 3
    assert X.shape[1] > 0
    assert len(fe.get_feature_names()) == X.shape[1]


def test_classical_classifiers():
    corpus = [
        "deep neural networks reinforcement learning transformer",
        "deep learning convolutional networks vision image recognition",
        "natural language processing sequence models transformer attention",
        "quantum mechanics superconductors entanglement physics matter",
        "quantum phase transition magnetic field physics particle",
        "general relativity black holes spacetime gravitational physics",
        "algebraic topology cohomology rings banach spaces math",
        "differential equations riemannian manifold lie algebra math",
        "complex analysis modular forms functional analysis math"
    ]
    labels = [
        "Computer Science", "Computer Science", "Computer Science",
        "Physics", "Physics", "Physics",
        "Mathematics", "Mathematics", "Mathematics"
    ]
    
    fe = FeatureExtractor(max_features=100)
    X = fe.fit_transform(corpus)

    for model_type in ["calibrated_svm", "complement_nb", "logistic_regression", "ensemble_voting"]:
        clf = ClassicalMLClassifier(model_type=model_type)
        clf.fit(X, labels)
        preds = clf.predict(X[:2])
        assert len(preds) == 2
        probs = clf.predict_proba(X[:2])
        assert probs.shape == (2, 3)
        # Check probability sums to ~1
        assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_pytorch_classifier():
    texts = [
        "deep neural networks transformer models computer science",
        "quantum physics superconductor entanglement",
        "single cell genomics crispr biology"
    ]
    labels = ["Computer Science", "Physics", "Biology"]
    
    dl = PyTorchAbstractClassifier(epochs=3, batch_size=2, embed_dim=32, hidden_dim=32)
    dl.fit(texts, labels)
    preds = dl.predict(["neural networks"])
    assert len(preds) == 1
    probs = dl.predict_proba(["neural networks"])
    assert probs.shape == (1, 3)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
