"""
Feature Extraction and Vectorization for Scientific Paper Abstracts.
Supports TF-IDF with unigrams, bigrams, and custom vocabulary options.
"""

import os
import joblib
from typing import List, Optional, Tuple, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessor import TextPreprocessor


class FeatureExtractor:
    """
    TF-IDF Feature Extractor optimized for scientific abstracts classification.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        min_df: int = 1,
        max_df: float = 0.95
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.max_df = max_df
        self.preprocessor = TextPreprocessor()
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df,
            max_df=self.max_df,
            preprocessor=self.preprocessor.clean_text,
            token_pattern=r"(?u)\b\w[\w-]+\b"
        )
        self.is_fitted = False

    def fit(self, raw_documents: List[str]):
        """Fit the vectorizer on a corpus of text documents."""
        self.vectorizer.fit(raw_documents)
        self.is_fitted = True
        return self

    def transform(self, raw_documents: List[str]):
        """Transform documents to TF-IDF sparse matrix."""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor has not been fitted yet. Call fit() first.")
        return self.vectorizer.transform(raw_documents)

    def fit_transform(self, raw_documents: List[str]):
        """Fit and transform in one step."""
        self.is_fitted = True
        return self.vectorizer.fit_transform(raw_documents)

    def get_feature_names(self) -> np.ndarray:
        """Returns array of feature names / n-grams."""
        if not self.is_fitted:
            return np.array([])
        return np.array(self.vectorizer.get_feature_names_out())

    def save(self, filepath: str):
        """Save vectorizer to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)

    def load(self, filepath: str):
        """Load fitted vectorizer from disk."""
        self.vectorizer = joblib.load(filepath)
        self.is_fitted = True
        return self


if __name__ == "__main__":
    docs = [
        "Attention is all you need for deep neural networks in computer science",
        "Topological quantum phase transitions in condensed matter physics",
        "Single-cell RNA sequencing reveals gene regulatory networks in biology",
        "Existence and uniqueness of elliptic partial differential equations in mathematics",
        "High-dimensional covariance matrix estimation and minimax asymptotic rates in statistics"
    ]
    fe = FeatureExtractor(max_features=20)
    X = fe.fit_transform(docs)
    print("TF-IDF shape:", X.shape)
    print("Features:", fe.get_feature_names())
