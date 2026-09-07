import os
import sys

# Ensure root directory is on Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from data.dataset_loader import MAIN_CATEGORIES, CATEGORY_COLORS
from src.preprocessor import TextPreprocessor
from src.feature_extractor import FeatureExtractor
from src.ml_models import ClassicalMLClassifier
from src.deep_models import PyTorchAbstractClassifier
from src.explainability import ExplainabilityEngine

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class PaperClassificationPipeline:
    """
    Production-ready Inference Pipeline for academic paper classification.
    """

    def __init__(
        self,
        model_type: str = "calibrated_svm",
        model_dir: str = DEFAULT_MODEL_DIR
    ):
        self.model_type = model_type
        self.model_dir = model_dir
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = None
        self.classifier = None
        self.explainability_engine = None
        self.is_loaded = False
        
        self.load_models()

    def load_models(self):
        """Loads vectorizer and requested classification model."""
        os.makedirs(self.model_dir, exist_ok=True)
        fe_path = os.path.join(self.model_dir, "tfidf_vectorizer.joblib")
        
        if self.model_type == "deep_bilstm":
            dl_path = os.path.join(self.model_dir, "deep_bilstm_attn.pt")
            if not os.path.exists(dl_path):
                print(f"Deep model not found at {dl_path}. Triggering training...")
                from src.trainer import train_and_evaluate_all
                train_and_evaluate_all(model_dir=self.model_dir)
                
            self.classifier = PyTorchAbstractClassifier()
            self.classifier.load(dl_path)
            
            # Load feature extractor for keyword explainability
            if os.path.exists(fe_path):
                self.feature_extractor = FeatureExtractor()
                self.feature_extractor.load(fe_path)
                
                # Also load linear surrogate for explainability
                surrogate = ClassicalMLClassifier("calibrated_svm")
                svm_path = os.path.join(self.model_dir, "calibrated_svm.joblib")
                if os.path.exists(svm_path):
                    surrogate.load(svm_path)
                self.explainability_engine = ExplainabilityEngine(self.feature_extractor, surrogate)
        else:
            model_path = os.path.join(self.model_dir, f"{self.model_type}.joblib")
            if not os.path.exists(model_path) or not os.path.exists(fe_path):
                print("Trained models not found. Running training pipeline...")
                from src.trainer import train_and_evaluate_all
                train_and_evaluate_all(model_dir=self.model_dir)
                
            self.feature_extractor = FeatureExtractor()
            self.feature_extractor.load(fe_path)
            
            self.classifier = ClassicalMLClassifier(model_type=self.model_type)
            self.classifier.load(model_path)
            
            self.explainability_engine = ExplainabilityEngine(self.feature_extractor, self.classifier)

        self.is_loaded = True

    def predict(
        self,
        title: str = "",
        abstract: str = "",
        threshold_secondary: float = 0.15,
        title_weight: int = 2
    ) -> Dict[str, Any]:
        """
        Classifies a single research paper abstract.
        Returns primary category, cross-disciplinary categories, probabilities, and highlights.
        """
        if not self.is_loaded:
            self.load_models()

        full_raw_text = f"{title}\n\n{abstract}".strip() if title else abstract
        combined_text = self.preprocessor.combine_title_abstract(title, abstract, title_weight=title_weight)
        
        if not combined_text.strip():
            return {
                "primary_category": "Unknown",
                "confidence": 0.0,
                "confidence_percent": "0.0%",
                "probabilities": {cat: 0.0 for cat in MAIN_CATEGORIES},
                "secondary_categories": [],
                "top_keywords": [],
                "highlighted_html": "<p>Empty text provided.</p>",
                "model_used": self.model_type
            }

        # Predict probability distribution
        if self.model_type == "deep_bilstm":
            probs = self.classifier.predict_proba([combined_text])[0]
            classes = list(self.classifier.classes_)
        else:
            X_vec = self.feature_extractor.transform([combined_text])
            probs = self.classifier.predict_proba(X_vec)[0]
            classes = list(self.classifier.classes_)

        # Build probability map
        prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        top_cat, top_prob = sorted_probs[0]
        
        # Cross-disciplinary / secondary categories (above threshold and not the top)
        secondary = [
            {"category": cat, "probability": prob, "percent": f"{prob * 100:.1f}%"}
            for cat, prob in sorted_probs[1:]
            if prob >= threshold_secondary
        ]

        # Explainability & salient keywords
        explanation = {}
        if self.explainability_engine:
            explanation = self.explainability_engine.explain_text(full_raw_text, target_class=top_cat, top_k=7)

        return {
            "primary_category": top_cat,
            "confidence": float(top_prob),
            "confidence_percent": f"{top_prob * 100:.1f}%",
            "category_color": CATEGORY_COLORS.get(top_cat, "#3b82f6"),
            "probabilities": {cat: prob_dict.get(cat, 0.0) for cat in MAIN_CATEGORIES if cat in prob_dict},
            "ranked_probabilities": sorted_probs,
            "secondary_categories": secondary,
            "is_multidisciplinary": len(secondary) > 0,
            "top_keywords": explanation.get("top_contributing_tokens", []),
            "highlighted_html": explanation.get("highlighted_html", ""),
            "model_used": self.model_type
        }

    def predict_batch(self, papers: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Classify a list of paper dicts with 'title' and 'abstract' keys."""
        return [self.predict(p.get("title", ""), p.get("abstract", "")) for p in papers]


if __name__ == "__main__":
    pipeline = PaperClassificationPipeline(model_type="calibrated_svm")
    res = pipeline.predict(
        title="Generative Adversarial Nets for Image Synthesis",
        abstract="We propose a framework for estimating generative models via an adversarial process in which we simultaneously train two models: a generative model G and a discriminative model D."
    )
    print("Primary Category:", res["primary_category"], f"({res['confidence_percent']})")
    print("Probabilities:", res["probabilities"])
    print("Top Keywords:", res["top_keywords"])
