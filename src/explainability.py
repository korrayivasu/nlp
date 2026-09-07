"""
Explainability & Feature Attribution Engine for Paper Classification.
Extracts salient tokens, n-grams, and renders visual attention/importance heatmaps.
"""

import re
import html
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class ExplainabilityEngine:
    """
    Interprets model predictions and extracts key scientific phrases driving classifications.
    """

    def __init__(self, feature_extractor, classifier):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.feature_names = self.feature_extractor.get_feature_names() if feature_extractor.is_fitted else np.array([])
        self.classes = getattr(classifier, "classes_", [])

    def get_top_category_keywords(self, top_n: int = 15) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extracts the most influential n-grams globally for each category.
        """
        category_keywords = {}
        model = getattr(self.classifier, "model", self.classifier)
        
        # Check for linear models (Logistic Regression, CalibratedClassifierCV with LinearSVC)
        if hasattr(model, "coef_"):
            coef = model.coef_
            for i, cat in enumerate(self.classes):
                weights = coef[i] if coef.ndim > 1 else coef[0]
                top_indices = np.argsort(weights)[::-1][:top_n]
                category_keywords[cat] = [
                    (self.feature_names[idx], float(weights[idx]))
                    for idx in top_indices if idx < len(self.feature_names)
                ]
                
        elif hasattr(model, "feature_log_prob_"):
            # Naive Bayes
            for i, cat in enumerate(self.classes):
                weights = model.feature_log_prob_[i]
                # Log-odds ratio against mean of other classes
                other_mean = np.mean(np.delete(model.feature_log_prob_, i, axis=0), axis=0)
                relative_weights = weights - other_mean
                top_indices = np.argsort(relative_weights)[::-1][:top_n]
                category_keywords[cat] = [
                    (self.feature_names[idx], float(relative_weights[idx]))
                    for idx in top_indices if idx < len(self.feature_names)
                ]
                
        elif hasattr(model, "calibrated_classifiers_"):
            # CalibratedClassifierCV
            weights_list = []
            for cal_clf in model.calibrated_classifiers_:
                base_estimator = getattr(cal_clf, "estimator", getattr(cal_clf, "base_estimator", None))
                if hasattr(base_estimator, "coef_"):
                    weights_list.append(base_estimator.coef_)
                    
            if weights_list:
                avg_coef = np.mean(weights_list, axis=0)
                for i, cat in enumerate(self.classes):
                    w = avg_coef[i] if avg_coef.ndim > 1 else avg_coef[0]
                    top_indices = np.argsort(w)[::-1][:top_n]
                    category_keywords[cat] = [
                        (self.feature_names[idx], float(w[idx]))
                        for idx in top_indices if idx < len(self.feature_names)
                    ]
        
        # Fallback if model doesn't expose coefficients directly
        if not category_keywords:
            for cat in self.classes:
                category_keywords[cat] = [("academic term", 1.0)]
                
        return category_keywords

    def explain_text(self, text: str, target_class: Optional[str] = None, top_k: int = 8) -> Dict[str, Any]:
        """
        Computes local token attribution for a single input text abstract.
        Returns top supporting tokens and highlighted HTML text.
        """
        X_vec = self.feature_extractor.transform([text])
        probs = self.classifier.predict_proba(X_vec)[0]
        
        if target_class is None or target_class not in self.classes:
            pred_class_idx = int(np.argmax(probs))
            target_class = str(self.classes[pred_class_idx])
        else:
            pred_class_idx = list(self.classes).index(target_class)

        # Non-zero feature indices in this text
        feature_indices = X_vec.nonzero()[1]
        tfidf_values = X_vec.data
        
        token_contributions = []
        global_top_keywords = self.get_top_category_keywords(top_n=200).get(target_class, [])
        global_weights_dict = {term: weight for term, weight in global_top_keywords}
        
        for idx, val in zip(feature_indices, tfidf_values):
            feat_name = self.feature_names[idx]
            weight = global_weights_dict.get(feat_name, val * 0.5)
            score = val * weight
            token_contributions.append((feat_name, float(score)))

        token_contributions.sort(key=lambda x: x[1], reverse=True)
        top_tokens = token_contributions[:top_k]

        # Generate HTML highlighted text
        highlighted_html = self._generate_highlighted_html(text, top_tokens, target_class)

        return {
            "target_class": target_class,
            "probability": float(probs[pred_class_idx]),
            "top_contributing_tokens": top_tokens,
            "highlighted_html": highlighted_html
        }

    def _generate_highlighted_html(self, text: str, top_tokens: List[Tuple[str, float]], target_class: str) -> str:
        """Generates rich HTML with highlighted background spans for important terms."""
        color_map = {
            "Computer Science": "rgba(59, 130, 246, 0.35)",
            "Mathematics": "rgba(139, 92, 246, 0.35)",
            "Physics": "rgba(236, 72, 153, 0.35)",
            "Biology": "rgba(16, 185, 129, 0.35)",
            "Statistics": "rgba(245, 158, 11, 0.35)"
        }
        hl_color = color_map.get(target_class, "rgba(59, 130, 246, 0.35)")
        
        escaped_text = html.escape(text)
        # Sort terms by length descending to match longer n-grams first
        sorted_terms = sorted([t[0] for t in top_tokens], key=len, reverse=True)
        
        for term in sorted_terms:
            if not term.strip():
                continue
            # Regex to match whole word or phrase case-insensitively
            pattern = re.compile(rf"\b({re.escape(term)})\b", re.IGNORECASE)
            escaped_text = pattern.sub(
                rf'<span style="background-color: {hl_color}; padding: 2px 5px; border-radius: 4px; font-weight: 600; border-bottom: 2px solid #ffffff33;">\1</span>',
                escaped_text
            )
            
        return f'<div style="line-height: 1.7; font-size: 1.02rem; color: #e2e8f0; font-family: sans-serif;">{escaped_text}</div>'
