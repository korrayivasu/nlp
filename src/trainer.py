import os
import sys

# Ensure root directory is on Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from data.dataset_loader import load_dataset, MAIN_CATEGORIES
from src.preprocessor import TextPreprocessor
from src.feature_extractor import FeatureExtractor
from src.ml_models import ClassicalMLClassifier
from src.deep_models import PyTorchAbstractClassifier

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def prepare_data(
    dataset_path: Optional[str] = None,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[List[str], List[str], List[str], List[str], pd.DataFrame]:
    """
    Loads dataset, combines title and abstract with weighting, and creates stratified split.
    """
    df = load_dataset(dataset_path)
    
    # Filter to only valid main categories
    df = df[df["category"].isin(MAIN_CATEGORIES)].copy()
    
    preprocessor = TextPreprocessor()
    df["full_text"] = df.apply(
        lambda row: preprocessor.combine_title_abstract(row.get("title", ""), row.get("abstract", "")),
        axis=1
    )
    
    X = df["full_text"].tolist()
    y = df["category"].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test, df


def train_and_evaluate_all(
    dataset_path: Optional[str] = None,
    model_dir: str = MODEL_DIR,
    train_dl: bool = True
) -> Dict[str, Any]:
    """
    Trains all ML & DL models, saves artifacts, and returns comprehensive evaluation benchmark.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    print("Loading and preparing dataset...")
    X_train, X_test, y_train, y_test, df = prepare_data(dataset_path=dataset_path)
    print(f"Total papers: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    
    # 1. Fit and save Feature Extractor
    print("Fitting TF-IDF Feature Extractor...")
    feature_extractor = FeatureExtractor(max_features=5000, ngram_range=(1, 2))
    X_train_vec = feature_extractor.fit_transform(X_train)
    X_test_vec = feature_extractor.transform(X_test)
    
    fe_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    feature_extractor.save(fe_path)
    print(f"Saved feature extractor to {fe_path}")
    
    results = {}
    
    # 2. Train Classical ML Models
    ml_models = [
        ("calibrated_svm", "Calibrated Linear SVM"),
        ("complement_nb", "Complement Naive Bayes"),
        ("logistic_regression", "Logistic Regression"),
        ("ensemble_voting", "Soft Voting Ensemble")
    ]
    
    for model_key, model_name in ml_models:
        print(f"Training {model_name}...")
        clf = ClassicalMLClassifier(model_type=model_key)
        clf.fit(X_train_vec, y_train)
        
        eval_metrics = clf.evaluate(X_test_vec, y_test)
        cm = confusion_matrix(y_test, eval_metrics["y_pred"], labels=list(clf.classes_)).tolist()
        eval_metrics["confusion_matrix"] = cm
        eval_metrics["classes"] = list(clf.classes_)
        
        # Save model
        save_path = os.path.join(model_dir, f"{model_key}.joblib")
        clf.save(save_path)
        
        results[model_key] = {
            "name": model_name,
            "metrics": eval_metrics
        }
        print(f"  -> {model_name} Accuracy: {eval_metrics['accuracy']:.4f} | Macro F1: {eval_metrics['macro_f1']:.4f}")
        
    # 3. Train PyTorch Deep Learning Model
    if train_dl:
        print("Training PyTorch BiLSTM + Attention Neural Classifier...")
        dl_clf = PyTorchAbstractClassifier(
            embed_dim=128,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
            epochs=20,
            batch_size=16
        )
        dl_clf.fit(X_train, y_train)
        
        dl_eval = dl_clf.evaluate(X_test, y_test)
        dl_cm = confusion_matrix(y_test, dl_eval["y_pred"], labels=list(dl_clf.classes_)).tolist()
        dl_eval["confusion_matrix"] = dl_cm
        dl_eval["classes"] = list(dl_clf.classes_)
        
        dl_save_path = os.path.join(model_dir, "deep_bilstm_attn.pt")
        dl_clf.save(dl_save_path)
        
        results["deep_bilstm"] = {
            "name": "PyTorch BiLSTM + Self-Attention",
            "metrics": dl_eval
        }
        print(f"  -> Deep BiLSTM Accuracy: {dl_eval['accuracy']:.4f} | Macro F1: {dl_eval['macro_f1']:.4f}")

    # Save benchmark metrics summary to JSON
    benchmark_path = os.path.join(model_dir, "benchmark_results.json")
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark results to {benchmark_path}")
    
    return results


if __name__ == "__main__":
    train_and_evaluate_all()
