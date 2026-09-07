"""
PyTorch Deep Learning Classifier for Academic Paper Abstracts.
Implements an Attention-based Bidirectional LSTM / TextCNN Neural Network.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.preprocessor import TextPreprocessor


class Vocabulary:
    """Vocabulary mapper from tokens to integer indices."""

    def __init__(self, max_vocab_size: int = 10000, min_freq: int = 1):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1
        self.word2idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}
        self.idx2word = {self.pad_idx: self.pad_token, self.unk_idx: self.unk_token}

    def build_vocab(self, texts: List[str]):
        word_counts = {}
        for text in texts:
            tokens = text.split()
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        idx = len(self.word2idx)
        for word, count in sorted_words:
            if count >= self.min_freq and idx < self.max_vocab_size:
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

    def text_to_indices(self, text: str, max_len: int = 256) -> List[int]:
        tokens = text.split()
        indices = [self.word2idx.get(tok, self.unk_idx) for tok in tokens[:max_len]]
        if len(indices) < max_len:
            indices += [self.pad_idx] * (max_len - len(indices))
        return indices

    def __len__(self):
        return len(self.word2idx)


class AbstractDataset(Dataset):
    """PyTorch Dataset for paper abstracts."""

    def __init__(self, texts: List[str], labels: Optional[List[int]], vocab: Vocabulary, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        input_ids = torch.tensor(self.vocab.text_to_indices(text, self.max_len), dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return input_ids, label
        return input_ids


class AttentionPooling(nn.Module):
    """Self-Attention Pooling over sequential hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, hiddens: torch.Tensor, mask: torch.Tensor = None):
        # hiddens: (batch_size, seq_len, hidden_dim)
        scores = self.attn(hiddens)  # (batch_size, seq_len, 1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        pooled = torch.sum(hiddens * weights, dim=1)  # (batch_size, hidden_dim)
        return pooled, weights.squeeze(-1)


class AbstractBiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with Self-Attention Pooling for Academic Paper Classification.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 5,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = AttentionPooling(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids: torch.Tensor):
        mask = (input_ids != 0)
        embeds = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.lstm(embeds)
        pooled, attn_weights = self.attention(lstm_out, mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits, attn_weights


class PyTorchAbstractClassifier:
    """
    High-level wrapper for training and inference with PyTorch deep learning models.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_len: int = 200,
        lr: float = 1e-3,
        epochs: int = 15,
        batch_size: int = 16,
        device: Optional[str] = None
    ):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_len = max_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.vocab = Vocabulary()
        self.preprocessor = TextPreprocessor()
        self.label2id = {}
        self.id2label = {}
        self.classes_ = []
        self.model = None

    def fit(self, texts: List[str], labels: List[str], val_texts: Optional[List[str]] = None, val_labels: Optional[List[str]] = None):
        """Train PyTorch deep neural network."""
        cleaned_texts = [self.preprocessor.clean_text(t) for t in texts]
        self.vocab.build_vocab(cleaned_texts)

        unique_labels = sorted(list(set(labels)))
        self.label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.id2label = {i: lbl for i, lbl in enumerate(unique_labels)}
        self.classes_ = np.array(unique_labels)
        
        y_train = [self.label2id[l] for l in labels]
        train_ds = AbstractDataset(cleaned_texts, y_train, self.vocab, self.max_len)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self.model = AbstractBiLSTMAttention(
            vocab_size=len(self.vocab),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_classes=len(unique_labels),
            num_layers=self.num_layers,
            dropout=self.dropout,
            pad_idx=self.vocab.pad_idx
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict class probability distribution."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
            
        self.model.eval()
        cleaned_texts = [self.preprocessor.clean_text(t) for t in texts]
        ds = AbstractDataset(cleaned_texts, None, self.vocab, self.max_len)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.device)
                logits, _ = self.model(batch_x)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict top class labels."""
        probs = self.predict_proba(texts)
        top_indices = np.argmax(probs, axis=1)
        return np.array([self.id2label[idx] for idx in top_indices])

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """Evaluate deep learning model."""
        y_pred = self.predict(texts)
        acc = accuracy_score(labels, y_pred)
        macro_f1 = f1_score(labels, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, y_pred, average="weighted", zero_division=0)
        report = classification_report(labels, y_pred, output_dict=True, zero_division=0)
        
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "classification_report": report,
            "y_true": list(labels),
            "y_pred": list(y_pred)
        }

    def save(self, filepath: str):
        """Save weights, vocab, and label mappings."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "model_state": self.model.state_dict() if self.model else None,
            "vocab_word2idx": self.vocab.word2idx,
            "vocab_idx2word": self.vocab.idx2word,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "classes": list(self.classes_),
            "config": {
                "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "max_len": self.max_len
            }
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str):
        """Load weights, vocab, and configuration."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.vocab.word2idx = checkpoint["vocab_word2idx"]
        self.vocab.idx2word = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in checkpoint["vocab_idx2word"].items()}
        self.label2id = checkpoint["label2id"]
        self.id2label = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in checkpoint["id2label"].items()}
        self.classes_ = np.array(checkpoint["classes"])
        
        cfg = checkpoint["config"]
        self.embed_dim = cfg["embed_dim"]
        self.hidden_dim = cfg["hidden_dim"]
        self.num_layers = cfg["num_layers"]
        self.dropout = cfg["dropout"]
        self.max_len = cfg["max_len"]

        self.model = AbstractBiLSTMAttention(
            vocab_size=len(self.vocab),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_classes=len(self.classes_),
            num_layers=self.num_layers,
            dropout=self.dropout,
            pad_idx=self.vocab.pad_idx
        ).to(self.device)
        
        if checkpoint["model_state"] is not None:
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
        return self


if __name__ == "__main__":
    texts = [
        "Attention mechanisms in transformer neural networks for computer science",
        "Quantum entanglement in superconductors and condensed matter physics",
        "Single-cell RNA sequencing for gene expression analysis in biology",
        "Banach spaces and partial differential equations in mathematics",
        "High dimensional covariance estimation and hypothesis testing in statistics"
    ]
    labels = ["Computer Science", "Physics", "Biology", "Mathematics", "Statistics"]
    dl_model = PyTorchAbstractClassifier(epochs=5, batch_size=2)
    dl_model.fit(texts, labels)
    preds = dl_model.predict(["Transformers neural network"])
    print("PyTorch Prediction:", preds)
