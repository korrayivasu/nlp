# 🔬 arXiv Research Paper Category Classification

A production-grade NLP system and interactive Web Dashboard for classifying scientific research paper abstracts into academic disciplines (**Computer Science**, **Mathematics**, **Physics**, **Biology**, and **Statistics**) using Calibrated Machine Learning, PyTorch Deep Learning, and the official arXiv API.

---

## 🌟 Key Features

1. **5 Core Academic Disciplines Supported**:
   - 💻 **Computer Science** (`cs.AI`, `cs.CL`, `cs.CV`, `cs.LG`, `cs.RO`, `cs.CR`, etc.)
   - 📐 **Mathematics** (`math.NT`, `math.AP`, `math.AG`, `math.CO`, `math.DS`, etc.)
   - ⚛️ **Physics** (`quant-ph`, `cond-mat`, `hep-th`, `astro-ph`, `gr-qc`, etc.)
   - 🧬 **Biology** (Quantitative Biology: `q-bio.GN`, `q-bio.BM`, `q-bio.NC`, `q-bio.MN`, etc.)
   - 📈 **Statistics & Quantitative Finance** (`stat.TH`, `stat.ML`, `stat.ME`, `q-fin.ST`, etc.)

2. **Cross-Disciplinary & Multi-Label Tagging**:
   - Predicts calibrated probability distribution across all 5 disciplines.
   - Automatically tags multi-disciplinary research (e.g. `Computer Science 70%` + `Statistics 25%`).

3. **Multi-Model ML & Deep Learning Suite**:
   - **Calibrated Linear SVM** (via `CalibratedClassifierCV` for reliable probabilities)
   - **Complement Naive Bayes** (tailored for text classification)
   - **Logistic Regression** (L2 regularized)
   - **Soft Voting Ensemble**
   - **PyTorch BiLSTM with Self-Attention Pooling**

4. **Explainability & Keyword Attribution (XAI)**:
   - Identifies top contributing tokens and n-grams driving each prediction.
   - Renders interactive highlighted text heatmaps in the UI.

5. **Live arXiv API Harvester**:
   - Query arXiv in real-time by search terms/topics and auto-classify newly retrieved papers.

6. **Interactive Web Application (Streamlit)**:
   - Modern dark glassmorphism dashboard with 5 specialized labs (Live Classifier, arXiv Harvester, Benchmark Lab, Discipline Explorer, Retraining Studio).

---

## 🚀 Quick Start

### 1. Run the Web Application
```powershell
streamlit run app.py
```

### 2. Command-Line Classification (CLI)

**Run Built-In Benchmark Samples:**
```powershell
python cli.py --sample
```

**Classify a Custom Paper:**
```powershell
python cli.py --title "Attention Is All You Need" --abstract "We propose the Transformer architecture based on self-attention..."
```

**Live arXiv Search & Classify:**
```powershell
python cli.py --arxiv "quantum machine learning" --max-results 3
```

### 3. Run Automated Tests
```powershell
python -m pytest tests/ -v
```

---

## 📊 Benchmark Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **Calibrated Linear SVM** | **96.97%** | **96.92%** | **96.97%** |
| **Complement Naive Bayes** | **96.97%** | **96.92%** | **96.97%** |
| **Soft Voting Ensemble** | **96.97%** | **96.92%** | **96.97%** |
| **Logistic Regression (L2)** | **90.91%** | **90.94%** | **90.91%** |
| **PyTorch BiLSTM Attention** | **70.00%** | **70.06%** | **70.00%** |

---

## 📁 Repository Structure

```
nlp/
├── data/
│   ├── dataset_loader.py         # arXiv Harvester, loader, mapping & pre-curated corpus
│   └── sample_arxiv_dataset.json # Offline balanced benchmark dataset
├── src/
│   ├── preprocessor.py           # LaTeX math normalization, academic token cleanup
│   ├── feature_extractor.py      # TF-IDF n-gram vectorizer
│   ├── ml_models.py              # Calibrated SVM, Naive Bayes, LR, Ensemble
│   ├── deep_models.py            # PyTorch BiLSTM + Self-Attention text classifier
│   ├── trainer.py                # Model training, validation, benchmark suite
│   ├── explainability.py         # Token attribution & salient keyword highlighter
│   └── pipeline.py               # Unified production inference pipeline
├── models/                       # Serialized model weights & vectorizers (.joblib, .pt)
├── tests/                        # Full pytest automated test suite
├── app.py                        # Streamlit web application
├── cli.py                        # Command-line interface
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```
