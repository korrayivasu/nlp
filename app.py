"""
Streamlit Web Application for arXiv Research Paper Category Classification.
Interactive dashboard with live classifier, arXiv harvester, benchmark lab, and explainability.
"""

import os
import sys

# Ensure root directory is on Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.dataset_loader import load_dataset, fetch_arxiv_papers, MAIN_CATEGORIES, CATEGORY_COLORS
from src.pipeline import PaperClassificationPipeline
from src.trainer import train_and_evaluate_all, MODEL_DIR

# Page configuration
st.set_page_config(
    page_title="arXiv Paper Category Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphism aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 1));
        color: #f8fafc;
    }
    
    /* Header Card */
    .header-box {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(88, 28, 135, 0.4) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .header-sub {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* Glass card */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        backdrop-filter: blur(10px);
    }
    
    /* Prediction result banner */
    .pred-badge {
        display: inline-block;
        font-size: 1.6rem;
        font-weight: 700;
        padding: 8px 20px;
        border-radius: 12px;
        margin-bottom: 12px;
        letter-spacing: 0.5px;
    }
    
    .sec-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .kw-chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc;
        padding: 3px 10px;
        border-radius: 20px;
        margin: 3px;
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Metric Card */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline(model_type: str = "calibrated_svm") -> PaperClassificationPipeline:
    """Cached initialization of classification pipeline."""
    return PaperClassificationPipeline(model_type=model_type)


@st.cache_data
def get_benchmark_data() -> dict:
    """Load benchmark results if available."""
    bm_path = os.path.join(MODEL_DIR, "benchmark_results.json")
    if os.path.exists(bm_path):
        with open(bm_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_cached_dataset() -> pd.DataFrame:
    """Load dataset."""
    return load_dataset()


# Sample papers for quick fill
SAMPLE_PAPERS = {
    "Computer Science": {
        "title": "Attention Is All You Need: Scalable Sequence Transduction with Transformers",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train."
    },
    "Mathematics": {
        "title": "Existence and Sobolev Regularity of Solutions for Non-Linear Elliptic Partial Differential Equations",
        "abstract": "We investigate the existence, uniqueness, and Sobolev boundary regularity of weak solutions to a class of quasilinear elliptic partial differential equations with singular p-Laplacian operators. Using Moser iteration techniques and De Giorgi-Nash-Moser estimates, we establish global Holder continuity and C^{1,alpha} interior regularity of the extremal solutions on Riemannian manifolds with bounded Ricci curvature."
    },
    "Physics": {
        "title": "Observation of Gravitational Waves from a Binary Black Hole Merger",
        "abstract": "The two detectors of the Laser Interferometer Gravitational-Wave Observatory (LIGO) simultaneously observed a transient gravitational-wave signal matching the waveform predicted by general relativity for the inspiral and merger of a pair of black holes and the ringdown of the resulting single black hole. The source lies at a luminosity distance of roughly 410 Mpc with component masses of 36 and 29 solar masses."
    },
    "Biology": {
        "title": "CRISPR-Cas9 Base Editing and Transcriptomic Profiling of Pluripotent Stem Cells",
        "abstract": "Genome editing technologies enable precise nucleotide substitutions without generating double-stranded DNA breaks. We utilize adenine and cytosine base editors in human pluripotent stem cells to introduce targeted single-nucleotide variants. Single-cell RNA sequencing (scRNA-seq) reveals minimal off-target transcriptome disruption and preserves pluripotency gene networks."
    },
    "Statistics": {
        "title": "High-Dimensional Covariance Matrix Estimation and Minimax Non-Linear Shrinkage",
        "abstract": "Estimating large covariance matrices when the dimension p exceeds the sample size n is a fundamental challenge in multivariate statistics. We propose a non-linear eigenvalue shrinkage estimator under the Marchenko-Pastur random matrix theory regime. We prove asymptotic minimax optimality under Frobenius and spectral matrix norms, demonstrating superior risk bounds compared to sample covariance matrices."
    },
    "Multi-Disciplinary (CS + Stat + Bio)": {
        "title": "Deep Variational Autoencoders for Single-Cell Gene Regulatory Network Inference",
        "abstract": "Deciphering high-dimensional transcriptomic dynamics requires scalable statistical learning algorithms. We formulate a deep generative variational autoencoder with Dirichlet prior latent variables to reconstruct cellular trajectory manifolds from single-cell RNA sequencing data. We establish conformal confidence intervals and test gene knockout perturbations across immune cell subpopulations."
    }
}


# Header
st.markdown("""
<div class="header-box">
    <div class="header-title">arXiv Research Paper Category Classifier</div>
    <p class="header-sub">Automated Academic Discipline Classification & Cross-Disciplinary Tagging using Calibrated NLP & Deep Learning</p>
</div>
""", unsafe_allow_html=True)


# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Classifier Configuration")
    
    model_choice = st.selectbox(
        "Select Classification Model:",
        options=[
            ("calibrated_svm", "Calibrated Linear SVM (Recommended)"),
            ("ensemble_voting", "Soft Voting Ensemble"),
            ("complement_nb", "Complement Naive Bayes"),
            ("logistic_regression", "Logistic Regression (L2)"),
            ("deep_bilstm", "PyTorch BiLSTM + Self-Attention")
        ],
        format_func=lambda x: x[1],
        index=0
    )[0]
    
    sec_threshold = st.slider(
        "Cross-Disciplinary Tag Threshold:",
        min_value=0.05,
        max_value=0.40,
        value=0.15,
        step=0.01,
        help="Confidence percentage threshold to tag paper as multidisciplinary (secondary category)"
    )
    
    title_weight = st.slider(
        "Paper Title Weighting Factor:",
        min_value=1,
        max_value=4,
        value=2,
        help="Title tokens carry high signal weight in academic classification"
    )
    
    st.markdown("---")
    st.markdown("### 📚 Supported Disciplines")
    for cat in MAIN_CATEGORIES:
        color = CATEGORY_COLORS.get(cat, "#3b82f6")
        st.markdown(f"<span style='color:{color}; font-weight:600;'>●</span> **{cat}**", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<small style='color:#94a3b8;'>Powered by scikit-learn & PyTorch</small>", unsafe_allow_html=True)


# Navigation Tabs
tabs = st.tabs([
    "🔬 Live Abstract Classifier",
    "🌐 Live arXiv Harvester",
    "📊 Model Analytics & Benchmark",
    "📖 Discipline & Keyword Explorer",
    "⚡ Retraining Studio"
])


# ==========================================
# TAB 1: LIVE ABSTRACT CLASSIFIER
# ==========================================
with tabs[0]:
    st.markdown("### Classify Paper Abstract")
    st.markdown("Enter a research paper title and abstract or select one of the presets to test classification.")

    # Preset selection buttons
    st.markdown("**Quick-Fill Sample Preset:**")
    cols = st.columns(len(SAMPLE_PAPERS))
    
    if "input_title" not in st.session_state:
        st.session_state.input_title = SAMPLE_PAPERS["Computer Science"]["title"]
        st.session_state.input_abstract = SAMPLE_PAPERS["Computer Science"]["abstract"]

    for i, (preset_name, preset_data) in enumerate(SAMPLE_PAPERS.items()):
        short_label = preset_name.split()[0] if "Multi" not in preset_name else "Multi-Disc"
        if cols[i].button(f"{short_label}", key=f"btn_preset_{i}", use_container_width=True):
            st.session_state.input_title = preset_data["title"]
            st.session_state.input_abstract = preset_data["abstract"]
            st.rerun()

    # Input Form
    with st.container():
        input_title = st.text_input(
            "Research Paper Title:",
            value=st.session_state.input_title,
            placeholder="e.g. Attention Is All You Need",
            key="field_title"
        )
        input_abstract = st.text_area(
            "Abstract Text:",
            value=st.session_state.input_abstract,
            height=160,
            placeholder="Paste paper abstract here...",
            key="field_abstract"
        )
        
        classify_col1, classify_col2 = st.columns([1, 4])
        run_btn = classify_col1.button("🚀 Classify Paper", type="primary", use_container_width=True)

    if run_btn or input_abstract:
        pipeline = get_pipeline(model_choice)
        with st.spinner("Analyzing text and evaluating discipline probabilities..."):
            result = pipeline.predict(
                title=input_title,
                abstract=input_abstract,
                threshold_secondary=sec_threshold,
                title_weight=title_weight
            )

        cat = result["primary_category"]
        conf = result["confidence_percent"]
        cat_color = result["category_color"]

        st.markdown("---")
        
        # Results Layout
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("#### Primary Prediction")
            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 6px solid {cat_color};">
                    <span class="pred-badge" style="background: {cat_color}25; color: {cat_color}; border: 1px solid {cat_color}60;">
                        {cat}
                    </span>
                    <div style="font-size: 1.15rem; margin-top: 4px;">
                        Confidence: <strong style="color: #f8fafc;">{conf}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Secondary / Multidisciplinary Tags
            if result["secondary_categories"]:
                st.markdown("#### Cross-Disciplinary Tags")
                tags_html = "".join([
                    f"<span class='sec-tag' style='border-color: {CATEGORY_COLORS.get(s['category'], '#fff')}60; color: {CATEGORY_COLORS.get(s['category'], '#fff')};'>"
                    f"🏷️ {s['category']}: <b>{s['percent']}</b></span>"
                    for s in result["secondary_categories"]
                ])
                st.markdown(f"<div style='margin-bottom: 15px;'>{tags_html}</div>", unsafe_allow_html=True)
            else:
                st.info("📌 High domain purity (No secondary cross-disciplinary tags exceeded threshold).")

            # Salient Keywords
            if result["top_keywords"]:
                st.markdown("#### Salient Domain Keywords Detected")
                kw_html = "".join([f"<span class='kw-chip'>{kw}</span>" for kw, _ in result["top_keywords"]])
                st.markdown(f"<div style='margin-bottom: 15px;'>{kw_html}</div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### Probability Distribution Across Disciplines")
            prob_df = pd.DataFrame(result["ranked_probabilities"], columns=["Discipline", "Probability"])
            prob_df["Percentage"] = prob_df["Probability"] * 100
            prob_df["Color"] = prob_df["Discipline"].map(CATEGORY_COLORS)

            fig = go.Figure(go.Bar(
                x=prob_df["Percentage"],
                y=prob_df["Discipline"],
                orientation="h",
                marker=dict(color=prob_df["Color"]),
                text=prob_df["Percentage"].apply(lambda x: f"{x:.1f}%"),
                textposition="auto"
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc", family="Outfit"),
                xaxis=dict(title="Probability (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(title="", autorange="reversed"),
                margin=dict(l=10, r=20, t=10, b=30),
                height=260
            )
            st.plotly_chart(fig, use_container_width=True)

        # Highlighted Abstract Inspector
        st.markdown("#### 🔍 Explainability & Keyword Highlight Inspector")
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;">
                    Highlighted phrases indicate influential domain terms mapped to <b style="color: {cat_color};">{cat}</b>:
                </div>
                {result['highlighted_html']}
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# TAB 2: LIVE ARXIV HARVESTER
# ==========================================
with tabs[1]:
    st.markdown("### 🌐 Live arXiv Harvester & Classifier")
    st.markdown("Query the official arXiv API in real-time, retrieve real academic paper abstracts, and run instant classification.")

    harv_col1, harv_col2 = st.columns([3, 1])
    with harv_col1:
        arxiv_query = st.text_input(
            "arXiv Search Query:",
            value="quantum machine learning",
            placeholder="e.g. topological insulators, crispr gene editing, causal inference"
        )
    with harv_col2:
        max_papers = st.selectbox("Max Papers to Fetch:", options=[3, 5, 8, 10], index=1)

    fetch_btn = st.button("🔍 Search & Fetch from arXiv API", type="primary")

    if fetch_btn or "fetched_papers" in st.session_state:
        if fetch_btn:
            with st.spinner(f"Querying arXiv API for '{arxiv_query}'..."):
                st.session_state.fetched_papers = fetch_arxiv_papers(arxiv_query, max_results=max_papers)

        papers = st.session_state.get("fetched_papers", [])
        if not papers:
            st.warning("No papers returned from arXiv API. Please check your query or internet connection.")
        else:
            st.success(f"Retrieved {len(papers)} papers from arXiv.")
            pipeline = get_pipeline(model_choice)

            for idx, paper in enumerate(papers):
                pred = pipeline.predict(
                    title=paper["title"],
                    abstract=paper["abstract"],
                    threshold_secondary=sec_threshold,
                    title_weight=title_weight
                )
                
                gt_cat = paper.get("primary_category", "N/A")
                gt_code = paper.get("primary_category_code", "N/A")
                pred_cat = pred["primary_category"]
                pred_conf = pred["confidence_percent"]
                p_color = pred["category_color"]
                
                match_badge = "✅ Category Matched" if gt_cat == pred_cat else "ℹ️ Interdisciplinary Shift"

                with st.expander(f"📄 [{paper.get('published', '')}] {paper['title'][:90]}...", expanded=(idx == 0)):
                    p_col1, p_col2 = st.columns([2, 1])
                    with p_col1:
                        st.markdown(f"**Authors:** {', '.join(paper.get('authors', [])[:4])}")
                        st.markdown(f"**arXiv ID / Link:** [{paper.get('id', '')}]({paper.get('id', '#')})")
                        st.markdown(f"**Original arXiv Category:** `{gt_code}` ({gt_cat})")
                        st.markdown(f"**Abstract:** {paper['abstract']}")
                    
                    with p_col2:
                        st.markdown(
                            f"""
                            <div class="glass-card" style="border-top: 4px solid {p_color};">
                                <div style="font-size:0.85rem; color:#94a3b8;">PREDICTED DISCIPLINE</div>
                                <div style="font-size:1.3rem; font-weight:700; color:{p_color}; margin: 4px 0;">{pred_cat}</div>
                                <div style="font-size:0.95rem;">Confidence: <b>{pred_conf}</b></div>
                                <div style="font-size:0.8rem; color:#a5b4fc; margin-top:6px;">{match_badge}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if pred["secondary_categories"]:
                            st.caption("Cross-disciplinary tags:")
                            for s in pred["secondary_categories"]:
                                st.caption(f"• {s['category']} ({s['percent']})")


# ==========================================
# TAB 3: MODEL BENCHMARK & ANALYTICS
# ==========================================
with tabs[2]:
    st.markdown("### 📊 Model Benchmark & Analytics Lab")
    st.markdown("Compare performance metrics across classical ML and deep neural architectures.")

    bm_data = get_benchmark_data()
    if not bm_data:
        st.info("No saved benchmark data found. Running training and benchmark suite...")
        with st.spinner("Training models and calculating benchmark metrics..."):
            bm_data = train_and_evaluate_all()

    # Metrics Summary Cards
    m_rows = []
    for m_key, m_info in bm_data.items():
        metrics = m_info["metrics"]
        m_rows.append({
            "Model": m_info["name"],
            "Accuracy": f"{metrics['accuracy'] * 100:.2f}%",
            "Macro F1": f"{metrics['macro_f1'] * 100:.2f}%",
            "Weighted F1": f"{metrics['weighted_f1'] * 100:.2f}%",
            "_acc_val": metrics["accuracy"],
            "_f1_val": metrics["macro_f1"],
            "key": m_key
        })

    m_df = pd.DataFrame(m_rows)
    
    # Top stats
    best_acc_row = m_df.loc[m_df["_acc_val"].idxmax()]
    stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
    stat_c1.markdown(f"<div class='metric-card'><div class='metric-value'>{best_acc_row['Accuracy']}</div><div class='metric-label'>Top Accuracy ({best_acc_row['Model'].split()[0]})</div></div>", unsafe_allow_html=True)
    stat_c2.markdown(f"<div class='metric-card'><div class='metric-value'>{best_acc_row['Macro F1']}</div><div class='metric-label'>Top Macro F1</div></div>", unsafe_allow_html=True)
    stat_c3.markdown(f"<div class='metric-card'><div class='metric-value'>{len(m_df)}</div><div class='metric-label'>Evaluated Models</div></div>", unsafe_allow_html=True)
    stat_c4.markdown(f"<div class='metric-card'><div class='metric-value'>5</div><div class='metric-label'>Target Disciplines</div></div>", unsafe_allow_html=True)

    st.markdown("#### Model Performance Leaderboard")
    st.dataframe(
        m_df[["Model", "Accuracy", "Macro F1", "Weighted F1"]],
        use_container_width=True,
        hide_index=True
    )

    # Confusion Matrix Visualization
    st.markdown("#### Interactive Confusion Matrix")
    selected_bm_model = st.selectbox(
        "Select Model for Confusion Matrix:",
        options=list(bm_data.keys()),
        format_func=lambda x: bm_data[x]["name"],
        index=0
    )

    if selected_bm_model in bm_data:
        cm_data = bm_data[selected_bm_model]["metrics"]
        cm = np.array(cm_data.get("confusion_matrix", []))
        classes = cm_data.get("classes", MAIN_CATEGORIES)

        if len(cm) > 0:
            fig_cm = px.imshow(
                cm,
                x=classes,
                y=classes,
                labels=dict(x="Predicted Discipline", y="True Discipline", color="Count"),
                color_continuous_scale="Viridis",
                text_auto=True
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc", family="Outfit"),
                margin=dict(l=10, r=10, t=30, b=10),
                height=450
            )
            st.plotly_chart(fig_cm, use_container_width=True)


# ==========================================
# TAB 4: DISCIPLINE & KEYWORD EXPLORER
# ==========================================
with tabs[3]:
    st.markdown("### 📖 Discipline & Category Keyword Explorer")
    st.markdown("Inspect characteristic terms and dataset distribution for each academic domain.")

    df_corpus = get_cached_dataset()
    
    col_k1, col_k2 = st.columns([1, 2])
    with col_k1:
        selected_cat = st.selectbox("Select Academic Discipline:", options=MAIN_CATEGORIES, index=0)
        c_color = CATEGORY_COLORS.get(selected_cat, "#3b82f6")
        cat_papers = df_corpus[df_corpus["category"] == selected_cat]
        st.markdown(f"**Total Papers in Corpus:** {len(cat_papers)}")

    with col_k2:
        # Top global keywords for this category
        pipeline = get_pipeline("calibrated_svm")
        if pipeline.explainability_engine:
            top_kws = pipeline.explainability_engine.get_top_category_keywords(top_n=12).get(selected_cat, [])
            st.markdown(f"#### Top Distinctive Terms for <span style='color:{c_color};'>{selected_cat}</span>", unsafe_allow_html=True)
            
            if top_kws:
                kw_df = pd.DataFrame(top_kws, columns=["Term", "Weight"])
                fig_kw = px.bar(
                    kw_df,
                    x="Weight",
                    y="Term",
                    orientation="h",
                    color_discrete_sequence=[c_color]
                )
                fig_kw.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc", family="Outfit"),
                    yaxis=dict(autorange="reversed", title=""),
                    xaxis=dict(title="Learned Importance Weight", gridcolor="rgba(255,255,255,0.08)"),
                    margin=dict(l=10, r=10, t=10, b=20),
                    height=300
                )
                st.plotly_chart(fig_kw, use_container_width=True)

    st.markdown("#### Sample Papers in Corpus")
    st.dataframe(
        cat_papers[["title", "abstract"]].head(6),
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# TAB 5: RETRAINING STUDIO
# ==========================================
with tabs[4]:
    st.markdown("### ⚡ Model Retraining & Fine-Tuning Studio")
    st.markdown("Retrain all classifiers with customized hyperparameters on the arXiv dataset.")

    with st.form("retrain_form"):
        rc1, rc2 = st.columns(2)
        with rc1:
            split_ratio = st.slider("Test Split Ratio:", min_value=0.15, max_value=0.40, value=0.25, step=0.05)
            retrain_dl = st.checkbox("Include PyTorch Deep BiLSTM Model in Training", value=True)
        with rc2:
            st.info("💡 Training optimizes Calibrated SVM, Complement Naive Bayes, Logistic Regression, Ensemble Voting, and PyTorch BiLSTM Attention.")
            
        retrain_btn = st.form_submit_button("🔄 Start Retraining Pipeline", type="primary")

    if retrain_btn:
        with st.spinner("Retraining models and evaluating metrics..."):
            new_bm = train_and_evaluate_all(train_dl=retrain_dl)
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("✅ Models retrained and saved successfully! View the Benchmark tab for updated leaderboards.")
            st.rerun()
