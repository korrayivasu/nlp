import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
from src.pipeline import PaperClassificationPipeline


def test_pipeline_prediction():
    pipeline = PaperClassificationPipeline(model_type="calibrated_svm")
    
    # Computer Science paper
    res_cs = pipeline.predict(
        title="Attention Is All You Need for Transformer Models",
        abstract="We propose the Transformer architecture based on self-attention mechanisms for deep neural network language modeling."
    )
    assert res_cs["primary_category"] == "Computer Science"
    assert res_cs["confidence"] > 0.5
    assert len(res_cs["ranked_probabilities"]) == 5
    assert "top_keywords" in res_cs

    # Physics paper
    res_phys = pipeline.predict(
        title="Gravitational Waves from Black Hole Mergers in General Relativity",
        abstract="LIGO observed gravitational waves from binary black hole inspiral with solar masses measured via interferometry."
    )
    assert res_phys["primary_category"] == "Physics"
    assert res_phys["confidence"] > 0.4

    # Biology paper
    res_bio = pipeline.predict(
        title="CRISPR Cas9 Base Editing in Human Pluripotent Stem Cells",
        abstract="We use single cell RNA sequencing to analyze gene expression and transcriptomic profiling in mammalian cells."
    )
    assert res_bio["primary_category"] == "Biology"
    assert res_bio["confidence"] > 0.4


def test_pipeline_empty_input():
    pipeline = PaperClassificationPipeline(model_type="calibrated_svm")
    res = pipeline.predict("", "")
    assert res["primary_category"] == "Unknown"
    assert res["confidence"] == 0.0
