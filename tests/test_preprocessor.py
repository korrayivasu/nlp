import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
from src.preprocessor import TextPreprocessor


def test_clean_text_basic():
    pre = TextPreprocessor()
    text = "We propose a novel deep learning model for image classification."
    cleaned = pre.clean_text(text)
    assert "deep learning" in cleaned
    assert "image classification" in cleaned


def test_clean_latex_math():
    pre = TextPreprocessor(replace_math=True)
    text = "Let $f(x) = \\sum_{i=1}^n x_i$ and $\\mathcal{O}(N \\log N)$ be the complexity."
    cleaned = pre.clean_text(text)
    assert "mathformula" in cleaned
    assert "\\sum" not in cleaned


def test_clean_citations_and_urls():
    pre = TextPreprocessor()
    text = "As shown in [1, 2] and (Vaswani et al., 2017), see https://arxiv.org/abs/1706.03762 for code."
    cleaned = pre.clean_text(text)
    assert "https" not in cleaned
    assert "[1, 2]" not in cleaned


def test_combine_title_abstract():
    pre = TextPreprocessor()
    title = "Attention Is All You Need"
    abstract = "We present the Transformer model."
    combined = pre.combine_title_abstract(title, abstract, title_weight=2)
    # Title words should be repeated based on weight
    assert "attention is all you need attention is all you need" in combined
    assert "transformer" in combined
