"""
Text Preprocessor for Academic Research Papers.
Handles LaTeX symbols, citations, formulas, and academic text normalization.
"""

import re
import string
from typing import List, Optional, Union


# Standard scientific stop words that don't help category classification
ACADEMIC_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
    # Generic meta-paper words (often appearing across all disciplines)
    "paper", "propose", "present", "demonstrate", "study", "results", "show",
    "using", "used", "approach", "method", "methods", "based", "also", "new"
}


class TextPreprocessor:
    """
    NLP Preprocessor specifically tuned for scientific & arXiv abstracts.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        replace_math: bool = True,
        preserve_scientific_terms: bool = True
    ):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.replace_math = replace_math
        self.preserve_scientific_terms = preserve_scientific_terms

    def clean_text(self, text: str) -> str:
        """
        Normalize LaTeX equations, URLs, citations, and special symbols in scientific text.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        
        # 2. Remove arXiv identifiers (e.g. arXiv:2104.12345v1)
        text = re.sub(r"arXiv:\d{4}\.\d{4,5}(?:v\d+)?", " ", text, flags=re.IGNORECASE)
        
        # 3. Clean LaTeX math environments and formulas
        if self.replace_math:
            # Inline math $...$
            text = re.sub(r"\$[^$]+\$", " mathformula ", text)
            # Display math $$...$$ or \[...\]
            text = re.sub(r"\$\$[^$]+\$\$", " mathformula ", text)
            text = re.sub(r"\\\[.*?\\\]", " mathformula ", text)
            # Common TeX commands: \mathcal{O}, \mathbb{R}, \sqrt{}, etc.
            text = re.sub(r"\\[a-zA-Z]+(?:\{[^}]*\})*", " ", text)
        
        # 4. Remove bracketed citations like [1, 2] or (Smith et al., 2021)
        text = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", text)
        text = re.sub(r"\([A-Z][a-zA-Z\s]+ et al\.,?\s*\d{4}\)", " ", text)
        
        # 5. Normalize hyphens and dashes (e.g., state-of-the-art -> state-of-the-art or state of the art)
        text = re.sub(r"[–—−]", "-", text)
        
        # 6. Lowercase if enabled
        if self.lowercase:
            text = text.lower()
            
        # 7. Strip isolated punctuation while keeping alphanumeric and hyphens
        # Replace non-word characters (except hyphens inside words) with space
        text = re.sub(r"[^\w\s-]", " ", text)
        text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)
        
        # 8. Filter stopwords if enabled
        if self.remove_stopwords:
            tokens = text.split()
            tokens = [t for t in tokens if t not in ACADEMIC_STOPWORDS and len(t) > 1]
            text = " ".join(tokens)

        # 9. Clean extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def combine_title_abstract(self, title: str, abstract: str, title_weight: int = 2) -> str:
        """
        Combines title and abstract with customizable title repetition weight.
        Title terms carry strong category signal.
        """
        title = (title or "").strip()
        abstract = (abstract or "").strip()
        
        if not title and not abstract:
            return ""
        
        weighted_title = " ".join([title] * max(1, title_weight))
        full_text = f"{weighted_title} {abstract}".strip()
        return self.clean_text(full_text)

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer after cleaning."""
        cleaned = self.clean_text(text)
        return cleaned.split() if cleaned else []


if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    sample_title = "Attention Is All You Need for $\\mathcal{O}(N^2)$ Transformers"
    sample_abs = "We present the Transformer architecture. Using self-attention mechanisms over $[1, 2]$, we achieve $99.4\\%$ accuracy."
    
    combined = preprocessor.combine_title_abstract(sample_title, sample_abs)
    print("Preprocessed combined text:\n", combined)
