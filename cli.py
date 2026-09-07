import os
import sys

# Ensure root directory is on Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
from data.dataset_loader import fetch_arxiv_papers
from src.pipeline import PaperClassificationPipeline


def format_prediction_output(title: str, abstract: str, result: dict):
    print("=" * 70)
    if title:
        print(f"TITLE:    {title}")
    if abstract:
        snippet = abstract[:180] + "..." if len(abstract) > 180 else abstract
        print(f"ABSTRACT: {snippet}")
    print("-" * 70)
    print(f"PREDICTED CATEGORY:  {result['primary_category']} (Confidence: {result['confidence_percent']})")
    print(f"MODEL USED:          {result['model_used']}")
    
    print("\nPROBABILITY BREAKDOWN:")
    for cat, prob in result["ranked_probabilities"]:
        bar = "#" * int(prob * 30)
        print(f"  {cat:<18}: {prob * 100:5.1f}% | {bar}")
        
    if result["secondary_categories"]:
        sec_str = ", ".join([f"{s['category']} ({s['percent']})" for s in result["secondary_categories"]])
        print(f"\nCROSS-DISCIPLINARY TAGS: {sec_str}")
        
    if result["top_keywords"]:
        kw_str = ", ".join([kw for kw, _ in result["top_keywords"][:5]])
        print(f"KEY SALIENT PHRASES:     {kw_str}")
    print("=" * 70 + "\n")


def run_sample_predictions(pipeline: PaperClassificationPipeline):
    samples = [
        (
            "Transformers in Multi-Modal Deep Neural Networks for Computer Vision",
            "We study self-attention transformer models applied to multi-modal representation learning in Computer Vision. We propose cross-attention fusion layers that align text tokens and visual embeddings into a shared semantic latent space."
        ),
        (
            "Existence and Regularity of Extremal Solutions for Elliptic Partial Differential Equations",
            "We investigate the existence, uniqueness, and Sobolev boundary regularity of weak solutions to a class of quasilinear elliptic partial differential equations with singular p-Laplacian operators on Riemannian manifolds."
        ),
        (
            "Observation of Gravitational Waves from a Binary Black Hole Inspiral",
            "The Laser Interferometer Gravitational-Wave Observatory observed a transient gravitational-wave signal matching general relativity waveforms for the inspiral and merger of binary black holes."
        ),
        (
            "High-Resolution Cryo-EM Structure and Dynamic Gating of Voltage-Gated Ion Channels",
            "Voltage-gated sodium and calcium ion channels control action potential generation. We determine the 2.7-Angstrom cryo-electron microscopy structure of human Nav1.7 in lipid nanodiscs."
        ),
        (
            "High-Dimensional Covariance Matrix Estimation and Minimax Shrinkage Rates",
            "Estimating large covariance matrices when the dimension p exceeds the sample size n is a fundamental challenge. We propose a non-linear eigenvalue shrinkage estimator under Marchenko-Pastur regime."
        )
    ]
    print("\n--- RUNNING BENCHMARK SAMPLES ACROSS 5 DISCIPLINES ---\n")
    for title, abstract in samples:
        res = pipeline.predict(title, abstract)
        format_prediction_output(title, abstract, res)


def main():
    parser = argparse.ArgumentParser(description="arXiv Research Paper Category Classifier")
    parser.add_argument("--title", type=str, default="", help="Research paper title")
    parser.add_argument("--abstract", type=str, default="", help="Research paper abstract text")
    parser.add_argument("--model", type=str, default="calibrated_svm",
                        choices=["calibrated_svm", "complement_nb", "logistic_regression", "ensemble_voting", "deep_bilstm"],
                        help="Classification model to use")
    parser.add_argument("--sample", action="store_true", help="Run on pre-configured diverse test papers")
    parser.add_argument("--arxiv", type=str, default="", help="Live query to search arXiv API and classify top papers")
    parser.add_argument("--max-results", type=int, default=3, help="Max results to fetch from arXiv API")

    args = parser.parse_args()

    pipeline = PaperClassificationPipeline(model_type=args.model)

    if args.sample:
        run_sample_predictions(pipeline)
    elif args.arxiv:
        print(f"\nFetching live papers from arXiv API for query: '{args.arxiv}'...")
        papers = fetch_arxiv_papers(args.arxiv, max_results=args.max_results)
        if not papers:
            print("No papers returned or network timeout.")
            return
        print(f"Retrieved {len(papers)} papers from arXiv. Classifying...\n")
        for p in papers:
            res = pipeline.predict(p["title"], p["abstract"])
            print(f"arXiv ID:     {p.get('id', 'N/A')}")
            print(f"arXiv Ground Truth Primary Code: {p.get('primary_category_code', 'N/A')} ({p.get('primary_category', 'N/A')})")
            format_prediction_output(p["title"], p["abstract"], res)
    elif args.title or args.abstract:
        res = pipeline.predict(args.title, args.abstract)
        format_prediction_output(args.title, args.abstract, res)
    else:
        print("Please provide --title and --abstract, or pass --sample or --arxiv 'query'.")
        parser.print_help()


if __name__ == "__main__":
    main()
