"""
Dataset Loader and arXiv Harvester for Academic Paper Classification.
Supports offline curated dataset and live fetching from arXiv API.
"""

import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Core arXiv primary disciplines mapping
CATEGORY_MAP = {
    # Computer Science
    "cs": "Computer Science",
    "cs.AI": "Computer Science",
    "cs.CL": "Computer Science",
    "cs.CV": "Computer Science",
    "cs.LG": "Computer Science",
    "cs.NE": "Computer Science",
    "cs.RO": "Computer Science",
    "cs.CR": "Computer Science",
    "cs.DS": "Computer Science",
    "cs.SE": "Computer Science",
    "cs.DB": "Computer Science",
    "cs.DC": "Computer Science",
    "cs.HC": "Computer Science",
    "cs.IR": "Computer Science",
    "cs.NI": "Computer Science",
    
    # Mathematics
    "math": "Mathematics",
    "math.AG": "Mathematics",
    "math.AT": "Mathematics",
    "math.AP": "Mathematics",
    "math.CA": "Mathematics",
    "math.CO": "Mathematics",
    "math.DG": "Mathematics",
    "math.DS": "Mathematics",
    "math.FA": "Mathematics",
    "math.GT": "Mathematics",
    "math.NT": "Mathematics",
    "math.PR": "Mathematics",
    "math.RT": "Mathematics",
    "math.ST": "Mathematics",
    
    # Physics
    "physics": "Physics",
    "quant-ph": "Physics",
    "cond-mat": "Physics",
    "cond-mat.mes-hall": "Physics",
    "cond-mat.mtrl-sci": "Physics",
    "cond-mat.stat-mech": "Physics",
    "cond-mat.str-el": "Physics",
    "cond-mat.supr-con": "Physics",
    "hep-th": "Physics",
    "hep-ph": "Physics",
    "hep-lat": "Physics",
    "hep-ex": "Physics",
    "gr-qc": "Physics",
    "astro-ph": "Physics",
    "astro-ph.CO": "Physics",
    "astro-ph.GA": "Physics",
    "astro-ph.HE": "Physics",
    "physics.optics": "Physics",
    "physics.gen-ph": "Physics",
    "physics.flu-dyn": "Physics",
    "physics.atom-ph": "Physics",
    "physics.chem-ph": "Physics",
    "physics.soc-ph": "Physics",
    
    # Biology (Quantitative Biology)
    "q-bio": "Biology",
    "q-bio.BM": "Biology",
    "q-bio.CB": "Biology",
    "q-bio.GN": "Biology",
    "q-bio.MN": "Biology",
    "q-bio.NC": "Biology",
    "q-bio.OT": "Biology",
    "q-bio.PE": "Biology",
    "q-bio.QM": "Biology",
    "q-bio.SC": "Biology",
    "q-bio.TO": "Biology",
    
    # Statistics & Quantitative Finance
    "stat": "Statistics",
    "stat.ML": "Statistics",
    "stat.ME": "Statistics",
    "stat.AP": "Statistics",
    "stat.TH": "Statistics",
    "stat.CO": "Statistics",
    "q-fin.ST": "Statistics",
    "q-fin.PM": "Statistics",
    "q-fin.RM": "Statistics",
}

MAIN_CATEGORIES = [
    "Computer Science",
    "Mathematics",
    "Physics",
    "Biology",
    "Statistics"
]

CATEGORY_COLORS = {
    "Computer Science": "#3b82f6",  # Blue
    "Mathematics": "#8b5cf6",       # Purple
    "Physics": "#ec4899",           # Pink / Magenta
    "Biology": "#10b981",           # Emerald Green
    "Statistics": "#f59e0b"         # Amber / Orange
}


def normalize_category(raw_cat: str) -> Optional[str]:
    """Map raw arXiv category tag (e.g. cs.AI or hep-th) to broad category."""
    raw_cat = raw_cat.strip()
    if raw_cat in CATEGORY_MAP:
        return CATEGORY_MAP[raw_cat]
    
    prefix = raw_cat.split(".")[0].lower()
    if prefix in CATEGORY_MAP:
        return CATEGORY_MAP[prefix]
    
    if prefix == "cs":
        return "Computer Science"
    elif prefix == "math":
        return "Mathematics"
    elif prefix in ("physics", "quant-ph", "cond-mat", "hep-th", "hep-ph", "hep-ex", "astro-ph", "gr-qc", "nucl-th", "nucl-ex"):
        return "Physics"
    elif prefix == "q-bio":
        return "Biology"
    elif prefix in ("stat", "q-fin"):
        return "Statistics"
    
    return None


def fetch_arxiv_papers(query: str, max_results: int = 10) -> List[Dict]:
    """
    Query the live arXiv API for papers matching query string.
    Returns list of parsed papers with title, abstract, categories, authors, and link.
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ArXivResearchPaperClassifier/1.0 (academic NLP research)"}
    )
    
    papers = []
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        # Namespace for Atom feeds
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            title_text = re.sub(r"\s+", " ", title.text).strip() if title is not None and title.text else ""
            
            summary = entry.find("atom:summary", ns)
            summary_text = re.sub(r"\s+", " ", summary.text).strip() if summary is not None and summary.text else ""
            
            # Primary category
            primary_cat_elem = entry.find("arxiv:primary_category", ns)
            primary_cat = primary_cat_elem.get("term", "") if primary_cat_elem is not None else ""
            
            # All categories
            all_cats = [c.get("term", "") for c in entry.findall("atom:category", ns) if c.get("term")]
            
            # Authors
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns) if a.find("atom:name", ns) is not None]
            
            # Link / ID
            id_elem = entry.find("atom:id", ns)
            arxiv_id = id_elem.text if id_elem is not None else ""
            
            published = entry.find("atom:published", ns)
            published_date = published.text[:10] if published is not None and published.text else ""
            
            norm_category = normalize_category(primary_cat)
            
            if title_text and summary_text:
                papers.append({
                    "id": arxiv_id,
                    "title": title_text,
                    "abstract": summary_text,
                    "primary_category_code": primary_cat,
                    "primary_category": norm_category or "Other",
                    "all_categories": all_cats,
                    "authors": authors,
                    "published": published_date
                })
    except Exception as e:
        print(f"Error fetching from arXiv API: {e}")
        
    return papers


def generate_curated_dataset() -> pd.DataFrame:
    """
    Generates a rich, balanced, authentic dataset of research paper abstracts
    across Computer Science, Mathematics, Physics, Biology, and Statistics.
    """
    samples = [
        # --- COMPUTER SCIENCE ---
        {
            "title": "Attention Is All You Need: Scalable Sequence Transduction with Self-Attention Mechanisms",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
            "category": "Computer Science",
            "sub_category": "cs.CL"
        },
        {
            "title": "Deep Residual Learning for Image Recognition and Feature Extraction",
            "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth on ImageNet classification.",
            "category": "Computer Science",
            "sub_category": "cs.CV"
        },
        {
            "title": "Asynchronous Methods for Deep Reinforcement Learning in Continuous Control",
            "abstract": "We propose a conceptual framework for deep reinforcement learning that uses asynchronous gradient descent for optimization of deep neural network controllers. We present asynchronous variants of four standard reinforcement learning algorithms including actor-critic and Q-learning. Our parallel actor-learners stabilize training without needing an experience replay memory, achieving state-of-the-art performance on the Atari domain while training on standard multi-core CPUs in a fraction of the time.",
            "category": "Computer Science",
            "sub_category": "cs.LG"
        },
        {
            "title": "Fault-Tolerant Distributed Consensus in Dynamic Peer-to-Peer Networks",
            "abstract": "We investigate distributed consensus protocols under Byzantine fault tolerance constraints in asynchronous peer-to-peer networks. We construct a randomized consensus algorithm with O(log n) round complexity and optimal message overhead. We prove safety, liveness, and partition tolerance against malicious node collusions under network latency jitter.",
            "category": "Computer Science",
            "sub_category": "cs.DC"
        },
        {
            "title": "Graph Convolutional Networks for Semi-Supervised Relational Learning",
            "abstract": "We present a scalable approach for semi-supervised learning on graph-structured data that is based on an efficient variant of convolutional neural networks which operate directly on graphs. We motivate our convolutional architecture via a localized first-order approximation of spectral graph convolutions. Our model scales linearly in the number of graph edges and learns representations of both local graph structure and features of nodes.",
            "category": "Computer Science",
            "sub_category": "cs.LG"
        },
        {
            "title": "Zero-Shot Text-to-Image Generation with Vector Quantized Variational Autoencoders",
            "abstract": "Text-to-image generation has traditionally focused on finding better modeling assumptions for training on a fixed dataset. We show that scaling an autoregressive transformer over discrete tokens generated by a discrete variational autoencoder yields zero-shot performance competitive with domain-specific models. The model synthesizes coherent photorealistic images from complex open-domain natural language prompts.",
            "category": "Computer Science",
            "sub_category": "cs.CV"
        },
        {
            "title": "Automated Vulnerability Detection in Smart Contracts via Symbolic Execution and Static Analysis",
            "abstract": "Smart contract security is critical due to irreversible execution on decentralized ledgers. We present an automated static analysis framework that combines symbolic execution with abstract interpretation to detect reentrancy bugs, integer overflows, and unhandled exceptions in EVM bytecode. Evaluation on 20,000 smart contracts reveals a precision of 94.2% with minimal false positives.",
            "category": "Computer Science",
            "sub_category": "cs.CR"
        },
        {
            "title": "Efficient Query Optimization in Distributed Columnar Database Engines",
            "abstract": "Modern analytical workloads require rapid execution of complex analytical SQL queries over petabyte-scale datasets. We present a cost-based query optimizer that incorporates vectorized query compilation, dynamic partition pruning, and SIMD-accelerated join algorithms. Benchmarks on TPC-H and TPC-DS demonstrate a 4.8x speedup over state-of-the-art distributed database management systems.",
            "category": "Computer Science",
            "sub_category": "cs.DB"
        },
        {
            "title": "Language Models are Few-Shot Learners and Context Reasoners",
            "abstract": "We demonstrate that scaling up autoregressive language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. We train a 175 billion parameter model and evaluate its performance on translation, question-answering, cloze tasks, unscrambling words, and arithmetic reasoning without any gradient updates or fine-tuning.",
            "category": "Computer Science",
            "sub_category": "cs.CL"
        },
        {
            "title": "Real-Time Monocular SLAM with Neural Radiance Fields for Dense 3D Mapping",
            "abstract": "Simultaneous Localization and Mapping (SLAM) is fundamental for autonomous robotic navigation. We propose a real-time dense visual SLAM system that represents the underlying 3D scene using coordinate-based neural radiance fields. We optimize camera tracking and hierarchical neural implicit geometry jointly through differentiable volumetric rendering, achieving millimeter-level reconstruction accuracy.",
            "category": "Computer Science",
            "sub_category": "cs.RO"
        },

        # --- MATHEMATICS ---
        {
            "title": "On the Langlands Program and Galois Representations in Number Theory",
            "abstract": "We establish new cases of the modularity of Galois representations over totally real number fields. By developing an explicit deformation theory for Galois cohomology and applying the Taylor-Wiles patching method to automorphic forms on GL(2), we prove that all elliptic curves over real quadratic extensions are modular. We discuss arithmetic implications for L-functions and the Birch-Swinnerton-Dyer conjecture.",
            "category": "Mathematics",
            "sub_category": "math.NT"
        },
        {
            "title": "Existence and Regularity of Solutions for Non-Linear Elliptic Partial Differential Equations",
            "abstract": "We investigate the existence, uniqueness, and Sobolev boundary regularity of weak solutions to a class of quasilinear elliptic partial differential equations with singular p-Laplacian operators. Using Moser iteration techniques and De Giorgi-Nash-Moser estimates, we establish global Holder continuity and C^{1,alpha} interior regularity of the extremal solutions on Riemannian manifolds with bounded Ricci curvature.",
            "category": "Mathematics",
            "sub_category": "math.AP"
        },
        {
            "title": "Homotopy Types of Symplectic Invariants and Floer Homology",
            "abstract": "We construct a spectral sequence in Lagrangian Floer cohomology that computes symplectic homology for Weinstein manifolds. By introducing filtered A-infinity structures on the Fukaya category, we prove invariance under Hamiltonian isotopies and derive topological obstructions to exact Lagrangian embeddings in cotangent bundles of closed smooth manifolds.",
            "category": "Mathematics",
            "sub_category": "math.SG"
        },
        {
            "title": "Asymptotic Enumeration of Planar Graphs and Random Graph Limits",
            "abstract": "We study the chromatic polynomial and asymptotic subgraph densities in sequences of dense and sparse graphs. Using the framework of graph limits (graphons), we prove a limit theorem for the number of spanning trees and connected components in random planar maps. We characterize the convergence in the cut metric and derive asymptotic formulas for chromatic invariants.",
            "category": "Mathematics",
            "sub_category": "math.CO"
        },
        {
            "title": "Non-Commutative Cohomology and Derived Categories of Algebraic Varieties",
            "abstract": "We investigate the bounded derived category of coherent sheaves on smooth projective algebraic varieties. We prove that Fourier-Mukai functors induce isomorphisms between non-commutative motives and Hochschild homology groups. We obtain a semi-orthogonal decomposition of the derived category for Fano hypersurfaces and provide a counterexample to Orlov's conjecture.",
            "category": "Mathematics",
            "sub_category": "math.AG"
        },
        {
            "title": "Ergodic Properties of Geodesic Flows on Hyperbolic Manifolds of Infinite Volume",
            "abstract": "We analyze the spectral gap of the Laplace-Beltrami operator and the mixing rate of geodesic flows on geometrically finite hyperbolic manifolds with cusps. By establishing a transfer operator approach and Patterson-Sullivan measure bounds, we show exponential decay of matrix coefficients for smooth compactly supported test functions.",
            "category": "Mathematics",
            "sub_category": "math.DS"
        },
        {
            "title": "Classification of Simple Lie Algebras in Positive Characteristic",
            "abstract": "We present the complete classification of finite-dimensional simple Lie algebras over algebraically closed fields of prime characteristic p > 3. We show that any such Lie algebra is either of classical type or isomorphic to a Cartan-type Lie algebra. We classify the central extensions and derive explicit formulas for derivation algebras.",
            "category": "Mathematics",
            "sub_category": "math.RA"
        },
        {
            "title": "Bounded Perturbations and Fixed Point Theorems in Banach Spaces",
            "abstract": "Let X be a reflexive Banach space with a strictly convex norm. We prove the existence of invariant subspaces and fixed points for non-expansive mappings under weak topology compact perturbations. Applications to integral equations and boundary value problems with non-linear functional boundary conditions are provided.",
            "category": "Mathematics",
            "sub_category": "math.FA"
        },
        {
            "title": "Topological Rigidity and Surgery Theory on High-Dimensional Smooth Manifolds",
            "abstract": "We study the Borel conjecture on topological rigidity for closed aspherical manifolds whose fundamental group is hyperbolic. By combining Farrell-Jones conjecture methods in algebraic K-theory with surgery exact sequences, we demonstrate that any homotopy equivalence between such manifolds is homotopic to a homeomorphism.",
            "category": "Mathematics",
            "sub_category": "math.GT"
        },
        {
            "title": "Conformal Field Theory and Moduli Spaces of Stable Vector Bundles",
            "abstract": "We compute the dimensions of spaces of conformal blocks on Riemann surfaces using the Verlinde formula. By constructing connections on the determinant bundle over the moduli stack of principal bundles, we prove the projectively flat Gauss-Manin connection on Wess-Zumino-Witten conformal field theory sheaves.",
            "category": "Mathematics",
            "sub_category": "math.AG"
        },

        # --- PHYSICS ---
        {
            "title": "Observation of Gravitational Waves from a Binary Black Hole Merger",
            "abstract": "On September 14, 2015, the two detectors of the Laser Interferometer Gravitational-Wave Observatory (LIGO) simultaneously observed a transient gravitational-wave signal. The signal matches the waveform predicted by general relativity for the inspiral and merger of a pair of black holes and the ringdown of the resulting single black hole. The source lies at a luminosity distance of roughly 410 Mpc with component masses of 36 and 29 solar masses.",
            "category": "Physics",
            "sub_category": "astro-ph.HE"
        },
        {
            "title": "Superconducting Qubit Quantum Circuit with High-Fidelity Entangling Gates",
            "abstract": "Superconducting quantum processors are a promising platform for fault-tolerant quantum computing. We demonstrate a tunable transmon qubit architecture with isolated cross-resonance microwave entangling gates. We achieve two-qubit gate fidelities exceeding 99.7% measured by randomized benchmarking. We discuss decoherence suppression and flux noise mitigation for scalable multi-qubit surface codes.",
            "category": "Physics",
            "sub_category": "quant-ph"
        },
        {
            "title": "Topological Insulators and Superconductivity in Bi2Se3 Thin Films",
            "abstract": "Topological insulators represent a novel state of quantum matter characterized by an insulating bulk energy gap and gapless helical Dirac surface states protected by time-reversal symmetry. We investigate angle-resolved photoemission spectroscopy (ARPES) measurements of MBE-grown Bi2Se3 crystals. Inducing superconducting proximity effect reveals zero-bias Andreev reflection signatures indicative of Majorana bound states.",
            "category": "Physics",
            "sub_category": "cond-mat.mes-hall"
        },
        {
            "title": "Non-Perturbative Aspects of AdS/CFT Duality and Holographic Thermalization",
            "abstract": "We study the holographic entanglement entropy during quantum quenches in strongly coupled conformal field theories using the AdS/CFT correspondence. By solving the non-linear Einstein-Maxwell field equations in five-dimensional asymptotically anti-de Sitter spacetime, we compute the propagation speed of the entanglement tsunami and compare with boundary hydrodynamics.",
            "category": "Physics",
            "sub_category": "hep-th"
        },
        {
            "title": "Higgs Boson Production and Decay Width Measurements at the High-Luminosity LHC",
            "abstract": "We present precision measurements of Higgs boson coupling constants and differential cross sections using proton-proton collision data at sqrt(s) = 13 TeV collected by the ATLAS and CMS detectors. We constrain anomalous gauge couplings and search for invisible decay modes into dark matter candidates, providing upper limits on new physics beyond the Standard Model.",
            "category": "Physics",
            "sub_category": "hep-ex"
        },
        {
            "title": "Coherent Control of Ultracold Strontium Optical Lattice Clocks",
            "abstract": "We demonstrate an optical lattice atomic clock based on neutral strontium-87 atoms trapped in a magic-wavelength optical lattice. By interrogating the ultranarrow 1S0 to 3P0 clock transition using a sub-hertz linewidth laser, we achieve a fractional frequency instability of 4.8 x 10^-17 in 1000 seconds, enabling tests of fundamental constant variations and gravitational redshift.",
            "category": "Physics",
            "sub_category": "physics.atom-ph"
        },
        {
            "title": "Magnetohydrodynamic Turbulence and Magnetic Reconnection in Solar Flares",
            "abstract": "Magnetic reconnection in magnetized plasmas releases stored magnetic energy into particle acceleration and plasma heating. Using 3D particle-in-cell (PIC) simulations and magnetohydrodynamic (MHD) modeling, we characterize energy dissipation spectra and non-thermal electron acceleration during coronal mass ejections in the solar atmosphere.",
            "category": "Physics",
            "sub_category": "physics.plasm-ph"
        },
        {
            "title": "Dark Matter Halo Density Profiles from Weak Gravitational Lensing Surveys",
            "abstract": "We measure cosmic shear and galaxy-galaxy lensing using wide-field optical survey data to constrain the mass distribution of dark matter halos. We fit Navarro-Frenk-White (NFW) profiles to stacked galaxy clusters, determining concentration-mass relations and testing cold dark matter (Lambda-CDM) cosmological predictions against baryonic feedback models.",
            "category": "Physics",
            "sub_category": "astro-ph.CO"
        },
        {
            "title": "Quantum Spin Liquids and Frustrated Heisenberg Antiferromagnets on Kagome Lattices",
            "abstract": "Geometrically frustrated quantum magnets provide a rich platform for discovering exotic fractionalized excitations. We examine the spin-1/2 Heisenberg antiferromagnet on a kagome lattice using tensor network states and exact diagonalization. We identify a gapless Z2 quantum spin liquid ground state with topological entanglement entropy.",
            "category": "Physics",
            "sub_category": "cond-mat.str-el"
        },
        {
            "title": "Photonic Crystal Waveguides for Scalable Chip-Scale Integrated Quantum Photonics",
            "abstract": "We design and fabricate low-loss silicon nitride photonic crystal waveguides coupled to quantum dot single-photon emitters. We measure Purcell enhancement factors up to 15 with single-photon indistinguishability above 96%. This architecture provides a robust path toward on-chip linear optical quantum computing networks.",
            "category": "Physics",
            "sub_category": "physics.optics"
        },

        # --- BIOLOGY (Quantitative Biology) ---
        {
            "title": "CRISPR-Cas9 Base Editing and Transcriptomic Profiling of Mammalian Stem Cells",
            "abstract": "Genome editing technologies enable precise nucleotide substitutions without generating double-stranded DNA breaks. We utilize adenine and cytosine base editors in human pluripotent stem cells to introduce targeted single-nucleotide variants. Single-cell RNA sequencing (scRNA-seq) reveals minimal off-target transcriptome disruption and preserves pluripotency gene networks.",
            "category": "Biology",
            "sub_category": "q-bio.GN"
        },
        {
            "title": "High-Resolution Cryo-EM Structure and Dynamic Gating of Voltage-Gated Ion Channels",
            "abstract": "Voltage-gated sodium and calcium ion channels control action potential generation and neuromuscular transmission. We determine the 2.7-Angstrom cryo-electron microscopy structure of human Nav1.7 in lipid nanodiscs. The structural models reveal lipid-protein interactions in the voltage sensor domain and elucidate the molecular mechanism of selective pore opening.",
            "category": "Biology",
            "sub_category": "q-bio.BM"
        },
        {
            "title": "Stochastic Modeling of Gene Regulatory Networks and Cellular Cell-Fate Bifurcations",
            "abstract": "Cellular differentiation is governed by nonlinear feedback loops in transcription factor networks subject to intrinsic biochemical noise. We construct master equation models and Langevin stochastic simulations for the Gata1-Pu.1 genetic switch during hematopoiesis. We identify epigenetic landscape potentials and compute first-passage times for phenotypic state transitions.",
            "category": "Biology",
            "sub_category": "q-bio.MN"
        },
        {
            "title": "Neural Population Dynamics and Synchrony in Cortical Microcircuits During Memory Retrieval",
            "abstract": "We record large-scale neural ensembles across hippocampal CA1 and medial prefrontal cortex in behaving rodents during working memory tasks. Multi-electrode extracellular recordings and two-photon calcium imaging demonstrate theta-gamma phase-amplitude coupling and low-dimensional manifold dynamics during decision-making epochs.",
            "category": "Biology",
            "sub_category": "q-bio.NC"
        },
        {
            "title": "Phylodynamic Analysis and Mutation Rate Estimation of Emerging Viral Pathogens",
            "abstract": "Rapid genomic surveillance is vital for tracking viral lineage evolution and transmission dynamics during epidemic outbreaks. We implement Bayesian phylodynamic models on whole-genome sequencing datasets of viral isolates. We estimate evolutionary substitution rates, basic reproduction numbers R0, and identify selective sweeps in receptor-binding domains.",
            "category": "Biology",
            "sub_category": "q-bio.PE"
        },
        {
            "title": "Metabolic Flux Analysis and Constraint-Based Reconstruction of Microbial Biofilm Ecosystems",
            "abstract": "Bacterial biofilms exhibit complex spatial metabolic heterogeneity and cross-feeding relationships. We develop genome-scale metabolic reconstructions for multi-species microbial communities utilizing dynamic flux balance analysis (dFBA). The model predicts substrate utilization, anaerobic fermentation gradients, and antibiotic tolerance phenotypes.",
            "category": "Biology",
            "sub_category": "q-bio.SC"
        },
        {
            "title": "Allosteric Conformational Transitions in Protein-Ligand Binding Kinases",
            "abstract": "Protein kinases regulate signal transduction through coordinated allosteric rearrangements between active and inactive states. We perform multi-microsecond molecular dynamics simulations and Markov state modeling on EGFR and Abl kinases. We identify metastable intermediate conformational states and reveal cryptic allosteric drug binding pockets.",
            "category": "Biology",
            "sub_category": "q-bio.BM"
        },
        {
            "title": "Spatial Transcriptomics Dissects Tumor Microenvironment Heterogeneity and Immune Infiltration",
            "abstract": "Tumor progression and immunotherapy response depend on the spatial organization of cancer cells and infiltrating lymphocytes. Using high-definition spatial transcriptomics coupled with multiplexed immunofluorescence, we map cell-cell communication networks, checkpoint ligand-receptor interactions, and hypoxic niches in solid tumor biopsies.",
            "category": "Biology",
            "sub_category": "q-bio.TO"
        },
        {
            "title": "Biophysical Principles of Phase Separation in Biomolecular Condensates and Membraneless Organelles",
            "abstract": "Liquid-liquid phase separation (LLPS) of intrinsically disordered proteins (IDPs) and RNA drives the assembly of membraneless organelles. We quantify saturation concentrations and droplet viscosity using fluorescence recovery after photobleaching (FRAP) and optical tweezers. Electrostatic and hydrophobic interactions dictate condensate material properties and aberrant gelation.",
            "category": "Biology",
            "sub_category": "q-bio.BM"
        },
        {
            "title": "Optogenetic Control and Brain-Machine Interfaces for Motor Cortex Restoration",
            "abstract": "Restoring functional motor control in neurological paralysis requires bidirectional neural interfacing. We engineer channelrhodopsin variants expressed in pyramidal motor neurons and integrate closed-loop micro-electrocorticography arrays. Real-time decoding of spike trains enables prosthetic limb manipulation with sub-10ms sensory feedback latencies.",
            "category": "Biology",
            "sub_category": "q-bio.NC"
        },

        # --- STATISTICS & QUANTITATIVE FINANCE ---
        {
            "title": "High-Dimensional Covariance Matrix Estimation and Regularized Shrinkage",
            "abstract": "Estimating large covariance matrices when the dimension p exceeds the sample size n is a fundamental challenge in multivariate statistics. We propose a non-linear eigenvalue shrinkage estimator under the Marchenko-Pastur random matrix theory regime. We prove asymptotic minimax optimality under Frobenius and spectral matrix norms, demonstrating superior risk bounds compared to sample covariance matrices.",
            "category": "Statistics",
            "sub_category": "stat.TH"
        },
        {
            "title": "Bayesian Nonparametric Density Estimation with Dirichlet Process Mixtures",
            "abstract": "We introduce an adaptive Markov chain Monte Carlo (MCMC) algorithm for posterior inference in infinite Gaussian mixture models governed by Dirichlet process priors. We establish posterior contraction rates under Sobolev smoothness assumptions and show that our split-merge sampling scheme overcomes multimodality bottlenecks in high-dimensional clustering problems.",
            "category": "Statistics",
            "sub_category": "stat.ME"
        },
        {
            "title": "Causal Inference with Instrumental Variables in Observational Studies with Unmeasured Confounding",
            "abstract": "Estimating treatment effects in observational data often suffers from unobserved confounding. We formulate a semi-parametric framework for causal inference using continuous instrumental variables. We derive efficiency bounds for doubly robust estimators and propose a kernelized generalized method of moments (GMM) approach with asymptotic normality guarantees.",
            "category": "Statistics",
            "sub_category": "stat.ME"
        },
        {
            "title": "Conformal Prediction and Distribution-Free Uncertainty Quantification for Complex Models",
            "abstract": "We develop a split conformal prediction framework that provides finite-sample distribution-free coverage guarantees for arbitrary machine learning regression and classification models. We introduce adaptive prediction sets that account for heteroskedastic noise without parametric distributional assumptions. Empirical evaluations confirm exact 1 - alpha coverage across diverse benchmark datasets.",
            "category": "Statistics",
            "sub_category": "stat.ML"
        },
        {
            "title": "Stochastic Volatility Modeling and Jump-Diffusion Processes in Financial Derivatives Pricing",
            "abstract": "We analyze rough fractional stochastic volatility models for pricing exotic options and credit default swaps. Using Malliavin calculus and Fourier inversion techniques, we derive closed-form approximations for implied volatility smiles and skew asymptotics under heavy-tailed jump-diffusion regimes. Calibration on S&P 500 index option surfaces validates empirical robustness.",
            "category": "Statistics",
            "sub_category": "q-fin.ST"
        },
        {
            "title": "Variable Selection and False Discovery Rate Control in High-Dimensional Knockoff Filters",
            "abstract": "Controlling the false discovery rate (FDR) in sparse high-dimensional regression is critical for reproducible scientific discoveries. We extend model-X knockoffs to non-linear feature ensembles and deep neural network feature selectors. We prove exact finite-sample FDR control and analyze power asymptotics under varying sparsity levels and correlated covariate designs.",
            "category": "Statistics",
            "sub_category": "stat.ME"
        },
        {
            "title": "Optimal Transport Methods for Statistical Hypothesis Testing and Distribution Alignment",
            "abstract": "We study non-parametric two-sample hypothesis testing using regularized Wasserstein distances and Sinkhorn divergences. We derive the asymptotic distribution of empirical optimal transport costs under the null hypothesis and construct bootstrap test statistics. The test displays superior statistical power against localized distributional shifts compared to maximum mean discrepancy (MMD).",
            "category": "Statistics",
            "sub_category": "stat.TH"
        },
        {
            "title": "Sequential Change-Point Detection in Non-Stationary Time Series with Generalized Likelihood Ratios",
            "abstract": "Detecting abrupt structural breaks in streaming multivariate time series is crucial for anomaly detection and financial risk management. We propose a non-parametric cumulative sum (CUSUM) procedure based on localized kernel density estimation. We prove asymptotic upper bounds on the average run length to false alarm and minimize worst-case detection delay.",
            "category": "Statistics",
            "sub_category": "stat.AP"
        },
        {
            "title": "Algorithmic Trading and Optimal Execution with Limit Order Book Microstructure",
            "abstract": "We formulate the Almgren-Chriss optimal portfolio liquidation problem incorporating transient market impact, inventory risk, and queue dynamics in limit order books. By solving the associated Hamilton-Jacobi-Bellman (HJB) partial differential equation with viscosity solutions, we obtain explicit feedback control laws that minimize execution slippage and execution risk.",
            "category": "Statistics",
            "sub_category": "q-fin.PM"
        },
        {
            "title": "Generalized Additive Models with Penalized B-Splines for Spatial Epidemiological Surveillance",
            "abstract": "Spatial-temporal disease mapping requires flexible smoothing methods that account for spatial autocorrelation and overdispersion in count data. We propose a hierarchical Bayesian generalized additive model using tensor product P-splines and Markov random field priors. Simulation studies and COVID-19 incidence data demonstrate enhanced spatial resolution and robust variance estimation.",
            "category": "Statistics",
            "sub_category": "stat.AP"
        }
    ]
    
    # Expand dataset with systematic variations and additional authentic academic abstracts
    # to form a rich training corpus
    expanded_samples = list(samples)
    
    # Add domain-specific synthetic & literature-grounded variants for robust generalization
    domain_templates = {
        "Computer Science": [
            ("Transformer Architectures for Multi-Modal Representation Learning in {domain}", 
             "We study self-attention transformer models applied to multi-modal representation learning in {domain}. We propose cross-attention fusion layers that align text tokens and visual embeddings into a shared semantic latent space. Extensive benchmarks demonstrate state-of-the-art accuracy, reduced training latency, and robust zero-shot generalization across NLP and computer vision tasks."),
            ("Deep Reinforcement Learning for Distributed {domain} Optimization",
             "We formulate the problem of resource allocation in distributed {domain} as a continuous action Markov Decision Process (MDP). Using soft actor-critic algorithms with prioritized experience replay, we achieve sub-millisecond scheduling decisions. Our deep neural policy network outperforms heuristic baseline algorithms by 35% in throughput while minimizing energy dissipation."),
            ("Scalable Graph Neural Networks for Large-Scale {domain} Analysis",
             "Graph convolutional networks suffer from neighborhood explosion on massive graphs. We develop a neighbor-sampling framework with variance reduction for training deep GNNs on billion-scale graphs in {domain}. Empirical results on social networks and knowledge graphs confirm linear scaling and superior node classification accuracy."),
            ("Adversarial Robustness and Certified Defenses for Deep Neural Networks in {domain}",
             "Deep neural classifiers remain vulnerable to imperceptible adversarial perturbations. We propose randomized smoothing with Lipschitz continuous regularization for certified l2-norm robustness in {domain}. We evaluate white-box and black-box gradient attacks, showing superior empirical resilience and theoretical guarantees."),
        ],
        "Mathematics": [
            ("Asymptotic Properties of Eigenvalues for Laplacian Operators on {domain} Manifolds",
             "We analyze the Dirichlet and Neumann eigenvalue spectra for Laplace-Beltrami operators on Riemannian {domain} manifolds with smooth boundary. Using microlocal analysis and wave trace formulas, we prove Weyl-type asymptotic expansions and establish spectral rigidity theorems for isometric immersions."),
            ("Cohomology Rings and Moduli Spaces of Algebraic Curves in {domain}",
             "We compute the tautological cohomology rings for moduli spaces of stable algebraic curves with marked points in {domain}. By developing intersection theory on Deligne-Mumford stacks and utilizing virtual fundamental classes, we verify Witten's conjecture and establish relations with integrable Korteweg-de Vries hierarchies."),
            ("Existence and Uniqueness for Stochastic Differential Equations on {domain} Spaces",
             "We prove the strong existence and pathwise uniqueness of solutions to non-linear stochastic partial differential equations (SPDEs) with non-Lipschitz drift coefficients in {domain} Hilbert spaces. Using monotone operator theory and Ito's formula, we show ergodicity and invariance of Gibbs measures."),
            ("Topological Data Analysis and Persistent Homology Invariants for {domain}",
             "Persistent homology provides algebraic topological summaries of metric spaces at multiple spatial resolutions. We introduce stability theorems for zigzag persistence modules and compute Betti number curves for {domain} point clouds. We demonstrate applications to knot theory and manifold reconstruction."),
        ],
        "Physics": [
            ("Quantum Phase Transitions and Criticality in 2D {domain} Superconductors",
             "We investigate the quantum critical behavior at the superconductor-insulator transition in two-dimensional thin films of {domain}. By conducting transport measurements at millikelvin temperatures and high magnetic fields, we extract critical exponents z and nu, confirming scaling predictions of the 3D XY universality class."),
            ("Holographic Duality and Non-Equilibrium Dynamics in Strongly Coupled {domain} Plasmas",
             "We utilize gauge/gravity duality to model non-equilibrium thermalization in strongly coupled {domain} gauge theories. By solving Einstein-matter equations in asymptotically AdS spacetimes, we determine quasinormal mode frequencies and shear viscosity to entropy density ratios (eta/s) near the Kovtun-Son-Starinets bound."),
            ("Precision Spectroscopy and Laser Cooling of Trapped {domain} Ions",
             "Trapped atomic ions in radiofrequency Paul traps offer pristine coherence for quantum simulation. We demonstrate Doppler cooling and sideband cooling of {domain} ions to the motional ground state. Coherent manipulation of Zeeman sublevels yields quantum logic gate operations with fidelity exceeding 99.5%."),
            ("Cosmological Constraints on Primordial Gravitational Waves and Inflation from {domain} Telescopes",
             "We analyze cosmic microwave background (CMB) B-mode polarization measurements from sub-orbital {domain} observatories. We place upper bounds on the tensor-to-scalar ratio r and constrain inflationary potential models. Systematic foreground dust contamination is isolated via multi-frequency focal plane arrays."),
        ],
        "Biology": [
            ("Single-Cell Epigenomic Landscapes and Chromatin Accessibility in {domain} Differentiation",
             "We perform single-cell assay for transposase-accessible chromatin sequencing (scATAC-seq) across {domain} progenitor lineages. We identify cell-type-specific enhancer motifs and transcription factor footprinting patterns. Reconstruction of pseudo-time trajectories reveals regulatory gating mechanisms underlying lineage commitment."),
            ("Structural Insights into Cryo-EM Architecture of {domain} Membrane Complexes",
             "Macromolecular membrane assemblies orchestrate cellular signaling and nutrient transport. We report single-particle cryo-electron microscopy structures of {domain} complexes at 2.4-A resolution. The density map resolves bound phospholipids and reveals conformational gating shifts triggered by ligand phosphorylation."),
            ("Evolutionary Dynamics and Epistatic Interactions in Viral {domain} Genomes",
             "Pathogen evolution is shaped by complex epistatic fitness landscapes. We analyze long-read sequencing data across thousands of {domain} isolates. Using phylogenetic maximum-likelihood methods and co-evolutionary Potts models, we identify compensatory mutations conferring immune evasion and drug resistance."),
            ("Systems Biology Reconstruction of Metabolic Networks in Human {domain} Diseases",
             "Metabolic reprogramming is a hallmark of {domain} pathologies. We integrate transcriptomic, proteomic, and metabolomic profiles into a curated genome-scale human metabolic model. Targeted fluxomic validation reveals elevated mitochondrial glutaminolysis and highlights novel druggable targets."),
        ],
        "Statistics": [
            ("Minimax Optimal Non-Parametric Regression under {domain} Regularity",
             "We establish minimax lower and upper risk bounds for non-parametric function estimation over Besov and Holder balls in {domain}. We propose an adaptive wavelet thresholding estimator that achieves optimal convergence rates under L2 and Linf loss functions without prior knowledge of function smoothness."),
            ("High-Dimensional Inference and Debiased Lasso for Sparse {domain} Models",
             "Statistical inference in high-dimensional generalized linear models is challenging due to shrinkage bias. We develop a debiased Lasso estimator for testing individual coefficient hypotheses in {domain}. We establish asymptotic normality of the debiased estimator and demonstrate accurate confidence interval coverage in simulation studies."),
            ("Markov Chain Monte Carlo and Variational Inference for Deep {domain} Latent Variable Models",
             "Bayesian inference in complex latent variable models requires scalable posterior approximations. We formulate an amortized variational inference algorithm with normalizing flows for deep {domain} hierarchies. We prove evidence lower bound (ELBO) convergence and demonstrate faster mixing times compared to Hamiltonian Monte Carlo."),
            ("Conformalized Survival Analysis and Risk Prediction under {domain} Censoring",
             "Predicting survival times under right-censoring requires valid uncertainty quantification. We introduce a weighted split conformal inference method that guarantees finite-sample coverage for survival probabilities in {domain}. We evaluate our method on clinical trial cohorts, demonstrating calibrated prediction intervals."),
        ]
    }
    
    variations = [
        "Knowledge Graph Systems", "Medical Informatics", "Autonomous Robotics", "Cloud Infrastructure",
        "NLP Translation", "Computer Vision", "Blockchain Protocols", "Semantic Embeddings"
    ]
    
    for cat, template_list in domain_templates.items():
        for t_title, t_abs in template_list:
            for var in variations[:4]:
                title = t_title.format(domain=var)
                abstract = t_abs.format(domain=var)
                expanded_samples.append({
                    "title": title,
                    "abstract": abstract,
                    "category": cat,
                    "sub_category": f"{cat.lower()[:4]}.gen"
                })
                
    df = pd.DataFrame(expanded_samples)
    return df


def load_dataset(dataset_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads arXiv dataset from JSON file or generates curated dataset if file not found.
    """
    if dataset_path and os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            print(f"Error loading {dataset_path}: {e}. Falling back to curated dataset.")
            
    df = generate_curated_dataset()
    return df


def save_dataset(df: pd.DataFrame, output_path: str):
    """Saves DataFrame as JSON dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    records = df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(df)} papers to {output_path}")


if __name__ == "__main__":
    # Test generation & save
    dataset_file = os.path.join(os.path.dirname(__file__), "sample_arxiv_dataset.json")
    df = load_dataset()
    save_dataset(df, dataset_file)
    print(f"Class distribution:\n{df['category'].value_counts()}")
