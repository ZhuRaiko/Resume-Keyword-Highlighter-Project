% embedder_latex.md — readable LaTeX layout with highlighted comments

\section*{embedder.py}

\subsection*{Overview}
\textbf{Purpose:} Load and expose a SentenceTransformer BERT model and helper
functions to encode sentences into dense vector representations.

\textbf{Role in system:} Provides reusable BERT embeddings consumed by the
KNN classifier and other components that need semantic sentence vectors.

\subsection*{Notes on LaTeX view}
The explanatory text above is plain black. The full source is embedded in a
`lstlisting` block below; include the `listings` and `xcolor` packages and set
`commentstyle=\color{green}` to render inline comments in green while keeping
the overview and headers in black for readability.

\begin{verbatim}
\usepackage{xcolor}
\usepackage{listings}
\lstset{commentstyle=\color{green}, basicstyle=\ttfamily\small, breaklines=true}
\end{verbatim}

\subsection*{Source — full module}
\begin{lstlisting}[language=Python, basicstyle=\ttfamily\small, breaklines=true, frame=single, commentstyle=\color{green}]
"""
BERT Embedder - Core Component #1

What this module does:
    - Provide utilities to load a SentenceTransformer BERT model and produce
      dense sentence embeddings used across the project.

Why this module is necessary in the overall system:
    - Downstream classifiers and similarity components expect fixed-width
      vector representations for sentences. This module centralizes model
      loading and encoding so the rest of the pipeline uses consistent
      embeddings.

How this module connects to other parts of a larger NLP / ML pipeline:
    - The `load_bert_model()` loader is used by the KNN classifier in
      `models/knn_classifier.py` and by any component that needs semantic
      sentence vectors (scoring, clustering, matching).
    - The embedding functions are thin wrappers around the SentenceTransformer
      model and return numpy arrays that feed directly into scikit-learn
      models and vector distance computations.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer


# What this function does:
#   - Load and return a pre-trained SentenceTransformer model instance.
# Why this function exists:
#   - Centralize model loading and caching so the same model instance is
#     reused across multiple calls and modules, avoiding repeated downloads
#     and heavy initialization costs.
# Inputs expected:
#   - No runtime inputs; uses the hard-coded model name "all-MiniLM-L6-v2".
# Returns / side effects:
#   - Returns a `SentenceTransformer` instance. Side effect: Streamlit caches
#     the loaded model via `@st.cache_resource` so subsequent loads are fast.
# How it contributes to the larger NLP / ML system:
#   - Supplies the canonical encoder used to produce vectors for downstream
#     classification and similarity tasks.
@st.cache_resource
def load_bert_model():
    """
    Load the pre-trained BERT model for sentence embeddings.
    
    Returns:
        SentenceTransformer: The loaded BERT model (all-MiniLM-L6-v2)

    Expanded notes:
        - Uses Streamlit's `cache_resource` to persist the loaded model in
          memory across runs of a Streamlit app.
        - This function performs no text processing; it strictly returns the
          embedder object which exposes `encode()`.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


# What this function does:
#   - Encode a list of sentence strings into a matrix of BERT embeddings.
# Why this function exists:
#   - Provide a convenient wrapper that accepts either an explicit model
#     instance or lazily loads the canonical model via `load_bert_model()`.
# Inputs expected:
#   - sentences (list): list of sentence strings to encode.
#   - model (SentenceTransformer, optional): If provided, this model is used
#     to encode; otherwise `load_bert_model()` is called.
# Returns / side effects:
#   - Returns a numpy array of shape (n_sentences, embedding_dim). No side
#     effects beyond potential model loading when `model` is None.
# How it contributes to the larger NLP / ML system:
#   - Produces batched inputs that are directly consumable by KNN training,
#     classification, and similarity scoring functions elsewhere in the repo.
def encode_sentences(sentences, model=None):
    """
    Encode a list of sentences into BERT embeddings.
    
    Args:
        sentences (list): List of sentence strings
        model (SentenceTransformer, optional): Pre-loaded model. If None, loads default.
    
    Returns:
        numpy.ndarray: Matrix of sentence embeddings (n_sentences x 384)
    
    Expanded notes:
        - If `model` is None, this function calls `load_bert_model()` which
          may return a cached model instance.
        - The returned object is the same type returned by
          `SentenceTransformer.encode()` and is suitable for scikit-learn.
    """
    if model is None:
        model = load_bert_model()
    return model.encode(sentences)


# What this function does:
#   - Encode a single sentence, returning a one-row embedding vector.
# Why this function exists:
#   - Convenience wrapper for callers that need to embed single sentences
#     without handling batching themselves.
# Inputs expected:
#   - sentence (str): Text for a single sentence.
#   - model (SentenceTransformer, optional): Pre-loaded model; if omitted,
#     the default model is loaded via `load_bert_model()`.
# Returns / side effects:
#   - Returns a 1D numpy array representing the sentence embedding.
# How it contributes to the larger NLP / ML system:
#   - Used by single-sentence prediction helpers (e.g., per-sentence
#     self-promotion scoring) where callers pass the returned vector to a
#     classifier or similarity routine.
def encode_single_sentence(sentence, model=None):
    """
    Encode a single sentence into BERT embedding.
    
    Args:
        sentence (str): Input sentence
        model (SentenceTransformer, optional): Pre-loaded model. If None, loads default.
    
    Returns:
        numpy.ndarray: Sentence embedding vector (384-dim)
    
    Expanded notes:
        - Internally encodes the sentence as a single-element list and
          returns the embedding for that element.
        - This function intentionally mirrors the interface of the project
          that expects a 1D embedding for single-sentence pipelines.
    """
    if model is None:
        model = load_bert_model()
    return model.encode([sentence])
\end{lstlisting}
