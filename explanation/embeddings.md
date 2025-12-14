################################################################################
# Module: embeddings.py
#
# What this module does:
#   - Loads and exposes the NLP and sentence-embedding models used by the
#     system (spaCy for tokenization and a SentenceTransformer BERT model for
#     embedding sentences).
#   - Provides a tiny utility to encode a single sentence into an embedding
#     vector using the loaded BERT model.
#
# Why this module is necessary in the overall system:
#   - Embeddings are required for similarity search, KNN classification,
#     and semantic comparisons used by scoring and matching components. This
#     module centralizes model loading and a minimal encoding helper so other
#     modules do not duplicate model initialization logic.
#
# How this module connects to other parts of the NLP / ML pipeline:
#   - `load_models()` is called early (and cached) to provide a shared `nlp`
#     pipeline and `bert_model` instance used by `scoring.py`, `highlight.py`,
#     and any embedding-based components.
#   - `encode_sentence()` is called by higher-level code that needs a single
#     sentence vector for scoring, KNN lookup, or similarity computation.
#
################################################################################

"""BERT embeddings and model loading

This module keeps the original behavior of the project's `embeddings.py` but
adds inline, function-level comments and a module-level description. The code
behavior and public API are unchanged: `load_models()` returns `(nlp, bert_model)`
and `encode_sentence()` returns the BERT encoding for a sentence.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
import spacy


# Function: load_models
# What this function does:
#   - Loads the spaCy `en_core_web_sm` pipeline and the SentenceTransformer
#     model `all-MiniLM-L6-v2`, and returns both objects.
# Why this function exists:
#   - Model loading is expensive; this function centralizes loading and is
#     decorated with Streamlit's caching so callers get a shared, cached
#     instance rather than reloading repeatedly.
# Inputs expected:
#   - No arguments. Uses hard-coded model identifiers consistent with the
#     rest of the project.
# Returns / side effects:
#   - Returns a tuple `(nlp, bert_model)` where `nlp` is a spaCy language
#     object and `bert_model` is a SentenceTransformer instance.
# How it contributes to the larger NLP / ML system:
#   - Provides the canonical `nlp` and `bert_model` used for tokenization,
#     sentence splitting, and embedding generation across modules.
@st.cache_resource
def load_models():
    """Load spaCy and BERT models (cached)"""
    # Load spaCy small English model for tokenization and sentence segmentation
    nlp = spacy.load("en_core_web_sm")
    # Load a lightweight sentence transformer used for embedding sentences
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    # Return both objects so callers can use spaCy and embeddings from one place
    return nlp, bert_model


# Function: encode_sentence
# What this function does:
#   - Encodes a single sentence into an embedding vector using the provided
#     `bert_model` (SentenceTransformer).
# Why this function exists:
#   - A small helper to keep one-off sentence encodes consistent across the
#     codebase and to wrap the call with simple error handling.
# Inputs expected:
#   - bert_model: a SentenceTransformer instance (expects `.encode([...])`).
#   - sentence (str): the sentence text to encode into a vector.
# Returns / side effects:
#   - Returns the result of `bert_model.encode([sentence])` (embedding array).
#   - Raises an exception with a clear message if encoding fails.
# How it contributes to the larger NLP / ML system:
#   - Used by scoring and embedding-based lookups where callers only need a
#     single-sentence vector rather than batch encoding.
def encode_sentence(bert_model, sentence: str):
    """Encode a single sentence using BERT"""
    try:
        # Call SentenceTransformer.encode with a single-item list to get a vector
        return bert_model.encode([sentence])
    except Exception as e:
        # Re-raise with a clearer message while preserving original error info
        raise Exception(f"BERT encoding failed: {e}")
