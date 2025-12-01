"""
BERT Embedder - Core Component #1

This module handles contextual sentence embeddings using SentenceTransformer.
This is one of the two core academic components of the SkillHighlight system.

Model: all-MiniLM-L6-v2 (384-dimensional embeddings)
Purpose: Transform sentences into dense vector representations for KNN classification
"""

import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_bert_model():
    """
    Load the pre-trained BERT model for sentence embeddings.
    
    Returns:
        SentenceTransformer: The loaded BERT model (all-MiniLM-L6-v2)
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


def encode_sentences(sentences, model=None):
    """
    Encode a list of sentences into BERT embeddings.
    
    Args:
        sentences (list): List of sentence strings
        model (SentenceTransformer, optional): Pre-loaded model. If None, loads default.
    
    Returns:
        numpy.ndarray: Matrix of sentence embeddings (n_sentences x 384)
    """
    if model is None:
        model = load_bert_model()
    return model.encode(sentences)


def encode_single_sentence(sentence, model=None):
    """
    Encode a single sentence into BERT embedding.
    
    Args:
        sentence (str): Input sentence
        model (SentenceTransformer, optional): Pre-loaded model. If None, loads default.
    
    Returns:
        numpy.ndarray: Sentence embedding vector (384-dim)
    """
    if model is None:
        model = load_bert_model()
    return model.encode([sentence])
