"""BERT embeddings and model loading"""
import streamlit as st
from sentence_transformers import SentenceTransformer
import spacy


@st.cache_resource
def load_models():
    """Load spaCy and BERT models (cached)"""
    nlp = spacy.load("en_core_web_sm")
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, bert_model


def encode_sentence(bert_model, sentence: str):
    """Encode a single sentence using BERT"""
    try:
        return bert_model.encode([sentence])
    except Exception as e:
        raise Exception(f"BERT encoding failed: {e}")
