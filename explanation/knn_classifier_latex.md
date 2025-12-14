% knn_classifier_latex.md — readable LaTeX layout with highlighted comments

\section*{knn_classifier.py}

\subsection*{Overview}
\textbf{Purpose:} K-Nearest Neighbors classifier for detecting self-promotional sentences using BERT embeddings.

\textbf{Role in system:} Provides per-sentence self-promotion probabilities used by scoring and analysis components; includes a fallback `DummyKNN` when no model is available.

\subsection*{Notes on this LaTeX view}
The explanatory text above is plain black. The full source is embedded in a `lstlisting` block below; include the `listings` and `xcolor` packages and set `commentstyle=\color{green}` to render inline comments in green and retain the LaTeX headers in black.

\begin{verbatim}
\usepackage{xcolor}
\usepackage{listings}
\lstset{commentstyle=\color{green}, basicstyle=\ttfamily\small, breaklines=true}
\end{verbatim}

\subsection*{Source — full module}
\begin{lstlisting}[language=Python, basicstyle=\ttfamily\small, breaklines=true, frame=single, commentstyle=\color{green}]
################################################################################
# Module: knn_classifier.py
#
# What this module does:
#   - Implements K-Nearest Neighbors classification for detecting
#     self-promotional sentences using BERT embeddings.
#   - Provides a fallback `DummyKNN` when no model or dataset is available.
#   - Loads or trains a KNN model from a labeled CSV and offers a single
#     sentence prediction helper that returns a probability score.
#
# Why this module is necessary in the overall system:
#   - Self-promotion detection is a core analytic used to weight candidate
#     statements and to produce features for scoring and reporting.
#   - KNN on BERT embeddings provides a simple, interpretable classifier
#     that is fast to train and inspect for the project's dataset size.
#
# How this module connects to other parts of the NLP / ML pipeline:
#   - Relies on the project's BERT embedding loader (`load_bert_model`) to
#     convert sentences into vectors used by the KNN classifier.
#   - The `predict_self_promotion_score()` helper is called by scoring and
#     analysis modules that need per-sentence self-promotion probabilities.
#
"""
KNN Classifier - Core Component #2

This module implements K-Nearest Neighbors classification for self-promotion detection.
This is the second core academic component of the SkillHighlight system.

Configuration: k=5 neighbors
Purpose: Classify sentences as self-promotional based on BERT embeddings
Dataset: Trained on 6,752 labeled resume sentences
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from models.embedder import load_bert_model


# Class: DummyKNN
# What this class does:
#   - Provides a minimal fallback classifier object with the same interface
#     shape as a real classifier (supports `predict_proba` and `predict`).
# Why this class exists:
#   - When no training dataset or model is available, return a safe object so
#     callers can continue without additional error handling.
# Inputs expected:
#   - No constructor arguments; stateless.
# Returns / side effects:
#   - Instances expose `predict_proba(X)` and `predict(X)` methods that return
#     zeroed outputs matching expected shapes.
# How it contributes to the larger NLP / ML system:
#   - Avoids throwing errors in production when the trained model is missing.
class DummyKNN:
    """Fallback classifier when no training data is available."""
    def predict_proba(self, X):
        # Return zeroed probability matrix with two columns (negative/positive)
        return np.zeros((len(X), 2))
    
    def predict(self, X):
        # Return all-zero class predictions (non-promotional)
        return np.zeros(len(X), dtype=int)


# Function: load_or_train_knn
# What this function does:
#   - Attempts to load a cached KNN model from `model_path` or trains a new
#     KNN using sentences and labels from `csv_path` if loading fails.
# Why this function exists:
#   - Centralizes model loading and training logic so callers obtain a usable
#     classifier object without duplicating persistence/training code.
# Inputs expected:
#   - model_path (str): Path to load/save the serialized model file.
#   - csv_path (str): Path to the labeled training dataset CSV containing
#     `sentence` and `label` columns.
# Returns / side effects:
#   - Returns a trained `KNeighborsClassifier` instance or a `DummyKNN` on
#     failure. Trains and caches the model file when training succeeds.
# How it contributes to the larger NLP / ML system:
#   - Supplies the KNN model used by `predict_self_promotion_score()` and other
#     analysis code to compute self-promotion probabilities.
def load_or_train_knn(model_path="knn_model.pkl", csv_path="self_promotion_dataset.csv"):
    """
    Load existing KNN model or train new one from dataset.
    
    This function implements the core academic methodology:
    1. Load labeled dataset (10,000 sentences)
    2. Generate BERT embeddings for all sentences
    3. Train KNN classifier (k=5)
    4. Cache model for future use
    
    Args:
        model_path (str): Path to save/load the trained model
        csv_path (str): Path to the training dataset CSV
    
    Returns:
        KNeighborsClassifier or DummyKNN: The trained classifier
    """
    # Try loading cached model
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass  # Continue to training if loading fails
    
    # Check if dataset exists
    if not os.path.exists(csv_path):
        return DummyKNN()
    
    # Load and validate dataset
    df = pd.read_csv(csv_path)
    if df.empty or "sentence" not in df.columns or "label" not in df.columns:
        return DummyKNN()
    
    try:
        # Generate BERT embeddings (Core Component #1)
        bert_model = load_bert_model()
        X = bert_model.encode(df["sentence"].tolist())
        y = df["label"].astype(int).values
        
        # Train KNN classifier (Core Component #2)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, y)
        
        # Cache trained model
        joblib.dump(knn, model_path)
        return knn
    except Exception:
        return DummyKNN()


# Function: predict_self_promotion_score
# What this function does:
#   - Encodes a single sentence with BERT and then applies the KNN model to
#     return a self-promotion probability score in [0.0, 1.0].
# Why this function exists:
#   - Provides a compact, safe API for callers that need a numeric probability
#     for a single sentence rather than managing batching or model details.
# Inputs expected:
#   - sentence (str): Sentence text to classify.
#   - knn_model: Trained KNN classifier (or `DummyKNN`).
#   - bert_model: Loaded BERT model instance with `.encode()`.
# Returns / side effects:
#   - Returns a float probability for the positive (self-promotional) class.
#   - On errors or missing models, returns 0.0.
# How it contributes to the larger NLP / ML system:
#   - Used by scoring and analysis modules to assign probabilistic self-promotion
#     features to individual sentences.
def predict_self_promotion_score(sentence, knn_model, bert_model):
    """
    Predict self-promotion probability for a single sentence.
    
    This is the core academic prediction pipeline:
    1. Encode sentence with BERT (Component #1)
    2. Classify with KNN (Component #2)
    3. Return probability score
    
    Args:
        sentence (str): Input sentence to classify
        knn_model: Trained KNN classifier
        bert_model: Loaded BERT model
    
    Returns:
        float: Self-promotion probability [0.0, 1.0]
    """
    try:
        # Step 1: BERT encoding
        vec = bert_model.encode([sentence])
        
        # Step 2: KNN prediction
        if knn_model is None or isinstance(knn_model, DummyKNN):
            return 0.0
        
        if hasattr(knn_model, "predict_proba"):
            probs = knn_model.predict_proba(vec)
            # Return probability of positive class (self-promotional)
            if probs.shape[1] > 1:
                return float(probs[0][1])
            return float(probs[0][0])
        
        # Fallback to binary prediction
        return float(knn_model.predict(vec)[0])
    except Exception:
        return 0.0
\end{lstlisting}
