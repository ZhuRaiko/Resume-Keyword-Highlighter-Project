"""
KNN Classifier - Core Component #2

This module implements K-Nearest Neighbors classification for self-promotion detection.
This is the second core academic component of the SkillHighlight system.

Configuration: k=5 neighbors
Purpose: Classify sentences as self-promotional based on BERT embeddings
Dataset: Trained on ≈10,000 labeled resume sentences (see `data/self_promotion_dataset.csv`)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from models.embedder import load_bert_model


class DummyKNN:
    """Fallback classifier when no training data is available."""
    def predict_proba(self, X):
        return np.zeros((len(X), 2))
    
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


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
