"""
Model Evaluation Script
Evaluates the KNN self-promotion classifier and generates performance metrics.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
import json

def evaluate_knn_model():
    """Evaluate KNN model with train/test split and cross-validation."""
    
    print("Loading dataset...")
    df = pd.read_csv("data/self_promotion_dataset.csv")
    
    if df.empty or "sentence" not in df.columns or "label" not in df.columns:
        print("Error: Invalid dataset format. Need 'sentence' and 'label' columns.")
        return
    
    print(f"Dataset size: {len(df)} samples")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Load BERT model
    print("\nLoading SentenceTransformer model...")
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Generate embeddings
    print("Generating embeddings...")
    X = bert_model.encode(df["sentence"].tolist(), show_progress_bar=True)
    y = df["label"].astype(int).values
    
    # Train-test split (80/20)
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train KNN
    print("\nTraining KNN classifier (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Predictions
    print("Generating predictions...")
    y_pred = knn.predict(X_test)
    y_pred_proba = knn.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    
    # Cross-validation (5-fold)
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(knn, X, y, cv=5, scoring='f1')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Print results
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    print("\n📊 DATASET STATISTICS")
    print(f"Total samples: {len(df)}")
    print(f"Positive samples (self-promotional): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
    print(f"Negative samples (neutral): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
    
    print("\n📈 TEST SET PERFORMANCE")
    print(f"Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"Recall:    {recall:.3f} ({recall*100:.1f}%)")
    print(f"F1-Score:  {f1:.3f} ({f1*100:.1f}%)")
    
    print("\n🔄 CROSS-VALIDATION (5-fold)")
    print(f"Mean F1-Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print(f"CV Scores: {[f'{s:.3f}' for s in cv_scores]}")
    
    print("\n📋 CONFUSION MATRIX")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    
    print("\n📝 DETAILED CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, 
                                target_names=['Neutral', 'Self-Promotional'],
                                digits=3))
    
    # Save metrics to JSON
    metrics = {
        "dataset": {
            "total_samples": int(len(df)),
            "positive_samples": int((y == 1).sum()),
            "negative_samples": int((y == 0).sum()),
            "class_balance": f"{(y == 1).sum()/len(y)*100:.1f}% / {(y == 0).sum()/len(y)*100:.1f}%"
        },
        "test_performance": {
            "accuracy": float(f"{accuracy:.3f}"),
            "precision": float(f"{precision:.3f}"),
            "recall": float(f"{recall:.3f}"),
            "f1_score": float(f"{f1:.3f}")
        },
        "cross_validation": {
            "mean_f1": float(f"{cv_scores.mean():.3f}"),
            "std_f1": float(f"{cv_scores.std():.3f}"),
            "fold_scores": [float(f"{s:.3f}") for s in cv_scores]
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "model_config": {
            "algorithm": "KNeighborsClassifier",
            "k_neighbors": 5,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "train_test_split": "80/20",
            "random_state": 42
        }
    }
    
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n✅ Metrics saved to 'model_metrics.json'")
    print("="*60)
    
    return metrics

if __name__ == "__main__":
    try:
        metrics = evaluate_knn_model()
    except FileNotFoundError:
        print("Error: self_promotion_dataset.csv not found in current directory.")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
