"""
Model Evaluation Script
Evaluates the KNN self-promotion classifier and generates performance metrics.

This script tests the core academic components:
- Component #1: BERT embeddings (all-MiniLM-L6-v2)
- Component #2: KNN classifier (k=5)

Provides comprehensive evaluation including:
- Train/test split validation (80/20)
- 5-fold cross-validation
- Confusion matrix analysis
- Misclassification examples for error analysis
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
import os

def evaluate_knn_model(show_examples=True, n_examples=5):
    """
    Evaluate KNN model with comprehensive metrics and error analysis.
    
    Args:
        show_examples (bool): Whether to display misclassification examples
        n_examples (int): Number of examples to show per error type
    
    Returns:
        dict: Complete evaluation metrics
    """
    
    print("Loading dataset...")
    dataset_path = "data/self_promotion_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return None
    
    # Read CSV with encoding fallback for special characters
    try:
        df = pd.read_csv(dataset_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(dataset_path, encoding='cp1252')  # Windows encoding
    
    # Drop any unnamed columns (from trailing commas)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    if df.empty or "sentence" not in df.columns or "label" not in df.columns:
        print("Error: Invalid dataset format. Need 'sentence' and 'label' columns.")
        return None
    
    # Clean data: remove rows with missing values
    initial_size = len(df)
    df = df.dropna(subset=['sentence', 'label'])
    df = df[df['sentence'].str.strip() != '']  # Remove empty sentences
    
    if len(df) < initial_size:
        print(f"⚠️  Removed {initial_size - len(df)} rows with missing/empty data")
    
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
    
    # Error analysis - show misclassification examples
    if show_examples:
        print("\n" + "="*60)
        print("🔍 ERROR ANALYSIS - MISCLASSIFICATION EXAMPLES")
        print("="*60)
        
        # Get test data with predictions
        test_indices = np.arange(len(X))[len(X_train):]  # Reconstruct test indices
        test_sentences = df.iloc[test_indices]["sentence"].values
        
        # False positives (predicted self-promotional but actually neutral)
        fp_mask = (y_pred == 1) & (y_test == 0)
        if fp_mask.sum() > 0:
            print(f"\n❌ FALSE POSITIVES ({fp_mask.sum()} total)")
            print("   Predicted: Self-Promotional | Actual: Neutral")
            print("-" * 60)
            for i, idx in enumerate(np.where(fp_mask)[0][:n_examples]):
                prob = y_pred_proba[idx][1]
                print(f"{i+1}. [{prob:.3f}] {test_sentences[idx]}")
        
        # False negatives (predicted neutral but actually self-promotional)
        fn_mask = (y_pred == 0) & (y_test == 1)
        if fn_mask.sum() > 0:
            print(f"\n❌ FALSE NEGATIVES ({fn_mask.sum()} total)")
            print("   Predicted: Neutral | Actual: Self-Promotional")
            print("-" * 60)
            for i, idx in enumerate(np.where(fn_mask)[0][:n_examples]):
                prob = y_pred_proba[idx][1]
                print(f"{i+1}. [{prob:.3f}] {test_sentences[idx]}")
        
        # Show some correct predictions for comparison
        print(f"\n✅ CORRECT PREDICTIONS (sample)")
        print("-" * 60)
        
        # True positives
        tp_mask = (y_pred == 1) & (y_test == 1)
        if tp_mask.sum() > 0:
            print(f"\nTrue Positives ({tp_mask.sum()} total) - examples:")
            for i, idx in enumerate(np.where(tp_mask)[0][:3]):
                prob = y_pred_proba[idx][1]
                print(f"  [{prob:.3f}] {test_sentences[idx]}")
        
        # True negatives
        tn_mask = (y_pred == 0) & (y_test == 0)
        if tn_mask.sum() > 0:
            print(f"\nTrue Negatives ({tn_mask.sum()} total) - examples:")
            for i, idx in enumerate(np.where(tn_mask)[0][:3]):
                prob = y_pred_proba[idx][1]
                print(f"  [{prob:.3f}] {test_sentences[idx]}")
    
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
        },
        "error_rates": {
            "false_positive_rate": float(f"{fp/(fp+tn) if (fp+tn) > 0 else 0:.3f}"),
            "false_negative_rate": float(f"{fn/(fn+tp) if (fn+tp) > 0 else 0:.3f}"),
            "total_errors": int(fp + fn),
            "error_rate": float(f"{(fp+fn)/len(y_test):.3f}")
        }
    }
    
    # Save detailed results
    output_path = "model_evaluation/model_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n✅ Metrics saved to 'model_evaluation/model_metrics.json'")
    print("="*60)
    
    return metrics


def compare_with_threshold(threshold=0.5):
    """
    Test different probability thresholds for classification.
    
    Args:
        threshold (float): Probability threshold for positive classification
    """
    print(f"\n🔬 THRESHOLD ANALYSIS (testing threshold = {threshold})")
    print("="*60)
    
    try:
        df = pd.read_csv("data/self_promotion_dataset.csv", encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv("data/self_promotion_dataset.csv", encoding='cp1252')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(subset=['sentence', 'label'])
    df = df[df['sentence'].str.strip() != '']
    
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    X = bert_model.encode(df["sentence"].tolist(), show_progress_bar=False)
    y = df["label"].astype(int).values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Get probabilities
    y_pred_proba = knn.predict_proba(X_test)[:, 1]
    
    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\nThreshold | Precision | Recall | F1-Score | Accuracy")
    print("-" * 60)
    
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        print(f"  {t:.1f}    |   {prec:.3f}   |  {rec:.3f} |  {f1:.3f}  |  {acc:.3f}")


if __name__ == "__main__":
    import sys
    
    try:
        # Run main evaluation
        print("="*60)
        print("KNN SELF-PROMOTION CLASSIFIER EVALUATION")
        print("Core Academic Components: BERT + KNN")
        print("="*60 + "\n")
        
        metrics = evaluate_knn_model(show_examples=True, n_examples=5)
        
        if metrics is None:
            sys.exit(1)
        
        # Optional: threshold analysis
        if len(sys.argv) > 1 and sys.argv[1] == "--threshold":
            compare_with_threshold()
        
        print("\n✨ Evaluation complete!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure 'data/self_promotion_dataset.csv' exists.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
