Model Evaluation Figures

This folder contains evaluation artifacts (PNGs and JSON) produced by `evaluate_full_system.py`.

Generated figures and suggested captions:

- `extraction_success_rate.png`:
  - Caption: "Text extraction success counts across test resumes (success vs failure). Shows how often the extractor returns usable text."

- `keyword_highlighting_confusion_matrix.png`:
  - Caption: "Counts for keyword highlighting: true positives, false negatives, false positives. Used to assess per-sentence detection behavior."

- `keyword_highlighting_metrics.png`:
  - Caption: "Precision, recall and F1 for keyword highlighting across curated test cases."

- `self_promotion_confusion_matrix.png`:
  - Caption: "Confusion matrix for self-promotion scoring (HIGH vs LOW) on curated test set."

- `self_promotion_metrics.png`:
  - Caption: "Summary metrics (accuracy, HIGH detection, LOW detection, separation) for the self-promotion scoring task."

- `self_promotion_roc.png`:
  - Caption: "ROC curve (AUC) for self-promotion scoring on the test set. Indicates classifier discrimination ability."

- `self_promotion_pr.png`:
  - Caption: "Precision-Recall curve (Average Precision) for self-promotion scoring. Important when classes are imbalanced."

- `self_promotion_score_hist.png`:
  - Caption: "Histogram of predicted self-promotion scores separated by true label (HIGH vs LOW). Visualizes score separation and thresholding."

- `knn_learning_curve.png`:
  - Caption: "KNN learning curve (training vs cross-validation accuracy) showing model behavior as training size increases."

- `knn_k_validation.png`:
  - Caption: "Validation curve across different `k` values for KNN. Used to select the hyperparameter `k`."

- `knn_test_confusion_matrix.png`:
  - Caption: "KNN confusion matrix on held-out test set (Promo vs Non-Promo)."

- `knn_nearest_neighbors.png`:
  - Caption: "Nearest-neighbor qualitative examples: query sentences from the test set with their top-3 nearest training neighbors (text). Useful for explaining KNN behavior."

- `bert_pca.png`:
  - Caption: "PCA 2D projection of BERT sentence embeddings (full dataset). Fast baseline for representation separability."

- `bert_umap.png` or `bert_tsne.png`:
  - Caption: "UMAP (preferred) or t-SNE 2D projection of a balanced subsample of BERT embeddings. Shows cluster structure and class separation."

Notes
- Captions are short suggestions. Expand them in the thesis figure captions with experimental details (dataset, subsample size, seed, metric values).
- If `umap-learn` is not installed, the script falls back to t-SNE for the subsample.
- For reproducibility, commit `full_system_metrics.json` and include `requirements.txt` when delivering the reproducible package.

If you want, I can also generate a `requirements.txt` and a short run command snippet to include in this README.
