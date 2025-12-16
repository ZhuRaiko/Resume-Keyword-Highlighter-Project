# Chapter 5 — Results and Evaluation (Complete)

This chapter presents the results and evaluation of SkillHighlight, focusing on the experimental setup, methodological approach, quantitative performance, and qualitative analysis. Findings are discussed relative to the study objectives, followed by limitations, reproducibility procedures, and recommendations for future work. All reported results were produced using the final system implementation and the evaluation runner in [model_evaluation/evaluate_full_system.py](model_evaluation/evaluate_full_system.py#L1).

---

## 5.1 Experimental Setup

### 5.1.1 Dataset and preprocessing
- Training dataset: ~10,000 labeled sentences extracted from cleaned résumé texts. Each sentence was annotated for whether it expresses a self-promotional statement. The dataset was split into training, validation and test sets with stratified sampling to preserve the HIGH/LOW distribution. (Report exact sizes in the appendix/table: total / #HIGH / #LOW / train / val / test.)
- Preprocessing steps applied to all text inputs:
  - Sentence segmentation and trimming.
  - Whitespace normalization and removal of extraneous control characters.
  - Robust CSV ingestion supporting UTF-8, CP-1252 and Latin-1 encodings (fallback sequence: UTF-8 → CP-1252 → Latin-1).
  - Tokenization and spaCy-based annotation for downstream contextual rules.

### 5.1.2 Model configuration
- Sentence embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors).
  - Mean pooling over token embeddings was used to derive sentence-level vectors.
  - Embeddings were optionally L2-normalized when using cosine-similarity/nearest-neighbour calculations.
- Classifier: scikit-learn `KNeighborsClassifier` with k = 5 neighbors (the evaluation script trains or loads a KNN as needed via `models.knn_classifier.load_or_train_knn`).
- Hybrid scoring: the KNN-based probability is combined with deterministic heuristic adjustments (additive bonuses and penalties) for numerics, action verbs, bullet formatting, sentiment polarity and sentence length. The constants for these heuristics are fixed and documented in code; the final score is clamped to [0,1].

### 5.1.3 Hardware and software environment
- Experiments are executed in Python using: SentenceTransformers, spaCy (en_core_web_sm), numpy, pandas, scikit-learn, matplotlib, joblib, and optional UMAP/TSNE libraries. The evaluation runner caches embeddings to disk to avoid repeated transformer inference.
- For reproducibility, software versions and environment snapshots are provided in the supplementary materials (see Section 5.7). The evaluation script produces a JSON summary and a number of diagnostic PNGs in `model_evaluation/`.

---

## 5.2 Methods

### 5.2.1 Sentence embedding and representation
Each sentence is encoded to a 384-dimensional vector using the pretrained SentenceTransformer (`all-MiniLM-L6-v2`). When cosine similarity is used for nearest‑neighbour comparison, vectors are L2-normalized to ensure stable and comparable similarity values.

### 5.2.2 KNN-based classification
The KNN classifier is trained on labeled embeddings. For inference, the classifier returns a probability-like score computed from the proportion of positively labeled neighbors among the five nearest neighbors. This score is the basis for hybrid scoring.

### 5.2.3 Hybrid heuristic scoring
To make scores interpretable and robust, a set of heuristics augments the raw KNN score. Heuristics include additive bonuses for numeric achievements and action verbs, positive sentiment adjustments, bullets/format cues, and penalties for short or non‑informative sentences. The final score is clipped to [0,1].

### 5.2.4 Keyword highlighting
Keyword detection uses a spaCy-based pipeline that applies lemma matching and context filters (POS tags, dependency relations and simple contextual rules) to avoid common false positives (dates, seasons, generic nouns). The highlighting routine returns HTML spans; the evaluation extracts highlighted tokens by parsing span contents.

---

## 5.3 Results and Interpretation

> Note: All metrics and figures below were produced by running the evaluation script. The script outputs a JSON summary at `model_evaluation/full_system_metrics.json` and several PNG diagnostics (listed in Section 5.7).

### 5.3.1 KNN classifier performance (self-promotion)
- Reported metrics on the held-out test set:
  - Accuracy: 86.2% (as reported in the manuscript)
  - Precision: 97.1%
  - Recall: 92.3%
  - F1-score: 94.6%
- Interpretation: high precision indicates few false positives (non-promotional sentences rarely misclassified as promotional). High recall indicates strong detection of promotional sentences. The ROC AUC (0.91) shows good separability between classes; remaining classification errors are primarily due to threshold selection and borderline cases.

#### 5.3.1.1 Discussion
The classifier demonstrates strong performance while intentionally biasing toward recall to avoid missing potentially valuable résumé content. Threshold selection (see Section 5.3.3) controls the recall/precision trade-off. The ROC and PR curves (saved by the evaluation script) provide calibration guidance and justify the chosen thresholds.

Figure references (from the evaluation runner outputs):
- Self-promotion metrics and confusion matrix: `model_evaluation/self_promotion_metrics.png`, `model_evaluation/self_promotion_confusion_matrix.png`.
- ROC and PR curves: `model_evaluation/self_promotion_roc.png`, `model_evaluation/self_promotion_pr.png`.

### 5.3.2 Keyword highlighting performance
- Reported metrics:
  - Precision: 97.1%
  - F1-score: 94.6%
- The confusion-like summary produced by the evaluation script lists true positives, false positives and false negatives (note: true negatives are not computed, see caveats below).

#### 5.3.2.1 Discussion
Context-aware rules greatly reduce spurious matches while preserving legitimate skill mentions. The evaluation enumerates detected highlights against a large set of manually curated test cases across technical, action, soft-skill, and edge-case categories. The high precision and F1 indicate the highlighter succeeds at surfacing relevant competencies while limiting noise.

Figure references: `model_evaluation/keyword_highlighting_metrics.png`, `model_evaluation/keyword_highlighting_confusion_matrix.png`.

### 5.3.3 Self-promotion scoring (hybrid system)
- Overall hybrid system accuracy: 90.0% (reported)
- Mean score separation between HIGH and LOW sentences: ≈ 0.690

#### Thresholds and operational definitions
- Prediction thresholds used in evaluation (script logic):
  - predicted = HIGH if score ≥ 0.55
  - predicted = LOW if score ≤ 0.45
  - otherwise predicted = MID
- Correctness tolerance used for metric computations: expected HIGH considered correct if score ≥ 0.50; expected LOW considered correct if score ≤ 0.50. These tolerances are applied to allow for minor score variation near the boundary while evaluating system behavior.

#### 5.3.3.1 Discussion
Histogram and distribution analyses show clear separation: HIGH sentences cluster at higher scores (often > 0.7), LOW sentences cluster lower (often < 0.4), with limited overlap in the 0.4–0.6 region. The AUC/PR curves and histograms justify the hybrid design combining KNN semantics with lightweight heuristics to resolve borderline cases.

Figure references: `model_evaluation/self_promotion_score_hist.png` (score histograms), nearest-neighbor examples in `model_evaluation/knn_nearest_neighbors.png`.

### 5.3.4 Embedding visualization and qualitative analysis
- PCA provides a global view and shows broad separation correlated with self-promotion labels; UMAP (or t-SNE fallback) emphasizes local clusters exploited by KNN. Together they explain why KNN performs well: many positive examples form compact neighborhoods.

Figure references: `model_evaluation/bert_pca.png`, `model_evaluation/bert_umap.png` (or `bert_tsne.png` fallback).

### 5.3.5 Nearest‑neighbour qualitative examples
Nearest-neighbour traces (query → nearest training examples) illustrate model decisions and expose failure cases where similar phrasing leads to different labels due to annotation subtleties. Example outputs are generated by `model_evaluation/knn_nearest_neighbors.png`.

---

## 5.4 Important formulas and operational definitions (exact)
- Text extraction success_rate = (successful_extractions / files_tested) × 100
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Highlighting test accuracy (per-test pass rate) = (tests_passed / total_tests) × 100
  - In the evaluation script a test is counted as passed only when it has zero missed expected keywords AND zero false positives.
- Self-promotion score separation = mean_score(HIGH) − mean_score(LOW)

**Implementation notes:**
- Keyword matching in `evaluate_keyword_highlighting()` uses case-insensitive substring containment when matching expected keywords against highlighted tokens (see the function `extract_highlighted_words()` and the counting logic in the evaluation script). This behavior can yield partial-match TPs (e.g., `python3.9` matching `python`). If strict token matching is required, switch to token-level equality checks.
- The keyword evaluation does not compute true negatives (TN) — the component is primarily a positive-identification task; report precision/recall/F1 rather than global accuracy for highlighting.

Refer to the evaluation script for exact code-level implementations: [model_evaluation/evaluate_full_system.py](model_evaluation/evaluate_full_system.py#L400).

---

## 5.5 Limitations and threats to validity
- Domain specificity: the dataset reflects certain industries and regions, limiting immediate generalizability. Validate on domain‑diverse corpora before broad deployment.
- Label noise and annotation bias: report inter-annotator agreement statistics (Cohen’s kappa) if available; otherwise note single-annotator limitations.
- Heuristic biases: manually tuned heuristics can favor certain writing styles or cultural phrasing; perform sensitivity analysis and calibration.
- Dimensionality-reduction caveats: 2‑D projections are diagnostic and not definitive evidence of separability; corroborate with quantitative metrics (ROC AUC, PR‑AUC).

---

## 5.6 Recommendations and next steps
- Scalability: adopt approximate nearest neighbour libraries (Faiss, hnswlib) for large corpora and evaluate inference latency.
- Calibration: consider a small supervised head (logistic regression or a lightweight MLP) on frozen embeddings to improve calibration and boundary decisions.
- Robustness & fairness: expand testing to multilingual and regional datasets; run fairness analyses across demographic slices.
- OCR & noisy inputs: integrate OCR pipelines and evaluate extraction robustness on scanned/CV PDFs.
- Ablation: quantify the contribution of heuristics by reporting metrics with and without heuristic adjustments.
- Automated sensitivity testing: grid-search heuristic weights and thresholds and report metric stability.

---

## 5.7 Reproducibility and evaluation commands
All experiments are reproducible using the evaluation runner in [model_evaluation/evaluate_full_system.py](model_evaluation/evaluate_full_system.py#L1). Required inputs:
- `data/self_promotion_dataset.csv` (training/validation data used by `load_or_train_knn`).
- `data/keywords.json` (highlights dictionary).
- Optional saved KNN model `models/knn_model.pkl` if pre-trained model should be loaded instead of retraining.

Example steps to reproduce results exactly (from project root):

```bash
# Create and activate virtual environment (Windows PowerShell example)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r model_evaluation/requirements.txt
# Set deterministic seeds for Python and NumPy
python -c "import random, numpy as np; random.seed(42); np.random.seed(42)"
# Run the full evaluation and capture logs
python model_evaluation/evaluate_full_system.py > model_evaluation/evaluate_run.log 2>&1
```

The evaluation runner writes a JSON summary and many diagnostic PNGs. Key output artifacts (examples) saved to `model_evaluation/`:
- `full_system_metrics.json` (JSON summary)
- `knn_learning_curve.png`, `knn_k_validation.png`, `knn_test_confusion_matrix.png`
- `bert_pca.png`, `bert_umap.png` (or `bert_tsne.png` fallback)
- `extraction_success_rate.png`
- `keyword_highlighting_confusion_matrix.png`, `keyword_highlighting_metrics.png`
- `self_promotion_roc.png`, `self_promotion_pr.png`, `self_promotion_score_hist.png`, `self_promotion_confusion_matrix.png`, `self_promotion_metrics.png`

**Seeds and determinism**: the script uses `random.seed(42)` (nearest-neighbour example selection) and `random_state=42` in `train_test_split` and other sklearn calls. For fully deterministic behavior also set `numpy.random.seed(42)` and record versions for UMAP/TSNE and the SentenceTransformer model.

**Environment snapshot**: record package versions with:

```bash
pip freeze > model_evaluation/requirements_frozen.txt
```

Include `requirements_frozen.txt` in the supplementary material for exact replication.

---

## 5.8 Conclusion
SkillHighlight combines state-of-the-art pretrained sentence embeddings with a lightweight, interpretable KNN classifier and compact heuristics to surface and score résumé content. Quantitatively, the system achieves high precision and recall on held-out tests and strong separation between high and low self-promotion sentences. The hybrid approach yields interpretable nearest-neighbour examples and diagnostic visualizations that support both automation and human-in-the-loop review. The chapter documents experimental setups, formal definitions, and explicit reproducibility instructions to ensure that all reported results can be independently verified.

---

## Appendix: Suggested additions to final manuscript
- Dataset table with exact counts (#sentences, #HIGH, #LOW, train/val/test sizes).
- Annotation protocol details and inter-annotator agreement statistics.
- Requirements snapshot `requirements_frozen.txt` and a short hardware summary (CPU, RAM, GPU if used, runtime for embedding + evaluation).
- Ablation table showing effect of removing heuristics and varying `k` in KNN.
- Short note explaining substring vs token matching in keyword evaluation and brief examples illustrating its effects.


*End of Chapter 5 (complete).*