# CHAPTER 5 — Results and Evaluation (Machine Learning)

This chapter documents the experimental setup, evaluation procedures, results, and interpretation for SkillHighlight. It is structured to make it easy to copy into your thesis: each subsection lists what to include in text, which figures/tables to show, how to generate them from the repository, and short interpretation notes.

---

## 5.1 Experimental setup

**What to include in the thesis text**
- Short description of SkillHighlight: BERT-based sentence embeddings + KNN classifier + SpaCy keyword validation + Streamlit UI.
- Data used: training set size (e.g., 10,000 labeled sentences), held-out test set size, any preprocessing (sentence split, cleaning, encoding notes).
- Model and hyperparameters: embedding model `all-MiniLM-L6-v2` (384-d), KNN with `k=5` (or tuned value), heuristic bonus values used in scoring.
- Software & hardware: Python version, main libraries (SentenceTransformers, scikit-learn, spaCy, Streamlit, umap-learn), and hardware used (CPU/GPU). Mention caching of embeddings and `joblib` for model persistence.

**Files and code paths**
- Evaluation runner: `model_evaluation/evaluate_full_system.py`
- Dataset: `data/self_promotion_dataset.csv`
- KNN model (trained/pickled): `models/knn_model.pkl` (if saved)
- Figures output folder: `model_evaluation/` (PNG files saved here)

**Reproducible commands (PowerShell)**
- Install core dependencies (if not already installed):
```powershell
python -m pip install -r requirements.txt
# If you don't have a requirements file, install core packages:
python -m pip install sentence-transformers scikit-learn pandas matplotlib seaborn joblib spacy umap-learn plotly streamlit pdfminer.six python-docx
```
- Run the evaluation script (uninterrupted):
```powershell
cd "c:\Users\Evance Adriano\Desktop\School\Resume Keyword Highlighter Project"
python -u model_evaluation/evaluate_full_system.py 2>&1 | tee model_evaluation/evaluate_run.log
```
Notes: use `-u` to avoid buffering; redirect stdout/stderr so you can keep logs. If packages missing, install them first; UMAP is preferred for visualization and t-SNE is fallback.

---

## 5.2 Evaluation plan and metrics (what to show and how to interpret)

For each subsection below: include a short paragraph summarizing the result, one figure/table, and an interpretation sentence that ties the number/plot back to research questions.

### 5.2.1 KNN classifier performance
- What to show:
  - Table of main metrics: Accuracy, Precision, Recall, F1, (and support) for the positive (self-promotion) class and overall.
  - Confusion matrix figure.
  - ROC curve (AUC) and Precision-Recall curve (AP).
- Files to generate/check:
  - `model_evaluation/self_promotion_metrics.png` (metrics table/summary)
  - `model_evaluation/self_promotion_confusion_matrix.png`
  - `model_evaluation/self_promotion_roc.png`
  - `model_evaluation/self_promotion_pr.png`
- How to interpret:
  - Report values (e.g., Accuracy 86.2%, Precision 97.1%, Recall 92.3%, F1 94.6%) and briefly say what they mean in practice (few false positives, good recall of self-promotion sentences).
  - Discuss calibration: if precision >> recall or vice versa, explain trade-offs and whether you prefer precision (to avoid false credit) or recall (to surface potential claims).

### 5.2.2 Keyword highlighting quality
- What to show:
  - Confusion matrix or precision/recall bar for keyword detection vs. gold annotations.
  - Example before/after highlighted resume snippet (PNG or screenshot).
- Files to generate/check:
  - `model_evaluation/keyword_highlight_confusion.png`
  - `model_evaluation/keyword_highlight_examples.png`
- How to interpret:
  - Report precision and recall (e.g., 97.1% precision, 94.6% F1). Note that context-aware filtering reduces false positives caused by dates, non-relevant contexts, and common words.

### 5.2.3 Self-promotion scoring (hybrid score)
- What to show:
  - Histogram / kernel density of scores separated by gold label (high vs low quality).
  - Report mean score separation (e.g., 0.54–0.69) and classification accuracy when using a threshold.
- Files to generate/check:
  - `model_evaluation/score_histogram.png`
  - `model_evaluation/score_separation_table.csv` (per-sentence scores and labels)
- How to interpret:
  - Explain how the hybrid score is formed (KNN base + bonuses) and why score separation indicates the scoring model's usefulness for feedback.

### 5.2.4 Embedding visualization (PCA / UMAP / t-SNE)
- What to show:
  - 2D scatter plots colored by label (self-promotion vs. not) and optionally by other metadata (e.g., resume source, industry).
- Files to generate/check:
  - `model_evaluation/embeddings_pca.png`
  - `model_evaluation/embeddings_umap.png` (or `embeddings_tsne.png` fallback)
- How to interpret:
  - Discuss clustering patterns: whether positive examples cluster, whether there is clear separation, and any interesting sub-clusters (e.g., metrics-rich sentences).
  - Note caveats: 2D projections lose information — don't over-interpret minor overlaps.

### 5.2.5 Nearest-neighbor qualitative examples
- What to show:
  - Example queries with their top-5 neighbors, distance/score, and labels — laid out in a figure or table.
- Files to generate/check:
  - `model_evaluation/nearest_neighbors_examples.png`
- How to interpret:
  - Use to demonstrate concrete behavior (why a sentence was scored high/low), and to show failure modes (similar wording but different context).

### 5.2.6 Extraction & preprocessing diagnostics
- What to show:
  - Bar chart of successful vs. fallback extraction methods, and common extraction errors (encoding issues, failed PDF->DOCX conversion).
- Files to generate/check:
  - `model_evaluation/extraction_success.png`
  - Check logs: `model_evaluation/knn_plot_error.log` and `model_evaluation/evaluate_run.log`
- How to interpret:
  - Use this to justify robustness steps (encoding fallbacks) and to recommend OCR if many scanned resumes exist.

---

## 5.3 Results (recommended structure for thesis text)

Use the following template subsections to populate results:

### 5.3.1 Dataset description and preprocessing
- State the train/validation/test split (or cross-validation strategy), number of sentences, number of positive/negative labels, and any data cleaning steps.
- Include a small table: counts per split and class distribution.

### 5.3.2 KNN classifier: quantitative results
- Paste the metrics table and confusion matrix.
- Short paragraph interpreting metrics and their implications for ranking and automated screening.
- Mention cross-validation results (Mean F1 ± std). If available, show a validation learning curve to demonstrate training stability.

### 5.3.3 Keyword highlighting: quantitative & qualitative
- Show precision/recall numbers and a few before/after text examples.
- Explain how context rules reduced false positives (give 1–2 concrete examples).

### 5.3.4 Self-promotion scoring
- Show score histogram and a short table linking threshold → accuracy/precision/recall for the binary decision if you threshold scores.
- Explain which heuristics contributed most on average (report mean bonus contribution per feature if tracked).

### 5.3.5 Embedding visualizations and qualitative analysis
- Present PCA/UMAP plots and describe clustering.
- Explain nearest-neighbour examples and any identified failure modes.

### 5.3.6 Reproducibility checks and runtime
- Report runtime (encoding + embedding + KNN inference) for a representative resume and for whole dataset training on your hardware.
- Note the reproducible command used and the environment (Python packages + versions). Save `evaluate_run.log` and reference it.

---

## 5.4 Reproducibility: exact steps to regenerate figures and tables

**Checklist before running**
- Ensure Python packages are installed (see 5.1 commands).
- Confirm `data/self_promotion_dataset.csv` exists and is readable; if encoding errors occur, the evaluation runner has fallbacks (utf-8 → cp1252 → latin-1) and will log which encoding was used.
- Ensure `models/` is writable if training and saving the model.

**Run the evaluation**
- PowerShell command (again):
```powershell
cd "c:\Users\Evance Adriano\Desktop\School\Resume Keyword Highlighter Project"
python -u model_evaluation/evaluate_full_system.py 2>&1 | tee model_evaluation/evaluate_run.log
```
- After run completes, inspect `model_evaluation/` for PNGs and `model_evaluation/evaluate_run.log` and `model_evaluation/knn_plot_error.log` for any errors.

**If PNGs are missing**
- Open `model_evaluation/evaluate_run.log` to find early exceptions.
- If a UnicodeDecodeError appears, the runner ideally already handled it — otherwise open `data/self_promotion_dataset.csv` in a text editor and try saving with UTF-8 or CP1252.

---

## 5.5 Limitations and threats to validity

Include these points and expand as appropriate:
- Dataset bias: annotated sentences may reflect annotator biases or a limited domain (student resumes). Generalizability to other domains is not guaranteed.
- Annotation quality: if labels come from single annotators the gold labels can be noisy. Report inter-annotator agreement if available.
- Feature dependence: heuristic bonuses are human-designed and may not generalize; document their effect and sensitivity analysis if possible.
- 2D visualizations: projections may misrepresent distances; use them as qualitative evidence only.
- Small-sample phenomena: if some classes are underrepresented, metrics like accuracy can be misleading; prefer per-class precision/recall.

---

## 5.6 Recommendations and next steps (machine-learning focus)

 - Run a fairness/bias audit across demographic slices if data includes such metadata.
 - Compare exact KNN to indexed search or other scalable retrieval approaches for scale and speed improvements.
 - Evaluate fine-tuned transformer classifiers (train a small classifier head on top of the embeddings) and compare to KNN on accuracy and inference cost.
- Add OCR pre-processing for scanned resumes and extend evaluation to multilingual datasets.

---

## Appendix: figure filename → suggested caption and placement

- `self_promotion_confusion_matrix.png`: "Confusion matrix for KNN self-promotion classification (test set)." Place in 5.3.2.
- `self_promotion_metrics.png`: "Classification metrics (precision, recall, F1, accuracy) for self-promotion detection." Place in 5.3.2.
- `self_promotion_roc.png`: "ROC curve and AUC for KNN classifier." Place in 5.3.2.
- `self_promotion_pr.png`: "Precision-Recall curve and Average Precision for self-promotion detection." Place in 5.3.2.
- `keyword_highlight_confusion.png`: "Keyword highlighting confusion matrix vs. annotated gold." Place in 5.3.3.
- `keyword_highlight_examples.png`: "Before/after examples of context-aware keyword highlighting." Place in 5.3.3.
- `score_histogram.png`: "Distribution of hybrid self-promotion scores for positive and negative classes." Place in 5.3.4.
- `embeddings_pca.png`, `embeddings_umap.png`: "2D embedding visualizations (PCA, UMAP)." Place in 5.3.5.
- `nearest_neighbors_examples.png`: "Nearest-neighbour qualitative examples showing top-5 neighbors and labels." Place in 5.3.5.
- `extraction_success.png`: "Extraction method success/failure rates and fallbacks used." Place in 5.3.6.

**Suggested LaTeX figure snippet (copy into thesis LaTeX)**
```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.85\textwidth]{model_evaluation/self_promotion_confusion_matrix.png}
  \caption{Confusion matrix for the KNN self-promotion classifier (test set).}
  \label{fig:knn_confusion}
\end{figure}
```

---

## Quick checklist for writing each subsection (What to do per part)

- KNN results (5.3.2): paste numeric metrics, include confusion matrix PNG, write one paragraph interpreting error types and calibration.
- Keyword highlighting (5.3.3): paste precision/recall values, include before/after snippets, add 1–2 failure examples and how rules fix them.
- Scoring (5.3.4): include histogram PNG, compute mean/median per class, and report suggested threshold for binary usage.
- Embeddings (5.3.5): include PCA/UMAP PNGs, comment on cluster separation, include nearest-neighbor table for 3 representative queries.
- Extraction (5.3.6): include extraction success plot and explain any preprocessing decisions taken.

---

If you want, I can:
- Run `model_evaluation/evaluate_full_system.py` here to generate the expected PNGs (I will need permission to install packages if any are missing), then list the PNGs and produce LaTeX-ready figure blocks for each.
- Or generate a `CHAPTER_5.tex` fragment that you can paste into your thesis with the exact figure placements and captions filled in.

Tell me which next step you want and I'll proceed. 
