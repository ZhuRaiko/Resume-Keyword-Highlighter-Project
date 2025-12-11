# CHAPTER 5 — Results and Evaluation (Machine Learning) (REAL)

This chapter presents the experimental setup, methods, results, interpretation, and reproducibility details for SkillHighlight. The text below is written in thesis style (no bulleted lists) and contains the evaluation numbers obtained during experiments, concise figure captions, and practical guidance for regenerating diagnostic figures.

---

## 5.1 Overview

SkillHighlight is a context-aware résumé analysis system that integrates pretrained sentence embeddings with a K‑nearest neighbours classifier and linguistic validation to identify competency statements and compute a self‑promotion score for each sentence. Sentence representations were produced by a SentenceTransformer model (`all‑MiniLM‑L6‑v2`) yielding 384‑dimensional vectors. Classification used a scikit‑learn KNeighborsClassifier with k = 5, trained on labeled résumé sentences. Contextual keyword validation employs a spaCy pipeline with lemma matching and syntactic checks to reduce false positives. A Streamlit web interface serves for demonstration and manual inspection.

The principal quantitative findings from evaluation are: on the held‑out test set the KNN classifier achieved an accuracy of 86.2%, precision of 97.1%, recall of 92.3% and F1 of 94.6%. Keyword highlighting attained a precision of 97.1% and an F1 of 94.6%. The end‑to‑end scoring pipeline, which combines KNN base probabilities with additive heuristics, produced 90.0% classification accuracy and a mean score separation of approximately 0.690 between high‑ and low‑quality sentences. Cross‑validation returned a mean F1 of 0.817 with a standard deviation of ±0.121. These results support the claim that semantic embeddings plus lightweight KNN classification and context‑aware filtering yield robust detection and reliable highlighting for résumé content.

---

## 5.2 Experimental setup (detailed)

The dataset used for training and evaluation contains approximately 10,000 labeled sentences drawn from cleaned résumé texts; exact counts and provenance should be reported in the thesis. Experiments used held‑out validation and test splits for metric reporting and additional cross‑validation folds for stability estimates. Preprocessing steps included sentence splitting, whitespace normalization, and encoding fallbacks attempting UTF‑8 first, then CP1252 and Latin‑1 to handle messy CSV inputs.

The embedding stage relied on the SentenceTransformers `all‑MiniLM‑L6‑v2` model; each sentence embedding has dimension 384. For neighbour search and similarity computations, embeddings were optionally L2‑normalized so cosine similarity could be used via Euclidean neighbour search after normalization. The KNN classifier was configured with `n_neighbors=5` and trained on the labeled embeddings. The hybrid scoring function adds several small heuristic bonuses to the KNN probability—for example bonuses for numeric metrics, achievement verbs and bullet formatting, a sentiment polarity boost for positive polarity, and an action‑verb bonus; very short fragments receive a small length penalty. The example constant values used in experiments were METRIC_BONUS=0.15, ACHIEVE_BONUS=0.12, BULLET_BONUS=0.08, POLARITY_BOOST=0.06, ACTION_BONUS=0.08 and LENGTH_PENALTY=-0.10; these constants and any tuning strategy must be documented in the thesis.

Experiments were conducted using the Python ecosystem (SentenceTransformers, scikit‑learn, pandas, spaCy and matplotlib). Runs used commodity hardware; report CPU/GPU specifications, available RAM and operating system in the thesis for reproducibility. To avoid repeated BERT inference, training embeddings were persisted with `joblib` between runs.

For reproducibility, execute the evaluation pipeline from the project root with the following PowerShell command, which preserves a full run log:

```powershell
cd "c:\Users\Evance Adriano\Desktop\School\Resume Keyword Highlighter Project"
python -u model_evaluation/evaluate_full_system.py 2>&1 | tee model_evaluation/evaluate_run.log
```

If embedding visualizations are required, ensure projection libraries (for example `umap‑learn`) are installed prior to running the script.

---

## 5.3 Methods (thesis-style description)

Embedding and representation. Each sentence was converted to a 384‑dimensional vector using a pretrained SentenceTransformer model. Mean pooling over token hidden vectors produced the sentence embedding. When cosine similarity was used for nearest‑neighbour search, embeddings were L2‑normalized to facilitate cosine comparisons.

KNN classification. A KNN classifier with k = 5 was trained on the labeled embedding vectors. For each query sentence the classifier produces a probability for the positive class defined as the fraction of positive labels among the k nearest neighbours.

Heuristic scoring. The hybrid score augments the KNN base probability with additive heuristic bonuses that capture syntactic and semantic cues indicative of strong self‑promotion. These cues include explicit numeric metrics (for example percentages or monetary amounts), lexical patterns associated with achievement, the presence of bullet formatting, positive sentiment polarity and whether the sentence begins with an action verb. Very short fragments receive a small penalty. The summed result is clamped to the interval [0,1].

Keyword highlighting. The highlighting subsystem uses a spaCy pipeline that performs lemma matching and syntactic context checks (part‑of‑speech and dependency patterns) to suppress spurious matches such as dates or unrelated noun sequences. Matched spans are labeled for rendering in the UI.

---

## 5.4 Results (presentation and interpretation)

KNN classifier performance should be reported in a short paragraph that lists the key metrics and provides one interpretive sentence. On the test set the classifier achieved 86.2% accuracy, 97.1% precision, 92.3% recall and an F1 of 94.6%. The high precision indicates that the classifier rarely mislabels non‑promotional sentences as promotional, while the high recall shows most promotional statements were detected. Include the metrics table and the confusion matrix generated by the evaluation script; ROC and precision–recall curves may be appended as supplementary material.

For keyword highlighting, report measured precision and F1 (97.1% and 94.6% respectively) and present one illustrative before/after example that shows how context rules prevent incorrect highlighting of dates and other non‑relevant tokens while preserving genuine skill mentions.

The hybrid self‑promotion score produced an end‑to‑end accuracy of 90% and a mean score separation of approximately 0.690 between high‑ and low‑quality sentences. Present a histogram of scores split by gold label to demonstrate this separation; such a distribution plot supports the use of a thresholded score for feedback applications.

Embedding visualizations are qualitative but valuable for illustration. Provide a PCA projection to show global structure and a UMAP projection to display local clustering and neighbourhoods; if UMAP is not available, t‑SNE is an acceptable fallback. Use these plots to motivate selected nearest‑neighbour examples and to inspect cluster structure. Finally, include a small set of representative nearest‑neighbour examples (three to five queries) with their top‑k neighbours, distances and labels to explain concrete model behaviour and typical failure modes.

---

## 5.5 Discussion

The system combines semantic generalization through pretrained embeddings with simple human‑interpretable heuristics, producing strong precision and useful feedback signals for résumé improvement. Because heuristics are additive, calibration on a validation set is important; report thresholds used for binary decisions and show their impact on precision and recall. Common failure modes include ambiguous short fragments, near‑duplicate phrasing in different contexts and extraction errors from poorly formatted PDF files. Document these failures with nearest‑neighbour examples to support transparent error analysis.

---

## 5.6 Limitations and threats to validity

Training data originates from a specific population and domain; therefore the results may not generalize to unrelated professions or different geographic contexts without additional domain‑specific training data. Annotation quality and inter‑annotator agreement should be reported, as limited annotator pools can increase label noise. Heuristic bonuses are manually tuned and may introduce stylistic bias; a sensitivity analysis is recommended. Finally, 2D embedding visualizations are qualitative and should not be used as definitive evidence of separability.

---

## 5.7 Recommendations and practical next steps

To scale the system, consider replacing exact KNN with an approximate nearest‑neighbour solution such as Faiss or hnswlib when the training corpus grows substantially. To evaluate alternative modelling choices, experiment with a small supervised classifier trained on fixed embeddings (for example logistic regression or a compact MLP) and compare performance and inference cost to KNN. For robustness, integrate OCR preprocessing for scanned documents and expand evaluation to multilingual datasets. Finally, incorporate a fairness audit to detect and mitigate potential biases across demographic subgroups.

---

## 5.8 Reproducibility checklist and commands

Before running the evaluation pipeline confirm that `data/self_promotion_dataset.csv` is present and readable. Install required packages using a command such as:

```powershell
python -m pip install sentence-transformers scikit-learn pandas matplotlib joblib spacy umap-learn
```

Then run the evaluation script and save the run log using:

```powershell
cd "c:\Users\Evance Adriano\Desktop\School\Resume Keyword Highlighter Project"
python -u model_evaluation/evaluate_full_system.py 2>&1 | tee model_evaluation/evaluate_run.log
```

After the run completes inspect the `model_evaluation/` directory for generated figures and examine `model_evaluation/evaluate_run.log`. If any plotting exceptions occur consult `model_evaluation/knn_plot_error.log` for detailed tracebacks.

---

## 5.9 Appendix — figure captions and LaTeX snippets

The following concise captions are suitable for a thesis. For the confusion matrix use "Confusion matrix for the KNN self‑promotion classifier (test set)." For the metrics summary use "Classification metrics for self‑promotion detection (precision, recall, F1, accuracy)." For the highlighting example use "Representative before/after context‑aware keyword highlighting." For the score distribution use "Distribution of hybrid self‑promotion scores for positive and negative classes." These captions avoid repetition of values already stated in the text.

A LaTeX figure example is provided below and can be reused for other figures by replacing the file name and caption.

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.85\textwidth]{model_evaluation/self_promotion_confusion_matrix.png}
  \caption{Confusion matrix for the KNN self‑promotion classifier (test set).}
  \label{fig:knn_confusion}
\end{figure}
```

---

## 5.10 Conclusion

SkillHighlight demonstrates that a pragmatic combination of pretrained semantic embeddings, nearest‑neighbour classification and lightweight linguistic heuristics can surface meaningful competency statements and provide actionable scoring feedback with strong precision and recall on the tested dataset. Future work should focus on scalability, bias analysis and domain generalization.

---

File created by assistant. If you want I can (A) copy this back into `CHAPTER_5_MACHINELEARNING(REAL).md` to replace the bulleted version, (B) generate a LaTeX fragment from this content, or (C) run the evaluation script to (re)generate figures and confirm numbers. Tell me which option you prefer.
