# 4.4 Data Analysis Methods

Section 4.4 documents the analysis methods used to evaluate SkillHighlight. It describes metrics, statistical procedures, thresholding and calibration, ablation and sensitivity analyses, and reproducibility checks. Where appropriate the exact formulas and operational definitions used by `model_evaluation/evaluate_full_system.py` are given.

## Purpose
Explain how experimental results are computed, summarized and interpreted in Chapter 5. This section ensures that evaluation choices are transparent and reproducible.

## Metrics and exact formulas
- True Positives (TP): count of correctly detected positive items.
- False Positives (FP): count of incorrectly detected positive items.
- False Negatives (FN): count of missed positive items.

Formulas used in code and reporting:

- Precision:

  $$\mathrm{Precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}$$

- Recall (Sensitivity):

  $$\mathrm{Recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}$$

- F1-score (harmonic mean of precision and recall):

  $$F_1=2\cdot\frac{\mathrm{Precision}\times\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}$$

- Text extraction success rate:

  $$\mathrm{Success\;Rate}=100\times\frac{\text{successful\_extractions}}{\text{files\_tested}}\;[\%]$$

- Highlighting test accuracy (per-test pass rate):

  $$\mathrm{PassRate}=100\times\frac{\text{tests\_passed}}{\text{total\_tests}}\;[\%]$$

  In the evaluation script a test is counted as passed only when it has zero missed expected keywords and zero false positives.

- Score separation (between HIGH and LOW sets):

  $$\mathrm{Separation}=\overline{s}_{\text{HIGH}}-\overline{s}_{\text{LOW}}$$

  where $\overline{s}_{\text{HIGH}}$ and $\overline{s}_{\text{LOW}}$ are the mean scores for the HIGH and LOW subsets respectively.

Notes on implementation details that affect metric computation:
- Keyword matching in the highlighting evaluation relies on case-insensitive substring containment when comparing expected keywords to highlighted tokens. This can produce partial-match TPs (e.g., `python3.9` matching `python`).
- The keyword evaluation does not compute true negatives (TN); therefore global accuracy is not used for highlighting reporting.

## Evaluation protocol
- Data splits: training / validation / test splits are created with stratified sampling to preserve HIGH/LOW balance. The script uses `train_test_split(..., random_state=42)` for reproducibility.
- Cross-validation: learning and validation curves for KNN are computed using sklearn cross-validation (`cv=5`) to assess stability when applicable.
- Seeds: set seeds for deterministic behavior where supported: `random.seed(42)`, `numpy.random.seed(42)`, and `random_state=42` for sklearn functions. Note: some algorithms (UMAP, t-SNE, certain BLAS operations) may remain non-deterministic without additional configuration.

## Statistical methods
- Confidence intervals: bootstrap CIs (e.g., 95% percentile bootstrap) are recommended to quantify metric variability, but bootstrap CI plots and numeric intervals are not included in the present figures and are listed as future work.
- Hypothesis tests: paired tests (e.g., McNemar's test or bootstrap differences) are recommended when comparing variants; these tests were not applied to the figures shown and are optional future analyses.

## Calibration & thresholding
- ROC and Precision-Recall curves are plotted for the HIGH vs LOW binary subset and are included among the figures (see Chapter 5). Reliability diagrams / reliability (calibration) plots are recommended but are not included in the current figures and are noted as future work.
- Thresholds used in the evaluation script:
  - predicted = HIGH if score >= 0.55
  - predicted = LOW if score <= 0.45
  - otherwise predicted = MID
- For metric correctness the script applies a tolerance: expected HIGH is considered correct if score >= 0.50; expected LOW considered correct if score <= 0.50.

## Ablation & sensitivity analyses
- Ablation and sensitivity analyses (for example, toggling heuristics, varying `k` in KNN, or sweeping thresholds) are recommended to understand component contributions. These specific analyses were not included among the current figures and are listed as future work; the evaluation code includes hooks for these experiments.

## Error analysis and qualitative inspection
- Qualitative inspection of errors is essential. The presented figures include representative confusion matrices and nearest-neighbour visualizations (PCA/UMAP) to help locate clusters of errors. Example-by-example nearest-neighbour images are recommended for a deeper manual audit but are not included in the current figures; these can be produced by the evaluation script when requested.

## Data handling decisions
- Encoding fallbacks when reading CSV: UTF-8 → CP-1252 → Latin-1 to maximize robustness to real-world résumé encodings.
- Tokenization: spaCy tokenization and lemmatization are used for contextual rules in highlighting; tests should document whether matching is token-level or substring-level.

## Reproducibility checklist (to include with experiments)
- Save a pinned `requirements_frozen.txt` from the runtime used for reported experiments (`pip freeze > model_evaluation/requirements_frozen.txt`).
- Record seeds before running: `random.seed(42); numpy.random.seed(42)`.
- Record the SentenceTransformer model id (e.g., `sentence-transformers/all-MiniLM-L6-v2`) and spaCy model (`en_core_web_sm`).
- Record whether `models/knn_model.pkl` was loaded or retrained, and include the CSV `data/self_promotion_dataset.csv` used for training.
- Save all diagnostic PNGs and `full_system_metrics.json` produced by `model_evaluation/evaluate_full_system.py`.

## Tools and packages
List core packages used for analysis and plotting (include versions in `requirements_frozen.txt`):
- Python 3.8+ (report exact)
- numpy, pandas
- scikit-learn
- sentence-transformers (model id recorded separately)
- spaCy + `en_core_web_sm`
- matplotlib
- joblib
- umap-learn (optional)

## Cross references
- Refer to `model_evaluation/evaluate_full_system.py` for exact code-level implementations and the location where metrics are computed and saved.


*End of 4.4 Data Analysis Methods.*
