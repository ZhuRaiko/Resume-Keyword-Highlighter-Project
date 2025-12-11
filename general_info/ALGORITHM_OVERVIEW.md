# SkillHighlight — Deep Algorithm & System Guide

This document provides a comprehensive, technical, and defense-ready explanation of every major algorithm and system component in SkillHighlight. Each section includes: plain language, technical details, equations, pseudocode, time complexity analysis, edge cases, tradeoffs, and supporting info for thesis defense.

---

## 1. System Summary

SkillHighlight is a modular pipeline for résumé analysis. It extracts sentences, generates semantic embeddings, classifies self-promotion, scores sentences, highlights keywords, and presents results in a web UI. Each step is optimized for accuracy, speed, and robustness.

**Defense notes:**
- Modular design allows for easy extension and debugging.
- Each stage is independently testable and replaceable.
- Caching and batching are used to optimize performance.

---

## 2. Keyword Extraction & Highlighting

**Plain language:**
Finds and highlights important words/phrases (skills, metrics, actions) in résumés, making strengths visible to reviewers.

**Technical details (merged):**
- Inputs: raw text, keyword dictionary, spaCy NLP pipeline.
- Preprocessing: Canonicalize keywords (lemmatize, lowercase, join tokens).
- Matching: For each sentence, scan tokens with sliding windows (max keyword length to 1). For each window, check for keyword match and context rules.
- Context rules: Reject date-like matches for temporal terms, require verb context for actions, prefer multi-token lemma matches.
- Highlighting: Mark matched spans, avoid overlaps by marking occupied tokens.
- Lemma tuple for multi-word keyword matching: $(\ell_1, \ell_2, ..., \ell)$
- Joined alphanumeric variant for matching: $\text{joined} = \text{concat}(\text{lower}(t_1), ..., \text{lower}(t))$

**Formula and Mathematical Computations:**
Let:
- $n$ = number of tokens in the résumé
- $k$ = number of keywords in the dictionary
Tokenization and dependency parsing: $O(n)$
Keyword matching across all tokens and all keywords: $O(nk)$
Context validation for each token: $O(n)$
Overall dominant complexity: $O(nk)$

**Pseudocode (full):**
```
ALGORITHM: KEYWORD_HIGHLIGHTER
INPUT: text (raw resume string), keyword_lists (dict of labeled keyword lists), nlp (spaCy pipeline)
OUTPUT: highlighted_html (string with inline spans for highlighting)
GLOBAL VARIABLES:
    None persistent; keyword maps built at runtime

FUNCTION preprocess_keywords(keyword_lists):
    // Build lemma-tuple and joined-alnum maps for fast lookup
    map_by_tuple = {}
    map_by_joined = {}
    FOR EACH label, kw IN keyword_lists DO
        toks = nlp(kw)
        lem_tup = [LOWER(tok.lemma_) FOR tok IN toks]
        joined = CONCAT_ALNUM([LOWER(tok.text) FOR tok IN toks])
        map_by_tuple[TUPLE(lem_tup)] = (label, kw)
        map_by_joined[joined] = (label, kw)
    END FOR
    RETURN map_by_tuple, map_by_joined
END FUNCTION

FUNCTION is_valid_context(window_tokens, label):
    // Heuristics: reject date-contexts for HARD, require verb properties for ACTION, etc.
    // Returns TRUE if the keyword should be accepted in this token window
    APPLY rules from implementation
    RETURN TRUE or FALSE
END FUNCTION

FUNCTION highlight_text(text, keyword_lists, nlp):
    map_by_tuple, map_by_joined = preprocess_keywords(keyword_lists)
    doc = nlp(text)
    tokens = LIST(doc)
    occupied = SET()
    replacements = []
    max_kw_len = MAX_KEYWORD_LENGTH(map_by_tuple)
    i = 0
    WHILE i < LEN(tokens) DO
        IF i IN occupied THEN
            i = i + 1
            CONTINUE
        END IF
        matched = FALSE
        FOR L = MIN(max_kw_len, LEN(tokens)-i) DOWNTO 1 DO
            window = tokens[i : i+L]
            lem_tup = TUPLE(LOWER(t.lemma_) FOR t IN window)
            joined = CONCAT_ALNUM(LOWER(t.text) FOR t IN window)
            meta = NULL
            IF lem_tup IN map_by_tuple THEN
                meta = map_by_tuple[lem_tup]
            ELSE IF joined IN map_by_joined THEN
                meta = map_by_joined[joined]
            END IF
            IF meta IS NOT NULL THEN
                label, display_kw = meta
                IF NOT is_valid_context(window, label) THEN
                    CONTINUE
                END IF
                s = window[0].idx
                e = window[-1].idx + LEN(window[-1].text)
                html = RENDER_SPAN(text[s:e], label)
                APPEND (s, e, html) TO replacements
                FOR t_idx = i TO i+L-1 DO
                    ADD t_idx TO occupied
                END FOR
                matched = TRUE
                BREAK
            END IF
        END FOR
        IF NOT matched THEN
            i = i + 1
        ELSE
            i = i + 1
        END IF
    END WHILE
    // Apply replacements in reverse order
    SORT replacements BY start DESC
    highlighted = text
    FOR EACH (s, e, html) IN replacements DO
        highlighted = highlighted[:s] + html + highlighted[e:]
    END FOR
    RETURN highlighted
END FUNCTION
```

**Time complexity:**
- Tokenization: O(n) (n = tokens)
- Sliding window matching: O(n * k_max) (k_max = max keyword length)
- Context checks: O(1) per window (POS, lemma, etc.)
- Overall: Near-linear for practical k_max (usually ≤5)

**Edge cases & tradeoffs:**
- Overlapping keywords: handled by marking tokens as occupied.
- Ambiguous matches: context rules reduce false positives.
- Large keyword sets: precompute lookup maps for O(1) match checks.

**Defense notes:**
- spaCy’s linguistic features (POS, lemma) improve accuracy over naive substring matching.
- Right-to-left highlighting prevents index shift bugs.
- System is robust to noisy input and supports multi-word phrases.

---

## 3. Sentence Embedding Generation (BERT)

**Plain language:**
Transforms each sentence into a high-dimensional vector that encodes its meaning, enabling semantic comparison.

**Technical details:**
- Model: SentenceTransformer (BERT-based, e.g., all-MiniLM-L6-v2)
- Embedding dimension: 384
- Method: Mean pooling over token hidden states
- Batching: Sentences processed in batches (size 32–128) for efficiency
- Caching: Embeddings for training data are cached to avoid recomputation

**Equation:**
Mean pooling of token embeddings:
$$
\mathbf{e}(s) = \frac{1}{T} \sum_{t=1}^T \mathbf{h}_t
$$
Cosine similarity between two embeddings:
$$
\cos(u, v) = \frac{u \cdot v}{\|u\| \times \|v\|}
$$
Expanded cosine similarity form:
$$
\cos(u, v) = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}
$$

**Pseudocode:**
```
ALGORITHM: BERT_EMBEDDING
INPUT: sentences (list of strings), model_name (string)
OUTPUT: embeddings (array)
GLOBAL VARIABLES:
    model_name = "all-MiniLM-L6-v2"
    batch_size = 64

FUNCTION load_model(name = model_name):
    model = SentenceTransformer(name)  // cached if using streamlit cache
    RETURN model
END FUNCTION

FUNCTION batch_encode(sentences, model, batch_size = batch_size):
    embeddings = []
    FOR start = 0 TO LEN(sentences)-1 STEP batch_size DO
        batch = sentences[start : start + batch_size]
        emb = model.encode(batch)
        APPEND emb TO embeddings
    END FOR
    RETURN CONCATENATE(embeddings)
END FUNCTION
```

**Time complexity:**
- Per sentence: O(b) (b = BERT forward pass)
- Batch: O(q * b) (q = sentences)
- Model loading: O(1) amortized

**Edge cases & tradeoffs:**
- Long sentences: BERT truncates to max token length (usually 128–512).
- GPU vs CPU: GPU accelerates batch encoding, but may hit memory limits.
- Large datasets: Cache embeddings to disk (e.g., with joblib or numpy)

**Defense notes:**
- Mean pooling is standard for sentence-level semantics.
- Model choice (MiniLM) balances speed and accuracy for real-time use.
- Caching and batching are critical for scalability.

---

## 4. KNN Classification (Self-Promotion Detection)

**Plain language:**
Classifies sentences as self-promotion by comparing their embeddings to labeled examples and using majority vote among nearest neighbors.

**Technical details:**
- Model: scikit-learn KNeighborsClassifier (k=5 typical)
- Distance metric: Cosine similarity (preferred for embeddings), Euclidean (supported)
- Training: Fit on labeled embeddings (from CSV)
- Prediction: For each query, find k nearest neighbors, compute positive-class probability

**Equations:**
Cosine similarity:
$$
\mathrm{cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}
$$
KNN probability:
$$
P_{KNN}(s) = \frac{1}{k} \sum_{j \in \mathcal{N}_k(s)} \mathbf{1}\{y_j = 1\}
$$
Hybrid scoring function:
$$
\text{score}(x) = \alpha \cdot P(\text{positive}|x) + \beta \cdot \text{actionVerb}(x) + \gamma \cdot \text{metric}(x) + \delta \cdot \text{achievement}(x) + \epsilon \cdot \text{length}(x)
$$

**Pseudocode:**
```
ALGORITHM: KNN_TRAIN_AND_PREDICT
INPUT: csv_path (training csv), model_path (pkl), bert_model
OUTPUT: knn_model, predict_proba for queries
GLOBAL:
    K = 5

FUNCTION load_or_train_knn(csv_path, model_path, bert_model):
    IF FILE_EXISTS(model_path) THEN
        TRY
            RETURN JOBLIB_LOAD(model_path)
        EXCEPT
            PASS
        END TRY
    END IF
    df = PANDAS_READ_CSV(csv_path)
    X = batch_encode(df['sentence'].tolist(), bert_model)
    y = df['label'].astype(int).values
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X, y)
    JOBLIB_DUMP(knn, model_path)
    RETURN knn
END FUNCTION

FUNCTION predict_proba(knn, bert_model, queries):
    Xq = batch_encode(queries, bert_model)
    RETURN knn.predict_proba(Xq)
END FUNCTION
```

**Time complexity:**
- Training: O(m * d) (m = training samples, d = embedding dim)
- Query: O(m * d) per query (brute-force)
- Batch prediction: O(q * m * d) (q = queries)
- Approximate NN (Faiss, Annoy): O(d * log m) or sublinear

**Edge cases & tradeoffs:**
- Large m: Exact KNN is slow; use approximate NN libraries for scalability.
- Imbalanced classes: KNN can be biased; consider class weighting or threshold tuning.
- Curse of dimensionality: High d can reduce KNN effectiveness; dimensionality reduction (PCA, UMAP) may help.

**Defense notes:**
- Cosine similarity is robust for semantic embeddings.
- KNN is interpretable: can show nearest neighbors for each prediction.
- Approximate NN methods are industry standard for large-scale search.

---

## 5. Self-Promotion Scoring (Hybrid)

**Plain language:**
Combines KNN probability with heuristic bonuses (metrics, achievements, bullets, sentiment, action verbs, length penalty) to produce a final score for each sentence.

**Technical details:**
- Score formula:
$$
\mathrm{Score}(s) = \operatorname{clamp}\left(P_{KNN}(s) + B_{metric} + B_{achieve} + B_{bullet} + B_{sentiment} + B_{action} + P_{length}\right)
$$
- Bonuses:
  - Metric (number present): +0.15
  - Achievement verb: +0.12
  - Bullet: +0.08
  - Positive sentiment: +0.06
  - Action verb start: +0.08
  - Short sentence penalty: -0.10
- Clamp: Ensures score ∈ [0, 1]

**Pseudocode:**
```
def score_sentence(sentence, knn_prob, features):
    score = knn_prob
    if features['metric']: score += 0.15
    if features['achieve']: score += 0.12
    if features['bullet']: score += 0.08
    if features['sentiment'] > threshold: score += 0.06
    if features['action_verb']: score += 0.08
    if features['short']: score -= 0.10
    return clamp(score, 0, 1)
```

**Time complexity:**
- Dominated by KNN: O(m * d) per sentence
- Heuristics: O(1) per sentence (regex, lookup)

**Edge cases & tradeoffs:**
- Bonus values: Tuned on validation set; may require calibration for new datasets.
- Heuristics: Add interpretability but may introduce bias.
- Clamp: Prevents out-of-range scores, but may mask overconfident predictions.

**Defense notes:**
- Hybrid scoring improves recall for achievement sentences.
- Each component is logged for interpretability and ablation studies.
- System is extensible: new heuristics can be added as needed.

---

## 6. Extraction & Preprocessing

**Plain language:**
Reads résumé files (TXT, PDF, DOCX), extracts and normalizes text, handles encoding and formatting issues.

**Technical details:**
- TXT: Try UTF-8, fallback to Latin-1
- DOCX: Use python-docx or docx2txt
- PDF: Try pdf2docx (preserves layout), fallback to docling or pdfminer.six
- Cleaning: Normalize whitespace, unify bullet characters, smart sentence splitting

**Pseudocode:**
```
def extract_text(filename):
    if filename.endswith('.txt'):
        try: return read_utf8(filename)
        except: return read_latin1(filename)
    elif filename.endswith('.docx'):
        try: return read_docx(filename)
        except: return fallback_docx(filename)
    elif filename.endswith('.pdf'):
        try: docx = pdf2docx(filename); return read_docx(docx)
        except: try: return docling(filename)
        except: return pdfminer(filename)
    else: raise ValueError('Unsupported format')
```

**Time complexity:**
- File read: O(f) (f = file size)
- Text cleaning: O(n) (n = characters/tokens)
- PDF conversion: O(f) but may be slow for large/messy files

**Edge cases & tradeoffs:**
- PDF extraction: Many failure modes; pipeline of fallbacks increases robustness.
- Encoding issues: Try/catch for multiple encodings.
- Formatting: Normalize bullets and whitespace for consistent downstream processing.

**Defense notes:**
- Multi-path extraction maximizes success rate across file types.
- Logging and error handling ensure traceability for failed extractions.
- Preprocessing is modular and can be extended for new formats.

---

## 7. UI Rendering (Streamlit)

**Plain language:**
Interactive web app for uploading résumés, viewing highlighted text and sentence scores, and exploring results.

**Technical details:**
- Loads models and embeddings at startup (cached)
- Waits for user upload, runs extraction, highlighting, scoring
- Renders highlighted HTML and score tables
- Uses Streamlit’s cache for fast repeated interactions

**Time complexity:**
- Rendering: O(n) (n = tokens/characters)
- Overall: O(q * b + q * m * d) (embedding + KNN for q sentences)

**Edge cases & tradeoffs:**
- Large files: Progress bars and preview mode for responsiveness
- Caching: Reduces latency for repeated uploads
- UI: Designed for clarity and interpretability

**Defense notes:**
- Streamlit is widely used for rapid prototyping and data apps.
- Caching and batching ensure scalability for real-world use.
- UI is modular and can be extended for new features.

---

## 8. Model Evaluation & Visualizations

**Plain language:**
Evaluates classifier performance, visualizes embeddings, and generates publication-quality figures for thesis.

**Technical details:**
- Plots: Learning curve, validation curve, confusion matrix, ROC/PR curves, embedding scatterplots (PCA, UMAP, t-SNE), nearest neighbor examples
- Metrics: Precision, recall, F1, ROC AUC, average precision

**Equations:**
Precision, Recall, F1:
$$
\mathrm{Precision} = \frac{TP}{TP + FP},\quad \mathrm{Recall} = \frac{TP}{TP + FN},\quad F_1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$
ROC AUC:
$$
\mathrm{AUC} = \int_0^1 TPR(FPR^{-1}(t)) dt
$$
Average precision:
$$
\mathrm{AP} = \sum_n (R_n - R_{n-1}) P_n
$$

**Time complexity:**
- Confusion matrix, ROC/PR: O(n) (n = samples)
- t-SNE: O(n^2) (slow for large n)
- UMAP/PCA: O(n * d) (preferred for speed)

**Edge cases & tradeoffs:**
- t-SNE: High quality but slow; use UMAP/PCA for large datasets
- CSV encoding: Try/catch for cp1252, latin-1, utf-8
- Plotting: Log errors for reproducibility

**Defense notes:**
- Evaluation suite covers all standard ML metrics and visualizations
- Figures are thesis-ready and reproducible
- Embedding plots provide qualitative insight into model behavior

---

## 9. Practical Tips & Pitfalls

- Encoding: Always try multiple encodings for CSVs
- Caching: Persist embeddings and models to avoid recomputation
- Approximate NN: Use for large training sets
- Reproducibility: Fix random seeds, log model/library versions
- Figure generation: Run scripts in uninterrupted sessions, check logs

---


## 10. Overall Program Time Complexity

The total time complexity of SkillHighlight combines all major steps:

- Extraction & Preprocessing: $O(f + n)$, where $f$ is file size and $n$ is number of tokens/characters.
- Keyword Extraction & Highlighting: $O(n \cdot k_{max})$ (usually near-linear, $k_{max} \leq 5$)
- Sentence Embedding Generation (BERT): $O(q \cdot b)$, where $q$ is number of sentences and $b$ is BERT forward pass cost.
- KNN Classification: $O(q \cdot m \cdot d)$, where $m$ is training samples, $d$ is embedding dimension.
- Heuristic Scoring: $O(q)$ (simple lookups and regex per sentence)
- UI Rendering: $O(n)$ (HTML rendering, negligible compared to above)

	extbf{Combined Overall Complexity:}

For a résumé with $q$ sentences, $n$ tokens, and $m$ training samples:
$$
O(f + n \cdot k_{max} + q \cdot b + q \cdot m \cdot d)
$$

- $f$ — file size (preprocessing)
- $n \cdot k_{max}$ — keyword extraction
- $q \cdot b$ — sentence embedding
- $q \cdot m \cdot d$ — KNN classification (dominant for large $m$)
- $q$ — heuristics and rendering (minor)

	extbf{In practice:}
- For small $m$ (training set), the program is near-linear in résumé size.
- For large $m$, KNN classification dominates; use approximate NN for scalability.

If you want a more detailed breakdown or want to discuss bottlenecks and optimizations, let me know!

---

## 11. Appendix — Reference

- BERT mean pooling:
  $$
  \mathbf{e}(s) = \frac{1}{T} \sum_{t=1}^T \mathbf{h}_t
  $$
- Cosine similarity:
  $$
  \mathrm{cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}
  $$
- KNN probability:
  $$
  P_{KNN}(s) = \frac{1}{k} \sum_{j \in \mathcal{N}_k(s)} \mathbf{1}\{y_j = 1\}
  $$
- Hybrid score:
  $$
  \mathrm{Score}(s) = \operatorname{clamp}\left(P_{KNN}(s) + \sum_i B_i + P_{length}\right)
  $$
- Complexity quick list:
  - Tokenization: O(n)
  - Keyword matching: O(n * k_max)
  - Embeddings: O(q * b)
  - KNN: O(m * d)
  - t-SNE: O(n^2)

---


## 12. Thesis Defense Practice: Q&A and Talking Points

Here are sample questions and strong talking points for defending each algorithm:

### Keyword Extraction & Highlighting
- Q: Why use spaCy and linguistic features instead of simple string matching?
  - A: spaCy provides part-of-speech and lemma info, reducing false positives and handling word variations. This increases accuracy and robustness, especially for multi-word phrases and noisy input.
- Q: How do you avoid overlapping highlights?
  - A: We mark tokens as occupied during highlighting, ensuring no overlaps and correct rendering.
- Q: What is the time complexity and how does it scale?
  - A: Near-linear, $O(n \cdot k_{max})$, with $k_{max}$ small. Precomputed lookup maps make matching efficient.

### Sentence Embedding Generation (BERT)
- Q: Why use mean pooling for sentence embeddings?
  - A: Mean pooling is standard for capturing overall sentence meaning and works well for semantic similarity tasks.
- Q: How do you handle long sentences?
  - A: BERT truncates to a max token length; most résumé sentences fit within this limit.
- Q: How do you optimize for speed?
  - A: We batch sentences and clamp embeddings, reducing redundant computation.

### KNN Classification
- Q: Why use KNN and cosine similarity?
  - A: KNN is interpretable and robust for semantic embeddings; cosine similarity is effective for high-dimensional spaces.
- Q: How do you scale to large training sets?
  - A: We use approximate nearest neighbor libraries (Faiss, Annoy) for sublinear search time.
- Q: What about class imbalance?
  - A: We can tune thresholds or use class weighting to address imbalance.

### Self-Promotion Scoring (Hybrid)
- Q: Why combine KNN with heuristics?
  - A: Heuristics boost recall for achievement sentences and add interpretability. Each component is logged for analysis.
- Q: How do you tune the bonus values?
  - A: We calibrate on validation data and can adjust for new datasets.
- Q: How do you prevent overconfident scores?
  - A: We clamp scores to [0, 1] and monitor calibration.

### Extraction & Preprocessing
- Q: How do you handle different file formats and encoding issues?
  - A: We use a pipeline of extraction methods and try/catch for multiple encodings, maximizing robustness.
- Q: What if extraction fails?
  - A: We log errors and provide traceability for debugging.

### UI Rendering
- Q: Why use Streamlit?
  - A: Streamlit is fast for prototyping and supports caching and modular UI design.
- Q: How do you handle large files?
  - A: We use progress bars and preview modes for responsiveness.

### Model Evaluation & Visualizations
- Q: What metrics do you use and why?
  - A: Precision, recall, F1, ROC AUC, and average precision are standard for classification. Embedding plots (PCA, UMAP, t-SNE) provide qualitative insight.
- Q: How do you ensure reproducibility?
  - A: We fix random seeds, log library versions, and check logs for errors.

---

Use these points to confidently answer panelist questions and demonstrate deep understanding of each algorithm and design choice.


- Generate thesis-ready figures and plots
- Run evaluation scripts for results
- Practice defending each algorithm with supporting info above

---

If you need further expansion, ablation studies, or sample Q&A for panelists, request specific sections!
