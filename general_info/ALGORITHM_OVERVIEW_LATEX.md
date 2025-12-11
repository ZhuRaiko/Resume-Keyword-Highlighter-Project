\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{geometry}
\usepackage{hyperref}
\geometry{margin=1in}
\title{SkillHighlight --- Deep Algorithm \\ \& System Guide}
\author{Resume Keyword Highlighter Project}
\date{\today}
\begin{document}
\maketitle

\section*{System Summary}
SkillHighlight is a modular pipeline for résumé analysis. It extracts sentences, generates semantic embeddings, classifies self-promotion, scores sentences, highlights keywords, and presents results in a web UI. Each step is optimized for accuracy, speed, and robustness.

\textbf{Defense notes:}
\begin{itemize}
  \item Modular design allows for easy extension and debugging.
  \item Each stage is independently testable and replaceable.
  \item Caching and batching are used to optimize performance.
\end{itemize}

\section*{Keyword Extraction \& Highlighting}

\textbf{Technical details (merged):}
\begin{itemize}
  \item Inputs: raw text, keyword dictionary, spaCy NLP pipeline.
  \item Preprocessing: Canonicalize keywords (lemmatize, lowercase, join tokens).
  \item Matching: For each sentence, scan tokens with sliding windows (max keyword length to 1). For each window, check for keyword match and context rules.
  \item Context rules: Reject date-like matches for temporal terms, require verb context for actions, prefer multi-token lemma matches.
  \item Highlighting: Mark matched spans, avoid overlaps by marking occupied tokens.
  \item Lemma tuple for multi-word keyword matching: $(\ell_1, \ell_2, ..., \ell)$
  \item Joined alphanumeric variant for matching: $\text{joined} = \text{concat}(\text{lower}(t_1), ..., \text{lower}(t))$
\end{itemize}

\textbf{Formula and Mathematical Computations:}
Let:
\begin{itemize}
  \item $n$ = number of tokens in the résumé
  \item $k$ = number of keywords in the dictionary
\end{itemize}
Tokenization and dependency parsing: $O(n)$
Keyword matching across all tokens and all keywords: $O(nk)$
Context validation for each token: $O(n)$
Overall dominant complexity: $O(nk)$

% --- Explanation of Keyword Extraction Pseudocode ---
\textbf{Pseudocode Explanation:}
The pseudocode for keyword extraction begins by preprocessing the keyword lists to create fast lookup maps using both lemma tuples and joined alphanumeric variants. For each keyword, the spaCy pipeline is used to tokenize and lemmatize, ensuring robust matching. The main highlighting function iterates through the résumé tokens, using a sliding window approach to check for multi-word keyword matches. For each window, it checks both the lemma tuple and joined variant against the lookup maps. If a match is found, it applies context validation rules (e.g., rejecting date contexts for hard skills, requiring verbs for actions). Matched spans are marked and replaced with HTML for highlighting, and occupied tokens are tracked to prevent overlapping highlights. Replacements are applied in reverse order to avoid index shifting issues, resulting in a context-aware, color-coded résumé output.

\textbf{Pseudocode (full):}
\begin{verbatim}
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
\end{verbatim}

\textbf{Time complexity:}
\begin{itemize}
  \item Tokenization: $O(n)$ (n = tokens)
  \item Sliding window matching: $O(n \cdot k_{max})$ ($k_{max}$ = max keyword length)
  \item Context checks: $O(1)$ per window (POS, lemma, etc.)
  \item Overall: Near-linear for practical $k_{max}$ (usually $\leq 5$)
\end{itemize}

\textbf{Edge cases \& tradeoffs:}
\begin{itemize}
  \item Overlapping keywords: handled by marking tokens as occupied.
  \item Ambiguous matches: context rules reduce false positives.
  \item Large keyword sets: precompute lookup maps for $O(1)$ match checks.
\end{itemize}

\textbf{Defense notes:}
\begin{itemize}
  \item spaCy’s linguistic features (POS, lemma) improve accuracy over naive substring matching.
  \item Right-to-left highlighting prevents index shift bugs.
  \item System is robust to noisy input and supports multi-word phrases.
\end{itemize}

\section*{Sentence Embedding Generation (BERT)}
\textbf{Plain language:} Transforms each sentence into a high-dimensional vector that encodes its meaning, enabling semantic comparison.

\textbf{Technical details:}
\begin{itemize}
  \item Model: SentenceTransformer (BERT-based, e.g., all-MiniLM-L6-v2)
  \item Embedding dimension: 384
  \item Method: Mean pooling over token hidden states
  \item Batching: Sentences processed in batches (size 32--128) for efficiency
  \item Caching: Embeddings for training data are cached to avoid recomputation
\end{itemize}

\textbf{Equation:}
Mean pooling of token embeddings:
\begin{equation*}
\mathbf{e}(s) = \frac{1}{T} \sum_{t=1}^T \mathbf{h}_t
\end{equation*}
Cosine similarity between two embeddings:
\begin{equation*}
\cos(u, v) = \frac{u \cdot v}{\|u\| \times \|v\|}
\end{equation*}
Expanded cosine similarity form:
\begin{equation*}
\cos(u, v) = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}
\end{equation*}

% --- Explanation of BERT Embedding Pseudocode ---
\textbf{Pseudocode Explanation:}
The BERT embedding pseudocode loads a pre-trained SentenceTransformer model ("all-MiniLM-L6-v2") and processes sentences in batches for efficiency. Each batch is encoded into embeddings using the model, and the results are concatenated. This approach leverages mean pooling over token hidden states to generate a fixed-size vector for each sentence, capturing its semantic meaning. Batching and caching are used to optimize speed and resource usage, making the embedding process scalable for large datasets.

\textbf{Pseudocode:}
\begin{verbatim}
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
\end{verbatim}

\textbf{Time complexity:}
\begin{itemize}
  \item Per sentence: $O(b)$ ($b$ = BERT forward pass)
  \item Batch: $O(q \cdot b)$ ($q$ = sentences)
  \item Model loading: $O(1)$ amortized
\end{itemize}

\textbf{Edge cases \& tradeoffs:}
\begin{itemize}
  \item Long sentences: BERT truncates to max token length (usually 128--512).
  \item GPU vs CPU: GPU accelerates batch encoding, but may hit memory limits.
  \item Large datasets: Cache embeddings to disk (e.g., with joblib or numpy)
\end{itemize}

\textbf{Defense notes:}
\begin{itemize}
  \item Mean pooling is standard for sentence-level semantics.
  \item Model choice (MiniLM) balances speed and accuracy for real-time use.
  \item Caching and batching are critical for scalability.
\end{itemize}

\section*{KNN Classification (Self-Promotion Detection)}
\textbf{Plain language:} Classifies sentences as self-promotion by comparing their embeddings to labeled examples and using majority vote among nearest neighbors.

\textbf{Technical details:}
\begin{itemize}
  \item Model: scikit-learn KNeighborsClassifier ($k=5$ typical)
  \item Distance metric: Cosine similarity (preferred for embeddings), Euclidean (supported)
  \item Training: Fit on labeled embeddings (from CSV)
  \item Prediction: For each query, find $k$ nearest neighbors, compute positive-class probability
\end{itemize}

\textbf{Equations:}
Cosine similarity:
\begin{equation*}
\mathrm{cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}
\end{equation*}
KNN probability:
\begin{equation*}
P_{KNN}(s) = \frac{1}{k} \sum_{j \in \mathcal{N}_k(s)} \mathbf{1}\{y_j = 1\}
\end{equation*}
Hybrid scoring function:
\begin{equation*}
\text{score}(x) = \alpha \cdot P(\text{positive}|x) + \beta \cdot \text{actionVerb}(x) + \gamma \cdot \text{metric}(x) + \delta \cdot \text{achievement}(x) + \epsilon \cdot \text{length}(x)
\end{equation*}

% --- Explanation of KNN Classification Pseudocode ---
\textbf{Pseudocode Explanation:}
The KNN classification pseudocode first checks if a trained model exists and loads it if available; otherwise, it trains a new KNeighborsClassifier using sentence embeddings and labels from a CSV file. The classifier uses $k=5$ neighbors and fits on the BERT-encoded training data. For prediction, new sentences are encoded into embeddings and passed to the classifier to obtain class probabilities. This process allows the system to classify sentences as self-promotional or not, based on their proximity to labeled examples in the embedding space. The approach is interpretable, as nearest neighbors can be inspected for each prediction, and is extensible for new training data.

\textbf{Pseudocode:}
\begin{verbatim}
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
\end{verbatim}

\textbf{Time complexity:}
\begin{itemize}
  \item Training: $O(m \cdot d)$ ($m$ = training samples, $d$ = embedding dim)
  \item Query: $O(m \cdot d)$ per query (brute-force)
  \item Batch prediction: $O(q \cdot m \cdot d)$ ($q$ = queries)
  \item Approximate NN (Faiss, Annoy): $O(d \cdot \log m)$ or sublinear
\end{itemize}

\textbf{Edge cases \& tradeoffs:}
\begin{itemize}
  \item Large $m$: Exact KNN is slow; use approximate NN libraries for scalability.
  \item Imbalanced classes: KNN can be biased; consider class weighting or threshold tuning.
  \item Curse of dimensionality: High $d$ can reduce KNN effectiveness; dimensionality reduction (PCA, UMAP) may help.
\end{itemize}

\textbf{Defense notes:}
\begin{itemize}
  \item Cosine similarity is robust for semantic embeddings.
  \item KNN is interpretable: can show nearest neighbors for each prediction.
  \item Approximate NN methods are industry standard for large-scale search.
\end{itemize}

\section*{Self-Promotion Scoring (Hybrid)}
\textbf{Plain language:} Combines KNN probability with heuristic bonuses (metrics, achievements, bullets, sentiment, action verbs, length penalty) to produce a final score for each sentence.

\textbf{Technical details:}
\begin{itemize}
  \item Score formula:
\end{itemize}
\begin{equation*}
\mathrm{Score}(s) = \operatorname{clamp}\left(P_{KNN}(s) + B_{metric} + B_{achieve} + B_{bullet} + B_{sentiment} + B_{action} + P_{length}\right)
\end{equation*}
\begin{itemize}
  \item Bonuses:
    \begin{itemize}
      \item Metric (number present): +0.15
      \item Achievement verb: +0.12
      \item Bullet: +0.08
      \item Positive sentiment: +0.06
      \item Action verb start: +0.08
      \item Short sentence penalty: -0.10
    \end{itemize}
  \item Clamp: Ensures score $\in [0, 1]$
\end{itemize}

\textbf{Pseudocode:}
\begin{verbatim}
def score_sentence(sentence, knn_prob, features):
    score = knn_prob
    if features['metric']: score += 0.15
    if features['achieve']: score += 0.12
    if features['bullet']: score += 0.08
    if features['sentiment'] > threshold: score += 0.06
    if features['action_verb']: score += 0.08
    if features['short']: score -= 0.10
    return clamp(score, 0, 1)
\end{verbatim}

\textbf{Time complexity:}
\begin{itemize}
  \item Dominated by KNN: $O(m \cdot d)$ per sentence
  \item Heuristics: $O(1)$ per sentence (regex, lookup)
\end{itemize}

\textbf{Edge cases \& tradeoffs:}
\begin{itemize}
  \item Bonus values: Tuned on validation set; may require calibration for new datasets.
  \item Heuristics: Add interpretability but may introduce bias.
  \item Clamp: Prevents out-of-range scores, but may mask overconfident predictions.
\end{itemize}

\textbf{Defense notes:}
\begin{itemize}
  \item Hybrid scoring improves recall for achievement sentences.
  \item Each component is logged for interpretability and ablation studies.
  \item System is extensible: new heuristics can be added as needed.
\end{itemize}

\section*{Extraction \& Preprocessing}
\textbf{Plain language:} Reads résumé files (TXT, PDF, DOCX), extracts and normalizes text, handles encoding and formatting issues.

\textbf{Technical details:}
\begin{itemize}
  \item TXT: Try UTF-8, fallback to Latin-1
  \item DOCX: Use python-docx or docx2txt
  \item PDF: Try pdf2docx (preserves layout), fallback to docling or pdfminer.six
  \item Cleaning: Normalize whitespace, unify bullet characters, smart sentence splitting
\end{itemize}

\textbf{Pseudocode:}
\begin{verbatim}
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
\end{verbatim}

\textbf{Time complexity:}
\begin{itemize}
  \item File read: $O(f)$ ($f$ = file size)
  \item Text cleaning: $O(n)$ ($n$ = characters/tokens)
  \item PDF conversion: $O(f)$ but may be slow for large/messy files
\end{itemize}

\textbf{Edge cases \& tradeoffs:}
\begin{itemize}
  \item PDF extraction: Many failure modes; pipeline of fallbacks increases robustness.
  \item Encoding issues: Try/catch for multiple encodings.
  \item Formatting: Normalize bullets and whitespace for consistent downstream processing.
\end{itemize}

\textbf{Defense notes:}
\begin{itemize}
  \item Multi-path extraction maximizes success rate across file types.
  \item Logging and error handling ensure traceability for failed extractions.
  \item Preprocessing is modular and can be extended for new formats.
\end{itemize}

\section*{UI Rendering (Streamlit)}
\textbf{Plain language:} Interactive web app for uploading résumés, viewing highlighted text and sentence scores, and exploring results.

\textbf{Technical details:}
\begin{itemize}
  \item Loads models and embeddings at startup (cached)
  \item Waits for user upload, runs extraction, highlighting, scoring
  \item Renders highlighted HTML and score tables
  \item Uses Streamlit’s cache for fast repeated interactions
\end{itemize}

\textbf{Time complexity:}
\begin{itemize}
  \item Rendering: $O(n)$ ($n$ = tokens/characters)
  \item Overall: $O(q \cdot b + q \cdot m \cdot d)$ (embedding + KNN for $q$ sentences)
\end{itemize}

\textbf{Edge cases \& tradeoffs:}
\begin{itemize}
  \item Large files: Progress bars and preview mode for responsiveness
  \item Caching: Reduces latency for repeated uploads
  \item UI: Designed for clarity and interpretability
\end{itemize}

\textbf{Defense notes:}
\begin{itemize}
  \item Streamlit is widely used for rapid prototyping and data apps.
  \item Caching and batching ensure scalability for real-world use.
  \item UI is modular and can be extended for new features.
\end{itemize}

\section*{Model Evaluation \& Visualizations}
\textbf{Plain language:} Evaluates classifier performance, visualizes embeddings, and generates publication-quality figures for thesis.

\textbf{Technical details:}
\begin{itemize}
  \item Plots: Learning curve, validation curve, confusion matrix, ROC/PR curves, embedding scatterplots (PCA, UMAP, t-SNE), nearest neighbor examples
  \item Metrics: Precision, recall, F1, ROC AUC, average precision
\end{itemize}

\textbf{Equations:}
\begin{align*}
\mathrm{Precision} &= \frac{TP}{TP + FP} \\
\mathrm{Recall} &= \frac{TP}{TP + FN} \\
F_1 &= 2 \cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}} \\
\mathrm{AUC} &= \int_0^1 TPR(FPR^{-1}(t)) dt \\
\mathrm{AP} &= \sum_n (R_n - R_{n-1}) P_n
\end{align*}

\textbf{Time complexity:}
\begin{itemize}
  \item Confusion matrix, ROC/PR: $O(n)$ ($n$ = samples)
  \item t-SNE: $O(n^2)$ (slow for large $n$)
  \item UMAP/PCA: $O(n \cdot d)$ (preferred for speed)
\end{itemize}

\textbf{Edge cases \& tradeoffs:}
\begin{itemize}
  \item t-SNE: High quality but slow; use UMAP/PCA for large datasets
  \item CSV encoding: Try/catch for cp1252, latin-1, utf-8
  \item Plotting: Log errors for reproducibility
\end{itemize}

\textbf{Defense notes:}
\begin{itemize}
  \item Evaluation suite covers all standard ML metrics and visualizations
  \item Figures are thesis-ready and reproducible
  \item Embedding plots provide qualitative insight into model behavior
\end{itemize}

\section*{Practical Tips \& Pitfalls}
\begin{itemize}
  \item Encoding: Always try multiple encodings for CSVs
  \item Caching: Persist embeddings and models to avoid recomputation
  \item Approximate NN: Use for large training sets
  \item Reproducibility: Fix random seeds, log model/library versions
  \item Figure generation: Run scripts in uninterrupted sessions, check logs
\end{itemize}

\section*{Overall Program Time Complexity}
The total time complexity of SkillHighlight combines all major steps:
\begin{itemize}
  \item Extraction \& Preprocessing: $O(f + n)$, where $f$ is file size and $n$ is number of tokens/characters.
  \item Keyword Extraction \& Highlighting: $O(n \cdot k_{max})$ (usually near-linear, $k_{max} \leq 5$)
  \item Sentence Embedding Generation (BERT): $O(q \cdot b)$, where $q$ is number of sentences and $b$ is BERT forward pass cost.
  \item KNN Classification: $O(q \cdot m \cdot d)$, where $m$ is training samples, $d$ is embedding dimension.
  \item Heuristic Scoring: $O(q)$ (simple lookups and regex per sentence)
  \item UI Rendering: $O(n)$ (HTML rendering, negligible compared to above)
\end{itemize}

	extbf{Combined Overall Complexity:}

For a résumé with $q$ sentences, $n$ tokens, and $m$ training samples:
\begin{equation*}
O(f + n \cdot k_{max} + q \cdot b + q \cdot m \cdot d)
\end{equation*}

\begin{itemize}
  \item $f$ --- file size (preprocessing)
  \item $n \cdot k_{max}$ --- keyword extraction
  \item $q \cdot b$ --- sentence embedding
  \item $q \cdot m \cdot d$ --- KNN classification (dominant for large $m$)
  \item $q$ --- heuristics and rendering (minor)
\end{itemize}

	extbf{In practice:}
\begin{itemize}
  \item For small $m$ (training set), the program is near-linear in résumé size.
  \item For large $m$, KNN classification dominates; use approximate NN for scalability.
\end{itemize}

If you want a more detailed breakdown or want to discuss bottlenecks and optimizations, let me know!

\section*{Appendix --- Reference}
\begin{itemize}
  \item BERT mean pooling:
    $\mathbf{e}(s) = \frac{1}{T} \sum_{t=1}^T \mathbf{h}_t$
  \item Cosine similarity:
    $\mathrm{cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$
  \item KNN probability:
    $P_{KNN}(s) = \frac{1}{k} \sum_{j \in \mathcal{N}_k(s)} \mathbf{1}\{y_j = 1\}$
  \item Hybrid score:
    $\mathrm{Score}(s) = \operatorname{clamp}\left(P_{KNN}(s) + \sum_i B_i + P_{length}\right)$
  \item Complexity quick list:
    \begin{itemize}
      \item Tokenization: $O(n)$
      \item Keyword matching: $O(n \cdot k_{max})$
      \item Embeddings: $O(q \cdot b)$
      \item KNN: $O(m \cdot d)$
      \item t-SNE: $O(n^2)$
    \end{itemize}
\end{itemize}

\section*{Thesis Defense Practice: Q\&A and Talking Points}
Here are sample questions and strong talking points for defending each algorithm:

\subsection*{Keyword Extraction \& Highlighting}
\begin{itemize}
  \item \textbf{Why use spaCy and linguistic features instead of simple string matching?}\\
    spaCy provides part-of-speech and lemma info, reducing false positives and handling word variations. This increases accuracy and robustness, especially for multi-word phrases and noisy input.
  \item \textbf{How do you avoid overlapping highlights?}\\
    We mark tokens as occupied during highlighting, ensuring no overlaps and correct rendering.
  \item \textbf{What is the time complexity and how does it scale?}\\
    Near-linear, $O(n \cdot k_{max})$, with $k_{max}$ small. Precomputed lookup maps make matching efficient.
\end{itemize}

\subsection*{Sentence Embedding Generation (BERT)}
\begin{itemize}
  \item \textbf{Why use mean pooling for sentence embeddings?}\\
    Mean pooling is standard for capturing overall sentence meaning and works well for semantic similarity tasks.
  \item \textbf{How do you handle long sentences?}\\
    BERT truncates to a max token length; most résumé sentences fit within this limit.
  \item \textbf{How do you optimize for speed?}\\
    We batch sentences and cache embeddings, reducing redundant computation.
\end{itemize}

\subsection*{KNN Classification}
\begin{itemize}
  \item \textbf{Why use KNN and cosine similarity?}\\
    KNN is interpretable and robust for semantic embeddings; cosine similarity is effective for high-dimensional spaces.
  \item \textbf{How do you scale to large training sets?}\\
    We use approximate nearest neighbor libraries (Faiss, Annoy) for sublinear search time.
  \item \textbf{What about class imbalance?}\\
    We can tune thresholds or use class weighting to address imbalance.
\end{itemize}

\subsection*{Self-Promotion Scoring (Hybrid)}
\begin{itemize}
  \item \textbf{Why combine KNN with heuristics?}\\
    Heuristics boost recall for achievement sentences and add interpretability. Each component is logged for analysis.
  \item \textbf{How do you tune the bonus values?}\\
    We calibrate on validation data and can adjust for new datasets.
  \item \textbf{How do you prevent overconfident scores?}\\
    We clamp scores to [0, 1] and monitor calibration.
\end{itemize}

\subsection*{Extraction \& Preprocessing}
\begin{itemize}
  \item \textbf{How do you handle different file formats and encoding issues?}\\
    We use a pipeline of extraction methods and try/catch for multiple encodings, maximizing robustness.
  \item \textbf{What if extraction fails?}\\
    We log errors and provide traceability for debugging.
\end{itemize}

\subsection*{UI Rendering}
\begin{itemize}
  \item \textbf{Why use Streamlit?}\\
    Streamlit is fast for prototyping and supports caching and modular UI design.
  \item \textbf{How do you handle large files?}\\
    We use progress bars and preview modes for responsiveness.
\end{itemize}

\subsection*{Model Evaluation \& Visualizations}
\begin{itemize}
  \item \textbf{What metrics do you use and why?}\\
    Precision, recall, F1, ROC AUC, and average precision are standard for classification. Embedding plots (PCA, UMAP, t-SNE) provide qualitative insight.
  \item \textbf{How do you ensure reproducibility?}\\
    We fix random seeds, log library versions, and check logs for errors.
\end{itemize}

\bigskip
Use these points to confidently answer panelist questions and demonstrate deep understanding of each algorithm and design choice.

\begin{itemize}
  \item Generate thesis-ready figures and plots
  \item Run evaluation scripts for results
  \item Practice defending each algorithm with supporting info above
\end{itemize}

\end{document}
