CHAPTER 2

2.1 Algorithms and Its Rules
2.1.1 Keyword Extraction and Highlighting
The Keyword Extraction and Highlighting module is the foundation of the SkillHighlight system. Its primary function is to scan résumé text, identify relevant skill-related terms, and visually highlight them with contextual accuracy. Using spaCy for linguistic processing, the algorithm performs tokenization, part-of-speech tagging, and dependency parsing to ensure that each word or phrase is evaluated in relation to its syntactic and semantic context. This prevents incorrect detections, such as mistaking dates for numerical skills or confusing verbs for technical competencies. The final output is a context-aware, color-coded résumé that helps users understand which skills are properly emphasized and which areas may require improvement.
2.1.1.1 Implications of Keyword Extraction

Context-aware keyword extraction is aligned with research emphasizing the need for deeper semantic and syntactic understanding in résumé analysis systems. Traditional rule-based keyword spotting is prone to false positives, especially when words appear in irrelevant contexts. By integrating lemma comparison, combined-token matching, and contextual filters, the SkillHighlight system adheres to modern approaches that reduce noise and capture more meaningful skill patterns. This ensures that subtle variations—such as "machine learning," "machine-learning," or "ML"—are correctly recognized under a unified skill representation. The method also reflects findings that résumé evaluation tools must interpret terms in the correct context to avoid misclassifying user competencies, thereby improving fairness and accuracy in automated screening systems.

2.1.1.2 Formula and Mathematical Computations
Let:
- n = number of tokens in the résumé
- k = number of keywords in the dictionary
Tokenization and dependency parsing:
	O(n)   
Keyword matching across all tokens and all keywords:

	O(nk)

Context validation for each token:

	O(n)

Overall dominant complexity:

	O(nk)

Lemma tuple used for multi-word keyword matching:

	(ℓ₁, ℓ₂, ..., ℓ)

Joined alphanumeric variant for matching:

	joined = concat(lower(t₁), ..., lower(t))

2.1.1.3 Time Complexity

The final time complexity of keyword extraction is O(nk). This is efficient even for multi-page résumés, as spaCy's operations run in linear time and k remains manageable.

2.1.1.4 Pseudocode of Keyword Extraction

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

2.1.1.5 Relevance to the Application

This algorithm plays a crucial role in SkillHighlight because it determines how effectively the system identifies and emphasizes the user's skills. Accurate detection ensures that users receive reliable feedback on which competencies stand out and which may be underrepresented. By minimizing incorrect highlights and ensuring clean contextual interpretation, the system supports fair, interpretable résumé evaluation aligned with common expectations in professional screening.

2.1.2 K-Nearest Neighbors Classification

The K-Nearest Neighbors classifier is the main decision-making component of SkillHighlight's sentence-level self-promotion scoring. After the system converts each résumé sentence into a vector representation using BERT embeddings, the KNN classifier compares the new sentence with labeled examples to determine whether it reflects strong, weak, or neutral self-promotion. The classifier's ability to draw inferences based on proximity to similar sentences makes it well-suited for evaluating how confidently and effectively a résumé communicates accomplishments.

2.1.2.1 Implications of KNN Classification

KNN's suitability in high-dimensional embedding spaces is supported by current work in text classification and similarity-based analysis. Because it is a non-parametric model, its predictions adapt seamlessly as new training examples are added without requiring retraining of model parameters. This makes it appropriate for résumé analysis systems that continuously expand and refine their datasets. The classifier's interpretability also aligns with explainable AI frameworks, allowing similarity-based justification for each prediction. Given that résumé scoring can affect applicants' perceptions of fairness and quality, having traceable reasoning strengthens transparency. Additionally, SkillHighlight's hybrid scoring strategy—combining KNN similarity with action verb recognition, metric detection, achievement patterns, sentence length heuristics, and polarity shifts—reflects established guidelines in résumé writing research emphasizing quantification, clarity, and assertive phrasing.

2.1.2.2 Formula and Mathematical Computations

Let:
- m = number of training examples
- d = embedding dimension (384)
- q = number of sentences in a résumé

Distance computation for one sentence:

	O(m · d)

Euclidean distance:

			

KNN positive-class probability:

Hybrid scoring function:

	

2.1.2.3 Time Complexity

Per sentence: O(m · d)

Full résumé scoring: O(q · m · d)

Because m and q are both relatively small in typical usage, the system produces results in real time.

2.1.2.4 Pseudocode of KNN Classification

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

2.1.2.5 Relevance to the Application

The KNN classifier determines the strength of the résumé's self-promotional language, helping users evaluate whether their statements are assertive, measurable, and accomplishment-oriented. This is important because many applicants struggle to present their achievements convincingly. By computing similarity to strong and weak examples and enhancing the score through linguistic cues, SkillHighlight helps users rewrite vague or generic statements into impactful, competitive résumé content.

2.1.3 BERT Embedding Model

BERT Embedding Generation in SkillHighlight uses the "all-MiniLM-L6-v2" BERT model to convert sentences into 384-dimensional embeddings that capture their semantic meaning. The model processes text through multiple transformer layers, enabling it to understand context, tone, and intent. These embeddings serve as the foundation for both similarity comparison and classification, ensuring that the algorithm evaluates meaning instead of relying on surface-level patterns.

2.1.3.1 Implications of BERT

Transformer-based models such as BERT have become the standard for semantic text representation due to their ability to understand nuanced language. In résumé analysis, this capability is crucial because the same idea can be expressed with different wording, and strong statements often differ from weak ones only in subtle phrasing. BERT's contextual embeddings allow SkillHighlight to distinguish between sentences like "Led a team of five developers" and "Worked with a team of developers" even though the vocabulary overlaps significantly. This property ensures that the system evaluates statements with human-like semantic awareness, improving fairness and consistency.

2.1.3.2 Formula and Mathematical Computations

Mean pooling of token embeddings:

	

Cosine similarity between two embeddings:

			cos(u, v) = (u · v) / (||u|| × ||v||)

Expanded cosine similarity form:

	

2.1.3.3 Time Complexity

Let b = constant processing cost per sentence for the BERT model.

Embedding generation for n sentences:

	O(n · b)

Since the model is lightweight and optimized, embedding computation remains efficient during real-world usage.

2.1.3.4 Pseudocode of BERT Embedding

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

2.1.3.5 Relevance to the Application

BERT embeddings ensure that SkillHighlight understands résumé content beyond keyword matching. By interpreting the semantic meaning of each sentence, the system can reliably assess strength of wording, detect achievement-oriented phrasing, and support the KNN classifier with accurate representations. This makes scoring more consistent and aligned with real HR review patterns, ultimately improving the quality of feedback given to users.


