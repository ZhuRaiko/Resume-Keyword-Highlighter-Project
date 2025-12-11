# SkillHighlight: Models, Features, and Time Complexity

This document summarizes the main models and features of the SkillHighlight system, along with their theoretical time complexity.

---

## 1. Keyword Extraction & Highlighting
- **Description:** Scans résumé text for keywords and highlights them, using spaCy for context validation.
- **Steps:**
  - Tokenization (spaCy): O(n), where n = number of words in the résumé.
  - Keyword matching (with regex): O(n * k), where k = number of keywords.
  - Context validation (spaCy): O(n), as each token is checked for part-of-speech and dependencies.
- **Overall Complexity:** O(n * k) (dominated by keyword matching).

## 2. BERT Embedding Generation
- **Description:** Converts each sentence to a 384-dimensional vector using a pre-trained BERT model (all-MiniLM-L6-v2).
- **Steps:**
  - Sentence splitting: O(n), where n = number of sentences.
  - Embedding (BERT): O(n * b), where b = average BERT inference time per sentence (constant for fixed model size).
- **Overall Complexity:** O(n), linear in the number of sentences (since b is constant for a fixed model).

## 3. KNN Classification
- **Description:** For each sentence, finds the k nearest neighbors among all labeled training samples using cosine similarity.
- **Steps:**
  - Distance computation: O(m * d), where m = number of training samples, d = embedding dimension (384).
  - Sorting/selecting k nearest: O(m) per query.
- **Overall Complexity (per query):** O(m * d), where m is typically large (e.g., 10,000).
- **Batch Complexity:** O(q * m * d), where q = number of sentences to classify.

## 4. Self-Promotion Scoring
- **Description:** Uses KNN output and heuristics to assign a self-promotion score to each sentence.
- **Steps:**
  - KNN classification: O(m * d) per sentence (see above).
  - Heuristic scoring: O(1) per sentence.
- **Overall Complexity:** O(q * m * d), dominated by KNN.

## 5. UI Rendering (Streamlit)
- **Description:** Renders highlighted résumé, configuration panel, and results in a web interface.
- **Steps:**
  - Rendering: O(n), where n = number of tokens/words to display.
  - User interaction (checkboxes, toggles): O(1) per event.
- **Overall Complexity:** O(n) per render.

## 6. Extraction & Preprocessing
- **Description:** Reads and cleans résumé files (PDF, DOCX, TXT).
- **Steps:**
  - File reading: O(f), where f = file size.
  - Text cleaning: O(n), where n = number of characters/words.
- **Overall Complexity:** O(f + n).

---

## Summary Table
| Feature/Model            | Main Steps                        | Time Complexity         |
|-------------------------|-----------------------------------|------------------------|
| Keyword Highlighting    | Tokenize, match, context check    | O(n * k)               |
| BERT Embedding          | Sentence split, embed             | O(n)                   |
| KNN Classification      | Distance, sort/select             | O(m * d) per query     |
| Self-Promotion Scoring  | KNN, heuristics                   | O(q * m * d)           |
| UI Rendering            | Render, interact                  | O(n)                   |
| Extraction/Preprocessing| Read, clean                       | O(f + n)               |

**Notes:**
- n = number of words or sentences in the résumé
- k = number of keywords
- m = number of training samples
- d = embedding dimension (384)
- q = number of sentences to classify
- f = file size

This analysis assumes typical input sizes and does not account for hardware-specific optimizations or parallelization.

---

## Model Evaluation Metrics
- **KNN classifier:** Achieved 86.2% accuracy, 97.1% precision, 92.3% recall, and 94.6% F1-score on the held-out test set.
- **Keyword highlighting:** Achieved 97.1% precision with 94.6% F1-score, demonstrating reliable and accurate detection.
- **End-to-end scoring:** Achieved 90% classification accuracy with 0.690 score separation between high and low quality sentences.
- **Cross-validation:** Demonstrated consistent performance (Mean F1: 0.817 ± 0.121).
- **Context-aware highlighting:** Successfully prevented common false positives.

### MATHEMATICAL EQUATIONS (Mapped to Algorithms)

Below are the principal mathematical formulas used across the algorithms and where they apply. When inserting into the thesis, render these with LaTeX/KaTeX for clarity.

- **BERT embedding (mean pooling)** — used in `BERT_EMBEDDING` / `get_embedding`:

    $$\mathbf{e}(s)=\frac{1}{T}\sum_{t=1}^{T}\mathbf{h}_t$$

    where $\mathbf{h}_t$ is the last‑layer token hidden state for token $t$ and $T$ is the number of tokens in sentence $s$.

- **Cosine similarity** — used in `BERT_AI_Chatbot_Semantic_Search` and for retrieval:

    $$\mathrm{cosine}(\mathbf{u},\mathbf{v})=\frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|}=\frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2}\,\sqrt{\sum_{i=1}^d v_i^2}}$$

- **Euclidean distance** — typical KNN distance metric (used in `KNN_TRAIN_AND_PREDICT`):

    $$d(\mathbf{u},\mathbf{v})=\sqrt{\sum_{i=1}^d (u_i-v_i)^2}$$

- **KNN positive-class probability** — how the KNN base probability is computed:

    $$P_{KNN}(s)=\frac{1}{k}\sum_{j\in\mathcal{N}_k(s)}\mathbf{1}\{y_j=1\}$$

    where $\mathcal{N}_k(s)$ are the $k$ nearest neighbours of $s$ and $y_j\in\{0,1\}$ are labels.

- **Final hybrid score (SELF_PROMOTION_SCORING):**

    $$\mathrm{Score}(s)=\operatorname{clamp}\Big(P_{KNN}(s)+B_{metric}+B_{achieve}+B_{bullet}+B_{sentiment}+B_{action}+P_{length}\Big)$$

    with $\operatorname{clamp}(x)=\min(1,\max(0,x))$.

- **Standard classification metrics** (used across evaluation):
    $$\mathrm{Precision}=\frac{TP}{TP+FP},\qquad \mathrm{Recall}=\frac{TP}{TP+FN},$$
    $$F_1=2\cdot\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}$$

- **ROC AUC (binary)** — conceptual definition used when plotting ROC:
    $$\mathrm{AUC}=\int_0^1 TPR(FPR^{-1}(t))\,dt$$

- **Average precision (PR curve area)** — practical discrete form:
    $$\mathrm{AP}=\sum_n (R_n-R_{n-1})P_n$$

- **Complexity references:** BERT embedding batch of size $q$ is $O(q\cdot b)$ (where $b$ is per‑sentence BERT cost); KNN per query is $O(m\cdot d)$.

These equations map directly to the pseudocode below (e.g., cosine similarity inside `BERT_AI_Chatbot_Semantic_Search`, Euclidean and $P_{KNN}$ in `KNN_TRAIN_AND_PREDICT`, and scoring in `SELF_PROMOTION_SCORING`).

### ALGORITHM: BERT_AI_Chatbot_Semantic_Search

```text
INPUT: user_input (text string)
OUTPUT: bot_reply (text string)

GLOBAL VARIABLES:
    tokenizer = BERT tokenizer ("bert-base-uncased")
    model = BERT model ("bert-base-uncased")
    adhd_embeddings = pre-computed ADHD response embeddings (tensor)
    adhd_texts = pre-computed ADHD response texts (list)
    knowledge_base = {
        "hello": "Hello! You can ask me anything about ADHD, studying, or productivity.",
        "study tips": "Use the Pomodoro method, take small breaks, and remove distractions.",
        "focus tips": "Break tasks into small pieces and reduce your environment stimuli.",
        "what is nlp": "NLP stands for Natural Language Processing, letting computers understand language."
    }

FUNCTION preprocess(text):
    text = CONVERT_TO_LOWERCASE(text)
    text = REMOVE all non-alphanumeric characters EXCEPT spaces
    text = TRIM_WHITESPACE(text)
    RETURN text
END FUNCTION

FUNCTION get_embedding(text):
    // Tokenize input text
    tokens = tokenizer.TOKENIZE(text, return_tensors="pt")
    
    // Generate embeddings without gradient computation
    WITH no_gradient_tracking DO
        output = model.FORWARD(tokens)
    END WITH
    
    // Extract last hidden state and compute mean pooling
    last_hidden_state = output.last_hidden_state  // Shape: [1, n_tokens, 768]
    sentence_embedding = MEAN(last_hidden_state, axis=1)  // Shape: [1, 768]
    
    RETURN sentence_embedding
END FUNCTION

FUNCTION adhd_reply(user_input):
    // Preprocess user input
    processed_text = preprocess(user_input)
    
    // Get user input embedding
    user_embedding = get_embedding(processed_text)
    
    // Compute cosine similarity with all ADHD embeddings
    scores = COSINE_SIMILARITY(user_embedding, adhd_embeddings)
    
    // Find best match
    best_index = INDEX_OF_MAX(scores)
    best_score = scores[best_index]
    
    // Check if similarity exceeds threshold
    IF best_score < 0.40 THEN
        RETURN NULL
    END IF
    
    // Return matching ADHD response
    RETURN adhd_texts[best_index]
END FUNCTION

FUNCTION simple_bert_reply(user_input):
    // Preprocess user input
    processed_text = preprocess(user_input)
    
    // Get user input embedding
    user_embedding = get_embedding(processed_text)
    
    best_score = -1
    best_answer = NULL
    
    // Compare with each knowledge base entry
    FOR EACH (question, answer) IN knowledge_base DO
        question_embedding = get_embedding(question)
        score = COSINE_SIMILARITY(user_embedding, question_embedding)
        
        IF score > best_score THEN
            best_score = score
            best_answer = answer
        END IF
    END FOR
    
    // Check if similarity exceeds threshold
    IF best_score < 0.5 THEN
        RETURN NULL
    END IF
    
    RETURN best_answer
END FUNCTION

FUNCTION get_bot_reply(text):
    // Check for simple greetings
    lowercase_text = CONVERT_TO_LOWERCASE(text)
    
    IF "hello" IN lowercase_text OR "hi" IN lowercase_text THEN
        RETURN "Hello! How can I help you today?"
    END IF
    
    IF "bye" IN lowercase_text THEN
        RETURN "Goodbye! Stay productive and take care."
    END IF
    
    // Try ADHD dataset semantic search
    adhd_answer = adhd_reply(text)
    IF adhd_answer IS NOT NULL THEN
        RETURN adhd_answer
    END IF
    
    // Try knowledge base fallback
    bert_answer = simple_bert_reply(text)
    IF bert_answer IS NOT NULL THEN
        RETURN bert_answer
    END IF
    
    // Final fallback
    RETURN "Hmm… I didn't fully understand that. Try rephrasing!"
END FUNCTION

FUNCTION send_message():
    user_text = GET user input from text field
    user_text = TRIM_WHITESPACE(user_text)
    
    IF user_text IS empty THEN
        RETURN
    END IF
    
    // Display user message
    add_message("You", user_text)
    
    // Get bot response
    bot_reply = get_bot_reply(user_text)
    
    // Display bot message
    add_message("Bot", bot_reply)
    
    // Clear input field
    CLEAR user input field
    
    // Scroll to bottom of chat
    SCHEDULE scroll_to_bottom()
END FUNCTION

FUNCTION add_message(sender, text):
    // Create message label
    label = CREATE_LABEL(
        text = "[bold]" + sender + ":[/bold] " + text,
        markup = TRUE,
        size_hint_y = NULL,
        font_size = 16,
        color = (0, 0, 0, 1)
    )
    
    // Calculate text height
    UPDATE label texture
    label.height = label.texture_height + 10
    
    // Add to chat log
    ADD label TO chat_log
END FUNCTION

FUNCTION scroll_to_bottom():
    SET scroll position TO bottom (scroll_y = 0)
END FUNCTION

// HELPER FUNCTION: Cosine Similarity
FUNCTION COSINE_SIMILARITY(vector_A, vector_B):
    // Compute dot product
    dot_product = 0
    FOR i = 0 TO LENGTH(vector_A) - 1 DO
        dot_product = dot_product + (vector_A[i] × vector_B[i])
    END FOR
    
    // Compute magnitudes
    magnitude_A = SQRT(SUM(vector_A[i]² for all i))
    magnitude_B = SQRT(SUM(vector_B[i]² for all i))
    
    // Compute cosine similarity
    similarity = dot_product / (magnitude_A × magnitude_B)
    
    RETURN similarity
END FUNCTION
```

---

### ALGORITHM: KEYWORD_HIGHLIGHTER
```text
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
            IF lem_tup IN map_by_tuple THEN meta = map_by_tuple[lem_tup]
            ELSE IF joined IN map_by_joined THEN meta = map_by_joined[joined]
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

---

### ALGORITHM: BERT_EMBEDDING (human-style)
```text
INPUT: sentences (list of strings), model_name (string)
OUTPUT: embeddings (array)

GLOBAL VARIABLES:
    model_name = "all-MiniLM-L6-v2"
    batch_size = 64

FUNCTION load_model(name = model_name):
    model = SentenceTransformer(name) // cached if using streamlit cache
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

---

### ALGORITHM: KNN_TRAIN_AND_PREDICT (human-style)
```text
INPUT: csv_path (training csv), model_path (pkl), bert_model
OUTPUT: knn_model, predict_proba for queries

GLOBAL:
    K = 5

FUNCTION load_or_train_knn(csv_path, model_path, bert_model):
    IF FILE_EXISTS(model_path) THEN
        TRY RETURN JOBLIB_LOAD(model_path)
        EXCEPT PASS
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

---

### ALGORITHM: SELF_PROMOTION_SCORING (human-style)
```text
INPUT: text (resume), knn_model, bert_model, action_verbs
OUTPUT: list of (sentence, score), average_score

GLOBAL CONSTANTS:
    METRIC_BONUS = 0.15
    ACHIEVE_BONUS = 0.12
    BULLET_BONUS = 0.08
    POLARITY_BOOST = 0.06
    ACTION_BONUS = 0.08
    LENGTH_PENALTY = -0.10

FUNCTION contains_metric(sentence):
    RETURN REGEX_SEARCH(r"(\b\d+%|\b\d{1,3}\b|\$\d+[kKmM]?)", sentence)
END FUNCTION

FUNCTION detect_achievement_pattern(sentence):
    lower = LOWER(sentence)
    achievement_words = {achieved, delivered, improved, increased, reduced, led, managed, developed, created, launched, implemented, drove, exceeded, optimized, streamlined, established, built, designed, spearheaded, pioneered}
    impact_words = {resulting, leading to, by, saved, generated, boosted, enhanced}
    RETURN (ANY(w IN lower FOR w IN achievement_words) AND ANY(p IN lower FOR p IN impact_words))
END FUNCTION

FUNCTION score_sentence(knn_model, bert_model, sentence):
    base = predict_proba(knn_model, bert_model, [sentence])[0][1] // positive-class prob
    metric = METRIC_BONUS IF contains_metric(sentence) ELSE 0
    achieve = ACHIEVE_BONUS IF detect_achievement_pattern(sentence) ELSE 0
    bullet = BULLET_BONUS IF REGEX_MATCH(r'^\s*[-•]', sentence) ELSE 0
    polarity = TEXTBLOB_POLARITY(sentence)
    pol = POLARITY_BOOST IF polarity > 0.15 ELSE 0
    action = ACTION_BONUS IF FIRST_WORD_IN(sentence, action_verbs) ELSE 0
    length = LENGTH_PENALTY IF LEN(sentence.split()) < 5 ELSE 0
    score = CLAMP(base + metric + achieve + bullet + pol + action + length, 0.0, 1.0)
    RETURN score
END FUNCTION

FUNCTION analyze_text(nlp, knn_model, bert_model, text, action_verbs):
    sentences = SMART_SENTENCE_SPLIT(nlp, text)
    scored = []
    FOR s IN sentences DO
        scored.APPEND((s, score_sentence(knn_model, bert_model, s)))
    END FOR
    avg = MEAN([sc for (_, sc) IN scored]) IF scored ELSE 0.0
    RETURN scored, avg
END FUNCTION
```

---

### ALGORITHM: STREAMLIT_APP_FLOW (human-style)
```text
INPUT: user uploads
OUTPUT: highlights and scores rendered in UI

FUNCTION main():
    bert_model = load_model()
    knn_model = load_or_train_knn('data/self_promotion_dataset.csv', 'models/knn_model.pkl', bert_model)
    uploaded = UI_FILE_UPLOADER()
    IF uploaded IS NONE THEN
        SHOW_INSTRUCTIONS()
        RETURN
    END IF
    raw = extract_from_file(uploaded)
    text = normalize_resume_text(raw)
    highlighted_html = highlight_text(text, keyword_lists, nlp)
    scored, avg = analyze_text(nlp, knn_model, bert_model, text, action_verbs)
    RENDER_RESULTS(highlighted_html, scored, avg)
END FUNCTION
```

---

### ALGORITHM: EXTRACT_AND_NORMALIZE (human-style)
```text
INPUT: uploaded_file (PDF/DOCX/TXT)
OUTPUT: normalized_text

FUNCTION extract_from_file(uploaded_file):
    data = uploaded_file.read()
    name = LOWER(uploaded_file.name)
    IF name ENDSWITH('.txt') THEN
        TRY
            RETURN data.decode('utf-8')
        EXCEPT
            RETURN data.decode('latin-1', errors='ignore')
        END TRY
    END IF

    IF name ENDSWITH('.pdf') THEN
        TRY
            // Try pdf2docx conversion first
            tmp_pdf = SAVE_TEMP(data)
            docx_path = PDF2DOCX_CONVERT(tmp_pdf)
            text = READ_DOCX(docx_path)
            RETURN normalize_resume_text(text)
        EXCEPT
            TRY
                md = DOCLING_CONVERT_TO_MARKDOWN(data)
                RETURN CLEAN_MARKDOWN(md)
            EXCEPT
                RETURN PDFMINER_EXTRACT_TEXT(data)
            END TRY
        END TRY
    END IF

    IF name ENDSWITH('.docx') THEN
        TRY
            RETURN normalize_resume_text(READ_DOCX_FROM_BYTES(data))
        EXCEPT
            RETURN DOCX2TXT_FALLBACK(data)
        END TRY
    END IF

    RAISE Exception('Unsupported file type')
END FUNCTION
```

INPUT: user_input (text string)
OUTPUT: bot_reply (text string)

GLOBAL VARIABLES:
    tokenizer = BERT tokenizer ("bert-base-uncased")
    model = BERT model ("bert-base-uncased")
    adhd_embeddings = pre-computed ADHD response embeddings (tensor)
    adhd_texts = pre-computed ADHD response texts (list)
    knowledge_base = {
        "hello": "Hello! You can ask me anything about ADHD, studying, or productivity.",
        "study tips": "Use the Pomodoro method, take small breaks, and remove distractions.",
        "focus tips": "Break tasks into small pieces and reduce your environment stimuli.",
        "what is nlp": "NLP stands for Natural Language Processing, letting computers understand language."
    }

FUNCTION preprocess(text):
    text = CONVERT_TO_LOWERCASE(text)
    text = REMOVE all non-alphanumeric characters EXCEPT spaces
    text = TRIM_WHITESPACE(text)
    RETURN text
END FUNCTION

FUNCTION get_embedding(text):
    // Tokenize input text
    tokens = tokenizer.TOKENIZE(text, return_tensors="pt")
    
    // Generate embeddings without gradient computation
    WITH no_gradient_tracking DO
        output = model.FORWARD(tokens)
    END WITH
    
    // Extract last hidden state and compute mean pooling
    last_hidden_state = output.last_hidden_state  // Shape: [1, n_tokens, 768]
    sentence_embedding = MEAN(last_hidden_state, axis=1)  // Shape: [1, 768]
    
    RETURN sentence_embedding
END FUNCTION

FUNCTION adhd_reply(user_input):
    // Preprocess user input
    processed_text = preprocess(user_input)
    
    // Get user input embedding
    user_embedding = get_embedding(processed_text)
    
    // Compute cosine similarity with all ADHD embeddings
    scores = COSINE_SIMILARITY(user_embedding, adhd_embeddings)
    
    // Find best match
    best_index = INDEX_OF_MAX(scores)
    best_score = scores[best_index]
    
    // Check if similarity exceeds threshold
    IF best_score < 0.40 THEN
        RETURN NULL
    END IF
    
    // Return matching ADHD response
    RETURN adhd_texts[best_index]
END FUNCTION

FUNCTION simple_bert_reply(user_input):
    // Preprocess user input
    processed_text = preprocess(user_input)
    
    // Get user input embedding
    user_embedding = get_embedding(processed_text)
    
    best_score = -1
    best_answer = NULL
    
    // Compare with each knowledge base entry
    FOR EACH (question, answer) IN knowledge_base DO
        question_embedding = get_embedding(question)
        score = COSINE_SIMILARITY(user_embedding, question_embedding)
        
        IF score > best_score THEN
            best_score = score
            best_answer = answer
        END IF
    END FOR
    
    // Check if similarity exceeds threshold
    IF best_score < 0.5 THEN
        RETURN NULL
    END IF
    
    RETURN best_answer
END FUNCTION

FUNCTION get_bot_reply(text):
    // Check for simple greetings
    lowercase_text = CONVERT_TO_LOWERCASE(text)
    
    IF "hello" IN lowercase_text OR "hi" IN lowercase_text THEN
        RETURN "Hello! How can I help you today?"
    END IF
    
    IF "bye" IN lowercase_text THEN
        RETURN "Goodbye! Stay productive and take care."
    END IF
    
    // Try ADHD dataset semantic search
    adhd_answer = adhd_reply(text)
    IF adhd_answer IS NOT NULL THEN
        RETURN adhd_answer
    END IF
    
    // Try knowledge base fallback
    bert_answer = simple_bert_reply(text)
    IF bert_answer IS NOT NULL THEN
        RETURN bert_answer
    END IF
    
    // Final fallback
    RETURN "Hmm… I didn't fully understand that. Try rephrasing!"
END FUNCTION

FUNCTION send_message():
    user_text = GET user input from text field
    user_text = TRIM_WHITESPACE(user_text)
    
    IF user_text IS empty THEN
        RETURN
    END IF
    
    // Display user message
    add_message("You", user_text)
    
    // Get bot response
    bot_reply = get_bot_reply(user_text)
    
    // Display bot message
    add_message("Bot", bot_reply)
    
    // Clear input field
    CLEAR user input field
    
    // Scroll to bottom of chat
    SCHEDULE scroll_to_bottom()
END FUNCTION

FUNCTION add_message(sender, text):
    // Create message label
    label = CREATE_LABEL(
        text = "[bold]" + sender + ":[/bold] " + text,
        markup = TRUE,
        size_hint_y = NULL,
        font_size = 16,
        color = (0, 0, 0, 1)
    )
    
    // Calculate text height
    UPDATE label texture
    label.height = label.texture_height + 10
    
    // Add to chat log
    ADD label TO chat_log
END FUNCTION

FUNCTION scroll_to_bottom():
    SET scroll position TO bottom (scroll_y = 0)
END FUNCTION

// HELPER FUNCTION: Cosine Similarity
FUNCTION COSINE_SIMILARITY(vector_A, vector_B):
    // Compute dot product
    dot_product = 0
    FOR i = 0 TO LENGTH(vector_A) - 1 DO
        dot_product = dot_product + (vector_A[i] × vector_B[i])
    END FOR
    
    // Compute magnitudes
    magnitude_A = SQRT(SUM(vector_A[i]² for all i))
    magnitude_B = SQRT(SUM(vector_B[i]² for all i))
    
    // Compute cosine similarity
    similarity = dot_product / (magnitude_A × magnitude_B)
    
    RETURN similarity
END FUNCTION




---

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
            IF lem_tup IN map_by_tuple THEN meta = map_by_tuple[lem_tup]
            ELSE IF joined IN map_by_joined THEN meta = map_by_joined[joined]
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



ALGORITHM: BERT_EMBEDDING (human-style)

INPUT: sentences (list of strings), model_name (string)
OUTPUT: embeddings (array)

GLOBAL VARIABLES:
    model_name = "all-MiniLM-L6-v2"
    batch_size = 64

FUNCTION load_model(name = model_name):
    model = SentenceTransformer(name) // cached if using streamlit cache
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


ALGORITHM: KNN_TRAIN_AND_PREDICT (human-style)

INPUT: csv_path (training csv), model_path (pkl), bert_model
OUTPUT: knn_model, predict_proba for queries

GLOBAL:
    K = 5

FUNCTION load_or_train_knn(csv_path, model_path, bert_model):
    IF FILE_EXISTS(model_path) THEN
        TRY RETURN JOBLIB_LOAD(model_path)
        EXCEPT PASS
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


ALGORITHM: SELF_PROMOTION_SCORING (human-style)

INPUT: text (resume), knn_model, bert_model, action_verbs
OUTPUT: list of (sentence, score), average_score

GLOBAL CONSTANTS:
    METRIC_BONUS = 0.15
    ACHIEVE_BONUS = 0.12
    BULLET_BONUS = 0.08
    POLARITY_BOOST = 0.06
    ACTION_BONUS = 0.08
    LENGTH_PENALTY = -0.10

FUNCTION contains_metric(sentence):
    RETURN REGEX_SEARCH(r"(\b\d+%|\b\d{1,3}\b|\$\d+[kKmM]?)", sentence)
END FUNCTION

FUNCTION detect_achievement_pattern(sentence):
    lower = LOWER(sentence)
    achievement_words = {achieved, delivered, improved, increased, reduced, led, managed, developed, created, launched, implemented, drove, exceeded, optimized, streamlined, established, built, designed, spearheaded, pioneered}
    impact_words = {resulting, leading to, by, saved, generated, boosted, enhanced}
    RETURN (ANY(w IN lower FOR w IN achievement_words) AND ANY(p IN lower FOR p IN impact_words))
END FUNCTION

FUNCTION score_sentence(knn_model, bert_model, sentence):
    base = predict_proba(knn_model, bert_model, [sentence])[0][1] // positive-class prob
    metric = METRIC_BONUS IF contains_metric(sentence) ELSE 0
    achieve = ACHIEVE_BONUS IF detect_achievement_pattern(sentence) ELSE 0
    bullet = BULLET_BONUS IF REGEX_MATCH(r'^\s*[-•]', sentence) ELSE 0
    polarity = TEXTBLOB_POLARITY(sentence)
    pol = POLARITY_BOOST IF polarity > 0.15 ELSE 0
    action = ACTION_BONUS IF FIRST_WORD_IN(sentence, action_verbs) ELSE 0
    length = LENGTH_PENALTY IF LEN(sentence.split()) < 5 ELSE 0
    score = CLAMP(base + metric + achieve + bullet + pol + action + length, 0.0, 1.0)
    RETURN score
END FUNCTION

FUNCTION analyze_text(nlp, knn_model, bert_model, text, action_verbs):
    sentences = SMART_SENTENCE_SPLIT(nlp, text)
    scored = []
    FOR s IN sentences DO
        scored.APPEND((s, score_sentence(knn_model, bert_model, s)))
    END FOR
    avg = MEAN([sc for (_, sc) IN scored]) IF scored ELSE 0.0
    RETURN scored, avg
END FUNCTION


ALGORITHM: STREAMLIT_APP_FLOW (human-style)

INPUT: user uploads
OUTPUT: highlights and scores rendered in UI

FUNCTION main():
    bert_model = load_model()
    knn_model = load_or_train_knn('data/self_promotion_dataset.csv', 'models/knn_model.pkl', bert_model)
    uploaded = UI_FILE_UPLOADER()
    IF uploaded IS NONE THEN
        SHOW_INSTRUCTIONS()
        RETURN
    END IF
    raw = extract_from_file(uploaded)
    text = normalize_resume_text(raw)
    highlighted_html = highlight_text(text, keyword_lists, nlp)
    scored, avg = analyze_text(nlp, knn_model, bert_model, text, action_verbs)
    RENDER_RESULTS(highlighted_html, scored, avg)
END FUNCTION


ALGORITHM: EXTRACT_AND_NORMALIZE (human-style)

INPUT: uploaded_file (PDF/DOCX/TXT)
OUTPUT: normalized_text

FUNCTION extract_from_file(uploaded_file):
    data = uploaded_file.read()
    name = LOWER(uploaded_file.name)
    IF name ENDSWITH('.txt') THEN
        TRY
            RETURN data.decode('utf-8')
        EXCEPT
            RETURN data.decode('latin-1', errors='ignore')
        END TRY
    END IF

    IF name ENDSWITH('.pdf') THEN
        TRY
            // Try pdf2docx conversion first
            tmp_pdf = SAVE_TEMP(data)
            docx_path = PDF2DOCX_CONVERT(tmp_pdf)
            text = READ_DOCX(docx_path)
            RETURN normalize_resume_text(text)
        EXCEPT
            TRY
                md = DOCLING_CONVERT_TO_MARKDOWN(data)
                RETURN CLEAN_MARKDOWN(md)
            EXCEPT
                RETURN PDFMINER_EXTRACT_TEXT(data)
            END TRY
        END TRY
    END IF

    IF name ENDSWITH('.docx') THEN
        TRY
            RETURN normalize_resume_text(READ_DOCX_FROM_BYTES(data))
        EXCEPT
            RETURN DOCX2TXT_FALLBACK(data)
        END TRY
    END IF

    RAISE Exception('Unsupported file type')
END FUNCTION