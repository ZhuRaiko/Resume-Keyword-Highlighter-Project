# Chapter 2 — Algorithms and Implementation Details
2.1 SentenceTransformer (all‑MiniLM‑L6‑v2) — Sentence Embeddings

Description

The system employs the SentenceTransformers `all‑MiniLM‑L6‑v2` model to produce fixed‑length, 384‑dimensional vector representations of sentences. These embeddings serve as the semantic substrate for downstream similarity computations and for the KNN classifier. The implementation loads the model once and reuses it for batch encoding to reduce end‑to‑end latency.

Pseudocode

```
FUNCTION load_embedding_model():
  return SentenceTransformer('all-MiniLM-L6-v2')

FUNCTION encode_sentences(sentences, model, batch_size):
  vectors = []
  FOR batch in chunk(sentences, batch_size):
    v = model.encode(batch)
    vectors.extend(v)
  return vectors
```

Complexity

- Notation: let s be the number of sentences, d = 384 the embedding dimension, q the number of queries, and b the batch size used for encoding.
- Encoding time (amortized with batching): O(q · b) for batched encoding workloads; space: O(s · d).

Relevance

The Resume Keyword Highlighter represents each sentence as a 384‑dimensional dense real‑valued vector (an embedding) produced by a pretrained SentenceTransformer encoder. Dense embeddings encode semantic information so that sentences with similar meaning occupy nearby positions in the vector space despite lexical variation; this enables semantic matching of sentences that do not share identical surface tokens. The `all‑MiniLM‑L6‑v2` model was selected for its favorable trade‑off between representational fidelity and encoding throughput, providing high‑quality semantic comparisons while maintaining latency acceptable for interactive use.

Representative implementation excerpt

The following excerpt is taken verbatim from the repository's embedder module. It shows model loading and the encoding function used by the system.

```python
@st.cache_resource
def load_bert_model():
    """
    Load the pre-trained BERT model for sentence embeddings.
    Returns:
        SentenceTransformer: The loaded BERT model (all-MiniLM-L6-v2)
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


def encode_sentences(sentences, model=None):
    """
    Encode a list of sentences into BERT embeddings.
    """
    if model is None:
        model = load_bert_model()
    return model.encode(sentences)
```

```python
# Loads and caches the sentence embedding model so repeated calls reuse the same resource.
@st.cache_resource
def load_bert_model():
  """
  Load the pre-trained BERT model for sentence embeddings.
  Returns:
    SentenceTransformer: The loaded BERT model (all-MiniLM-L6-v2)
  """
  # The function returns a SentenceTransformer instance; the decorator caches it.
  return SentenceTransformer("all-MiniLM-L6-v2")


def encode_sentences(sentences, model=None):
  """
  Encode a list of sentences into BERT embeddings.
  """
  # If the caller did not provide a pre-loaded model, load the cached model here.
  if model is None:
    model = load_bert_model()
  # `model.encode` transforms the input list of strings into a numpy array of vectors.
  return model.encode(sentences)
```

---

## 2.2 K‑Nearest Neighbors (KNN) — Self‑Promotion Scoring

Description

The system uses a K‑Nearest Neighbors classifier trained on sentence embeddings to estimate the probability that a sentence expresses self‑promotion. The classifier is persisted when trained; during inference a query embedding is compared with stored labeled examples and a probability is returned.

Pseudocode

```
FUNCTION train_knn(X_train, y_train, k):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  save_model(knn)
  return knn

FUNCTION predict_knn(knn, bert_model, sentence):
  vec = bert_model.encode([sentence])
  IF knn supports predict_proba:
    probs = knn.predict_proba(vec)
    return probs[0][positive_class]
  ELSE:
    return knn.predict(vec)[0]
```

Complexity

- Notation: let m denote the number of stored labeled embeddings, d the embedding dimensionality, and q the number of queries.
- Storage / preparation cost (vector storage): O(m · d).
- Brute‑force inference for q queries: O(q · m · d) (per‑query cost O(m · d)).

Relevance

KNN supplies an interpretable, example‑based score that integrates naturally with the embedding representation. For the dataset sizes used in the project, brute‑force KNN with scikit‑learn provides acceptable latency.

Representative implementation excerpt

The excerpt below is taken verbatim from the system's KNN module. It shows the training/loading logic and the prediction helper used in the codebase.

```python
def load_or_train_knn(model_path="knn_model.pkl", csv_path="self_promotion_dataset.csv"):
    """
    Load existing KNN model or train new one from dataset.
    """
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    if not os.path.exists(csv_path):
        return DummyKNN()
    df = pd.read_csv(csv_path)
    bert_model = load_bert_model()
    X = bert_model.encode(df["sentence"].tolist())
    y = df["label"].astype(int).values
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    joblib.dump(knn, model_path)
    return knn


def predict_self_promotion_score(sentence, knn_model, bert_model):
    try:
        vec = bert_model.encode([sentence])
        if hasattr(knn_model, "predict_proba"):
            probs = knn_model.predict_proba(vec)
            if probs.shape[1] > 1:
                return float(probs[0][1])
            return float(probs[0][0])
        return float(knn_model.predict(vec)[0])
    except Exception:
        return 0.0
```

```python
# Attempt to load a serialized KNN model; otherwise train from CSV and cache the trained model.
def load_or_train_knn(model_path="knn_model.pkl", csv_path="self_promotion_dataset.csv"):
  """
  Load existing KNN model or train new one from dataset.
  """
  # Try loading a cached model to avoid retraining on each run
  if os.path.exists(model_path):
    try:
      return joblib.load(model_path)
    except Exception:
      pass
  # If no dataset is available, return a safe fallback classifier
  if not os.path.exists(csv_path):
    return DummyKNN()
  # Read labeled sentences and labels, then encode with the embedding model
  df = pd.read_csv(csv_path)
  bert_model = load_bert_model()
  X = bert_model.encode(df["sentence"].tolist())  # produces m x d matrix
  y = df["label"].astype(int).values
  # Train a scikit-learn KNN classifier and persist it for later reuse
  knn = KNeighborsClassifier(n_neighbors=5)
  knn.fit(X, y)
  joblib.dump(knn, model_path)
  return knn


def predict_self_promotion_score(sentence, knn_model, bert_model):
  try:
    # Encode the single query sentence into a 1 x d vector
    vec = bert_model.encode([sentence])
    # Prefer probabilistic output when available (predict_proba gives class probabilities)
    if hasattr(knn_model, "predict_proba"):
      probs = knn_model.predict_proba(vec)
      # Return the probability of the positive (self-promotion) class when present
      if probs.shape[1] > 1:
        return float(probs[0][1])
      return float(probs[0][0])
    # Fallback to deterministic class prediction
    return float(knn_model.predict(vec)[0])
  except Exception:
    # On any failure, return a neutral 0.0 score
    return 0.0
```

---

## 2.3 spaCy — Tokenization, Parsing, and Keyword Highlighting

Description

The repository implements a spaCy‑based module for keyword detection and context validation. The spaCy pipeline is used for tokenization, part‑of‑speech tagging, lemmatization, and sentence segmentation. The highlighting function applies category‑specific heuristics (hard skills, soft skills, recruiter terms, action verbs) and emits HTML spans for presentation.

Pseudocode

```
FUNCTION highlight_keywords(nlp, text, keyword_lists):
  doc = nlp(text)
  build lookup maps for normalized keywords
  FOR each token position i in doc:
    FOR window_length from max_kw_len downto 1:
      window = tokens[i:i+window_length]
      IF normalized(window) in lookup:
        IF context_is_valid(window, category):
          mark span for highlighting
          advance i past window
          BREAK
  render marked spans into HTML and return
```

Complexity

- Notation: let n be the number of tokens in the document and m the (small) maximum keyword token length.
- Tokenization and syntactic parsing with spaCy: O(n).
- Matching and longest‑first window heuristics: in typical usage near‑linear, but worst‑case behaviour may grow to O(n^2) for adversarial inputs or very large keyword sets; therefore report both O(n) and O(n^2) depending on the phase and input characteristics.

Relevance

spaCy's syntactic information enables context‑aware filtering that substantially reduces false positives relative to naive substring matching, improving the precision of highlights presented to users.

Representative implementation excerpt

The excerpt below is taken verbatim from the repository's highlighting module. It shows the public `highlight_keywords` function signature and the initial excluded‑span detection used to avoid highlighting within emails and URLs.

```python
def highlight_keywords(
    nlp,
    text: str,
    hard_skills: list,
    soft_skills: list,
    recruiter_keywords: list,
    action_verbs: list,
    disabled_labels=None,
    token_aligned=True,
    relax_hard=False,
    relax_action=False,
    relax_soft=True,
    relax_recruiter=True,
    soft_neg_threshold=-0.1,
    render_legacy=False
) -> str:
    """
    Highlight keywords in resume text with context validation and HTML rendering.
    """
    if disabled_labels is None:
        disabled_labels = set()

    # Find excluded spans (e.g., emails, URLs) to avoid highlighting inside them
    excluded = []
    for m in re.finditer(r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b|https?://\S+|www\.\S+", text):
        excluded.append((m.start(), m.end()))
```

```python
# Public function to highlight keywords with spaCy parsing and context heuristics.
def highlight_keywords(
  nlp,
  text: str,
  hard_skills: list,
  soft_skills: list,
  recruiter_keywords: list,
  action_verbs: list,
  disabled_labels=None,
  token_aligned=True,
  relax_hard=False,
  relax_action=False,
  relax_soft=True,
  relax_recruiter=True,
  soft_neg_threshold=-0.1,
  render_legacy=False
) -> str:
  """
  Highlight keywords in resume text with context validation and HTML rendering.
  """
  if disabled_labels is None:
    disabled_labels = set()

  # Detect spans to exclude from matching (emails, URLs); this prevents false positives
  excluded = []
  for m in re.finditer(r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b|https?://\S+|www\.\S+", text):
    excluded.append((m.start(), m.end()))
  # The function continues by building normalized keyword maps, parsing the text with `nlp`,
  # computing sentence-level features (such as polarity via TextBlob and verb presence),
  # detecting enumerations, and performing longest-first window matching with category-specific
  # context validation; matches are converted to HTML spans and returned as an HTML string.
```

---

## 2.4 Combined scoring and usage

The implemented system combines the KNN score with explicit keyword counts and heuristic penalties to compute a final sentence score. The production code uses weighted linear combination of the KNN probability and normalized keyword counts, with adjustments for negation or negative sentiment when indicated by the spaCy/TextBlob analysis.

For reproducibility, the exact code excerpts are included above; the repository contains the complete functions referenced here.

---

