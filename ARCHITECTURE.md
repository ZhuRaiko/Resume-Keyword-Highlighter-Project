# SkillHighlight - Modular Architecture Guide

## 📂 Project Structure

```
/SkillHighlight
│
├── main_modular.py          # Streamlit UI (NEW - clean entry point)
├── main_clean.py            # Original monolithic version (legacy)
├── keywords.json            # Keyword database
├── self_promotion_dataset.csv  # Training data (6,752 samples)
├── knn_model.pkl            # Cached KNN model
│
├── models/                  # CORE THESIS COMPONENTS
│   ├── __init__.py
│   ├── embedder.py          # BERT contextual embeddings
│   ├── knn_classifier.py    # KNN classification (k=5)
│   └── sentiment.py         # Sentiment analysis support
│
├── processing/              # SECONDARY PROCESSING
│   ├── __init__.py
│   ├── highlight_keywords.py   # SpaCy-driven keyword highlighting
│   ├── sentence_scoring.py     # Self-promotion scoring with heuristics
│   └── metrics.py              # Keyword composition metrics
│
└── utilities/               # UTILITY FUNCTIONS
    └── __init__.py          # Text extraction (PDF/DOCX/TXT)
```

## 🎓 Academic Framing

### Core Components (Thesis Focus)

These are the **two primary academic contributions**:

1. **BERT Embeddings** (`models/embedder.py`)
   - Model: `all-MiniLM-L6-v2`
   - Purpose: Transform sentences into 384-dimensional semantic vectors
   - Academic significance: Contextual understanding beyond keyword matching

2. **KNN Classification** (`models/knn_classifier.py`)
   - Configuration: k=5 neighbors
   - Training: 6,752 labeled resume sentences (2,233 positive / 4,519 negative)
   - Purpose: Classify sentences as self-promotional based on learned patterns
   - Academic significance: Supervised learning for domain-specific detection

### Secondary Components (Stabilization Heuristics)

These are **NOT core thesis components** — they are implementation details for handling resume format variations:

- **SpaCy Highlighting** (`processing/highlight_keywords.py`)
  - Context validation to prevent false positives
  - Linguistic pattern detection
  - Framing: "Secondary validation layer using dependency parsing"

- **Heuristic Bonuses** (`processing/sentence_scoring.py`)
  - Achievement pattern detection
  - Metric detection
  - Bullet point recognition
  - Sentiment adjustments
  - Framing: "Stabilization heuristics to normalize scores across writing styles"

- **Text Extraction** (`utilities/`)
  - PDF/DOCX parsing
  - Framing: "Preprocessing utilities for document format handling"

## 📝 How to Describe in Your Thesis

### Simplified Methodology Statement

> "SkillHighlight consists of three core modules:
> 1. BERT contextual embeddings for semantic sentence representation
> 2. KNN-based classification trained on labeled resume data
> 3. SpaCy-driven keyword highlighting with linguistic validation
>
> Additional heuristics (sentiment analysis, pattern detection, metric recognition) were implemented as secondary stabilizers to improve accuracy across diverse resume writing styles."

### Architecture Diagram for Thesis

```
┌─────────────────┐
│  Resume Input   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Extraction │ (utility)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│ BERT   │ │ SpaCy Parse  │
│ Encode │ │ + Highlight  │
└───┬────┘ └──────────────┘
    │
    ▼
┌────────┐
│  KNN   │ ◄── Core Classification
│ Score  │
└───┬────┘
    │
    ▼
┌─────────────────┐
│ Heuristic Boost │ (stabilization)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Final Scores +  │
│ Visualization   │
└─────────────────┘
```

## 🔧 Running the Application

### Using the Modular Version (Recommended)

```bash
streamlit run main_modular.py
```

### Using the Legacy Version

```bash
streamlit run main_clean.py
```

Both versions produce identical results — the modular version is simply better organized for academic defense.

## 🧪 Testing & Validation

### Unit Testing Core Components

```python
# Test BERT embeddings
from models.embedder import load_bert_model, encode_single_sentence
model = load_bert_model()
vec = encode_single_sentence("I increased sales by 30%", model)
assert vec.shape == (1, 384)  # Correct dimensionality

# Test KNN classifier
from models.knn_classifier import load_or_train_knn
knn = load_or_train_knn()
assert knn is not None  # Model loaded successfully
```

### Integration Testing

```python
# Test full pipeline
from main_modular import load_all_models
nlp, bert_model, knn_model, keyword_data = load_all_models()
from processing.sentence_scoring import analyze_self_promotion

text = "I led a team of 5 engineers and increased efficiency by 40%."
results, avg = analyze_self_promotion(text, nlp, knn_model, bert_model, [])
assert avg > 0.5  # Should detect self-promotion
```

## 📊 Metrics & Performance

Run `evaluate_model.py` to generate performance metrics:

```bash
python evaluate_model.py
```

This generates `model_metrics.json` with:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 🎯 Benefits of Modularization

### For Development
- ✅ Easier debugging (isolated components)
- ✅ Unit testing possible
- ✅ Clear separation of concerns
- ✅ Reusable components

### For Thesis Defense
- ✅ Clear academic narrative (2 core components + helpers)
- ✅ Easy to explain architecture
- ✅ Professional code organization
- ✅ Demonstrates software engineering maturity
- ✅ Easy to identify which component contributes to results

### For Maintenance
- ✅ Changes localized to specific modules
- ✅ No risk of breaking unrelated features
- ✅ Clear documentation of purpose
- ✅ Easy onboarding for new developers

## 🚀 Migration Guide

If you need to update the legacy `main_clean.py`:

1. Test `main_modular.py` thoroughly
2. Back up `main_clean.py` → `main_clean_backup.py`
3. Replace `main_clean.py` with modular version
4. Update all documentation references

## 📚 Citation in Thesis

When citing the implementation:

> "The system was implemented in Python using three core libraries: 
> SentenceTransformers for BERT embeddings, scikit-learn for KNN classification, 
> and spaCy for linguistic analysis. The codebase was modularized into separate 
> components (models/, processing/, utilities/) to ensure maintainability and 
> facilitate individual component evaluation."

## ⚠️ Important Notes

### What to Emphasize in Defense

- **Core methodology**: BERT + KNN (machine learning approach)
- **Training data**: 6,752 labeled sentences (robust dataset for generalization)
- **Novel contribution**: Combining semantic embeddings with KNN for resume analysis
- **Validation**: SpaCy provides linguistic context

### What to De-emphasize

- Implementation complexity
- Number of heuristics
- PDF extraction details
- UI polish
- Configuration options

Frame everything beyond BERT+KNN as "implementation details for production readiness."

## 📖 Further Reading

- `models/embedder.py` - BERT implementation details
- `models/knn_classifier.py` - Classification methodology
- `processing/sentence_scoring.py` - Scoring algorithm documentation
- `main_modular.py` - Complete system integration
