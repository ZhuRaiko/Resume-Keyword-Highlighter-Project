# SkillHighlight - Modular Architecture Guide

## 📂 Project Structure

```
/SkillHighlight
│
├── main_modular.py              # Streamlit UI (main entry point)
├── evaluate_model.py            # Model evaluation script
│
├── data/                        # DATA FILES
│   ├── keywords.json            # Keyword database (4 categories)
│   └── self_promotion_dataset.csv  # Training data (10,000 samples)
│
├── fonts/                       # CUSTOM FONTS
│   ├── Morganite-*.ttf          # Morganite font family (titles/headers)
│   └── deliware-*.otf           # Deliware font family (labels)
│
├── models/                      # CORE THESIS COMPONENTS
│   ├── __init__.py
│   ├── embedder.py              # BERT contextual embeddings
│   ├── knn_classifier.py        # KNN classification (k=5)
│   ├── knn_model.pkl            # Cached KNN model
│   └── sentiment.py             # Sentiment analysis support
│
├── modules/                     # PROCESSING MODULES
│   ├── __init__.py
│   ├── counters.py              # Keyword composition metrics
│   ├── embeddings.py            # Embedding utilities
│   ├── extractor.py             # Text extraction (PDF→DOCX conversion)
│   ├── highlight.py             # SpaCy-driven keyword highlighting
│   └── scoring.py               # Self-promotion scoring with heuristics
│
└── backups/                     # LEGACY FILES
    ├── main_clean.py            # Original monolithic version
    └── main.py                  # Earlier implementation
```

## 🎨 UI/UX Design

### Theme & Typography

The application uses a **brutalist design aesthetic** with:

- **Dark Theme**: Background color `#0e1117`, sidebar `#262730`
- **Morganite Font Family**: Used for headers (Black 900, ExtraBold 800, Bold 700)
- **Deliware Font Family**: Used for toggle labels and UI elements
- **Color-Coded Categories**:
  - Hard Skills: Teal (#26a69a)
  - Soft Skills: Purple (#7e57c2)
  - Recruiter Keywords: Orange (#ff9f43)
  - Action Verbs: Red (#ef5350)

### Interactive Features

- **Toggle Switches**: Enable/disable keyword highlighting per category with immediate visual feedback
- **Dynamic Score Gradients**: Self-promotion score block changes color based on score:
  - Green gradient (>0.8): Excellent
  - Yellow/Orange gradient (0.5-0.8): Good
  - Red gradient (<0.5): Needs improvement
- **Progress Bars**: Compact 12px bars showing keyword composition percentages

### Third-Party UI Components

- **streamlit-toggle-switch**: Custom colored toggle switches for category control
- **pdf2docx**: PDF to DOCX conversion (primary extraction)
- **docling**: PDF text extraction (fallback #1)
- **pdfminer.six**: PDF text extraction (fallback #2)

## 🎓 Academic Framing

### Core Components (Thesis Focus)

These are the **two primary academic contributions**:

1. **BERT Embeddings** (`models/embedder.py`)
   - Model: `all-MiniLM-L6-v2`
   - Purpose: Transform sentences into 384-dimensional semantic vectors
   - Academic significance: Contextual understanding beyond keyword matching

2. **KNN Classification** (`models/knn_classifier.py`)
   - Configuration: k=5 neighbors
   - Training: 10,000 labeled resume sentences (4,730 positive / 5,270 negative)
   - Performance: 89.9% accuracy, 91.4% precision, 86.9% recall, 89.1% F1-score
   - Purpose: Classify sentences as self-promotional based on learned patterns
   - Academic significance: Supervised learning for domain-specific detection

### Secondary Components (Stabilization Heuristics)

These are **NOT core thesis components** — they are implementation details for handling resume format variations:

- **SpaCy Highlighting** (`modules/highlight.py`)
  - Context validation to prevent false positives
  - Linguistic pattern detection
  - Framing: "Secondary validation layer using dependency parsing"

- **Heuristic Bonuses** (`modules/scoring.py`)
  - Achievement pattern detection
  - Metric detection
  - Bullet point recognition
  - Sentiment adjustments
  - Framing: "Stabilization heuristics to normalize scores across writing styles"

- **Text Extraction** (`modules/extractor.py`)
  - PDF→DOCX conversion for consistent extraction
  - Aggressive text normalization
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
│ (PDF/DOCX/TXT)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Extraction │ (modules/extractor.py)
│ PDF→DOCX Conv.  │
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

```bash
streamlit run main_modular.py
```

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
from modules.scoring import analyze_sentences
results, avg = analyze_sentences(nlp, knn_model, bert_model, text, action_verbs, file_type)
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

## 📚 Citation in Thesis

When citing the implementation:

> "The system was implemented in Python using three core libraries: 
> SentenceTransformers for BERT embeddings, scikit-learn for KNN classification, 
> and spaCy for linguistic analysis. The codebase was modularized into separate 
> components (models/, modules/) to ensure maintainability and 
> facilitate individual component evaluation."

## ⚠️ Important Notes

### What to Emphasize in Defense

- **Core methodology**: BERT + KNN (machine learning approach)
- **Training data**: ≈10,000 labeled sentences (see `data/self_promotion_dataset.csv`) 
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
- `modules/scoring.py` - Scoring algorithm documentation
- `main_modular.py` - Complete system integration
