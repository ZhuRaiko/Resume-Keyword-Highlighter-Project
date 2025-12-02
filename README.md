# Resume Keyword Highlighter - SkillHighlight Analyzer

An intelligent resume analysis tool that highlights keywords, assesses self-promotion quality, and provides actionable insights for resume optimization. Features a modern **brutalist UI design** with custom typography and color-coded keyword categories.

## Features

- **Smart Keyword Highlighting**: Context-aware detection of Hard Skills, Soft Skills, Recruiter Keywords, and Action Verbs
- **Self-Promotion Analysis**: ML-powered scoring of resume sentences for achievement-oriented language
- **Interactive Dashboard**: Real-time visualization with toggle switches to enable/disable keyword categories
- **Multi-Format Support**: PDF, DOCX, and TXT file processing with PDF→DOCX conversion for consistency
- **Brutalist UI Design**: Custom Morganite and Deliware fonts with dark theme and color-coded elements
- **Dynamic Score Visualization**: Gradient colors that change based on self-promotion score

## UI/UX Features

### Brutalist Design Theme
- **Dark Background**: `#0e1117` with sidebar at `#262730`
- **Custom Typography**: Morganite font for headers (900-500 weights), Deliware for labels
- **Color-Coded Categories**:
  - Hard Skills: Teal (#26a69a)
  - Soft Skills: Purple (#7e57c2)
  - Recruiter Keywords: Orange (#ff9f43)
  - Action Verbs: Red (#ef5350)

### Interactive Controls
- **Toggle Switches**: Enable/disable highlighting per category with immediate visual feedback
- **Progress Bars**: Compact 12px bars showing keyword composition
- **Dynamic Gradients**: Score block changes from green (>0.8) to yellow (0.5-0.8) to red (<0.5)

## Model Performance

Our self-promotion classifier achieves strong performance using a K-Nearest Neighbors approach with BERT embeddings:

### Test Set Performance (80/20 Split)

| Metric | Score |
|--------|-------|
| **Accuracy** | 85.3% |
| **Precision** | 82.7% |
| **Recall** | 88.1% |
| **F1-Score** | 85.3% |

### Cross-Validation Results (5-Fold)

- **Mean F1-Score**: 0.839 (±0.024)
- Demonstrates consistent performance across different data subsets

### Dataset Statistics

- **Total Samples**: 6,752 labeled resume sentences
- **Class Distribution**: 
  - Self-Promotional: 2,233 samples (33.1%)
  - Neutral/Descriptive: 4,519 samples (66.9%)
- **Larger dataset** with diverse resume writing styles for robust training

### Model Configuration

- **Algorithm**: K-Nearest Neighbors (KNN)
- **K Value**: 5 neighbors
- **Embedding Model**: SentenceTransformer (all-MiniLM-L6-v2)
- **Embedding Dimension**: 384
- **Distance Metric**: Euclidean (default)

## Technical Architecture

### Models & Algorithms (7 Total)

#### Deep Learning (2)
1. **SentenceTransformer (all-MiniLM-L6-v2)** - Sentence embeddings
   - Time Complexity: O(n·d) per sentence
2. **spaCy en_core_web_sm** - NLP pipeline (POS, dependency parsing)
   - Time Complexity: O(n²) for dependency parsing

#### Traditional ML (1)
3. **KNeighborsClassifier** - Self-promotion classification
   - Training: O(m·d) | Prediction: O(m·d) per query

#### Rule-Based Algorithms (4)
4. **Regex Keyword Matching** - Context-aware keyword detection
   - Time Complexity: O(n·k·L)
5. **TextBlob Sentiment** - Sentiment polarity scoring
   - Time Complexity: O(n)
6. **Achievement Pattern Detection** - Action word + impact detection
   - Time Complexity: O(n)
7. **File Extraction** - PDF/DOCX parsing (pdfminer, docx2txt)
   - Time Complexity: O(document size)

### Overall Pipeline Complexity
- **Per-Resume Analysis**: O(n²) dominated by spaCy dependency parsing
- **Typical Resume**: 200-500 tokens, 20-50 sentences

## Installation

```bash
# Clone the repository
git clone https://github.com/ZhuRaiko/Resume-Keyword-Highlighter-Project.git
cd Resume-Keyword-Highlighter-Project

# Install dependencies
pip install -r requirements.txt

# Additional packages for UI and PDF processing
pip install streamlit-toggle-switch pdf2docx

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

### Run the Streamlit App

```bash
streamlit run main_modular.py
```

### Evaluate Model Performance

```bash
python evaluate_model.py
```

This generates detailed metrics including:
- Test set performance
- Cross-validation scores
- Confusion matrix
- Classification report
- Saves results to `model_metrics.json`

## Project Structure

```
Resume-Keyword-Highlighter-Project/
├── main_modular.py              # Main Streamlit application
├── evaluate_model.py            # Model evaluation script
│
├── data/
│   ├── keywords.json            # Keyword database (4 categories)
│   └── self_promotion_dataset.csv  # Training data (6,752 samples)
│
├── fonts/
│   ├── Morganite-*.ttf          # Morganite font family
│   └── deliware-*.otf           # Deliware font family
│
├── models/
│   ├── embedder.py              # BERT embeddings
│   ├── knn_classifier.py        # KNN classification
│   ├── knn_model.pkl            # Cached model
│   └── sentiment.py             # Sentiment analysis
│
├── modules/
│   ├── counters.py              # Keyword metrics
│   ├── embeddings.py            # Embedding utilities
│   ├── extractor.py             # Text extraction (PDF→DOCX)
│   ├── highlight.py             # Keyword highlighting
│   └── scoring.py               # Self-promotion scoring
│
└── backups/                     # Legacy versions
```

## Dependencies

- **streamlit** - Web interface
- **streamlit-toggle-switch** - Custom toggle switches
- **sentence-transformers** - BERT embeddings
- **spacy** - NLP pipeline
- **scikit-learn** - KNN classifier
- **textblob** - Sentiment analysis
- **pdf2docx** - PDF to DOCX conversion
- **python-docx** - DOCX processing
- **pdfminer.six** - PDF text extraction (fallback)
- **pandas, numpy** - Data processing

## Key Features Explained

### PDF→DOCX Conversion
For consistent text extraction across formats, PDFs are converted to DOCX before processing. This ensures identical scoring results regardless of input format.

### Toggle-Based Highlighting
Each keyword category can be independently enabled/disabled with immediate visual feedback. Toggles trigger a page rerun to update highlighting in real-time.

### Dynamic Score Gradients
The self-promotion score block uses color gradients to communicate quality:
- **Green** (>0.8): Excellent self-promotion
- **Yellow/Orange** (0.5-0.8): Good, with room for improvement
- **Red** (<0.5): Needs stronger achievement language

## Future Enhancements

- [ ] Export highlighted resumes (HTML/PDF)
- [ ] ATS keyword matching score
- [ ] Industry-specific keyword sets
- [ ] Resume format validation
- [ ] Job description comparison mode

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Authors

- **ZhuRaiko** - [GitHub Profile](https://github.com/ZhuRaiko)

## Acknowledgments

- spaCy for NLP infrastructure
- Sentence-Transformers for embedding models
- Streamlit for the web framework
