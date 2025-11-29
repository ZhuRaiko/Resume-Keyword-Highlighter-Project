# Resume Keyword Highlighter - SkillHighlight Analyzer

An intelligent resume analysis tool that highlights keywords, assesses self-promotion quality, and provides actionable insights for resume optimization.

## Features

- **Smart Keyword Highlighting**: Context-aware detection of Hard Skills, Soft Skills, Recruiter Keywords, and Action Verbs
- **Self-Promotion Analysis**: ML-powered scoring of resume sentences for achievement-oriented language
- **Interactive Dashboard**: Real-time visualization of keyword composition and sentence quality
- **Multi-Format Support**: PDF, DOCX, and TXT file processing
- **Configurable Heuristics**: Sidebar controls for customizing matching strictness

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

- **Total Samples**: 3,013 labeled resume sentences
- **Class Distribution**: 
  - Self-Promotional: 1,758 samples (58.4%)
  - Neutral: 1,254 samples (41.6%)
- **Well-balanced dataset** ensures unbiased model training

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

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

### Run the Streamlit App

```bash
streamlit run main_clean.py
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

## Configuration

Configure matching behavior via sidebar controls:

- **Token-Aligned Mode**: Highlight whole tokens (safer for punctuation)
- **Relax HARD Skills**: Allow without strict noun/object requirements
- **Relax ACTION Verbs**: Allow without direct object requirement
- **Relax SOFT Skills**: Skip negative sentiment suppression
- **Relax RECRUITER Keywords**: Allow in verbless bullets
- **Sentiment Threshold**: Adjust soft skill sentiment filter (-1.0 to 0.0)

## File Structure

```
Resume-Keyword-Highlighter-Project/
├── main_clean.py                    # Main application (production)
├── main.py                          # Original implementation
├── keywords.json                    # Keyword database (4 categories)
├── self_promotion_dataset.csv       # Training data (3,013 samples)
├── evaluate_model.py               # Model evaluation script
├── knn_model.pkl                   # Cached KNN model
├── model_metrics.json              # Evaluation results
└── tests/
    └── highlight_test.py           # Unit tests

```

## Dependencies

- **streamlit** - Web interface
- **sentence-transformers** - BERT embeddings
- **spacy** - NLP pipeline
- **scikit-learn** - KNN classifier
- **textblob** - Sentiment analysis
- **pdfminer.six** - PDF extraction
- **docx2txt** - DOCX extraction
- **pandas, numpy** - Data processing

## Context-Aware Features

### Skill Enumeration Detection
Highlights complete phrases like "Strong communication and leadership skills" as a single unit.

### Context Validation
Uses dependency parsing to prevent false positives:
- "quality" in "quality assurance" → not highlighted alone
- Validates POS tags and grammatical relationships

### Achievement Pattern Recognition
Detects patterns like:
- "Achieved X by doing Y"
- Bullet points with metrics
- Action verbs with measurable results

## Performance Optimizations

- **Single spaCy Pass**: Cached tokens, POS tags, dependencies
- **Lemma-Based Normalization**: Pre-computed keyword maps
- **Longest-First Matching**: Minimizes redundant checks
- **Model Caching**: `@st.cache_resource` for fast loading

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
