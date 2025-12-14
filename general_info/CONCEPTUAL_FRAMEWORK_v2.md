
# Conceptual Framework: SkillHighlight Resume Self-Promotion Analyzer (Revised)

```
┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
│   INPUT & PREPROCESSING      │  FEATURE REPRESENTATION      │  CLASSIFICATION & SCORING    │  OUTPUT & EVALUATION         │
├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ ┌─────────────────────────┐  │ ┌─────────────────────────┐  │ ┌─────────────────────────┐  │ ┌─────────────────────────┐  │
│ │ Resume Files (PDF,      │  │ │ Sentence Embeddings     │  │ │ KNN Classifier         │  │ │ Highlighted Resume      │  │
│ │ DOCX, TXT)              │  │ │ (SentenceBERT)          │  │ │ (Self-promotion,       │  │ │ (Color-coded,           │  │
│ └─────────────┬───────────┘  │ │ Keyword Features        │  │ │  Keyword Category)     │  │ │  Interactive)           │  │
│               │              │ │ (JSON, Pattern Rules)   │  │ │ Heuristic Scoring      │  │ │ Model Metrics            │  │
│ ┌─────────────▼───────────┐  │ │ Linguistic Features     │  │ │ Sentiment Adjustment   │  │ │ (Accuracy, F1, etc.)     │  │
│ │ Preprocessing:          │  │ │ (spaCy, POS, NER)       │  │ └─────────────┬─────────┘  │ └─────────────┬───────────┘  │
│ │ Cleaning, Normalization │  │ └─────────────┬──────────┘  │               │            │               │              │
│ │ PDF→DOCX, spaCy NLP     │  │               │             │               │            │               │              │
│ └─────────────┬───────────┘  │               │             │               │            │               │              │
│               │              │               │             │               │            │               │              │
└───────────────┼──────────────┴───────────────┼─────────────┴───────────────┼────────────┴───────────────┘
              │                              │                             │
              ▼                              ▼                             ▼
       ┌─────────────────────────┐    ┌─────────────────────────┐   ┌─────────────────────────┐
       │  Cleaned & Annotated    │    │  Embeddings & Features  │   │  Classification &       │
       │  Text                   │────▶  (Vectors, Keywords)   │───▶  Scoring               │───▶ Output & Evaluation
       └─────────────────────────┘    └─────────────────────────┘   └─────────────────────────┘

    (Dashed arrow from Output & Evaluation back to Classification & Scoring for future feedback loop)
```

---


## Component Details (Aligned to Revised Framework)

### 1. Input & Preprocessing
| Component | Description |
|-----------|-------------|
| **Resume Files** | Input resumes in PDF, DOCX, or TXT formats |
| **Preprocessing** | Cleaning, normalization, PDF→DOCX conversion |
| **Linguistic Annotation** | spaCy NLP pipeline: POS tagging, NER |

### 2. Feature Representation
| Component | Description |
|-----------|-------------|
| **Sentence Embeddings** | SentenceBERT (all-MiniLM-L6-v2), 384-dim vectors |
| **Keyword Features** | JSON-based keyword database, pattern rules |
| **Linguistic Features** | POS, NER, dependency parsing (spaCy) |

### 3. Classification & Scoring
| Component | Description |
|-----------|-------------|
| **KNN Classifier** | scikit-learn KNN (k=5), predicts self-promotion and keyword category |
| **Heuristic Scoring** | Achievement-aware, rule-based adjustments |
| **Sentiment Adjustment** | TextBlob polarity scoring for fine-tuning |

### 4. Output & Evaluation
| Component | Description |
|-----------|-------------|
| **Highlighted Resume** | Color-coded, interactive output with toggles |
| **Model Metrics** | Accuracy: 89.9%, Precision: 91.4%, Recall: 86.9%, F1: 89.1% |
| **User Feedback (Future)** | Potential for feedback loop to improve model |

---


## Data Flow Summary (Linear)

```
Resume Document (PDF/DOCX/TXT)
       │
       ▼
Preprocessing (Cleaning, Normalization, spaCy NLP)
       │
       ▼
Feature Extraction (SentenceBERT Embeddings, Keyword Features)
       │
       ▼
Classification & Scoring (KNN, Heuristics, Sentiment)
       │
       ▼
Output (Highlighted Resume, Metrics)
```

---


## Technology Stack
| Layer | Technology | Role |
|-------|------------|------|
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) | Semantic sentence encoding |
| **Classification** | scikit-learn KNN (k=5) | Self-promotion prediction |
| **NLP Pipeline** | spaCy (en_core_web_sm) | Linguistic analysis & validation |
| **Sentiment** | TextBlob | Polarity scoring for adjustments |
| **Document Processing** | pdf2docx, Docling, pdfminer, python-docx | Multi-format text extraction |
| **Web Interface** | Streamlit | Interactive user interface |

---


## Research Gaps & Limitations (Aligned to Revised Framework)

### 1. Input & Preprocessing
| Gap | Description | Impact |
|-----|-------------|--------|
| **Language Limitation** | Only English resumes supported | Excludes non-English job markets |
| **PDF Extraction Quality** | Complex layouts may not extract cleanly | Loss of structure/context |
| **No OCR Support** | Cannot process scanned/image-based resumes | Limits input compatibility |

### 2. Feature Representation
| Gap | Description | Impact |
|-----|-------------|--------|
| **Generic Pre-trained Model** | Not fine-tuned on resumes | May miss domain-specific meaning |
| **Static Keyword Database** | Manual updates required | May become outdated |
| **Fixed Embedding Dimensions** | 384-dim may not capture all nuances | Information compression risk |

### 3. Classification & Scoring
| Gap | Description | Impact |
|-----|-------------|--------|
| **Simple Classifier** | KNN over deep learning for interpretability | May sacrifice accuracy |
| **Binary Labels** | 0/1 scoring, not continuous | Loses nuance in borderline cases |
| **Dataset Size** | 10,000 samples, not industry-scale | May not generalize fully |
| **No Online Learning** | Model is static | Cannot adapt to new data |

### 4. Output & Evaluation
| Gap | Description | Impact |
|-----|-------------|--------|
| **No Recruiter Validation** | No real-world recruiter feedback | Uncertain effectiveness |
| **No Outcome Tracking** | Cannot measure real-world impact | No empirical success metrics |
| **Industry Agnostic** | Same scoring for all industries | May not reflect domain norms |
| **No Personalization** | No adaptation to job postings | Generic feedback only |

---

### Scope Delimitations
| Exclusion | Rationale |
|-----------|-----------|
| **Real-time recruiter feedback loop** | Requires industry partnership and longitudinal study |
| **Multi-language support** | Would require separate NLP pipelines and datasets |
| **Job posting matching** | Different research problem (job-resume matching) |
| **ATS compatibility scoring** | Proprietary systems with unknown criteria |
| **Resume generation/rewriting** | Focus is on analysis, not content creation |

---

### Future Research Opportunities
1. Fine-tune BERT on resume corpus
2. Multi-class/continuous scoring
3. Industry-specific models
4. Recruiter validation study
5. Continuous learning pipeline
6. Multi-lingual expansion
7. ATS simulation

---

*Framework Version: 2.1 | December 2025 (Revised)*
