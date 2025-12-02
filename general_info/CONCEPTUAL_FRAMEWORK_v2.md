# Conceptual Framework: SkillHighlight Resume Self-Promotion Analyzer

```
┌──────────────────────┬───────────────────────────┬────────────────────────────┬──────────────────────────┐
│   DATA & PREPARATION │  REPRESENTATION & ENCODING│   LEARNING & EXTRACTION    │  KNOWLEDGE APPLICATION   │
├──────────────────────┼───────────────────────────┼────────────────────────────┼──────────────────────────┤
│                      │                           │                            │                          │
│  ┌────────────────┐  │  ┌─────────────────────┐  │  ┌──────────────────────┐  │  ┌────────────────────┐  │
│  │     DATA       │  │  │ FEATURE EXTRACTION  │  │  │  TRANSFORMER KERNEL  │  │  │   SELF-PROMOTION   │  │
│  │    Resume      │  │  │     EMBEDDING       │  │  │                      │  │  │      SCORING       │  │
│  │    Corpus      │  │  │                     │  │  │  Contextual encoder  │  │  │                    │  │
│  │                │  │  │  SentenceBERT       │  │  │  Domain-adapted      │  │  │  Achievement-aware │  │
│  │  • PDF files   │  │  │  Contextual encoding│  │  │  resume signals      │  │  │  Q&A guidance      │  │
│  │  • DOCX files  │  │  │  Domain-adapted     │  │  │                      │  │  │                    │  │
│  │  • TXT input   │  │  │  resume embeddings  │  │  └──────────┬───────────┘  │  └─────────┬──────────┘  │
│  └───────┬────────┘  │  └──────────┬──────────┘  │             │              │            │             │
│          │           │             │             │             ▼              │            │             │
│          ▼           │             │             │  ┌──────────────────────┐  │            │             │
│  ┌────────────────┐  │             │             │  │ CONCEPT DETECTION    │  │            ▼             │
│  │ PREPROCESSING  │  │             │             │  │                      │  │  ┌────────────────────┐  │
│  │                │  │             │             │  │  Token span +        │  │  │    EVALUATION      │  │
│  │  Cleaning,     │  │             │             │  │  keyword classes     │  │  │                    │  │
│  │  Text parsing, │  │             │             │  │  Max-margin with     │  │  │  Accuracy: 89.9%   │  │
│  │  PDF→DOCX,     │  │             │             │  │  class weighting     │  │  │  Precision: 91.4%  │  │
│  │  Normalization │  │             │             │  │                      │  │  │  Recall: 86.9%     │  │
│  │                │  │             │             │  └──────────┬───────────┘  │  │  F1-Score: 89.1%   │  │
│  └───────┬────────┘  │             │             │             │              │  │                    │  │
│          │           │             │             │             ▼              │  └────────────────────┘  │
│          ▼           │             ▼             │  ┌──────────────────────┐  │                          │
│  ┌────────────────┐  │  ┌─────────────────────┐  │  │ KEYWORD              │  │                          │
│  │  LINGUISTIC &  │  │  │       LABEL         │  │  │ CLASSIFICATION       │  │                          │
│  │  ANNOTATION    │  │  │                     │  │  │                      │  │                          │
│  │                │  │  │  Keyword Classes &  │  │  │  • Hard Skills       │──┼──────────────────────┐   │
│  │  SpaCy NLP,    │  │  │  Self-Promotion     │  │  │  • Soft Skills       │  │                      │   │
│  │  POS Tagging,  │  │  │  Score Type         │  │  │  • Recruiter KWs     │  │                      ▼   │
│  │  Dependency    │  │  │                     │  │  │  • Action Verbs      │  │  ┌────────────────────┐  │
│  │  Parsing,      │  │  │  • 0 = Neutral      │  │  │                      │  │  │   HIGHLIGHTED      │  │
│  │  NER           │  │  │  • 1 = Self-Promo   │  │  │  Uses SpaCy NLP      │  │  │   RESUME OUTPUT    │  │
│  │                │  │  │                     │  │  │  for validation      │  │  │                    │  │
│  └───────┬────────┘  │  └──────────┬──────────┘  │  │                      │  │  │  Color-coded       │  │
│          │           │             │             │  └──────────┬───────────┘  │  │  keyword display   │  │
│          │           │             │             │             │              │  │  with toggles      │  │
│          │           │             ▼             │             ▼              │  │                    │  │
│          │           │  ┌─────────────────────┐  │  ┌──────────────────────┐  │  └────────────────────┘  │
│          │           │  │ KEYWORD FEATURES    │  │  │ TRAINING & TUNING    │  │                          │
│          │           │  │                     │  │  │                      │  │                          │
│          │           │  │  Keyword Database   │  │  │  Contextual encoder  │  │                          │
│          │           │  │  (JSON)             │  │  │  Domain-adapted      │  │                          │
│          │           │  │                     │  │  │  resume signals      │  │                          │
│          │           │  │  • 4 Categories     │  │  │                      │  │                          │
│          │           │  │  • Pattern rules    │  │  │  KNN (k=5)           │  │                          │
│          │           │  │  • Context filters  │  │  │  10,000 samples      │  │                          │
│          │           │  │                     │  │  │                      │  │                          │
│          │           │  └─────────────────────┘  │  └──────────────────────┘  │                          │
│          │           │                           │                            │                          │
└──────────┼───────────┴───────────────────────────┴────────────────────────────┴──────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                          │
│   CLEANED TOKENS/SPANS          EMBEDDINGS, LABELED SPANS         PREDICTED KEYWORDS AND                 │
│   + LINGUISTIC FEATURES    ───► AND KEYWORD FEATURES          ───► SELF-PROMOTION SCORES                 │
│   READY FOR ENCODING            READY FOR CLASSIFICATION           (STRUCTURED RESULTS)                  │
│                                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Column 1: DATA & PREPARATION

| Component | Description |
|-----------|-------------|
| **DATA** | Resume corpus input (PDF, DOCX, TXT formats) |
| **PREPROCESSING** | Text cleaning, PDF→DOCX conversion, Unicode normalization, whitespace handling |
| **LINGUISTIC & ANNOTATION** | SpaCy NLP pipeline for POS tagging, dependency parsing, and Named Entity Recognition |

### Column 2: REPRESENTATION & ENCODING

| Component | Description |
|-----------|-------------|
| **FEATURE EXTRACTION EMBEDDING** | SentenceBERT (all-MiniLM-L6-v2) for 384-dimensional contextual embeddings |
| **LABEL** | Binary classification labels (0=Neutral, 1=Self-Promotional) for training |
| **KEYWORD FEATURES** | JSON-based keyword database with 4 categories and linguistic context rules |

### Column 3: LEARNING & EXTRACTION

| Component | Description |
|-----------|-------------|
| **TRANSFORMER KERNEL** | BERT-based contextual encoder adapted for resume domain signals |
| **CONCEPT DETECTION** | Token span identification with keyword class weighting |
| **KEYWORD CLASSIFICATION** | SpaCy-validated classification into Hard Skills, Soft Skills, Recruiter Keywords, Action Verbs |
| **TRAINING & TUNING** | KNN classifier (k=5) trained on 10,000 labeled resume sentences |

### Column 4: KNOWLEDGE APPLICATION

| Component | Description |
|-----------|-------------|
| **SELF-PROMOTION SCORING** | Achievement-aware scoring with heuristic adjustments for metrics, patterns, and sentiment |
| **EVALUATION** | Model performance: 89.9% accuracy, 91.4% precision, 86.9% recall, 89.1% F1-score |
| **HIGHLIGHTED RESUME OUTPUT** | Interactive color-coded display with category toggles and composition percentages |

---

## Data Flow Summary

```
INPUT                           PROCESSING                        OUTPUT
─────                           ──────────                        ──────

Resume Document          ───►   Text Extraction           ───►   Self-Promotion Score
(PDF/DOCX/TXT)                  & Normalization                  (0.0 - 1.0)

                         ───►   BERT Encoding             ───►   Highlighted Keywords
                                (384-dim vectors)                (4 color-coded categories)

                         ───►   KNN Classification        ───►   Sentence-by-Sentence
                                (k=5, 10K samples)               Analysis & Feedback
```

---

## Technology Stack

| Layer | Technology | Role |
|-------|------------|------|
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) | Semantic sentence encoding |
| **Classification** | scikit-learn KNN (k=5) | Self-promotion prediction |
| **NLP Pipeline** | spaCy (en_core_web_sm) | Linguistic analysis & validation |
| **Sentiment** | TextBlob | Polarity scoring for adjustments |
| **Document Processing** | pdf2docx, pdfminer, python-docx | Multi-format text extraction |
| **Web Interface** | Streamlit | Interactive user interface |

---

*Framework Version: 2.0 | December 2024*
