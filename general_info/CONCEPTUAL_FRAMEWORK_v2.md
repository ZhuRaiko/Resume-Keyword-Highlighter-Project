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
| **Document Processing** | pdf2docx, Docling, pdfminer, python-docx | Multi-format text extraction |
| **Web Interface** | Streamlit | Interactive user interface |

---

## Research Gaps & Limitations

This section identifies the current gaps and limitations within each phase of the framework, providing transparency about the system's boundaries and opportunities for future research.

### Gap Analysis by Framework Column

```
┌──────────────────────┬───────────────────────────┬────────────────────────────┬──────────────────────────┐
│   DATA & PREPARATION │  REPRESENTATION & ENCODING│   LEARNING & EXTRACTION    │  KNOWLEDGE APPLICATION   │
│        GAPS          │          GAPS             │          GAPS              │          GAPS            │
├──────────────────────┼───────────────────────────┼────────────────────────────┼──────────────────────────┤
│                      │                           │                            │                          │
│  ⚠ Limited to        │  ⚠ Pre-trained model      │  ⚠ KNN is simpler than     │  ⚠ No real recruiter     │
│    English resumes   │    not fine-tuned on      │    deep learning           │    validation of         │
│                      │    resume-specific data   │    alternatives            │    scoring accuracy      │
│  ⚠ PDF extraction    │                           │                            │                          │
│    fallback chain    │  ⚠ 384-dim embeddings     │  ⚠ Binary classification   │  ⚠ No A/B testing with   │
│    may lose          │    may miss nuanced       │    (0/1) loses             │    actual job            │
│    formatting        │    resume semantics       │    granularity             │    applications          │
│                      │                           │                            │                          │
│  ⚠ No image/chart    │  ⚠ Static keyword         │  ⚠ Dataset limited to      │  ⚠ Industry-agnostic     │
│    text extraction   │    database requires      │    10,000 samples          │    scoring (no domain    │
│                      │    manual updates         │                            │    customization)        │
│                      │                           │                            │                          │
│  ⚠ Multi-column      │  ⚠ No multi-lingual       │  ⚠ No continuous           │  ⚠ User feedback not     │
│    layouts may       │    support                │    learning/model          │    incorporated into     │
│    extract poorly    │                           │    updates                 │    model improvement     │
│                      │                           │                            │                          │
└──────────────────────┴───────────────────────────┴────────────────────────────┴──────────────────────────┘
```

### Detailed Gap Descriptions

#### Column 1: Data & Preparation Gaps

| Gap | Description | Impact |
|-----|-------------|--------|
| **Language Limitation** | System only processes English-language resumes | Excludes non-English job markets |
| **PDF Extraction Quality** | Complex PDF layouts (multi-column, tables, graphics) may not extract cleanly | Loss of structural information |
| **No OCR Support** | Scanned PDFs or image-based resumes cannot be processed | Limits input format compatibility |
| **Formatting Loss** | Original resume formatting (bold, italics, bullets) not preserved in analysis | Context from visual hierarchy lost |

#### Column 2: Representation & Encoding Gaps

| Gap | Description | Impact |
|-----|-------------|--------|
| **Generic Pre-trained Model** | all-MiniLM-L6-v2 trained on general text, not resume-specific corpus | May miss domain-specific semantics |
| **Static Keyword Database** | Keywords manually curated; no automatic discovery of emerging terms | Database may become outdated |
| **Fixed Embedding Dimensions** | 384 dimensions may not capture all nuances of professional language | Potential information compression |
| **No Contextual Keyword Weighting** | All keyword categories weighted equally regardless of job type | One-size-fits-all approach |

#### Column 3: Learning & Extraction Gaps

| Gap | Description | Impact |
|-----|-------------|--------|
| **Simple Classifier** | KNN chosen over deep learning (LSTM, transformers) for interpretability | May sacrifice accuracy for simplicity |
| **Binary Labels** | Self-promotion scored as 0 or 1, not on a continuous scale | Loses nuance in borderline cases |
| **Dataset Size** | 10,000 samples sufficient but smaller than industry-scale datasets | May not generalize to all resume styles |
| **No Online Learning** | Model is static; doesn't improve from user interactions | Cannot adapt to new patterns |
| **Single Dataset Source** | Training data from one curated source | May have inherent biases |

#### Column 4: Knowledge Application Gaps

| Gap | Description | Impact |
|-----|-------------|--------|
| **No Recruiter Validation** | Scores not validated against actual recruiter preferences | Uncertain real-world effectiveness |
| **No Outcome Tracking** | Cannot measure if highlighted resumes lead to more interviews | No empirical success metrics |
| **Industry Agnostic** | Same scoring criteria for all industries (tech, healthcare, finance) | May not reflect domain norms |
| **No Personalization** | Cannot adapt recommendations to specific job postings | Generic feedback only |
| **Heuristic Adjustments** | Score adjustments (metrics, achievements) are rule-based, not learned | May not capture all patterns |

---

### Scope Delimitations

The following are **intentional exclusions** from the current research scope:

| Exclusion | Rationale |
|-----------|-----------|
| **Real-time recruiter feedback loop** | Requires industry partnership and longitudinal study |
| **Multi-language support** | Would require separate NLP pipelines and datasets |
| **Job posting matching** | Different research problem (job-resume matching) |
| **ATS compatibility scoring** | Proprietary systems with unknown criteria |
| **Resume generation/rewriting** | Focus is on analysis, not content creation |

---

### Future Research Opportunities

Based on identified gaps, future work could address:

1. **Fine-tuning BERT on resume corpus** - Improve domain-specific embeddings
2. **Multi-class scoring** - Replace binary with 5-point self-promotion scale
3. **Industry-specific models** - Train separate classifiers per domain
4. **Recruiter validation study** - Partner with HR professionals for ground truth
5. **Continuous learning pipeline** - Incorporate user feedback for model updates
6. **Multi-lingual expansion** - Add support for Spanish, Mandarin, etc.
7. **ATS simulation** - Reverse-engineer common ATS scoring patterns

---

*Framework Version: 2.0 | December 2024*
