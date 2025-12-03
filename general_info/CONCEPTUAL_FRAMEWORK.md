# Conceptual Framework: SkillHighlight Resume Self-Promotion Analyzer

## Overview

This conceptual framework illustrates the theoretical foundation and system architecture of the SkillHighlight Resume Self-Promotion Analyzer—an intelligent system that combines machine learning, natural language processing, and rule-based analysis to evaluate and enhance resume quality.

---

## 1. Theoretical Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THEORETICAL FOUNDATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐     │
│   │  SELF-PROMOTION  │    │    SEMANTIC      │    │   SUPERVISED     │     │
│   │      THEORY      │    │   UNDERSTANDING  │    │    LEARNING      │     │
│   ├──────────────────┤    ├──────────────────┤    ├──────────────────┤     │
│   │ • Achievement-   │    │ • Contextual     │    │ • Pattern        │     │
│   │   oriented       │    │   word meaning   │    │   recognition    │     │
│   │   language       │    │ • Sentence-level │    │ • Classification │     │
│   │ • Action verbs   │    │   semantics      │    │   from labeled   │     │
│   │ • Quantifiable   │    │ • Beyond keyword │    │   examples       │     │
│   │   impact         │    │   matching       │    │ • Generalization │     │
│   └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘     │
│            │                       │                       │                │
│            └───────────────────────┼───────────────────────┘                │
│                                    ▼                                        │
│                    ┌───────────────────────────────┐                        │
│                    │   INTEGRATED RESUME ANALYSIS  │                        │
│                    │         METHODOLOGY           │                        │
│                    └───────────────────────────────┘                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. System Architecture (IPO Model)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INPUT → PROCESS → OUTPUT MODEL                            │
└─────────────────────────────────────────────────────────────────────────────┘

╔═══════════════╗      ╔════════════════════════════╗      ╔════════════════╗
║     INPUT     ║      ║         PROCESS            ║      ║     OUTPUT     ║
╠═══════════════╣      ╠════════════════════════════╣      ╠════════════════╣
║               ║      ║                            ║      ║                ║
║  Resume File  ║      ║  ┌──────────────────────┐  ║      ║  Self-Promo    ║
║  • PDF        ║ ───► ║  │   Text Extraction    │  ║      ║  Score         ║
║  • DOCX       ║      ║  │   & Normalization    │  ║      ║  (0.0 - 1.0)   ║
║  • TXT        ║      ║  └──────────┬───────────┘  ║      ║                ║
║               ║      ║             │              ║      ╠════════════════╣
║  ─────────    ║      ║             ▼              ║      ║                ║
║               ║      ║  ┌──────────────────────┐  ║      ║  Highlighted   ║
║  Text Input   ║      ║  │   BERT Embeddings    │  ║ ───► ║  Keywords      ║
║  (paste)      ║      ║  │   (384 dimensions)   │  ║      ║  • Hard Skills ║
║               ║      ║  └──────────┬───────────┘  ║      ║  • Soft Skills ║
║  ─────────    ║      ║             │              ║      ║  • Recruiter   ║
║               ║      ║             ▼              ║      ║  • Action Verbs║
║  Keyword      ║      ║  ┌──────────────────────┐  ║      ║                ║
║  Database     ║      ║  │   KNN Classification │  ║      ╠════════════════╣
║  (JSON)       ║      ║  │   (k=5 neighbors)    │  ║      ║                ║
║               ║      ║  └──────────┬───────────┘  ║      ║  Sentence-by-  ║
║               ║      ║             │              ║      ║  Sentence      ║
║               ║      ║             ▼              ║      ║  Analysis      ║
║               ║      ║  ┌──────────────────────┐  ║      ║                ║
║               ║      ║  │  SpaCy NLP Pipeline  │  ║      ╠════════════════╣
║               ║      ║  │  + Keyword Matching  │  ║      ║                ║
║               ║      ║  └──────────────────────┘  ║      ║  Keyword       ║
║               ║      ║                            ║      ║  Composition   ║
║               ║      ║                            ║      ║  Percentages   ║
║               ║      ║                            ║      ║                ║
╚═══════════════╝      ╚════════════════════════════╝      ╚════════════════╝
```

---

## 3. Core Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROCESSING PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                              RESUME INPUT
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │    DOCUMENT EXTRACTION      │
                    │    ────────────────────     │
                    │    • PDF → DOCX conversion  │
                    │    • Text normalization     │
                    │    • Unicode handling       │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │    SENTENCE TOKENIZATION    │
                    │    ────────────────────     │
                    │    • SpaCy sentence split   │
                    │    • Bullet point handling  │
                    │    • Resume format parsing  │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│   BERT EMBEDDING    │ │   SPACY ANALYSIS    │ │  KEYWORD MATCHING   │
│   ───────────────   │ │   ──────────────    │ │  ────────────────   │
│                     │ │                     │ │                     │
│ • all-MiniLM-L6-v2  │ │ • POS Tagging       │ │ • Hard Skills       │
│ • 384-dim vectors   │ │ • Dependency Parse  │ │ • Soft Skills       │
│ • Semantic meaning  │ │ • NER Recognition   │ │ • Recruiter KWs     │
│ • Context capture   │ │ • Lemmatization     │ │ • Action Verbs      │
│                     │ │                     │ │                     │
└──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
           │                       │                       │
           ▼                       ▼                       ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│  KNN CLASSIFICATION │ │ CONTEXT VALIDATION  │ │  KEYWORD COUNTING   │
│  ─────────────────  │ │ ──────────────────  │ │  ────────────────   │
│                     │ │                     │ │                     │
│ • k=5 neighbors     │ │ • Linguistic rules  │ │ • Category totals   │
│ • 10,000 samples    │ │ • Sentiment check   │ │ • Percentage calc   │
│ • 89.9% accuracy    │ │ • False + filtering │ │ • Composition       │
│                     │ │                     │ │                     │
└──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │      SCORE AGGREGATION      │
                    │      ─────────────────      │
                    │    • Heuristic bonuses      │
                    │    • Sentiment adjustment   │
                    │    • Final score (0-1)      │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │     VISUALIZATION OUTPUT    │
                    │     ───────────────────     │
                    │    • Highlighted text       │
                    │    • Score gradient         │
                    │    • Sentence breakdown     │
                    └─────────────────────────────┘
```

---

## 4. Machine Learning Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   COMPONENT #1: BERT SENTENCE EMBEDDINGS                                    │
│   ══════════════════════════════════════                                    │
│                                                                              │
│   Model: all-MiniLM-L6-v2 (SentenceTransformers)                            │
│                                                                              │
│   ┌─────────────┐         ┌─────────────────┐         ┌─────────────┐       │
│   │   Input     │         │   Transformer   │         │   Output    │       │
│   │  Sentence   │  ─────► │     Layers      │  ─────► │   Vector    │       │
│   │   (text)    │         │   (6 layers)    │         │  (384-dim)  │       │
│   └─────────────┘         └─────────────────┘         └─────────────┘       │
│                                                                              │
│   Purpose: Convert sentences into dense semantic representations            │
│   Significance: Captures contextual meaning beyond keyword matching         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   COMPONENT #2: KNN CLASSIFIER                                              │
│   ════════════════════════════                                              │
│                                                                              │
│   Algorithm: K-Nearest Neighbors (k=5)                                      │
│                                                                              │
│                        Training Data                                         │
│                     ┌─────────────────┐                                     │
│                     │  10,000 Labeled │                                     │
│                     │    Sentences    │                                     │
│                     │                 │                                     │
│                     │ • 4,730 Pos     │                                     │
│                     │ • 5,270 Neg     │                                     │
│                     └────────┬────────┘                                     │
│                              │                                              │
│                              ▼                                              │
│          ┌───────────────────────────────────────┐                          │
│          │         KNN Decision Process          │                          │
│          │                                       │                          │
│          │   Query ──► Find 5 Nearest ──► Vote   │                          │
│          │   Point     Neighbors          Label  │                          │
│          │                                       │                          │
│          │         ●                             │                          │
│          │        /│\      ● = Query             │                          │
│          │       / │ \     ○ = Class 0 (Neutral) │                          │
│          │      ○  ●  ●    ● = Class 1 (Self-P)  │                          │
│          │         │                             │                          │
│          │         ○       Result: 3/5 = Class 1 │                          │
│          │                                       │                          │
│          └───────────────────────────────────────┘                          │
│                                                                              │
│   Performance Metrics:                                                       │
│   ┌─────────────┬─────────────┬─────────────┬─────────────┐                 │
│   │  Accuracy   │  Precision  │   Recall    │  F1-Score   │                 │
│   │   89.9%     │    91.4%    │    86.9%    │    89.1%    │                 │
│   └─────────────┴─────────────┴─────────────┴─────────────┘                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Keyword Classification System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      KEYWORD CLASSIFICATION SYSTEM                           │
└─────────────────────────────────────────────────────────────────────────────┘

                           RESUME TEXT
                               │
                               ▼
              ┌────────────────────────────────┐
              │      SpaCy NLP Processing      │
              │   • Tokenization               │
              │   • POS Tagging                │
              │   • Dependency Parsing         │
              │   • Named Entity Recognition   │
              └────────────────┬───────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       │                       │                       │
       ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│   Pattern    │       │   Context    │       │  Sentiment   │
│   Matching   │       │  Validation  │       │   Analysis   │
│              │       │              │       │              │
│ • Regex      │       │ • Noun check │       │ • TextBlob   │
│ • Lemma      │       │ • Verb deps  │       │ • Polarity   │
│ • Token      │       │ • Structure  │       │ • Threshold  │
└──────┬───────┘       └──────┬───────┘       └──────┬───────┘
       │                      │                      │
       └──────────────────────┼──────────────────────┘
                              │
                              ▼
              ┌────────────────────────────────┐
              │        KEYWORD CATEGORIES      │
              └────────────────────────────────┘

    ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
    │  HARD SKILLS   │  │  SOFT SKILLS   │  │   RECRUITER    │  │  ACTION VERBS  │
    │    (Teal)      │  │   (Purple)     │  │    (Orange)    │  │     (Red)      │
    ├────────────────┤  ├────────────────┤  ├────────────────┤  ├────────────────┤
    │                │  │                │  │                │  │                │
    │ • Python       │  │ • Leadership   │  │ • Experienced  │  │ • Developed    │
    │ • JavaScript   │  │ • Teamwork     │  │ • Proficient   │  │ • Implemented  │
    │ • Machine      │  │ • Communication│  │ • Expert       │  │ • Designed     │
    │   Learning     │  │ • Problem      │  │ • Skilled      │  │ • Led          │
    │ • SQL          │  │   Solving      │  │ • Results-     │  │ • Achieved     │
    │ • Cloud        │  │ • Adaptability │  │   driven       │  │ • Increased    │
    │   Computing    │  │ • Creativity   │  │ • Self-starter │  │ • Optimized    │
    │                │  │                │  │                │  │                │
    └────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘
```

---

## 6. Scoring Algorithm

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCORING ALGORITHM                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                         Input Sentence
                              │
                              ▼
                 ┌────────────────────────┐
                 │   BERT EMBEDDING       │
                 │   (384-dimensional)    │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │   KNN PREDICTION       │
                 │   Base Score: 0.0-1.0  │
                 └───────────┬────────────┘
                             │
                             ▼
    ┌────────────────────────────────────────────────────┐
    │              HEURISTIC ADJUSTMENTS                 │
    │                                                    │
    │  ┌────────────────────────────────────────────┐   │
    │  │  Achievement Pattern Bonus (+0.10)         │   │
    │  │  • Action verb + measurable result         │   │
    │  │  • "Increased sales by 30%"                │   │
    │  └────────────────────────────────────────────┘   │
    │                                                    │
    │  ┌────────────────────────────────────────────┐   │
    │  │  Metric Detection Bonus (+0.05)            │   │
    │  │  • Numbers, percentages, quantities        │   │
    │  │  • "$1M", "50%", "100+ users"              │   │
    │  └────────────────────────────────────────────┘   │
    │                                                    │
    │  ┌────────────────────────────────────────────┐   │
    │  │  Bullet Point Bonus (+0.05)                │   │
    │  │  • Proper resume formatting                │   │
    │  │  • "• Led team of 5 engineers"             │   │
    │  └────────────────────────────────────────────┘   │
    │                                                    │
    │  ┌────────────────────────────────────────────┐   │
    │  │  Sentiment Adjustment (±0.05)              │   │
    │  │  • Positive language boost                 │   │
    │  │  • Negative language penalty               │   │
    │  └────────────────────────────────────────────┘   │
    │                                                    │
    └────────────────────────────────────────────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │   FINAL SCORE          │
                 │   Clamped: 0.0 - 1.0   │
                 └───────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  > 0.8   │  │ 0.5-0.8  │  │  < 0.5   │
        │  STRONG  │  │ MODERATE │  │   WEAK   │
        │  (Green) │  │ (Yellow) │  │  (Red)   │
        └──────────┘  └──────────┘  └──────────┘
```

---

## 7. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA FLOW DIAGRAM                                 │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────┐                                              ┌─────────────┐
    │  USER   │                                              │  KEYWORDS   │
    │         │                                              │    JSON     │
    └────┬────┘                                              └──────┬──────┘
         │                                                          │
         │ Upload/Paste                                             │ Load
         ▼                                                          ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │                         STREAMLIT APPLICATION                        │
    │                            (main_modular.py)                         │
    │                                                                      │
    │  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐       │
    │  │   extractor    │   │   embeddings   │   │    scoring     │       │
    │  │    module      │──►│     module     │──►│     module     │       │
    │  │                │   │                │   │                │       │
    │  │ • PDF parse    │   │ • Load BERT    │   │ • KNN predict  │       │
    │  │ • DOCX parse   │   │ • Load SpaCy   │   │ • Heuristics   │       │
    │  │ • Normalize    │   │ • Encode text  │   │ • Aggregate    │       │
    │  └────────────────┘   └────────────────┘   └────────────────┘       │
    │           │                   │                    │                 │
    │           │                   ▼                    │                 │
    │           │           ┌────────────────┐          │                 │
    │           │           │   highlight    │◄─────────┘                 │
    │           │           │    module      │                            │
    │           │           │                │                            │
    │           │           │ • SpaCy NLP    │                            │
    │           │           │ • Context      │                            │
    │           │           │ • HTML render  │                            │
    │           │           └────────────────┘                            │
    │           │                   │                                      │
    │           │                   ▼                                      │
    │           │           ┌────────────────┐                            │
    │           └──────────►│   counters     │                            │
    │                       │    module      │                            │
    │                       │                │                            │
    │                       │ • Count KWs    │                            │
    │                       │ • Percentages  │                            │
    │                       └────────────────┘                            │
    │                                                                      │
    └──────────────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼
                               ┌───────────────┐
                               │    OUTPUT     │
                               │               │
                               │ • Score       │
                               │ • Highlights  │
                               │ • Analysis    │
                               └───────────────┘
```

---

## 8. Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODULE DEPENDENCIES                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                            main_modular.py
                                  │
          ┌───────────┬───────────┼───────────┬───────────┐
          │           │           │           │           │
          ▼           ▼           ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │extractor │ │embeddings│ │ scoring  │ │highlight │ │ counters │
    │  .py     │ │   .py    │ │   .py    │ │   .py    │ │   .py    │
    └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────────┘
         │            │            │            │
         │            ▼            │            ▼
         │     ┌──────────┐        │     ┌──────────┐
         │     │ embedder │        │     │  SpaCy   │
         │     │   .py    │        │     │  Model   │
         │     └────┬─────┘        │     └──────────┘
         │          │              │
         │          ▼              ▼
         │   ┌────────────┐  ┌──────────────┐
         │   │  Sentence  │  │knn_classifier│
         │   │Transformers│  │     .py      │
         │   └────────────┘  └──────┬───────┘
         │                          │
         ▼                          ▼
    ┌──────────┐              ┌──────────┐
    │ pdf2docx │              │ sklearn  │
    │ pdfminer │              │   KNN    │
    │  docx    │              │          │
    └──────────┘              └──────────┘


    EXTERNAL DEPENDENCIES:
    ─────────────────────
    • streamlit           - Web UI framework
    • sentence-transformers - BERT embeddings
    • spacy               - NLP pipeline
    • scikit-learn        - KNN classifier
    • textblob            - Sentiment analysis
    • pdf2docx            - PDF conversion (primary)
    • Docling             - PDF extraction (fallback #1)
    • pdfminer.six        - PDF extraction (fallback #2)
    • python-docx         - DOCX processing
```

---

## 9. User Interaction Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌─────────────────┐
    │  Open Web App   │
    │  (Streamlit)    │
    └────────┬────────┘
         │
         ▼
    ┌─────────────────┐      ┌─────────────────┐
    │ Upload Resume   │      │  Paste Resume   │
    │ (PDF/DOCX/TXT)  │  OR  │     Text        │
    └────────┬────────┘      └────────┬────────┘
             │                        │
             └───────────┬────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  System Processing   │
              │  ──────────────────  │
              │  • Extract text      │
              │  • Analyze sentences │
              │  • Classify keywords │
              │  • Calculate scores  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   View Results       │
              │   ────────────       │
              │  • Highlighted text  │
              │  • Self-promo score  │
              │  • Keyword breakdown │
              │  • Sentence analysis │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Toggle Categories   │◄─────┐
              │  ────────────────    │      │
              │  • Hard Skills ON/OFF│      │
              │  • Soft Skills ON/OFF│      │ Interactive
              │  • Recruiter  ON/OFF │      │ Loop
              │  • Action     ON/OFF │──────┘
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Adjust Settings     │◄─────┐
              │  ───────────────     │      │
              │  • Relaxation modes  │      │ Interactive
              │  • Sentiment thresh  │      │ Loop
              │  • Rendering style   │──────┘
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Interpret Results   │
              │  ─────────────────   │
              │  • Strong (>0.8)     │
              │  • Moderate (0.5-0.8)│
              │  • Weak (<0.5)       │
              └──────────┬───────────┘
                         │
                         ▼
                    ┌─────────┐
                    │   END   │
                    └─────────┘
```

---

## 10. Summary Table

| Component | Technology | Purpose | Academic Significance |
|-----------|------------|---------|----------------------|
| **Sentence Embeddings** | BERT (all-MiniLM-L6-v2) | Convert text to 384-dim vectors | Semantic understanding beyond keywords |
| **Classification** | KNN (k=5) | Predict self-promotion score | Supervised learning on labeled data |
| **NLP Pipeline** | spaCy (en_core_web_sm) | POS tagging, dependency parsing | Linguistic context validation |
| **Sentiment** | TextBlob | Polarity scoring | Language tone adjustment |
| **Text Extraction** | pdf2docx, Docling, pdfminer | Document parsing | Format normalization |
| **UI Framework** | Streamlit | Interactive web interface | User accessibility |

---

## 11. Research Variables

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RESEARCH VARIABLES                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    INDEPENDENT VARIABLES                    DEPENDENT VARIABLES
    ─────────────────────                    ───────────────────

    ┌─────────────────────┐                  ┌─────────────────────┐
    │  Resume Content     │                  │  Self-Promotion     │
    │  • Sentence text    │ ───────────────► │  Score (0.0 - 1.0)  │
    │  • Writing style    │                  │                     │
    │  • Achievement      │                  └─────────────────────┘
    │    statements       │
    └─────────────────────┘                  ┌─────────────────────┐
                                             │  Keyword Detection  │
    ┌─────────────────────┐                  │  • Hard skills %    │
    │  Model Parameters   │ ───────────────► │  • Soft skills %    │
    │  • k value (5)      │                  │  • Recruiter kws %  │
    │  • Embedding model  │                  │  • Action verbs %   │
    │  • Heuristic weights│                  └─────────────────────┘
    └─────────────────────┘
                                             ┌─────────────────────┐
    ┌─────────────────────┐                  │  Classification     │
    │  Training Data      │ ───────────────► │  Accuracy           │
    │  • 10,000 samples   │                  │  • 89.9% accuracy   │
    │  • Label quality    │                  │  • 91.4% precision  │
    │  • Class balance    │                  │  • 86.9% recall     │
    └─────────────────────┘                  └─────────────────────┘


    CONTROL VARIABLES
    ─────────────────

    ┌─────────────────────────────────────────────────────────────────────┐
    │  • Random state seed (42) for reproducibility                       │
    │  • Train/test split ratio (80/20)                                   │
    │  • BERT model version (all-MiniLM-L6-v2)                           │
    │  • SpaCy model version (en_core_web_sm)                            │
    │  • Heuristic bonus weights (fixed)                                  │
    └─────────────────────────────────────────────────────────────────────┘
```

---

## Citation

When referencing this framework in academic writing:

> "The SkillHighlight system employs a dual-component machine learning architecture: (1) BERT sentence embeddings (all-MiniLM-L6-v2) for semantic representation, and (2) K-Nearest Neighbors classification (k=5) trained on 10,000 labeled resume sentences. The system achieves 89.9% accuracy in self-promotion detection, with secondary SpaCy-based linguistic validation for context-aware keyword highlighting."

---

*Framework Version: 1.0*  
*Last Updated: December 2024*
