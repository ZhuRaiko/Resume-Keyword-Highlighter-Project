# SkillHighlight Resume Analyzer: Process, Models, Features, and Evaluation Explained

This document provides a comprehensive, defense-ready explanation of the SkillHighlight Resume Analyzer. It covers the end-to-end process, the role of each model and feature, and the meaning of all evaluation metrics. Use this as a reference for thesis defense, technical Q&A, or onboarding new contributors.

---

## 1. End-to-End Process Overview

### Step 1: Input & Preprocessing
- **Input:** User uploads a resume (PDF, DOCX, or TXT).
- **Preprocessing:**
  - **PDF→DOCX Conversion:** Ensures consistent text extraction across formats.
  - **Text Cleaning:** Removes extraneous whitespace, normalizes Unicode, and prepares text for analysis.
  - **Linguistic Annotation:** Uses spaCy to perform part-of-speech (POS) tagging, dependency parsing, and named entity recognition (NER), enriching the text with linguistic features.

### Step 2: Feature Representation
- **Sentence Embeddings:** Each sentence is converted into a 384-dimensional vector using SentenceTransformer (all-MiniLM-L6-v2), capturing semantic meaning.
- **Keyword Features:** Extracts context-aware features using a curated JSON database of keywords (Hard Skills, Soft Skills, Recruiter Keywords, Action Verbs) and pattern rules.
- **Linguistic Features:** spaCy provides POS tags, NER labels, and syntactic dependencies for each token.

### Step 3: Classification & Scoring
- **KNN Classifier:** Predicts whether each sentence is self-promotional or neutral using K-Nearest Neighbors (k=5) on the embedding vectors.
- **Keyword Category Assignment:** Classifies detected keywords into their respective categories.
- **Heuristic Scoring:** Adjusts self-promotion scores based on the presence of achievement patterns, metrics, and sentiment.
- **Sentiment Analysis:** Uses TextBlob to score the polarity of each sentence, further refining the self-promotion score.

### Step 4: Output & Visualization
- **Highlighted Resume:** Sentences and keywords are color-coded by category and self-promotion score. Users can toggle categories on/off for focused review.
- **Score Visualization:** A dynamic gradient block visually communicates the overall self-promotion quality.
- **Metrics Display:** Shows composition percentages, keyword counts, and overall scores.

### Step 5: Evaluation & Feedback
- **Model Metrics:** Presents accuracy, precision, recall, F1-score, and cross-validation results for both self-promotion and keyword classification.
- **User Feedback (Future):** Planned feedback loop for continuous model improvement.

---

## 2. Models and Algorithms Explained

### Deep Learning Models

#### 1. SentenceTransformer (all-MiniLM-L6-v2)
- **Role:** Converts each sentence into a dense vector (embedding) that captures its semantic meaning.
- **Why:** Enables the KNN classifier to compare sentences based on meaning, not just keywords.
- **Complexity:** O(n·d) per sentence (n = tokens, d = embedding dimension).

#### 2. spaCy en_core_web_sm
- **Role:** Provides linguistic annotation (POS, NER, dependency parsing) for each token.
- **Why:** Supplies rich features for both rule-based and ML components, enabling context-aware keyword detection and validation.
- **Complexity:** O(n²) for dependency parsing.

### Traditional Machine Learning

#### 3. KNeighborsClassifier (KNN)
- **Role:** Classifies each sentence as self-promotional or neutral based on its embedding.
- **Why:** Chosen for interpretability and strong performance on moderate-sized datasets.
- **Complexity:** Training: O(m·d); Prediction: O(m·d) per query (m = training samples).

### Rule-Based Algorithms

#### 4. Regex Keyword Matching
- **Role:** Detects keywords using context-aware regular expressions and pattern rules.
- **Why:** Ensures precise, category-specific keyword identification.
- **Complexity:** O(n·k·L) (n = tokens, k = patterns, L = pattern length).

#### 5. TextBlob Sentiment Analysis
- **Role:** Scores the polarity (positive/negative) of each sentence.
- **Why:** Used to adjust self-promotion scores, rewarding achievement-oriented language.
- **Complexity:** O(n).

#### 6. Achievement Pattern Detection
- **Role:** Identifies action verbs followed by quantifiable impacts (e.g., "increased sales by 20%").
- **Why:** Boosts self-promotion score for sentences that demonstrate measurable achievements.
- **Complexity:** O(n).

#### 7. File Extraction (pdf2docx, Docling, pdfminer)
- **Role:** Extracts text from PDF/DOCX files, with fallback mechanisms for complex layouts.
- **Why:** Ensures robust, format-agnostic text extraction.
- **Complexity:** O(document size).

---

## 3. Features and Their Purpose

- **Smart Keyword Highlighting:** Helps users see which skills and attributes are most visible to recruiters.
- **Self-Promotion Analysis:** Quantifies how effectively the resume markets the candidate’s achievements.
- **Interactive Dashboard:** Empowers users to focus on specific keyword categories and see instant feedback.
- **Multi-Format Support:** Accepts resumes in PDF, DOCX, or TXT, ensuring accessibility.
- **Brutalist UI Design:** Modern, readable interface with custom fonts and color-coded elements for clarity.
- **Dynamic Score Visualization:** Makes it easy to interpret self-promotion quality at a glance.

---

## 4. Evaluation Metrics: What They Mean

### Accuracy
- **Definition:** Proportion of correct predictions (true positives + true negatives) out of all predictions.
- **Interpretation:** High accuracy means the model is generally reliable, but may be misleading if classes are imbalanced.

### Precision
- **Definition:** Proportion of true positives out of all positive predictions.
- **Interpretation:** High precision means that when the model predicts self-promotion, it is usually correct (few false positives).

### Recall
- **Definition:** Proportion of true positives out of all actual positives.
- **Interpretation:** High recall means the model successfully finds most self-promotional sentences (few false negatives).

### F1-Score
- **Definition:** Harmonic mean of precision and recall.
- **Interpretation:** Balances the trade-off between precision and recall; a high F1-score indicates strong overall performance.

### Cross-Validation (5-Fold)
- **Definition:** Model is trained and tested on 5 different splits of the data; mean and standard deviation of F1-score are reported.
- **Interpretation:** Low standard deviation means the model is stable and not overfitting to a particular subset.

### Score Separation
- **Definition:** Measures how well the model distinguishes between high and low self-promotion sentences.
- **Interpretation:** Higher values indicate clearer separation, making the score more actionable.

---

## 5. Program Continuity and Future Directions

- **Robustness:** Multiple fallback mechanisms for file extraction ensure the program works across diverse resume formats.
- **Extensibility:** Modular codebase allows for easy addition of new keyword categories, models, or UI features.
- **Planned Enhancements:**
  - Export highlighted resumes (HTML/PDF)
  - ATS keyword matching score
  - Industry-specific keyword sets
  - Resume format validation
  - Job description comparison mode
  - User feedback loop for continuous model improvement

---

## 6. Summary Table: Models, Features, and Metrics

| Component                | Role/Purpose                                      | Metric(s)                |
|--------------------------|---------------------------------------------------|--------------------------|
| SentenceTransformer      | Semantic sentence encoding                        | Embedding quality        |
| spaCy NLP                | Linguistic annotation, context features           | N/A                      |
| KNN Classifier           | Self-promotion/neutral classification             | Accuracy, Precision, etc.|
| Regex Matching           | Keyword detection                                 | Precision, Recall        |
| TextBlob Sentiment       | Sentiment scoring for adjustment                  | N/A                      |
| Achievement Detection    | Detects measurable achievements                   | N/A                      |
| File Extraction          | Robust text extraction from various formats       | N/A                      |

---

## 7. Conclusion

SkillHighlight combines state-of-the-art NLP, interpretable ML, and user-centric design to deliver actionable resume insights. Each model and feature is chosen for its role in making resume analysis robust, transparent, and effective. Evaluation metrics demonstrate strong, consistent performance, and the modular architecture ensures the program can evolve with future needs.
