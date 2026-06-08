# SkillHighlight Interview Reviewer

This file is a quick reviewer for explaining the project during an interview, defense, or presentation. It is written as a practical memory aid, not as formal system documentation.

## One-Minute Project Summary

SkillHighlight Analyzer is a resume analysis tool built with Streamlit. It lets a user upload or paste a resume, extracts the text, highlights important resume keywords, and gives a self-promotion score for each sentence.

The project combines:

- Natural Language Processing for text parsing and sentence handling.
- Machine learning for self-promotion scoring.
- Rule-based keyword matching for resume skill highlighting.
- A Streamlit UI for interactive upload, highlighting, toggles, and score display.

Simple interview answer:

> I built a resume analyzer that extracts text from PDF, DOCX, or TXT resumes, highlights skill-related keywords, and scores how strongly each sentence presents achievements. The main ML part uses BERT-style sentence embeddings with a K-Nearest Neighbors classifier, then I combine that prediction with resume-specific heuristics like metrics, action verbs, and achievement patterns.

## What Machine Learning Model Did I Use?

The main machine learning model is K-Nearest Neighbors, or KNN.

In the code, this is implemented with:

```text
scikit-learn KNeighborsClassifier(n_neighbors=5)
```

Relevant files:

- `models/knn_classifier.py`
- `modules/scoring.py`
- `models/knn_model.pkl`
- `data/self_promotion_dataset.csv`

The KNN classifier is used for self-promotion detection. It classifies a sentence as more or less self-promotional based on similarity to labeled resume sentences in the training dataset.

Important nuance:

The project does not train a deep neural network classifier from scratch. It uses a pre-trained SentenceTransformer model to convert sentences into embeddings, then uses KNN on top of those embeddings.

## What Is KNN?

K-Nearest Neighbors is a supervised machine learning algorithm. It stores labeled training examples and classifies a new example by looking at the most similar examples nearby.

In this project:

1. Each resume sentence is converted into a numerical vector.
2. The KNN model compares the new sentence vector to labeled training sentence vectors.
3. It looks at the 5 nearest neighbors.
4. It estimates whether the sentence is self-promotional based on those neighbors.

Simple explanation:

> KNN works by similarity. If a new sentence is close to examples that were labeled as strong self-promotion, it receives a higher self-promotion score. If it is closer to neutral or task-description sentences, it receives a lower score.

## Why Use Sentence Embeddings?

Raw text cannot be directly used by KNN because KNN needs numbers. The project uses SentenceTransformer `all-MiniLM-L6-v2` to convert each sentence into a 384-dimensional vector.

That vector represents the meaning of the sentence. For example, sentences about improving performance, increasing sales, or leading a project should be closer to other achievement-oriented examples.

Simple answer:

> I used sentence embeddings so the model compares meaning instead of just exact words. This lets the system recognize that two different achievement sentences can be similar even if they do not use the exact same wording.

## Overall Pipeline

The main app is `main_modular.py`.

The pipeline is:

1. User uploads a PDF, DOCX, or TXT file, or pastes resume text.
2. `modules/extractor.py` extracts text from the file.
3. `normalize_resume_text()` cleans formatting issues like bullet artifacts, awkward line breaks, and PDF spacing problems.
4. `modules/counters.py` counts keyword matches in four categories:
   - Hard Skills
   - Soft Skills
   - Recruiter Keywords
   - Action Verbs
5. `modules/highlight.py` highlights keywords with category-specific colors.
6. `modules/scoring.py` splits the resume into sentence candidates.
7. Each sentence is embedded with SentenceTransformer.
8. KNN predicts a base self-promotion probability.
9. Rule-based scoring adjusts the score using resume-specific signals.
10. The UI displays highlighted text, keyword category percentages, sentence-level scores, and the average score.

## What Are The Four Keyword Categories?

| Category | Purpose | Example Meaning |
| --- | --- | --- |
| Hard Skills | Technical skills and tools | Python, SQL, machine learning |
| Soft Skills | Personal or interpersonal qualities | leadership, communication |
| Recruiter Keywords | Resume/ATS-friendly terms | experience, project management |
| Action Verbs | Achievement-oriented verbs | led, built, improved, managed |

The keyword source is:

```text
data/keywords.json
```

## How Does Self-Promotion Scoring Work?

The self-promotion score is not only the KNN output. It is a hybrid score.

The base score comes from KNN. Then the system adds or subtracts heuristic signals:

- Metric bonus if the sentence has numbers, percentages, or money values.
- Achievement-pattern bonus if it combines achievement verbs with impact language.
- Bullet bonus for resume-style bullet points.
- Positive sentiment boost.
- Action-verb opening bonus if the sentence starts with a strong action verb.
- Short-sentence penalty for very short or weak fragments.

The final score is clamped between `0.0` and `1.0`.

Simple answer:

> The ML model gives the base probability, but I also added resume-specific heuristics because strong resumes often use measurable results, action verbs, and achievement language. That makes the score more practical for the resume domain.

## Why Use A Hybrid ML And Rule-Based Approach?

This project has two different tasks:

1. Classifying whether a sentence sounds achievement-oriented.
2. Highlighting specific keywords in the resume.

KNN is useful for the first task because it can compare sentence meaning through embeddings. Rule-based matching is useful for the second task because keyword highlighting needs exact spans and predictable colors.

Simple answer:

> I used ML where semantic judgment was needed, and rules where exact keyword highlighting was needed. The KNN model handles the subjective part, while regex and NLP context checks handle precise highlighting.

## What NLP Tools Did I Use?

The project uses:

- spaCy `en_core_web_sm` for tokenization, sentence parsing, POS tags, and dependency context.
- SentenceTransformer `all-MiniLM-L6-v2` for sentence embeddings.
- TextBlob for simple sentiment polarity checks.

spaCy helps the highlighter avoid bad matches, such as highlighting a word in the wrong grammatical context. TextBlob helps with soft-skill and scoring context.

## What File Types Are Supported?

The app supports:

- PDF
- DOCX
- TXT
- Pasted text

For PDFs, the extractor first tries to convert PDF to DOCX using `pdf2docx`, then reads the converted DOCX. If that fails, it falls back to Docling, then `pdfminer.six`.

For DOCX, it reads the document with `python-docx`, with `docx2txt` as a fallback.

## What Did The Evaluation Measure?

The main evaluation script is:

```bash
python model_evaluation/evaluate_full_system.py
```

The latest saved metrics are in:

```text
model_evaluation/full_system_metrics.json
```

Latest committed evaluation snapshot:

| Component | Metric | Result |
| --- | --- | --- |
| Text Extraction | Success Rate | 100.0% over 4 files |
| Keyword Highlighting | Accuracy | 87.8% |
| Keyword Highlighting | Precision | 97.1% |
| Keyword Highlighting | Recall | 92.3% |
| Keyword Highlighting | F1 Score | 94.6% |
| Self-Promotion Scoring | Accuracy | 86.2% |
| Self-Promotion Scoring | HIGH Detection | 90.6% |
| Self-Promotion Scoring | LOW Detection | 81.8% |
| Self-Promotion Scoring | Score Separation | 0.537 |

If asked what those mean:

- Precision: when the system highlights something, how often it is correct.
- Recall: how many expected keywords the system successfully finds.
- F1 score: balance between precision and recall.
- Scoring accuracy: how often the self-promotion score correctly separates high and low examples.
- Score separation: how far apart the average high-scoring and low-scoring examples are.

## Likely Interview Questions And Answers

### Why did you choose KNN?

> KNN is simple, interpretable, and works well with embedding-based similarity. Since my dataset contains labeled resume sentences, KNN can compare a new sentence against nearby examples rather than learning a complex model from scratch. It was also practical for a school project because the training process is straightforward and the results are explainable.

### What is the input to KNN?

> The input is not raw text. Each sentence is converted into a 384-dimensional embedding using SentenceTransformer `all-MiniLM-L6-v2`. KNN then compares those vectors.

### What is the output of KNN?

> The KNN model returns a probability-like score for whether the sentence is self-promotional. The app then adjusts that base score with heuristics and displays a final score between 0 and 1.

### What does a high self-promotion score mean?

> It means the sentence looks achievement-oriented. It likely uses stronger resume language, measurable outcomes, action verbs, or impact-focused wording.

### What does a low score mean?

> It means the sentence is more descriptive or passive. For example, it may list responsibilities without showing results, metrics, or impact.

### Is the highlighting also machine learning?

> Mostly no. Keyword highlighting is mainly rule-based. It uses keyword lists, regex matching, spaCy context checks, and category colors. The ML part is mainly the self-promotion scoring.

### Is BERT the classifier?

> Not exactly. The SentenceTransformer model is used as an embedding model. It converts sentences into vectors. The classifier on top is KNN.

### Did you train BERT?

> No. I used a pre-trained SentenceTransformer model, `all-MiniLM-L6-v2`. I trained or loaded the KNN classifier using the labeled resume sentence dataset.

### How is the model saved?

> The trained KNN model is cached with `joblib` as `models/knn_model.pkl`, so the app can load it instead of retraining every time.

### What are the limitations?

> The system depends on the quality of the labeled dataset and keyword list. KNN can be slower as the dataset grows because it compares against stored examples. The scoring is also partly heuristic, so it may overvalue sentences with numbers or action verbs even if the writing is not truly strong. PDF extraction can also vary depending on formatting.

### How would you improve it?

> I would add job-description comparison, industry-specific keyword lists, a cleaner dependency file, larger evaluation data, and possibly compare KNN against logistic regression, SVM, or a fine-tuned transformer classifier.

## Strong Phrases To Remember

- "The ML component is KNN over sentence embeddings."
- "SentenceTransformer converts resume sentences into 384-dimensional vectors."
- "KNN classifies by similarity to labeled resume examples."
- "The system is hybrid: ML for scoring, rule-based NLP for highlighting."
- "The score combines KNN probability with resume-specific heuristics."
- "The app uses Streamlit for the interface and modular Python files for extraction, highlighting, counting, and scoring."

## Honest Technical Summary

If you want the most accurate short explanation, say this:

> This project uses a hybrid NLP and ML pipeline. I use a pre-trained SentenceTransformer model to embed each resume sentence, then a KNN classifier with 5 neighbors to estimate whether the sentence is self-promotional. I combine that ML score with rule-based resume heuristics such as metrics, action verbs, and achievement patterns. Separately, keyword highlighting is handled with keyword lists, regex matching, and spaCy context checks. The whole system is wrapped in a Streamlit app for uploading resumes and viewing highlighted results.
