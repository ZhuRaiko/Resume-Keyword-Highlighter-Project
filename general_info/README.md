# SkillHighlight Analyzer

SkillHighlight Analyzer is a Streamlit application for resume keyword highlighting and self-promotion analysis. It accepts PDF, DOCX, and TXT resumes, extracts and normalizes the text, highlights resume keywords by category, and scores each sentence for achievement-oriented language.

This README reflects the current modular implementation in `main_modular.py`, `modules/`, `models/`, and `model_evaluation/`.

## Current Status

- Main app entry point: `main_modular.py`
- Evaluation entry point: `model_evaluation/evaluate_full_system.py`
- Latest committed full-system metrics: `model_evaluation/full_system_metrics.json`
- Training data: `data/self_promotion_dataset.csv`
- Keyword database: `data/keywords.json`
- Cached classifier: `models/knn_model.pkl`

## What The App Does

1. Loads spaCy `en_core_web_sm` and SentenceTransformer `all-MiniLM-L6-v2`.
2. Loads or trains a KNN classifier with `k=5` using 384-dimensional sentence embeddings.
3. Accepts uploaded resumes in PDF, DOCX, or TXT format, or pasted text.
4. Extracts and normalizes resume text for more consistent sentence splitting.
5. Counts keyword matches across four categories:
   - Hard Skills
   - Soft Skills
   - Recruiter Keywords
   - Action Verbs
6. Highlights matched keywords with category-specific colors.
7. Scores each sentence using KNN probability plus resume-specific heuristics.
8. Displays an average self-promotion score and sentence-level feedback.

## UI Features

- Dark Streamlit interface with custom Morganite and Deliware fonts.
- Category toggles for Hard Skills, Soft Skills, Recruiter Keywords, and Action Verbs.
- Sidebar settings for stricter or more relaxed matching behavior.
- Token-aligned highlighting mode for safer punctuation handling.
- Sentiment threshold control for soft-skill filtering.
- Legacy HTML rendering toggle for simpler highlight spans.
- Dynamic self-promotion score block:
  - Green when score is greater than `0.8`
  - Yellow/orange when score is greater than `0.5`
  - Red when score is `0.5` or lower

## Keyword Colors

| Category | Label | Color |
| --- | --- | --- |
| Hard Skills | `HARD` | `#26a69a` |
| Soft Skills | `SOFT` | `#7e57c2` |
| Recruiter Keywords | `RECRUITER` | `#ff9f43` |
| Action Verbs | `ACTION` | `#ef5350` |

## Processing Pipeline

### 1. Text Extraction

`modules/extractor.py` handles supported file formats.

- TXT files are decoded directly.
- PDF files first try PDF-to-DOCX conversion through `pdf2docx`, then extract document-order text with `python-docx`.
- If PDF-to-DOCX conversion fails, PDF extraction falls back to Docling.
- If Docling fails, PDF extraction falls back to `pdfminer.six`.
- DOCX files are read with `python-docx`, with `docx2txt` as a fallback.

After extraction, `normalize_resume_text()` cleans PDF artifacts, normalizes bullets, preserves skill-list structure, merges obvious continuation lines, and keeps resume sections readable for downstream scoring.

### 2. Keyword Counting

`modules/counters.py` counts direct keyword matches from `data/keywords.json`. The displayed percentages represent category composition across all keyword matches. If there are no keyword matches, the function falls back to percentages relative to total word count.

### 3. Keyword Highlighting

`modules/highlight.py` uses spaCy tokenization and category-specific context checks before rendering HTML spans. The highlighter avoids matching inside email addresses and URLs, supports token-aligned matching, and can relax checks for hard skills, soft skills, recruiter keywords, and action verbs through sidebar controls.

### 4. Self-Promotion Scoring

`modules/scoring.py` splits the resume into sentence candidates and scores each sentence with:

- KNN self-promotion probability from BERT embeddings.
- Metric bonus for numbers, percentages, and money-like values.
- Achievement-pattern bonus for action plus impact language.
- Bullet bonus for resume-style bullet points.
- Positive sentiment boost.
- Action-verb opening bonus.
- Short-sentence penalty.

Scores are clamped between `0.0` and `1.0`, then averaged for the overall self-promotion score.

## Model And Algorithms

| Component | Implementation | Purpose |
| --- | --- | --- |
| Sentence embeddings | SentenceTransformer `all-MiniLM-L6-v2` | Converts sentences into 384-dimensional vectors |
| NLP parsing | spaCy `en_core_web_sm` | Tokenization, sentence parsing, POS/dependency context |
| Classification | scikit-learn `KNeighborsClassifier(n_neighbors=5)` | Predicts self-promotion probability |
| Keyword matching | Regex plus spaCy context checks | Finds category-specific resume keywords |
| Sentiment | TextBlob | Helps suppress or boost soft-skill and scoring contexts |
| Extraction | `pdf2docx`, Docling, `pdfminer.six`, `python-docx`, `docx2txt` | Converts uploaded files into plain text |

## Latest Evaluation Snapshot

These values come from `model_evaluation/full_system_metrics.json`.

| Component | Metric | Value |
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

The metrics file records an evaluation timestamp of `2025-12-10T18:17:56.694874`.

## Running The App

Install the project dependencies in your Python environment, then download the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

Run the Streamlit app:

```bash
streamlit run main_modular.py
```

## Running The Evaluation

Run the full evaluation script from the repository root:

```bash
python model_evaluation/evaluate_full_system.py
```

This script evaluates:

- Text extraction success on files in `model_evaluation/test_resumes/`
- Keyword highlighting precision, recall, F1, and confusion counts
- Self-promotion scoring accuracy and score separation
- KNN diagnostic plots, including learning and validation curves
- BERT embedding visualizations through PCA and UMAP or t-SNE

Generated artifacts are saved in `model_evaluation/`, including `full_system_metrics.json` and PNG figures. See `model_evaluation/README.md` for figure captions.

## Project Structure

```text
Resume-Keyword-Highlighter-Project/
|-- main_modular.py
|-- data/
|   |-- keywords.json
|   `-- self_promotion_dataset.csv
|-- fonts/
|   |-- Morganite-*.ttf
|   `-- deliware-*.otf
|-- models/
|   |-- embedder.py
|   |-- knn_classifier.py
|   |-- knn_model.pkl
|   `-- sentiment.py
|-- modules/
|   |-- counters.py
|   |-- embeddings.py
|   |-- extractor.py
|   |-- highlight.py
|   `-- scoring.py
|-- model_evaluation/
|   |-- evaluate_full_system.py
|   |-- full_system_metrics.json
|   |-- README.md
|   `-- test_resumes/
|-- backups/
`-- general_info/
    `-- README.md
```

## Key Dependencies

- `streamlit`
- `streamlit-toggle-switch`
- `sentence-transformers`
- `spacy`
- `scikit-learn`
- `textblob`
- `pdf2docx`
- `docling`
- `pdfminer.six`
- `python-docx`
- `docx2txt`
- `pandas`
- `numpy`
- `matplotlib`

Note: this repository currently does not include a root `requirements.txt`. The evaluation folder includes `model_evaluation/requirements_frozen.txt` for the evaluation environment.

## Consistency Notes

The general documentation is now aligned with these current repository facts:

- The active app file is `main_modular.py`, not a legacy backup script.
- The active full-system evaluation command is `python model_evaluation/evaluate_full_system.py`.
- The repository does not currently contain `evaluate_model.py`, even though some older evaluation documentation still references it.
- PDF extraction is best described as a primary PDF-to-DOCX path with Docling and pdfminer fallbacks, not as a single guaranteed conversion path.
- The UI includes sidebar controls for matching strictness, sentiment threshold, token alignment, and rendering mode.
- The README performance numbers match `model_evaluation/full_system_metrics.json`.

## Future Enhancements

- Export highlighted resumes as HTML or PDF.
- Add job-description comparison mode.
- Add industry-specific keyword sets.
- Add ATS-style keyword coverage scoring.
- Add a root dependency file for easier setup.
- Consolidate older evaluation docs that still reference removed or renamed scripts.
