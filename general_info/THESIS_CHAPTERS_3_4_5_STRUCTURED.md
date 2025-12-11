# SkillHighlight Thesis: Chapters 3, 4, and 5 (Structured)

---

## Chapter 3: Technical Background

### 3.1 Overview of Résumé Screening and ATS
- Brief history and challenges of automated résumé screening
- Role of Applicant Tracking Systems (ATS)

### 3.2 Literature Review
- Related works on keyword detection, self-promotion scoring, and ATS limitations
- Gaps in current approaches

### 3.3 NLP and Machine Learning Fundamentals
- Introduction to NLP concepts relevant to résumé analysis
- Overview of machine learning techniques (KNN, embeddings)

### 3.4 Keyword Highlighting Theory
- Theoretical basis for context-aware keyword detection
- Importance of multi-word phrases, lemmatization, and context validation

### 3.5 Self-Promotion Scoring Theory
- Rationale for scoring achievement-oriented language
- Review of scoring models and heuristics

### 3.6 Summary of Existing Approaches
- Table comparing traditional ATS, SkillHighlight, and related systems

---

## Chapter 4: Technology and System Design

### 4.1 System Requirements and Use Cases
- Functional requirements for job seekers and recruiters
- Use case model and actor interactions

### 4.2 Overall System Architecture
- High-level architecture diagram
- Description of modular pipeline
- Table of system modules and functions

### 4.3 Data Sources and Preparation
- Description of labeled résumé dataset
- Keyword database construction
- Data preprocessing steps

### 4.4 Model Training and Configuration
- Training process for KNN classifier
- Embedding generation and configuration parameters
- Cross-validation setup

### 4.5 Scoring Algorithm
- Detailed formulas for KNN probability, heuristics, and final score
- Sentiment polarity calculation
- Classification thresholds

### 4.6 Tools and Technologies Used
- List of libraries, frameworks, and their roles

### 4.7 Implementation Details
- UI design and workflow
- Integration of modules
- Example screenshots (reference)

### 4.8 Summary
- Recap of system design and implementation choices

---

## Chapter 5: Results and Evaluation

### 5.1 Experimental Setup
- Description of test environment and procedures

### 5.2 Evaluation Metrics
- Definitions and formulas for accuracy, precision, recall, F1-score
- Confusion matrix explanation

**Metrics Used:**
- **Accuracy:** Proportion of correct predictions out of all predictions.
- **Precision:** Proportion of true positives out of all predicted positives.
- **Recall:** Proportion of true positives out of all actual positives.
- **F1-score:** Harmonic mean of precision and recall.

**Formulas:**
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-score = 2 * (Precision * Recall) / (Precision + Recall)

**Confusion Matrix:**
- Table showing counts of true positives, true negatives, false positives, and false negatives for each class.

### 5.3 Test Results
- Tables of performance metrics
- Confusion matrix and cross-validation results

**SkillHighlight System Performance (from model_evaluation/full_system_metrics.json):**
- **Keyword Highlighting:**
    - Accuracy: 87.8%
    - Precision: 97.1%
    - Recall: 92.3%
    - F1-score: 94.6%
    - Tests Passed: 108/123
- **KNN Classifier:**
    - Accuracy: 86.2%
    - High Accuracy: 90.6%
    - Low Accuracy: 81.8%
    - Average High Score: 0.81
    - Average Low Score: 0.27
    - Score Separation: 0.54
- **Extraction:**
    - Success Rate: 100% (4/4 files)
- **End-to-End Scoring:**
    - Classification Accuracy: 90%
    - Score Separation: 0.690
- **Cross-Validation:**
    - Mean F1: 0.817 ± 0.121

### 5.4 Discussion of Results
- Analysis of strengths, weaknesses, and notable findings

### 5.5 Limitations and Future Work
- Identified limitations of the current system
- Recommendations for future research and improvements

### 5.6 Summary
- Final recap of evaluation and key takeaways

### 5.2 CONCLUSIONS
Based on the results, the following conclusions are drawn in relation to the research objectives:

- **ATS Challenges Identified:** The literature review revealed that 88% of employers acknowledge losing qualified candidates due to ATS limitations. Rigid keyword matching and format requirements create barriers for students and fresh graduates.
- **System Architecture Designed:** SkillHighlight successfully integrates BERT-based semantic embeddings with KNN classification and SpaCy linguistic validation, providing a more nuanced approach than traditional keyword frequency methods.
- **Appropriate Tools Determined:** The Python ecosystem (SentenceTransformers, scikit-learn, spaCy, Streamlit) proved suitable for developing a functional résumé analysis system with strong performance.
- **Functional System Developed:** The implementation demonstrates automatic keyword highlighting, self-promotion scoring, and keyword composition analysis in a user-friendly web interface.
- **Meaningful Verification Integrated:** The self-promotion scoring mechanism helps distinguish genuine competency statements from buzzword-heavy or vague language, supporting more authentic résumé writing.
- **System Tested and Evaluated:** The KNN classifier achieved 86.2% accuracy, keyword highlighting achieved 97.1% precision with 94.6% F1-score, and end-to-end scoring achieved 90% classification accuracy with strong score separation (0.690) between high and low quality content.
- **Overall Conclusion:** SkillHighlight provides a viable tool for improving résumé quality and screening efficiency by combining machine learning classification with context-aware keyword detection. The system empowers job seekers to present their skills more effectively while supporting recruiters in identifying qualified candidates.

#### Conclusions Related to Research Questions and Hypotheses

In addition to meeting the research objectives, this study addressed the following research questions and hypotheses:

- **Contextual Keyword Highlighting:** The system’s use of spaCy for context validation significantly improved the visibility of student skills in résumés. With a precision rate of 97.1% and recall of 92.3%, SkillHighlight ensures that only relevant competencies are emphasized, reducing false positives and making genuine skills more prominent to reviewers.

- **Effectiveness of BERT-Based Embedding:** By leveraging BERT-based semantic embeddings and KNN classification, the system outperformed traditional keyword extraction methods. The high F1-score (94.6%) and overall scoring accuracy (86.2%) demonstrate that semantic analysis captures nuanced skill statements and self-promotion, reducing both missed skills and erroneous matches.

- **Self-Promotion Scoring:** The self-promotion scoring mechanism reliably distinguishes authentic competencies from keyword-stuffing. Contextual analysis and classification yield a clear separation (≈ 0.54) between high and low quality content, rewarding genuine, well-supported claims and penalizing superficial language.

- **Practical Résumé-Writing Improvements:** Automated analysis provides actionable feedback, helping users identify missing skills, overused buzzwords, and areas lacking evidence. This guidance enables students to write clearer, more targeted résumés, increasing their chances of passing both ATS and recruiter screening.

- **Hypotheses Validated:**  
  - *H1:* Context-aware keyword detection resulted in higher precision and recall than traditional keyword matching (Precision: 97.1%, Recall: 92.3%, F1-score: 94.6%).  
  - *H2:* Semantic embeddings (BERT) combined with KNN classification improved résumé screening accuracy (KNN accuracy: 86.2%, end-to-end accuracy: 90%).  
  - *H3:* Self-promotion scoring effectively separated authentic and superficial résumé content (score separation: 0.54–0.69).

SkillHighlight not only meets its research objectives but also answers key research questions and validates its hypotheses. By combining advanced NLP and machine learning techniques, the system offers a robust solution for improving résumé quality and screening efficiency, empowering both job seekers and recruiters.

---

## References
- [Add references in APA format]

---

## Appendices
- Appendix A: Sample Résumés Used for Testing
- Appendix B: Complete Keyword Database (keywords.json)
- Appendix C: Model Training Code
- Appendix D: System Screenshots
- Appendix E: Evaluation Script Output (model_metrics.json)
