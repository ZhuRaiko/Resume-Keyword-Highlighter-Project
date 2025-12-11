# Possible Panelist Questions for SkillHighlight Thesis Defense

---

## General/Introduction
- What motivated you to develop SkillHighlight?
- How does your system address the main challenges in résumé screening?
- What are the main features of your system?
- How is your work different from existing ATS or résumé tools?

## Technical Background
- Why did you choose a developmental research design?
- What are the main limitations of traditional ATS systems?
- Can you explain the importance of context-aware keyword detection?
- Why is self-promotion scoring important in résumé analysis?
- What are the theoretical foundations for your scoring algorithm?

## Data and Dataset
- How was your labeled résumé dataset constructed?
- How did you ensure the quality and diversity of your dataset?
- What is the class distribution of your dataset?
- How did you curate your keyword database?
- Are there any biases in your data?

## System Architecture and Implementation
- Can you walk us through your system architecture?
- Why did you choose a modular pipeline design?
- How do the different modules (extraction, embedding, classification, highlighting, scoring) interact?
- What are the main technologies and libraries used?
- How does your UI support user needs?

## NLP and Machine Learning
- Why did you choose BERT embeddings and KNN for classification?
- How does your system handle multi-word keywords and lemmatization?
- How do you ensure context validation for keyword highlighting?
- Can you explain the scoring algorithm in detail?
- How is sentiment polarity calculated and used?
- What are the thresholds for classifying self-promotion?

## Evaluation and Results
- How did you evaluate the performance of your system?
- What metrics did you use and why?
- Can you explain your confusion matrix and what it reveals?
- How does your system perform compared to traditional ATS?
- What are the strengths and weaknesses of your approach?
- How did you handle edge cases and negative test scenarios?

## Limitations and Future Work
- What are the main limitations of your current system?
- How could your system be improved in the future?
- How would you adapt your system for other domains or languages?
- What are the ethical considerations in automated résumé screening?

## Practical/Deployment
- How can SkillHighlight be integrated into real-world recruitment workflows?
- What are the hardware/software requirements for deployment?
- How scalable is your system?
- How user-friendly is your interface for non-technical users?

## Miscellaneous
- What was the most challenging part of your project?
- What did you learn during the development process?
- If you had more time/resources, what would you add or change?
- How would you explain your system to a non-technical audience?

## In-Depth Program-Specific Questions (with Answers)

### Configuration Panel
- **What is the purpose of the configuration panel in your UI?**
  - It lets users toggle which keyword categories are highlighted and may control other display options.
  
  > **In-depth:** The configuration panel is the user’s main control center for customizing how résumé analysis is displayed. It allows users to select which keyword categories (hard skills, soft skills, self-promotion, etc.) are highlighted. This makes the tool flexible for different use cases, such as focusing only on technical skills or soft skills depending on the job requirements.
- **How do the checkboxes and toggles in the configuration panel affect the system's behavior?**
  - They show or hide highlights for each category (e.g., hard skills, soft skills). Some toggles may be placeholders for future features.
  
  > **In-depth:** Each checkbox or toggle directly maps to a keyword category or display feature. When a user interacts with these controls, the UI updates in real time to show or hide highlights for the selected categories. This is achieved by updating the state in Streamlit, which then re-renders the highlighted résumé text. Some toggles may be present for planned features (e.g., toggling scoring display), but not all are fully implemented yet.
- **Why might some configuration options appear to have no effect? Are there features that are not yet fully implemented?**
  - Yes, some options are not fully wired up or are reserved for future updates.
  
  > **In-depth:** If a toggle does nothing, it’s likely a placeholder for a future feature or an option that hasn’t been connected to backend logic yet. This is common in iterative development, where UI elements are added before the underlying functionality is complete.
- **How would you improve the configuration panel for better user feedback and control?**
  - Make sure all toggles are functional, add tooltips, and provide instant visual feedback when options are changed.
  
  > **In-depth:** To make the panel more user-friendly, all toggles should be connected to actual features. Tooltips can explain what each option does. Instant feedback (such as changing highlight colors or showing a message) helps users know their actions had an effect. Disabling or hiding unfinished options also prevents confusion.

### Highlighting Module
- **How does the highlighting module determine which words or phrases to highlight?**
  - It matches words/phrases from the keyword database, then checks their context using spaCy to avoid false positives.
  
  > **In-depth:** The module scans résumé text for matches against the keyword database. For each match, it checks the context using spaCy NLP. This means it doesn’t just look for the word, but also analyzes the sentence structure and part of speech to ensure the keyword is used as intended (e.g., “Python” as a skill, not as an animal).
- **How does the system handle overlapping keywords or multi-word phrases?**
  - It prioritizes longer (multi-word) matches and prevents overlapping highlights.
  
  > **In-depth:** The system sorts keywords by length, so longer phrases are matched first (e.g., “project management” before “project”). Once a phrase is highlighted, overlapping shorter keywords are skipped to avoid double-highlighting. This ensures clarity and prevents clutter.
- **What role does spaCy play in context validation for highlighting?**
  - spaCy checks the part of speech and sentence structure to ensure keywords are used in the right context.
  
  > **In-depth:** spaCy parses each sentence to identify grammatical roles (noun, verb, etc.) and dependencies. For example, it can tell if “leadership” is used as a skill or just mentioned in passing. This reduces false positives (irrelevant highlights) and false negatives (missed relevant keywords).
- **How are false positives and false negatives minimized in the highlighting process?**
  - By using context validation and only highlighting when the usage matches the intended meaning.
  
  > **In-depth:** Context validation means a keyword is only highlighted if it fits the expected usage. For example, “Java” is highlighted only if it refers to the programming language, not the island. The system also uses negative keyword lists and part-of-speech filters to further reduce mistakes.
- **How does the system distinguish between different keyword categories?**
  - Each keyword is tagged by category in the database, and highlights use different colors/styles.
  
  > **In-depth:** Each keyword in the database is tagged with a category. When highlighted, the UI uses different colors or styles for each category (e.g., blue for hard skills, green for soft skills). This visual distinction helps users quickly identify strengths and gaps.
- **Can you explain how version numbers and special characters in keywords are handled?**
  - The system includes variants (e.g., Python3, C++) in the keyword list and uses regex for flexible matching.
  
  > **In-depth:** Keywords like “C++” or “Python3” are included as variants in the database. Regex patterns allow flexible matching, so “Python 3.8” or “C++11” are also detected. This ensures the system recognizes skills regardless of formatting.

### KNN Classifier
- **How does the KNN classifier work in your system?**
  - It compares the BERT embedding of a sentence to its 5 nearest neighbors in the training set and predicts based on their labels.
  
  > **In-depth:** The KNN (K-Nearest Neighbors) classifier takes the BERT embedding (a vector representation) of each sentence and compares it to all sentences in the training set. It finds the 5 closest matches (neighbors) based on vector distance and predicts the label (e.g., self-promotion, skill) by majority vote.
- **Why did you choose KNN over other classifiers?**
  - KNN is simple, effective for small datasets, and works well with high-quality embeddings.
  
  > **In-depth:** KNN is non-parametric and doesn’t require complex training. It’s ideal for small datasets and works well when embeddings are high-quality, as with BERT. It’s also easy to interpret: predictions are based on actual examples in the dataset.
- **How are BERT embeddings used as input to the KNN model?**
  - Each sentence is converted to a 384-dimensional vector using BERT, which KNN uses for distance calculation.
  
  > **In-depth:** BERT converts each sentence into a 384-dimensional vector that captures semantic meaning. These vectors allow the KNN model to compare sentences based on meaning, not just keywords.
- **What is the value of k, and how was it chosen?**
  - k=5, chosen by testing different values and picking the one with the best validation accuracy.
  
  > **In-depth:** The value of k (number of neighbors) was chosen by testing different values and selecting the one with the highest validation accuracy. k=5 provided a good balance between stability and sensitivity to outliers.
- **How does the KNN model handle ambiguous or borderline cases?**
  - If neighbors are mixed, the output probability is close to 0.5, and heuristics help refine the final score.
  
  > **In-depth:** If the nearest neighbors have mixed labels, the model’s output probability will be close to 0.5, indicating uncertainty. The system can use additional heuristics (like confidence thresholds or fallback rules) to refine the final decision.
- **How is the KNN model trained and evaluated?**
  - It is trained on labeled sentences and evaluated using accuracy, precision, recall, and F1-score.
  
  > **In-depth:** The model is trained on labeled sentences from the dataset. Performance is measured using standard metrics: accuracy (overall correctness), precision (true positives vs. false positives), recall (true positives vs. false negatives), and F1-score (balance between precision and recall).
- **What are the limitations of using KNN for this task?**
  - KNN can be slow with large datasets and may not generalize as well as more complex models.
  
  > **In-depth:** KNN can be slow if the dataset is large, since it compares each new sentence to all training examples. It also doesn’t learn complex decision boundaries, so it may not generalize as well as neural networks or ensemble models.

### System Integration and Edge Cases
- **How do the different modules communicate and share data?**
  - Data flows through the pipeline: extraction → embedding → classification → highlighting → scoring → UI.
  
  > **In-depth:** The system is modular:
  > 1. Extraction: Reads and cleans résumé text
  > 2. Embedding: Converts sentences to vectors
  > 3. Classification: Labels sentences (e.g., self-promotion, skill)
  > 4. Highlighting: Marks keywords in context
  > 5. Scoring: Calculates overall résumé scores
  > 6. UI: Displays results and allows user interaction
  > Data flows sequentially, with each module passing its output to the next.
- **How does the system handle resumes with unusual formatting or missing sections?**
  - It uses robust text extraction and fallback logic to process as much content as possible.
  
  > **In-depth:** The extraction module uses flexible parsing to handle different résumé formats (PDF, DOCX, TXT). If sections are missing or text is poorly formatted, fallback logic tries to extract as much information as possible, ensuring the system remains useful even with imperfect input.
- **What happens if a user uploads a file with unsupported content or encoding?**
  - The system shows an error message and asks the user to upload a supported file type.
  
  > **In-depth:** If a user uploads a file type or encoding the system can’t process, it detects the issue and shows a clear error message. The user is prompted to upload a supported format (e.g., PDF, DOCX, UTF-8 TXT).
- **How does the system ensure robustness and error handling in real-world usage?**
  - It uses try/except blocks, input validation, and clear user feedback for errors.
  
  > **In-depth:** The code uses try/except blocks to catch errors during processing. Input validation checks for empty files, unsupported formats, or corrupted data. When errors occur, the system provides clear feedback so users know what went wrong and how to fix it.

---

*Prepare concise, evidence-based answers for each question. Use visuals, tables, and code snippets as needed during your defense.*
