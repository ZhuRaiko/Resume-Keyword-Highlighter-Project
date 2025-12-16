# Evaluation Questions and Answers

This document provides possible questions that researchers might ask during the presentation of the SkillHighlight evaluation, along with clear, plain-language answers for each.

---

## 1. Precision-Recall Curve
**Q:** Why did you include a Precision-Recall (PR) curve in your evaluation?
**A:** The PR curve is especially useful when dealing with imbalanced data, as it focuses on the model’s ability to find relevant positive cases (self-promotion) without being misled by the abundance of negatives. It helps us show how well our model balances finding as many true self-promotional statements as possible while keeping false alarms low, which is critical for our highlighting tool.
**Q:** What does the Precision-Recall (PR) curve tell us about your model?
**A:** The PR curve shows how well our model balances precision (how many flagged sentences are truly self-promotional) and recall (how many true self-promotional sentences are found) at different thresholds. Our average precision (AP) of 0.85 means the model is good at finding most relevant content while keeping false alarms low.

---

## 2. ROC Curve and AUC
**Q:** Why did you include a ROC curve and AUC in your evaluation?
**A:** The ROC curve and AUC are standard metrics for evaluating how well a model can distinguish between two classes, regardless of the threshold. Including this shows that our model is not just tuned for one setting, but is generally good at ranking self-promotional versus neutral content across all possible thresholds.
**Q:** What is the significance of the ROC curve and its AUC value?
**A:** The ROC curve shows how well the model separates self-promotional from neutral sentences. The AUC of 0.91 means our model is very good at ranking true self-promotional content higher than neutral content, even if the threshold changes.
**Q:** How does the PR curve help in comparing different models?
**A:** The PR curve allows us to directly compare how different models perform in terms of balancing precision and recall, especially when positive cases are rare. A higher curve or AP means a better model for our use case.

**Q:** What would a poor PR curve look like?
**A:** A poor PR curve would drop quickly, showing that as we try to find more true positives, precision falls sharply—meaning the model makes too many mistakes as it tries to catch more positives.

---

## 3. Confusion Matrix
**Q:** Why did you include a confusion matrix in your evaluation?
**A:** The confusion matrix gives a detailed breakdown of the model’s predictions, showing exactly where it gets things right and wrong. This helps us and our audience understand not just overall performance, but the types of mistakes the model makes, which is important for practical use.
**Q:** Is a high AUC always enough to trust the model?
**A:** Not always. A high AUC means the model is good at ranking, but it doesn’t tell us about performance at a specific threshold. That’s why we also look at the PR curve and confusion matrix.

**Q:** What would a low AUC mean for your system?
**A:** A low AUC would mean the model struggles to tell self-promotional from neutral content, making it unreliable for our highlighting purpose.
**Q:** Why does the confusion matrix show more false positives (lower-quality sentences classified as high)?
**A:** This is intentional. We prefer to catch more potentially valuable content, even if it means a few extra false positives. Since our tool is for highlighting, not rejecting, this bias is appropriate.

---

## 4. Scoring Metrics (Accuracy, Determinacy, Separation)
**Q:** What do the numbers in each cell mean?
**A:** Each cell shows how many times the model made a certain type of prediction—true positives, false positives, true negatives, and false negatives. This helps us see exactly where the model is strong or weak.

**Q:** How do you use the confusion matrix to improve the model?
**A:** By analyzing which types of errors are most common, we can adjust the model or threshold to reduce those mistakes, or add more training data for the problematic cases.
**Q:** Why did you include these scoring metrics in your evaluation?
**A:** These metrics give a more complete picture of model performance. Accuracy shows overall correctness, determinacy shows confidence in predictions, and separation shows how well the model distinguishes between strong and weak self-promotion. Together, they help us explain the model’s strengths and limitations in plain terms.
**Q:** What do the accuracy, determinacy, and separation scores mean?
**A:** Accuracy shows how often the model is right overall. High determinacy means the model is confident when a sentence is clearly self-promotional or not. Separation shows how well the model distinguishes between strong and weak self-promotion.

---
**Q:** Why not just use accuracy alone?
**A:** Accuracy can be misleading if the data is imbalanced. That’s why we also use determinacy and separation to get a fuller picture of model performance.

**Q:** What does a low separation score mean?
**A:** It means the model has trouble distinguishing between strong and weak self-promotion, so its predictions are less reliable.

## 5. Score Distribution Histogram
**Q:** Why did you include a score distribution histogram in your evaluation?
**A:** The histogram lets us see how the model’s scores are distributed for each class. This helps us check if the model is really separating self-promotional from neutral content, or if there’s too much overlap. It’s a visual way to confirm the model’s effectiveness.
**Q:** What does the score distribution histogram show?
**A:** It shows that most truly self-promotional sentences get high scores, and most neutral ones get low scores. There’s a clear gap, meaning the model is good at telling them apart.
**Q:** What does it mean if the two classes overlap a lot in the histogram?
**A:** If there’s a lot of overlap, it means the model isn’t confident in telling the classes apart, which would reduce its usefulness for highlighting.

---

## 6. Embedding Visualizations (PCA, UMAP)
**Q:** Why did you include PCA and UMAP embedding visualizations in your evaluation?
**A:** These visualizations help us and our audience see that the model is learning meaningful differences between self-promotional and neutral sentences. If the points form clear groups, it means the model’s internal representations are working as intended, which builds trust in the system.
**Q:** What would it mean if the points were all mixed together?
**A:** If the points are mixed, it means the model’s internal features aren’t capturing the differences between self-promotional and neutral sentences, which would hurt performance.

**Q:** Why use both PCA and UMAP?
**A:** PCA and UMAP show different aspects of the data. PCA is good for showing overall structure, while UMAP can reveal finer details and local clusters.
**Q:** Why show PCA and UMAP plots of BERT embeddings?
**A:** These plots help us see that the model’s internal representations of self-promotional and neutral sentences form separate groups, which means the model is learning meaningful differences.

---

## 7. Keyword Highlighting Confusion Matrix
**Q:** What are the consequences of false positives or false negatives in keyword highlighting?
**A:** False positives mean highlighting unimportant words, which could distract users. False negatives mean missing important keywords, which could make the résumé less effective. We aim to minimize both.
**Q:** Why did you include a confusion matrix for keyword highlighting in your evaluation?
**A:** This confusion matrix shows exactly how well the keyword module finds important terms and avoids mistakes. It’s a clear, transparent way to demonstrate the module’s reliability and to identify any areas for improvement.
**Q:** How well does the keyword highlighting module work?
**A:** It finds almost all important keywords with very few mistakes, showing high precision and recall. This means it’s reliable for highlighting key résumé terms.

---
**Q:** How do you test robustness against new or unseen data?
**A:** We evaluate the model on data it hasn’t seen before, and check if the metrics stay high. If they do, it means the model generalizes well.

## 8. General Model Robustness
**Q:** Why did you include an overall robustness assessment in your evaluation?
**A:** By looking at all metrics and visualizations together, we can show that our system is not just good in one area, but is consistently strong and reliable. This reassures users and researchers that the tool is practical for real-world use.
**Q:** How do you know your system is robust and reliable?
**A:** Consistently high scores across all metrics, clear separation in visualizations, and strong performance on both self-promotion and keyword tasks show our system is technically sound and practical for real use.
**Q:** How do you choose the best threshold?
**A:** We look at the trade-off between precision and recall, and pick a threshold that matches our application’s needs—usually one that favors recall for highlighting.

---

## 9. Threshold Selection
**Q:** Why did you discuss threshold selection in your evaluation?
**A:** Threshold selection is important because it lets us adjust the model’s behavior for different needs—whether we want to be more strict or more inclusive. Discussing this shows we understand how to tune the system for different applications.
**Q:** How would you adapt the tool for a different use case?
**A:** If the tool were used for filtering or screening instead of highlighting, we could adjust the threshold to favor precision, so only the most certain cases are flagged.
**Q:** Can you adjust the model to reduce false positives?
**A:** Yes, by changing the decision threshold, we can make the model stricter or more lenient, depending on what’s needed for the application.

---

## 10. Practical Use
**Q:** Why did you focus on recall over precision for your tool?
**A:** For our application, it’s more important not to miss valuable résumé content, even if it means highlighting a few extra sentences. This focus matches the real-world needs of users who want to see all possible strong content, not risk missing something important.
**Q:** Why is recall more important than precision for your tool?
**A:** Because we want to make sure we don’t miss any valuable résumé content. It’s better to highlight a few extra sentences than to miss something important for the user.

---

Feel free to add more questions as needed based on the audience’s focus.