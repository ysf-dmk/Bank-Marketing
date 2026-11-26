# CONTRIBUTIONS.md - Individual Contributions

## Project: Bank Marketing - Predicting Term Deposit Subscription

## Team responsibilities

| Team member | Module | Responsibility |
|---|---|---|
| Chaima | Module 1 | Exploratory Data Analysis and Data Quality |
| Youssef Damak | Module 2 | Data Cleaning and Preprocessing |
| Mariem Dridi | Module 3 | Baseline Models and Evaluation |
| Adam | Module 4 | Alternative Models and Performance Comparison |

---

## Module 3 - Baseline Models and Evaluation

**Author:** Mariem Dridi

**Deliverable:** Baseline_Model.ipynb

### Completed tasks

- Verification of the cleaned dataset and target distribution.
- Reuse of the shared data preparation module.
- Implementation of a Dummy Classifier.
- Implementation of a Logistic Regression baseline.
- Construction of training-only preprocessing pipelines.
- Evaluation using accuracy, precision, recall, F1-score,
  ROC-AUC and Average Precision.
- Confusion matrix visualization.
- ROC and Precision-Recall curve analysis.
- Decision threshold optimization using validation F1-score.
- Comparison and interpretation of baseline results.
- Identification of methodological limitations.
- Preservation of the independent test set.

### AI governance contributions

- PROMPT_LOG.md: Entry 2, initial and constrained prompts.
- AI_AUDIT.md: identification, correction and experimental
  verification of AI-generated methodological and
  technical issues.

### Main experimental results

- Logistic Regression ROC-AUC: 0.8081.
- Logistic Regression Average Precision: 0.4742.
- Default threshold F1-score: 0.3350.
- Optimized threshold F1-score: 0.5156.
- Recall improvement: from 0.2209 to 0.5269.
- Selected validation threshold: approximately 0.2385.

### Individual accountability

Responsible for the implementation, verification,
interpretation and documentation of Module 3.

AI-generated recommendations were reviewed and
experimentally verified where applicable.

---

## Other individual contributions

Each team member must complete their own section
with their implemented tasks, deliverables,
experimental evidence and AI governance contributions.