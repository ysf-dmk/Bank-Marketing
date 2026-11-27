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

## Module 2 - Data Cleaning and Preprocessing

**Author:** Youssef Damak

**Deliverables:** src/data_prep.py, Cleaning_Preprocessing.ipynb,
bank_clean.csv

### Completed tasks

- Removal of the 12 exact duplicate rows (41,188 to 41,176).
- Removal of duration (only known after the call, so it
  leaks the outcome).
- Handling of pdays: 999 replaced by 0, plus the binary
  feature was_contacted_before.
- "unknown" kept as its own category.
- Target y encoded as 0/1.
- Stratified 60/20/20 split with random_state=42, done
  before anything is fitted.
- Unfitted preprocessor: StandardScaler on 9 numeric
  columns, OneHotEncoder on 10 categorical columns,
  was_contacted_before passed through.
- Notebook rewritten to call the shared module, and an
  assertions cell added.
- Preservation of the independent test set.

### Human decisions

- Keep the near-duplicate rows (identical on all features
  once duration is dropped) and document the cost.
- Name the cleaned column pdays everywhere.
- Use one shared module instead of repeating cleaning code
  in notebooks.

### AI governance contributions

- PROMPT_LOG.md: Entry 1, initial and follow-up prompts.
- AI_AUDIT.md: Audit 3 (near-duplicates) and Audit 4
  (pdays column mismatch), each with verification code
  and results.

### Main results

- 41,176 rows after cleaning, 11.27% positive.
- Split: 24,705 train / 8,235 validation / 8,236 test.
- Positive share: 11.26% train, 11.27% validation,
  11.27% test.
- 63 features after preprocessing.
- 2,009 rows identical to another row on all features
  (kept on purpose).
- All assertions in the notebook pass.

### Individual accountability

Responsible for the design decisions, verification and
documentation of Module 2.

src/data_prep.py was drafted with Claude Code from the
constrained prompt in PROMPT_LOG.md - Entry 1. The draft was
run, checked against the data and corrected as described in
AI_AUDIT.md. A second AI session (Claude chat) was used to
review the draft and to propose the assertion checks.

---

## Other individual contributions

Each team member must complete their own section
with their implemented tasks, deliverables,
experimental evidence and AI governance contributions.