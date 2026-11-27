# CONTRIBUTIONS.md - Individual Contributions

## Project: Bank Marketing - Predicting Term Deposit Subscription

## Team responsibilities

| Team member | Module | Responsibility |
|---|---|---|
| Chaima Garoui | Module 1 | Exploratory Data Analysis and Data Quality |
| Youssef Damak | Module 2 | Data Cleaning and Preprocessing |
| Mariem Dridi | Module 3 | Baseline Models and Evaluation |
| Adam Nasri | Module 4 | Alternative Models and Performance Comparison |

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


---

## Module 1 - Exploratory Data Analysis and Data Quality

**Author:** Chaima Garoui

**Deliverables:** EDA_and_Data_Quality.ipynb, README.md, bank-additional-full.csv

### Completed tasks

- Selection and initial exploration of the UCI Bank Marketing dataset.
- Definition of the business objective and binary classification problem.
- Identification of numerical and categorical features.
- Verification of dataset dimensions and data types.
- Detection of missing values and categorical values labelled "unknown".
- Identification of 12 exact duplicate rows.
- Analysis of the target distribution and class imbalance.
- Descriptive statistical analysis of numerical features.
- Visualization of customer characteristics and subscription rates.
- Analysis of subscription rates across age groups, occupations and education levels.
- Identification of potential data leakage associated with duration.
- Identification of the special value 999 in pdays.
- Communication of EDA findings and preprocessing recommendations to Module 2.
- Preparation of the Week 1 project specification.
- Implementation of reproducible dataset loading directly from GitHub.
- Creation of the project README.md.

### Human decisions

- Use bank-additional-full.csv as the original dataset.
- Define the prediction objective before the marketing call.
- Recommend excluding duration because it is unavailable before the call.
- Preserve the original dataset during EDA and leave cleaning decisions to Module 2.
- Avoid relying on accuracy alone because of the imbalanced target distribution.
- Interpret high subscription rates in small customer groups with caution.
- Avoid treating observed associations as causal relationships.
- Replace temporary Colab file paths with a shared GitHub dataset URL.

### AI governance contributions

- PROMPT_LOG.md: EDA, statistical interpretation, data quality analysis and reproducibility prompts.
- AI_AUDIT.md: individual audit entry to be completed after reviewing and verifying the identified AI-generated issues.

### Main results

- 41,188 observations and 21 columns.
- 20 input features: 10 numerical and 10 categorical.
- No standard NaN missing values.
- 12 exact duplicate rows, representing approximately 0.03% of the dataset.
- 36,548 negative observations (88.73%).
- 4,640 positive observations (11.27%).
- Average customer age: 40.02 years.
- Median customer age: 38 years.
- 8,597 unknown values in default.
- 1,731 unknown values in education.
- 990 unknown values in housing and 990 in loan.
- 330 unknown values in job.
- 80 unknown values in marital.
- Customers aged 65 and above had an observed subscription rate of approximately 47.21%.
- Students had an observed subscription rate of approximately 31.4%.
- The illiterate education category contained only 18 observations, making its subscription rate of 22.22% unreliable as a general conclusion.

### Individual accountability

Responsible for the implementation, verification, interpretation and documentation of Module 1.

ChatGPT was used to support the development of the EDA notebook, explain statistical observations and assist with GitHub integration.

The generated code was executed in Google Colab. Numerical findings were checked against the dataset outputs before being included in the documentation.

The original dataset was preserved, and the identified data quality issues were communicated to the preprocessing module owner.


## Other individual contributions

Each team member must complete their own section
with their implemented tasks, deliverables,
experimental evidence and AI governance contributions.
