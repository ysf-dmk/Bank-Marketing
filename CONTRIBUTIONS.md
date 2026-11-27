# CONTRIBUTIONS.md - Individual Contributions

## Project: Bank Marketing - Predicting Term Deposit Subscription

## Team responsibilities

| Team member | Module | Responsibility |
|---|---|---|
| Chaima | Module 1 | Exploratory Data Analysis and Data Quality |
| Youssef Damak | Module 2 | Data Cleaning and Preprocessing |
| Mariem Dridi | Module 3 | Baseline Models and Evaluation |
| Adam Nasri | Module 4 | Alternative Models and Performance Comparison |

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

- PROMPT_LOG.md: Entry 3, initial and constrained prompts.
- AI_AUDIT.md: identification, correction and experimental
  verification of AI-generated methodological and
  technical issues.

### Main experimental results — Random Split

- Logistic Regression ROC-AUC: 0.8081.
- Logistic Regression Average Precision: 0.4742.
- Default threshold F1-score: 0.3350.
- Optimized threshold F1-score: 0.5156.
- Recall improvement: from 0.2209 to 0.5269.
- Selected validation threshold: approximately 0.2385.

### Additional experiment — Temporal Sensitivity Analysis

- Reused the chronological time_split() function from
  src/data_prep.py.
- Verified temporal split sizes and class distributions.
- Retrained DummyClassifier and Logistic Regression using
  the temporal training set.
- Evaluated both models exclusively on temporal validation.
- Optimized the classification threshold independently.
- Generated confusion matrices and comparative visualizations.
- Compared random and temporal validation performance.
- Documented temporal generalization limitations.

### Temporal experimental results

- Training positive class rate: 4.81%.
- Validation positive class rate: 11.07%.
- ROC-AUC: 0.4836.
- Average Precision: 0.0984.
- Optimized F1-score: 0.2187.
- Optimized recall: 0.8147.
- Optimized precision: 0.1263.
- Optimized threshold: approximately 7.7963e-06.

### Individual accountability

Responsible for the implementation, verification,
interpretation and documentation of Module 3.

AI-generated recommendations were reviewed and
experimentally verified where applicable.

No test data was used for model evaluation or threshold
selection during Module 3 experiments.

---

## Module 4 - Alternative Models and Performance Comparison

**Author:** Adam Nasri

**Deliverable:** Alternative_Models_Comparison.ipynb

### Completed tasks

- Verification of the cleaned dataset (41,176 rows) and exact
  alignment with the shared stratified 60/20/20 split
  (`src/data_prep.py`, `random_state=42`).
- Reproduction of Module 3 baselines (`DummyClassifier` and
  `LogisticRegression`) on `X_val` for direct comparison.
- Implementation and 5-fold stratified hyperparameter tuning
  (`RandomizedSearchCV` optimizing `average_precision`) of
  Random Forest, XGBoost, and LightGBM inside leak-free
  scikit-learn `Pipeline` objects.
- Implementation of an out-of-course Calibrated Multi-Layer
  Perceptron (`MLPClassifier` wrapped with `CalibratedClassifierCV`),
  grounded in tabular deep learning and probability calibration
  literature (Gorishniy et al., 2021; Niculescu-Mizil & Caruana, 2005).
- Comparative ROC and Precision-Recall curve analysis on `X_val`.
- Global feature attribution via SHAP (`TreeExplainer`) and
  validation error profiling across TP, FP, FN, and TN outcomes.
- Chronological (temporal) 60/20/20 split robustness experiment
  quantifying macroeconomic concept drift (May 2008 – November 2010).
- Single, final holdout evaluation on the untouched `X_test` set.

### Human decisions

- Encapsulated `build_preprocessor()` inside every cross-validation
  `Pipeline` so scaling and one-hot encoding are refitted strictly
  within each training fold.
- Selected `average_precision` (PR-AUC) as the primary optimization
  criterion rather than accuracy due to the 88.73% / 11.27% class
  imbalance.
- Locked the champion decision threshold (`0.5514`) on `X_val` before
  running the single evaluation on `X_test`.

### AI governance contributions

- PROMPT_LOG.md: Entries documenting Module 4 experimental design,
  stratified CV tuning, out-of-course calibrated MLP justification,
  SHAP error analysis, and chronological drift evaluation.
- AI_AUDIT.md: Audit 6 (preventing preprocessing leakage inside CV
  and correcting uncalibrated neural probability outputs).

### Main experimental results

- Champion Model (`Random Forest Tuned`) Validation PR-AUC: **0.5027**
  (vs. **0.4742** Logistic Regression baseline), ROC-AUC: **0.8210**,
  F1-Score: **0.5315** (threshold **0.5514**, Recall **0.6002**).
- Out-of-Course Model (`Calibrated MLP`) Validation PR-AUC: **0.4798**,
  ROC-AUC: **0.8072**, F1-Score: **0.5194**.
- Holdout Test Set (`X_test`) Champion Results: PR-AUC **0.4632**
  (vs. **0.4423** baseline), ROC-AUC **0.7953**, F1-Score **0.5049**
  (Recall improved from **0.5032** to **0.5830**).
- Chronological Split Drift Check: Positive class rate shifted from
  **4.81%** (2008–2009 train) to **30.83%** (2010 test), reducing
  chronological validation PR-AUC to **0.1197** (ROC-AUC **0.5441**).

### Individual accountability

Responsible for the implementation, cross-validation tuning,
interpretability, temporal drift analysis, final holdout testing,
and documentation of Module 4.

AI-assisted code drafts were reviewed, executed, and verified
against the shared preprocessing assertions and baseline benchmarks.

---

## Other individual contributions

Chaima must complete her Module 1 section with her implemented
tasks, deliverables, experimental evidence and AI governance
contributions.