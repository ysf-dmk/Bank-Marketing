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
- Construction and validation-only selection of heterogeneous
  ensemble methods (`VotingClassifier` with soft voting and
  `StackingClassifier` with a Logistic Regression meta-learner)
  combining all four tuned candidate models.
- Comparative ROC and Precision-Recall curve analysis on `X_val`.
- Global feature attribution via SHAP (`TreeExplainer` on
  `LightGBM Tuned`) and validation error profiling across
  TP, FP, FN, and TN outcomes on the winning `Stacking` model.
- Chronological 60/20/20 split robustness experiment via the
  shared `src.data_prep.time_split()` helper, quantifying
  macroeconomic concept drift (May 2008 – November 2010).
- Single, final holdout evaluation on the untouched `X_test` set.

### Human decisions

- Encapsulated `build_preprocessor()` inside every cross-validation
  `Pipeline` so scaling and one-hot encoding are refitted strictly
  within each training fold.
- Selected `average_precision` (PR-AUC) as the primary optimization
  criterion rather than accuracy due to the 88.73% / 11.27% class
  imbalance.
- Routed SHAP attribution to `LightGBM (Tuned)` when the heterogeneous
  `Stacking` ensemble won on `X_val`, avoiding `TreeExplainer`
  incompatibility and runtime bottlenecks while keeping error
  profiling on the `Stacking` champion.
- Locked the champion decision threshold (`0.2469`) strictly on
  `X_val` before running the single evaluation on `X_test`.

### AI governance contributions

- PROMPT_LOG.md: Entry 4 and Entry 5 documenting Module 4
  experimental design, stratified CV tuning, out-of-course
  calibrated MLP justification, Voting/Stacking ensemble
  integration, SHAP error analysis, and `time_split()` alignment.
- AI_AUDIT.md: Audit 6 (preventing preprocessing leakage inside CV
  and correcting uncalibrated neural probability outputs) and
  Audit 7 (SHAP explainer compatibility with Stacking ensembles
  and shared `time_split()` synchronization).

### Main experimental results

- Champion Model (`Stacking — 4 Tuned Models`) Validation PR-AUC:
  **0.5038** (vs. **0.5027** Tuned Random Forest, **0.5006** Soft
  Voting, and **0.4742** Logistic Regression baseline), ROC-AUC:
  **0.8231**, F1-Score: **0.5362** (threshold **0.2469**, Recall
  **0.5787**, Precision **0.4995**).
- Individual Tuned Models on `X_val`: Random Forest PR-AUC **0.5027**,
  XGBoost **0.4920**, LightGBM **0.4919**, Out-of-Course Calibrated
  MLP **0.4798** (F1 **0.5194**).
- Holdout Test Set (`X_test`) Champion Results (`Stacking`): PR-AUC
  **0.4625** (vs. **0.4423** baseline), ROC-AUC **0.7975** (vs.
  **0.7908**), F1-Score **0.5086** (vs. **0.4812**), improving
  subscriber recall from **0.5032** to **0.5593** (519/928 captured).
- Shared `time_split()` Drift Check: Positive class rate shifted from
  **4.81%** (2008–2009 train) to **11.07%** (val) and **30.83%**
  (2010 test), reducing chronological validation PR-AUC to **0.0930**
  (ROC-AUC **0.4395**).

### Individual accountability

Responsible for the implementation, cross-validation tuning,
ensemble modeling, interpretability, temporal drift analysis,
final holdout testing, and documentation of Module 4.

AI-assisted code drafts were reviewed, executed, and verified
against the shared preprocessing assertions and baseline benchmarks.

---

## Module 1 - Exploratory Data Analysis, Data Quality and Model Error Analysis

**Author:** Chaima Garoui

**Deliverables:** `EDA_and_Data_Quality.ipynb`, `Model_Comparison_and_Error_Analysis.ipynb`

### Completed tasks — Exploratory Data Analysis and Data Quality

- Exploration of the original Bank Marketing dataset containing 41,188 observations and 21 variables.
- Identification of numerical and categorical features.
- Analysis of missing values and implicit missing information represented by the "unknown" category.
- Identification of 12 exact duplicate rows.
- Analysis of target class imbalance.
- Exploration of customer demographic, financial and campaign-related characteristics.
- Analysis of subscription rates across age, occupation, education and other customer characteristics.
- Identification of potential data leakage associated with the duration variable.
- Identification of the special meaning of the pdays value 999.
- Development of statistical visualizations and interpretation of observed patterns.
- Application of 95% Wilson confidence intervals to assess the reliability of subscription rates across education levels.

### Additional experiments — Week 3 Model Comparison and Error Analysis

- Comparison of Dummy Classifier, default Logistic Regression and threshold-optimized Logistic Regression on the validation set.
- Analysis of false positives and false negatives.
- Business cost sensitivity and break-even analysis under different hypothetical misclassification costs.
- Evaluation of model performance across age, occupation, education and marital status groups.
- Calculation and visualization of 95% Wilson confidence intervals for subgroup recall.
- Identification of subgroup performance disparities and statistical limitations.
- Development of cumulative gains and lift curves.
- Top-K customer targeting analysis under different marketing budget scenarios.
- Interpretation of experimental results and business implications.
- Full notebook execution and publication on GitHub.

### Human decisions

- Treat "unknown" values as implicit missing information rather than standard missing values.
- Recommend excluding duration from predictive modeling to prevent data leakage.
- Avoid relying exclusively on accuracy because of the imbalanced target distribution.
- Use Wilson confidence intervals rather than interpreting subgroup percentages without considering sample sizes.
- Use recall and false positive rates to investigate differences in model performance across customer groups.
- Preserve the independent test set during exploratory model comparison.
- Treat business misclassification costs as hypothetical scenarios rather than actual financial estimates.
- Avoid interpreting observed demographic disparities as proof of discrimination or causal relationships.

### AI governance contributions

- AI-assisted development of exploratory analysis, visualization and statistical evaluation code.
- Review and execution of AI-generated code in Google Colab.
- Statistical verification of potentially misleading subscription rates in small demographic groups.
- Verification of subgroup recall estimates using Wilson confidence intervals.
- Documentation of methodological assumptions, experimental limitations and human decisions.

### Main experimental results — Exploratory Data Analysis

- Original dataset: 41,188 observations and 21 variables.
- Target distribution: 88.73% non-subscribers and 11.27% subscribers.
- No conventional missing values, but several categorical variables contain "unknown" values.
- 12 exact duplicate observations identified.
- Customers aged 65 and above showed a subscription rate of approximately 47.21%.
- The illiterate education category showed a subscription rate of 22.22%, based on only 18 observations.
- The corresponding 95% Wilson confidence interval was approximately [9.00%, 45.21%], demonstrating substantial statistical uncertainty.

### Main experimental results — Model Error Analysis

- Validation dataset: 8,235 customers, including 928 actual subscribers.
- Default Logistic Regression: 91 false positives and 723 false negatives.
- Threshold-optimized Logistic Regression: 480 false positives and 439 false negatives.
- Recall improved from 22.09% to 52.69% after threshold optimization.
- Business cost break-even between Dummy Classifier and default Logistic Regression: approximately €0.4439 per false negative.
- Business cost break-even between default and optimized Logistic Regression: approximately €1.3697 per false negative, assuming a fixed false positive cost of €1.

### Subgroup performance results

- Customers aged 65 and above: recall of 90.16%, but a false positive rate of 75.38%.
- Customers aged 35–44: recall of 42.98%.
- Students: recall of 84.21%.
- Blue-collar workers: recall of 25.83%.
- Entrepreneurs: recall of 16.00%.
- University degree holders: recall of 59.53%, compared with 32.63% for customers with basic.9y education.
- Single customers: recall of 60.00%, compared with 47.98% for married customers.

Confidence intervals were used to assess the uncertainty of these estimates. Groups with few or no actual subscribers were interpreted cautiously.

### Business targeting results

- Targeting the top 5% of customers captured 28.23% of observed subscribers, with a lift of 5.64.
- Targeting the top 10% captured 46.88% of observed subscribers, with a lift of 4.68.
- Targeting the top 20% captured 65.19% of observed subscribers, with a lift of 3.26.
- The top 20% targeting strategy would reduce the number of contacts from 8,235 to 1,647 in the validation simulation.

These results demonstrate the potential value of probability-based customer ranking for marketing resource allocation. They do not establish actual campaign profitability or causal impact.

### Individual accountability

Responsible for the implementation, verification, interpretation and documentation of exploratory data analysis, data quality assessment, statistical reliability analysis, model error analysis and business-oriented evaluation.

AI-assisted code was reviewed, executed and interpreted by the contributor. Experimental results were checked against the dataset and shared baseline results.

The independent test set was not used in the individual model error analysis or customer targeting experiments.

**Status:** Individual EDA and baseline error analysis completed. Extension of the analysis to the team's alternative and ensemble models remains pending.
