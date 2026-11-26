# PROMPT_LOG.md - high-stakes AI interactions

---
## Entry 1 - Author: Youssef Damak (Module 2: preprocessing)
**Date:** 2026-11-25
**Prompt:**
Role: you are an engineering accelerator for Module 2 of a ML group project. I am accountable for every line, so write for review: short, commented, no clever tricks.

Data: UCI Bank Marketing, bank-additional-full.csv (sep=";"), 41,188 rows, 10 numeric + 10 categorical inputs, target y (yes/no, 11.27% yes). No NaN, "unknown" appears in categorical columns, 12 duplicate rows, pdays==999 means never contacted. File is date-sorted, no date column. Task: predict subscription BEFORE the call.

Deliver a single file src/data_prep.py with these functions only:
1. load_raw(path): read the CSV, return a DataFrame.
2. clean(df): drop duplicates, add was_contacted_before from pdays, clean pdays, drop duration (post-call leakage), map y to 0/1.
3. split(df): stratified 60/20/20 train/val/test, random_state=42.
4. build_preprocessor(): UNFITTED ColumnTransformer: StandardScaler on numeric, OneHotEncoder(handle_unknown="ignore") on categorical, "unknown" kept as its own category.

Hard constraints: nothing is fitted before the split; the test set is never touched; no baseline model, metrics or EDA.

Then answer separately:
A. For each function, one sentence on why each design choice was made.
B. Every assumption you made, and how I can verify each in the data.
C. Three ways this code could silently fail or leak, including what a random split hides given the date-sorted file.
D. Assertions I should run (row counts, no duplicates, no split overlap, class share per split, scaler fitted on train only).
E. Five oral-defence questions about this module.

If you are unsure of any syntax or library behaviour, say so instead of guessing.

**Follow-up prompt:**
Follow-up on src/data_prep.py (Module 2, UCI Bank Marketing). I reviewed your draft and ran it. Findings:
1. Dedup runs before duration is dropped, so 2,009 rows are identical on all remaining features (481 test rows match a train row). DECISION: keep them, only exact duplicates are removed. Add a comment in clean() stating this and the cost (slightly optimistic test scores).
2. The notebook Cleaning_Preprocessing.ipynb uses pdays_clean and drops pdays, but data_prep.py keeps pdays. DECISION: the column is named pdays everywhere.

Do exactly this and nothing else:
- Update Cleaning_Preprocessing.ipynb so it calls data_prep.load_raw, clean, split and build_preprocessor instead of repeating the cleaning code, and regenerate bank_clean.csv with a pdays column (no pdays_clean).
- Do not touch the baseline cells, and do not change the split sizes, random_state=42, or the preprocessor definition.
- Add an assertions cell: sizes 24,705 / 8,235 / 8,236, class share per split, no index overlap between splits, "pdays" in columns and "pdays_clean" not, preprocessor unfitted before fit, scaler mean computed from train only.
- Add a comment that a random split hides the date drift in this date-sorted file.

Show me a diff of every change, tell me anything you are unsure of instead of guessing, and list any other issue you notice without fixing it.

---

## Entry 3 - Author: Mariem Dridi (Module 3: Baseline and Evaluation)

**Date:** 2026-11-26

**AI tool:** ChatGPT

**Status:** Executed and reviewed.

### Initial prompt

Build a classification baseline for the Bank Marketing dataset and recommend how to evaluate and select its decision threshold.

### Refined prompt

Role: Act as an engineering assistant, not an autonomous analyst. Provide reviewable Python code and justify each methodological choice. Flag uncertainty instead of guessing.

Data: UCI Bank Marketing, bank_clean.csv, 41,176 rows, 20 predictors, binary target y (0/1), 11.27% positives. The post-call duration variable has been removed. Reuse split() and build_preprocessor() from src/data_prep.py. The shared stratified split is 60/20/20 with random_state=42.

Objective: Predict term deposit subscription before the call. Compare DummyClassifier(strategy="prior") and unweighted LogisticRegression(max_iter=2000).

Constraints:

- Fit separate preprocessing/model pipelines on training data only. Do not clean the encoded target again or access test data for model or threshold selection.
- Report validation accuracy, precision, recall, F1, ROC-AUC and Average Precision. Explain why accuracy alone is insufficient and distinguish Average Precision from trapezoidal PR-AUC.
- Use positive-class probabilities for ROC-AUC and Average Precision. Verify class ordering.
- Select a threshold maximizing validation F1; explain why this is not necessarily economically optimal. Preserve full threshold precision.
- Discuss independence of observations, duplicate feature vectors, temporal drift, convergence, logit linearity and probability calibration. Do not assume these properties have been verified.
- Return code, assumptions, checks, fallback procedures and limitations. Do not fabricate results.

Fallbacks:

- If schema or target checks fail, stop and investigate the shared preprocessing module.
- If class proportions or split sizes differ, verify dataset version and split configuration before proceeding.
- If convergence fails, inspect scaling and solver convergence before adjusting iterations.
- If duplicate feature vectors cross splits, quantify overlap and propose a group-aware sensitivity analysis; do not silently change the shared split.
- If calibration is poor, propose training-only cross-validated calibration before final evaluation.
- If business costs are unavailable, use F1 as an explicit proxy rather than claiming profit optimization.

### Prompt iteration and human review

Both prompts were executed using ChatGPT on 2026-11-26.

The initial response proposed appropriate baseline models,
imbalance-aware metrics and validation-based threshold
selection. However, it independently reconstructed the
preprocessing pipeline without following the shared
feature engineering methodology.

The refined prompt improved the response by introducing:

- Explicit dataset schema and target validation.
- Reuse of the shared preprocessing module.
- Explicit data leakage boundaries.
- Statistical assumption checks.
- Convergence monitoring.
- Positive-class probability verification.
- Duplicate feature-vector diagnostics.
- Probability calibration diagnostics.
- Explicit fallback procedures.

Despite these improvements, the refined response still
contained an incorrect assumption about the shared
preprocessing function signature.

Human review and experimental verification were conducted
to identify and correct the sentinel handling and function
signature issues.

The additional statistical diagnostics proposed by the AI
have not all been executed.

### Conclusion

Explicit prompt guardrails improved the quality,
consistency and reproducibility of the AI-generated
methodology. However, human statistical judgment,
code inspection and experimental verification remained
necessary.

**Related audit:** AI_AUDIT.md — Module 3: Baseline and
Evaluation, for detailed error analysis, corrected code
and experimental evidence.




---

## Entry 2 - Author: Chaima Garoui (Module 1: EDA and data quality)

**Date:** Week 1–2

**Note:** The following prompts are reconstructed from the AI-assisted work carried out during this module.

**Prompt 1:** Role: act as a Machine Learning tutor and guide me through the exploratory data analysis of our group project. Explain each step before moving to the next one.

Data: UCI Bank Marketing, bank-additional-full.csv, separator=";", 41,188 observations, 20 input features and a binary target variable y.

Objective: predict whether a customer will subscribe to a term deposit before the marketing call.

Help me perform the following tasks in Google Colab:

1. Load the dataset and verify its dimensions.
2. Identify numerical and categorical variables.
3. Check for missing values, including categorical values labelled "unknown".
4. Detect duplicate rows and calculate their percentage.
5. Analyze the target distribution and explain the impact of class imbalance.
6. Generate descriptive statistics and relevant visualizations.

Provide Python code one step at a time. Wait for my output before continuing. Do not clean the dataset or train any model yet, as these tasks belong to other team members.

Explain the results in simple English and do not invent numerical findings.

**Prompt 2:** I have completed the initial EDA. Help me interpret the following observations and identify the main issues that need to be communicated to the preprocessing team.

Findings:

- 41,188 observations and 21 columns, including the target.
- No standard NaN values.
- Several categorical features contain "unknown" values.
- 12 exact duplicate rows.
- 88.73% of customers did not subscribe, while 11.27% subscribed.
- Subscription rates vary across age groups, occupations and education levels.
- The duration variable is only available after the call.
- A value of 999 in pdays indicates no previous contact.

Explain:

A. Why accuracy alone is not sufficient for this classification problem.

B. Why duration introduces data leakage.

C. Whether the duplicate rows should be removed, considering that the dataset has no unique customer identifier.

D. How unknown categorical values and the special value in pdays could be handled during preprocessing.

E. Why high subscription rates in small customer groups should be interpreted carefully.

Distinguish observed results from assumptions. Do not interpret statistical associations as causal relationships.

**Prompt 3:** I have completed my EDA notebook in Google Colab and need to share my work with the other three members of our Machine Learning project.

Repository: ysf-dmk/Bank-Marketing.

The notebook currently loads the dataset from /content/bank-additional-full.csv, which requires manually uploading the file in every new Colab session.

Help me:

1. Upload the original dataset to the shared GitHub repository.
2. Replace the temporary dataset path with a GitHub raw URL.
3. Resolve any dataset loading errors.
4. Verify that the notebook runs successfully from beginning to end.
5. Upload the updated notebook under a filename that reflects my EDA contribution.
6. Create a README.md containing the project objective, dataset description, EDA findings and preprocessing recommendations.

Do not modify the original dataset or implement the preprocessing pipeline.

The documentation should clearly distinguish completed work from planned experiments. Do not report model performance before the baseline has been implemented.

Guide me step by step and explain the purpose of each action.
