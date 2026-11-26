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

## Entry 2 - Author: Mariem Dridi (Module 3: Baseline and Evaluation)

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