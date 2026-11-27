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

**Prompt 0 (sent first, before the prompt above; verbatim):**
I'm doing Module 2 (cleaning, feature engineering, split, unfitted preprocessor) of a ML group project on UCI Bank Marketing (bank-additional-full.csv, sep=";", 41,188 rows, target y, 11.27% yes). I will write the code myself. Don't write the solution. Give me ordered steps, each with what to do, why in one sentence, and the minimal pandas/sklearn syntax.

Scope: drop duplicates, handle pdays==999, drop duration (leakage), keep "unknown" as a category, stratified 60/20/20 split (random_state=42), and an unfitted ColumnTransformer (scaler + one-hot) that I fit on train only. Not my scope: EDA, baseline model, metrics.

After the steps, give me assertions to run (row counts, no duplicates, no split overlap, class share per split) and 5 oral-defence questions. Say which syntax you're unsure of. Then wait for my code and review it without rewriting it.

**Follow-up prompt (sent after I ran the draft; verbatim):**
Follow-up on src/data_prep.py (Module 2, UCI Bank Marketing). I reviewed your draft and ran it. Findings:
1. Dedup runs before duration is dropped, so 2,009 rows are identical on all remaining features (481 test rows match a train row). DECISION: keep them, only exact duplicates are removed. Add a comment in clean() stating this and the cost (slightly optimistic test scores).
2. The notebook Cleaning_Preprocessing.ipynb uses pdays_clean and drops pdays, but data_prep.py keeps pdays. DECISION: the column is named pdays everywhere.

Do exactly this and nothing else:
- Update Cleaning_Preprocessing.ipynb so it calls data_prep.load_raw, clean, split and build_preprocessor instead of repeating the cleaning code, and regenerate bank_clean.csv with a pdays column (no pdays_clean).
- Do not touch the baseline cells, and do not change the split sizes, random_state=42, or the preprocessor definition.
- Add an assertions cell: sizes 24,705 / 8,235 / 8,236, class share per split, no index overlap between splits, "pdays" in columns and "pdays_clean" not, preprocessor unfitted before fit, scaler mean computed from train only.
- Add a comment that a random split hides the date drift in this date-sorted file.

Show me a diff of every change, tell me anything you are unsure of instead of guessing, and list any other issue you notice without fixing it.

**Date:** 2026-11-25
**AI tool:** Claude Code (draft), Claude chat (review)
**Status:** Executed, reviewed, corrected.

### Prompt iteration and human review
- Initial prompt: asked for a constrained draft of src/data_prep.py (4 functions, leakage rules, assumptions, checks, oral questions).
- What the AI got right: [e.g. duration dropped, nothing fitted before the split, unfitted preprocessor, stratified split].
- What I found when I ran it: [dedup before dropping duration -> 2,009 near-duplicates; pdays vs pdays_clean mismatch between the module and the notebook].
- My decisions: kept the near-duplicates (no customer id, can't tell if they are the same customer); name the column pdays everywhere.
- Follow-up prompt: gave those decisions and asked for the notebook to call the shared module, plus an assertions cell. I read the diff and ran the assertions myself.
- Not verified: effect of the near-duplicates on scores.

### Conclusion
[1-2 sentences: what the guardrails improved, what still needed human checking.]

**Related audit:** AI_AUDIT.md - Audit 3 and Audit 4.

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

## Entry 4 - Author: Adam Nasri (Module 4: Alternative Models and Performance Comparison)

**Date:** 2026-11-27

**AI tool:** Gemini

**Status:** Executed, verified, and integrated into `Alternative_Models_Comparison.ipynb`.

### Initial prompt

Help me build my part (Module 4) of our group Machine Learning project on the UCI Bank Marketing dataset: train advanced models (Random Forest, XGBoost, LightGBM), add an out-of-course algorithm with a scientific reference, tune hyperparameters, analyze errors, and evaluate on the test set in VS Code.

### Refined prompt

Role: Act as a senior ML engineering assistant. Provide reviewable, self-contained Python cells and academic Markdown interpretations in English for `Alternative_Models_Comparison.ipynb`. Flag methodological pitfalls and never fabricate results.

Data & Shared Protocol: UCI Bank Marketing (`bank_clean.csv`, 41,176 rows, 20 predictors after dropping post-call `duration`, binary target `y` with 11.27% positives). Reuse `load_raw()`, `clean()`, `split()`, and `build_preprocessor()` from `src/data_prep.py` (`random_state=42`, stratified 60/20/20 split: 24,705 train / 8,235 val / 8,236 test).

Objective: Benchmark advanced model families against the Module 3 baselines (`DummyClassifier` and `LogisticRegression`, validation PR-AUC = 0.4742, ROC-AUC = 0.8081, F1 = 0.5156).

Constraints:

- Encapsulate `build_preprocessor()` inside every scikit-learn `Pipeline` passed to `RandomizedSearchCV` (`StratifiedKFold(n_splits=5)`) so scaling and one-hot encoding are refitted strictly inside each training fold.
- Optimize `average_precision` (PR-AUC) rather than raw accuracy due to the 88.73% / 11.27% class imbalance, and handle imbalance via `class_weight="balanced_subsample"` (Random Forest) and `scale_pos_weight` computed on `X_train` only (XGBoost, LightGBM).
- Include an out-of-course model grounded in peer-reviewed literature: a regularized `MLPClassifier` wrapped with post-hoc sigmoid calibration (`CalibratedClassifierCV`), citing Gorishniy et al. (2021) and Niculescu-Mizil & Caruana (2005).
- Tune decision thresholds strictly on `X_val` (maximizing F1-score). Never touch `X_test` during model selection or threshold tuning.
- Conduct global feature attribution via SHAP (`TreeExplainer`), validation error profiling across TP/FP/FN/TN outcomes, and a chronological 60/20/20 split check (May 2008 – November 2010) to quantify temporal concept drift.
- Execute a single, final evaluation on the untouched `X_test` set using the locked validation threshold.

### Prompt iteration and human review

Both prompts were executed using Gemini on 2026-11-27 while setting up the local VS Code environment.

The initial response outlined the candidate model families and evaluation metrics at a high level, without enforcing the exact shared split interface (`src/data_prep.py`) or isolating preprocessing inside cross-validation folds.

The refined prompt enforced strict engineering and statistical guardrails:

- Schema and split assertions verifying exact alignment with Module 2 and Module 3 (`24,705 / 8,235 / 8,236` rows, zero leakage of `duration`).
- Pipeline encapsulation inside `RandomizedSearchCV` to prevent scaling leakage across CV folds.
- Post-hoc probability calibration (`CalibratedClassifierCV`) for the out-of-course MLP model so predicted probabilities remain reliable under class imbalance.
- Chronological split evaluation exposing severe macroeconomic concept drift (positive rate shifting from `4.81%` in 2008–2009 to `30.83%` in 2010).
- Single-pass evaluation on `X_test` confirming that `Random Forest (Tuned)` outperformed the linear baseline (`Val PR-AUC = 0.5027`, `Test PR-AUC = 0.4632`).

### Conclusion

Constraining the AI to reuse the shared preprocessing module, nest transformers inside CV pipelines, and validate temporal drift prevented cross-validation leakage and produced reproducible, experimentally verified results across validation and holdout test sets.

**Related audit:** `AI_AUDIT.md` — Audit 5 (Module 4: Alternative Models and Performance Comparison).

---

## Entry 5 - Author: Adam Nasri (Module 4 Update: Voting & Stacking Ensembles + Shared `time_split()`)

**Date:** 2026-11-27

**AI tool:** Gemini

**Status:** Executed, verified, and integrated into `Alternative_Models_Comparison.ipynb`.

### Prompt

Update `Alternative_Models_Comparison.ipynb` to align with the final team synchronization protocol:
1. Combine the four tuned candidate models (`Random Forest`, `XGBoost`, `LightGBM`, and `Calibrated MLP`) using `VotingClassifier(voting="soft")` and `StackingClassifier` (with a `LogisticRegression` meta-learner and 5-fold stratified CV), selecting the champion model strictly on `X_val` before evaluating once on `X_test`.
2. Update ROC/PR comparison plots, SHAP attribution, and error profiling to reflect the ensemble champion without breaking `shap.TreeExplainer`.
3. Replace the manual chronological split in Section 4 with the shared `time_split()` function from `src/data_prep.py` to guarantee exact team-wide parity.

### Prompt iteration and human review

- **What the AI proposed:** Added `Soft Voting` and `Stacking` cells on top of the four tuned estimators, updated validation leaderboard sorting, and replaced the manual slicing in Section 4 with `src.data_prep.time_split()`.
- **What I caught and corrected during execution:**
  1. **SHAP Compatibility & Runtime Bottleneck:** Because `Stacking (4 Tuned Models)` outperformed all single models on `X_val` (`PR-AUC = 0.5038` vs. `0.5027` for Tuned RF), `shap.TreeExplainer` could not be applied directly to the heterogeneous stacking meta-estimator, and running `TreeExplainer` on a 200-tree `RandomForestClassifier` was computationally slow. I explicitly routed SHAP attribution to `LightGBM (Tuned)` while keeping the validation confusion matrix and error profile (`TP=537, FP=538, FN=391, TN=6769`) on the winning `Stacking` champion.
  2. **Module Reload & Protocol Verification:** Verified exact split dimensions (`24,705 / 8,235 / 8,236`) and class-prior shift (`4.81% -> 11.07% -> 30.83%`) under `time_split()`, and confirmed single-pass generalization on `X_test` (`Stacking Test PR-AUC = 0.4625`, `ROC-AUC = 0.7975`, `F1 = 0.5086` vs. `Logistic Regression Test PR-AUC = 0.4423`).

### Conclusion

Human review of SHAP estimator compatibility and strict enforcement of validation-only model selection ensured that combining all four tuned families via `Stacking` yielded the strongest generalization on `X_test` while remaining 100% aligned with the shared `src/data_prep.py` protocol.

**Related audit:** `AI_AUDIT.md` — Audit 6 (Module 4 Ensemble & Shared `time_split()` Update).

---

## Entry 6 - Author: Youssef Damak (Module 2 update: chronological split check, Week 3)

**Date:** 2026-11-27

**AI tool:** Claude Code

**Status:** Executed by the assistant on the real data; author review pending.

### Prompt 1 (verbatim): add time_split() and first checks

Role: engineering assistant for Module 2 of our ML group project (UCI Bank Marketing). I am accountable for every line, so write short, commented, reviewable code. Do not fabricate results.

Context: src/data_prep.py already has load_raw(), clean(), split() (stratified random 60/20/20, random_state=42, returns X_train, X_val, X_test, y_train, y_val, y_test) and build_preprocessor() (unfitted). bank-additional-full.csv is sorted by date but has no date column, so row order is our only time proxy. clean() keeps the original index (no reset_index) and drops 12 exact duplicates, leaving 41,176 rows.

Task (week 3): our spec promised to check whether splitting by time order fits how the bank would use the model.
1. Add time_split(df) to data_prep.py, same return signature as split(): first 60% of rows in original order = train, next 20% = validation, last 20% = test. No shuffling, no stratification. Sizes must be 24,705 / 8,235 / 8,236.
2. Write an assertions cell: index is increasing and every train index < every val index < every test index; no overlap; sizes as above; positive share of y in each block (print, do not assert equality).
3. Write a diagnostics cell on TRAIN vs VALIDATION only: for emp.var.rate, euribor3m, nr.employed, cons.price.idx, cons.conf.idx print the train min/max and the share of validation rows outside the train range.

Hard constraints: do not change split(), build_preprocessor() or random_state; nothing fitted before the split; the test set is never scored or inspected for model choices; no model training in this task.

Then answer separately:
A. One sentence on each design choice.
B. Every assumption you made (for example that row order equals time order) and how I can verify it in the data.
C. Three ways this could silently mislead us, including what changes about class balance and about the preprocessor's scaler fitted on early rows.
D. Which results a model must report on both splits so the comparison is fair.
E. Five oral-defence questions about this module.

If you are unsure of any syntax or library behaviour, say so instead of guessing. List any other issue you notice without fixing it. Then wait for my review.

### Prompt 2 (verbatim): stricter assertions, fuller drift diagnostics, inline-split check

Role: engineering assistant for Module 2 of our ML group project (UCI Bank Marketing). I am accountable for every line, so write short, commented, reviewable code. Do not fabricate results, and say so if you are unsure of any syntax or library behaviour.

Context:
- src/data_prep.py already has load_raw(), clean(), split() (stratified random 60/20/20, random_state=42), time_split() (chronological 60/20/20 on row order, same return signature as split(), raises if the index is not increasing) and build_preprocessor() (unfitted).
- clean() keeps the original index and returns 41,176 rows. The file is date-sorted with no date column, so row order is our time proxy.
- My notebook Cleaning_Preprocessing.ipynb has a cell that calls time_split(df_clean) and prints the sizes (variables Xt_train, Xt_val, Xt_test, yt_train, yt_val, yt_test). Teammates already report positive rates of about 4.81% in train, 11.07% in validation and 30.83% in test for the chronological split, and a large drop in scores on it.

Task: add two cells after that one. Do not change split(), time_split(), build_preprocessor() or random_state.
1. Assertions cell: sizes are exactly 24,705 / 8,235 / 8,236; the index is increasing; every train index is smaller than every validation index, and every validation index is smaller than every test index; no index appears in two parts; the three parts together equal df_clean's index; X and y have matching lengths; print (do not assert) the positive share of each block.
2. Drift diagnostics cell, TRAIN versus VALIDATION only (never test): for emp.var.rate, euribor3m, nr.employed, cons.price.idx and cons.conf.idx, print the train min and max, the validation min and max, and the share of validation rows outside the train [min, max]. Also print how many validation rows fall outside the train range on at least one of the five columns.
3. A short check that a split built inline as df.iloc[:int(0.6*n)], df.iloc[int(0.6*n):int(0.8*n)], df.iloc[int(0.8*n):] gives exactly the same indices as time_split(), so teammates can confirm their own chronological split matches the shared one.

Hard constraints: nothing is fitted; no model is trained; the test set is never scored and is not used for any statistics; no EDA beyond these cells.

Then answer separately: A. one sentence on each design choice; B. every assumption and how to verify it; C. three ways this could silently mislead us (include what a class share of 4.8% in train and 30.8% in test means for a threshold chosen on validation, and why macro columns act as time proxies); D. what to report next to the random-split results so the comparison is fair; E. five oral-defence questions. List any other issue you notice without fixing it. Then wait for my review.

### What the AI produced and what it measured (from the executed run)
- Prompt 1: added `time_split()` to `src/data_prep.py` (int-rounded 60/20/20 by row position, raises if the index is not increasing) and three notebook cells (call, assertions, drift diagnostics). Sizes 24,705 / 8,235 / 8,236.
- Prompt 2: replaced the assertions and diagnostics cells in place (same cell ids) with the stricter versions; the inline-split comparison sits at the end of the assertions cell. A deliberately off-by-one inline split was correctly flagged as different.
- Measured on the data: positive share 4.81% train, 11.07% validation, 30.83% test. Validation rows outside the train range: 99.76% for euribor3m, 63.97% for each of the other four columns; 8,215 of 8,235 validation rows are outside on at least one column. For four of the five columns the validation maximum equals the train minimum (validation lies entirely below train).

### Human review
- What the AI got right: [your review]
- What I found when I ran it: [your review]
- My decisions: [your review]
- Not verified: that row order equals time order (no date column; the month-run and euribor3m trend checks were suggested but not run); that `int(0.8*n)` equals `int(0.6*n)+int(0.2*n)` for any n other than 41,176.

### Conclusion
[1-2 sentences: what the guardrails improved, what still needed human checking.]

**Related audit:** AI_AUDIT.md - [add the audit number if one is written for this update].
---

## Entry 7 - Author: Chaima Garoui (Module 1 Update: Week 3 Model Comparison, Error Analysis and Business Evaluation)

**Date:** 2026-11-27

**AI tool:** ChatGPT

**Status:** Executed, reviewed and integrated into `Model_Comparison_and_Error_Analysis.ipynb`.

**Note:** The following prompts are reconstructed from the AI-assisted work carried out during Week 3. They are not verbatim conversation transcripts.

### Prompt 1 — Model comparison and evaluation

Role: Act as a Machine Learning tutor and engineering assistant. Guide me step by step through my individual contribution for Week 3.

Context: Our group project uses the UCI Bank Marketing dataset to predict term deposit subscription before the marketing call.

Reuse the shared preprocessing methodology and stratified 60/20/20 split with random_state=42.

Help me:

1. Compare the Dummy Classifier, default Logistic Regression and threshold-optimized Logistic Regression.
2. Evaluate their classification performance using appropriate metrics for an imbalanced dataset.
3. Compare confusion matrices and identify the trade-off between precision and recall.
4. Generate comparative visualizations.
5. Explain the results in simple English.

Constraints:

- Reuse the shared preprocessing module.
- Do not introduce data leakage.
- Perform comparisons on the validation set.
- Preserve the independent test set.
- Do not invent experimental results.
- Wait for my execution output before proceeding to the next step.

### Prompt 2 — Business cost sensitivity analysis

Help me extend the baseline comparison by evaluating the potential business consequences of false positives and false negatives.

Assume a hypothetical false positive cost of €1 and investigate different false negative costs.

Calculate:

1. Total misclassification cost for each model configuration.
2. Cost sensitivity under different scenarios.
3. Break-even points between model configurations.
4. Comparative cost visualizations.

Explain why the model maximizing F1-score is not necessarily the most economically appropriate model.

Clearly distinguish hypothetical costs from actual financial information. Do not claim that the results establish real campaign profitability.

### Prompt 3 — Demographic error analysis and statistical reliability

Help me investigate whether the threshold-optimized Logistic Regression model performs differently across customer groups.

Analyze performance according to:

- Age group.
- Occupation.
- Education level.
- Marital status.

For each subgroup, calculate the number of customers, actual subscribers, true positives, false positives, false negatives, recall and false positive rate.

Extend the analysis with 95% Wilson confidence intervals for recall.

Account for small subgroup sample sizes and handle groups containing no actual subscribers appropriately.

Generate visualizations and interpret the observed performance differences without claiming causality or discrimination.

### Prompt 4 — Customer targeting and campaign efficiency

Help me evaluate how predicted subscription probabilities could support marketing campaign targeting.

Rank validation customers according to their predicted probability of subscription.

Calculate:

1. Cumulative gains.
2. Lift.
3. Subscriber capture rates.
4. Precision among selected customers.
5. Number of contacts avoided.

Compare targeting strategies covering the top 5%, 10%, 20%, 30%, 50% and 100% of customers.

Generate cumulative gains and lift curves.

Explain the potential business implications while distinguishing retrospective validation simulations from actual campaign outcomes.

### Prompt iteration and human review

The AI-assisted workflow was conducted progressively in Google Colab. Code was executed and the resulting tables and visualizations were reviewed before continuing to subsequent experiments.

The initial analysis focused on standard classification metrics. Subsequent interactions extended the evaluation to include:

- Business cost sensitivity.
- Break-even analysis.
- Demographic error profiling.
- Statistical uncertainty of subgroup recall.
- Probability-based customer targeting.
- Cumulative gains and lift analysis.

### Human decisions and experimental verification

- Retained the shared preprocessing methodology and validation split.
- Used hypothetical misclassification costs because actual bank campaign costs were unavailable.
- Compared different decision thresholds instead of assuming that the default threshold was optimal.
- Extended subgroup recall analysis with Wilson confidence intervals to avoid overinterpreting small samples.
- Treated undefined recall separately from zero recall.
- Interpreted customer targeting results as simulations rather than realized financial benefits.
- Preserved the independent test set during the individual experiments.

### Main verified results

- Validation set: 8,235 customers, including 928 actual subscribers.
- Recall increased from 22.09% to 52.69% after threshold optimization.
- False negatives decreased from 723 to 439.
- False positives increased from 91 to 480.
- The unknown occupation group achieved 100% recall, but its 95% Wilson confidence interval was 56.55%–100.00%, based on only five actual subscribers.
- Targeting the top 20% of customers captured 65.19% of observed subscribers, with a lift of 3.26.

### Remaining limitations

- Actual business costs were unavailable.
- No formal statistical significance tests were performed between demographic groups.
- Potential dependencies between observations may affect confidence interval interpretation.
- The analysis focused on the Logistic Regression baseline rather than the final ensemble model.
- Actual campaign profitability and causal effects were not measured.

### Conclusion

AI assistance accelerated the development of the evaluation code, visualizations and statistical analysis. However, human review and experimental verification were necessary to ensure that the results were correctly interpreted.

The progressive prompt constraints helped maintain consistency with the shared project methodology and reduced the risk of unsupported statistical and business conclusions.

**Related audit:** `AI_AUDIT.md` — Audit 8 (Statistical Reliability of Subgroup Model Performance).

**Evidence:** `Model_Comparison_and_Error_Analysis.ipynb`.
