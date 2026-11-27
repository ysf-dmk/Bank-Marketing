# AI_AUDIT.md - Catch & Correct

## Author: Mariem Dridi
**Module:** Module 3 - Baseline and Evaluation
**Date:** 2026-11-26
**AI tool:** ChatGPT
**Related interaction:** PROMPT_LOG.md - Entry 3

---

## Audit 1 - Incorrect handling of the pdays sentinel

### 1. Catch: Identified flaw

The initial AI response independently reconstructed
the preprocessing pipeline using the raw dataset.

The AI treated pdays as an ordinary numerical feature
without accounting for the special value 999.

### 2. Root cause

The initial prompt did not explicitly specify the
semantic meaning of pdays or require reuse of the
shared preprocessing module.

The value 999 represents a customer who has never
been contacted previously. It does not represent
an actual elapsed duration of 999 days.

### 3. Statistical consequences

Treating this sentinel as a continuous numerical
value can:

- Distort the mean and standard deviation.
- Introduce an artificial numerical relationship.
- Affect the interpretation of Logistic Regression
  coefficients.
- Fail to distinguish previous contact status from
  elapsed time since previous contact.

### 4. Correct: Human intervention

The independently reconstructed preprocessing
pipeline was rejected.

The shared preprocessing module was retained.
It replaces the sentinel with 0 and introduces
the binary feature was_contacted_before.

The baseline notebook uses bank_clean.csv and
the shared preprocessing functions.

### 5. Experimental verification

Verification code:

    raw_df = pd.read_csv(
        "bank-additional-full.csv", sep=";"
    )
    clean_df = pd.read_csv("bank_clean.csv")

    print((raw_df["pdays"] == 999).sum())
    print((clean_df["pdays"] == 999).sum())
    print("was_contacted_before" in clean_df.columns)

Actual results:

    Raw pdays=999: 39673
    Clean pdays=999: 0
    Contact indicator exists: True

The checks confirm that the sentinel is no longer
present and that the contact indicator exists.

### 6. Remaining limitations

These checks do not establish the individual impact
of the correction on predictive performance or verify
the indicator values row by row.

**Status:** Correction retained; structural checks passed.

---

## Audit 2 - Incorrect preprocessing function signature

### 1. Catch: Identified flaw

The refined AI response proposed:

    preprocessor = build_preprocessor(X_train)

However, the shared function does not accept
any positional arguments.

### 2. Root cause

The AI assumed the function interface without
inspecting its actual implementation.

Although the AI acknowledged this uncertainty,
the proposed code was incompatible with the
existing project module.

### 3. Technical consequences

- The generated code raises a TypeError.
- The preprocessing pipeline cannot be constructed.
- Model training cannot proceed.
- Project reproducibility is affected.

### 4. Correct: Human intervention

Incorrect AI-generated code:

    preprocessor = build_preprocessor(X_train)

Corrected code:

    preprocessor = build_preprocessor()

The existing shared function was preserved instead
of modifying its interface to accommodate the
AI-generated code.

### 5. Experimental verification

Verification code:

    import inspect

    print(inspect.signature(build_preprocessor))

    try:
        build_preprocessor(X_train)
    except TypeError as e:
        print("AI-generated code failed:", e)

    preprocessor_check = build_preprocessor()
    print("Corrected function call: successful")

Actual results:

    Function signature: ()

    AI-generated code failed:
    build_preprocessor() takes 0 positional
    arguments but 1 was given

    Corrected function call: successful

**Status:** Corrected function call verified.

---

## Overall conclusion

The audit identified one methodological flaw and
one technical integration error in AI-generated
recommendations.

The refined prompt improved the methodological
quality of the response but did not eliminate
incorrect implementation assumptions.

Human inspection, statistical reasoning and
experimental verification were necessary to ensure
consistency with the shared project methodology.

The independent test set was not used for model
or threshold selection during this audit.

**Evidence:** Baseline_Model.ipynb

---

## Author: Youssef Damak
**Module:** Module 2 - Data Cleaning and Preprocessing
**Date:** 2026-11-25
**AI tool:** Claude Code (draft of src/data_prep.py), Claude chat (review of the draft)
**Related interaction:** PROMPT_LOG.md - Entry 1

---

## Audit 3 - Near-duplicate rows survive deduplication

### 1. Catch: Identified flaw

In the AI-drafted clean() function, drop_duplicates() ran
before the duration column was dropped. Rows that differ only
in duration were therefore not seen as duplicates. Once
duration is removed, they are identical on every remaining
feature.

### 2. Root cause

The prompt listed "drop duplicates" and "drop duration" as
separate steps without fixing their order or saying what
counts as a duplicate once the set of columns changes.

### 3. Statistical consequences

- The same feature vector can appear in train and in test.
- Validation and test scores can be slightly optimistic,
  mainly for models that can memorise rows.
- The dataset has no customer identifier, so these rows
  cannot be classified as true duplicates or as different
  customers with the same profile.

### 4. Correct: Human decision

The near-duplicates were kept on purpose. Only exact
duplicates (identical on all 21 original columns) are
removed. Reason: without an identifier, deleting them could
remove real customers and change the class balance.

The decision and its cost are written as a comment in
clean() in src/data_prep.py.

### 5. Experimental verification

Verification code:

    raw = pd.read_csv("bank-additional-full.csv", sep=";")
    df = data_prep.clean(
        data_prep.load_raw("bank-additional-full.csv")
    )
    X_train, X_val, X_test, y_train, y_val, y_test = (
        data_prep.split(df)
    )
    train_keys = set(map(tuple, X_train.astype(str).values))

    print(raw.duplicated().sum())
    print(df.drop(columns="y").duplicated().sum())
    print(sum(tuple(r) in train_keys
              for r in X_test.astype(str).values))
    print(sum(tuple(r) in train_keys
              for r in X_val.astype(str).values))

Actual results:

    Exact duplicates in raw file: 12
    Rows identical on all features after cleaning: 2009
    Test rows with a twin in train: 504 (of 8,236)
    Validation rows with a twin in train: 525 (of 8,235)

### 6. Remaining limitations

The effect of these rows on the scores was not measured.
A group-aware split or a sensitivity analysis without them
was not run.

**Status:** Kept by decision and documented; not removed.

---

## Audit 4 - Inconsistent pdays column between module and notebook

### 1. Catch: Identified flaw

The AI-drafted clean() in src/data_prep.py keeps the cleaned
column under the name pdays. The notebook cell that produced
bank_clean.csv named it pdays_clean and dropped pdays. Two
versions of the same cleaning existed with different column
names.

### 2. Root cause

The prompt specified the functions of the module but not the
output column names, and did not require the notebook to
reuse them.

### 3. Technical consequences

- build_preprocessor() cannot be fitted on the old
  bank_clean.csv, because its numeric column list contains
  pdays.
- The baseline owner would load a file that does not match
  the shared preprocessor.
- Duplicated cleaning code can drift apart.

### 4. Correct: Human intervention

The column is named pdays everywhere. The notebook was
rewritten to call data_prep.load_raw, clean, split and
build_preprocessor instead of repeating the cleaning code
(follow-up prompt in PROMPT_LOG.md - Entry 1).
bank_clean.csv was regenerated, and an assertions cell was
added to the notebook.

### 5. Experimental verification

Verification code:

    clean_df = pd.read_csv("bank_clean.csv")
    print("pdays" in clean_df.columns,
          "pdays_clean" in clean_df.columns)

    bad = clean_df.rename(
        columns={"pdays": "pdays_clean"}
    ).drop(columns="y")
    try:
        build_preprocessor().fit(bad)
    except Exception as e:
        print(type(e).__name__, e)

Actual results:

    True False
    ValueError A given column is not a column of the
    dataframe

The old file had pdays_clean in its header before
regeneration.

### 6. Remaining limitations

The check covers column names only. Row values of the old and
new files were not compared one by one.

**Status:** Corrected and verified.

---

## Overall conclusion (Module 2)

The AI draft ran and respected the main leakage rules
(nothing fitted before the split, duration removed, test set
untouched). The two flaws were an interface inconsistency
and an unstated assumption about what a duplicate is.
Running the code on the real data was needed to find both,
and the duplicate question required a human decision.

**Evidence:** src/data_prep.py, Cleaning_Preprocessing.ipynb
(assertions cell)

---

## Author: Chaima Garoui

**Module:** Module 1 - Exploratory Data Analysis and Data Quality

**Date:** 2026-11-27

**AI tool:** ChatGPT

**Related interaction:** PROMPT_LOG.md - Module 1 EDA interpretation

## Audit 5 - Statistical uncertainty in small customer groups

### 1. Catch: Identified limitation

The initial AI-assisted exploratory analysis compared
subscription rates across education categories.

The illiterate category appeared to have the highest
subscription rate, at 22.22%.

However, comparing percentages alone does not account
for differences in sample size or statistical uncertainty.

### 2. Root cause

The initial descriptive analysis focused on observed
subscription rates without quantifying their uncertainty.

The illiterate category contains only 18 customers.
Four of them subscribed to a term deposit.

A percentage calculated from such a small group is
sensitive to changes in individual observations.

### 3. Statistical consequences

- Small groups can appear artificially prominent in
  rankings based on observed subscription rates.
- The estimated subscription rate may be unstable.
- Differences between education categories may be
  overinterpreted.
- Marketing recommendations based only on these
  percentages could be misleading.

### 4. Correct: Human intervention

The descriptive analysis was extended to include
the number of customers, number of subscribers and
95% Wilson confidence intervals for each education
category.

The Wilson method was selected because it is generally
more suitable than the standard normal approximation
for small samples and binomial proportions.

The interpretation was revised to distinguish observed
subscription rates from the precision of their estimates.

### 5. Experimental verification

Verification code:

    from statsmodels.stats.proportion import proportion_confint

    education_audit = (
        df.groupby("education")["y"]
        .agg(
            total_customers="count",
            subscribers=lambda x: (x == "yes").sum()
        )
        .reset_index()
    )

    education_audit["subscription_rate"] = (
        education_audit["subscribers"]
        / education_audit["total_customers"]
    )

    lower, upper = proportion_confint(
        count=education_audit["subscribers"],
        nobs=education_audit["total_customers"],
        alpha=0.05,
        method="wilson"
    )

    education_audit["CI_lower"] = lower
    education_audit["CI_upper"] = upper
    education_audit["CI_width"] = upper - lower

    display(education_audit.round(4))

Actual results:

| Education | Customers | Subscribers | Rate | 95% CI | CI width |
|---|---:|---:|---:|---|---:|
| illiterate | 18 | 4 | 22.22% | 9.00%-45.21% | 36.21 pp |
| university.degree | 12168 | 1670 | 13.72% | 13.12%-14.35% | 1.22 pp |

The confidence interval for the illiterate category
is approximately 30 times wider than that of the
university degree category.

The verification confirms that the higher observed
subscription rate is associated with much greater
statistical uncertainty.

### 6. Remaining limitations

- The confidence intervals assume independent
  binomial observations.
- No formal hypothesis test comparing education
  categories was performed.
- Potential confounding variables such as age and
  occupation were not controlled.
- The analysis does not establish a causal
  relationship between education and subscription.
- This analysis is descriptive and uses the original
  dataset. It must not be used to select models or
  tune hyperparameters using the held-out test data.

**Status:** Statistical interpretation extended and
verified using the original dataset.

### Overall conclusion (Module 1)

The audit shows that descriptive percentages alone
are not sufficient when comparing groups with very
different sample sizes.

Adding confidence intervals made the interpretation
more statistically rigorous and reduced the risk of
drawing unsupported conclusions.

AI-assisted interpretations require human review,
especially when small groups produce apparently
strong results.

**Evidence:** EDA_and_Data_Quality.ipynb

---

## Author: Adam Nasri
**Module:** Module 4 - Alternative Models and Performance Comparison
**Date:** 2026-11-27
**AI tool:** Gemini
**Related interaction:** PROMPT_LOG.md - Entry 4 & Entry 5

---

## Audit 6 - Avoiding Preprocessing Leakage Inside Cross-Validation and Uncalibrated Neural Probabilities

### 1. Catch: Identified flaws

1. Pre-transforming `X_train` before passing it into `RandomizedSearchCV` fits `StandardScaler` across all 5 folds simultaneously, leaking validation fold statistics into training folds.
2. Using a raw `MLPClassifier` without probability calibration produces poorly scaled probabilities under an 88.73% / 11.27% class imbalance, destabilizing validation F1 threshold selection.

### 2. Root cause

Standard hyperparameter tuning templates often call `preprocessor.fit_transform(X_train)` outside the cross-validation loop for speed, while neural classifiers optimize log-loss without guaranteeing calibrated posterior probabilities on imbalanced tabular data.

### 3. Statistical consequences

- Fitting scalers outside CV introduces optimistic bias in cross-validated `average_precision` scores.
- Uncalibrated MLP outputs compress or skew predicted probabilities, making threshold transfer from `X_val` to `X_test` unreliable.

### 4. Correct: Human intervention

- Wrapped `build_preprocessor()` directly inside every scikit-learn `Pipeline` passed to `RandomizedSearchCV` so scaling and one-hot encoding are refitted strictly on the 4 training folds of each split.
- Wrapped the regularized `MLPClassifier` inside `CalibratedClassifierCV(method="sigmoid", cv=5)` (following Niculescu-Mizil & Caruana, 2005) before validation threshold optimization.

### 5. Experimental verification

- Verified that all candidate pipelines (`Random Forest`, `XGBoost`, `LightGBM`, `Calibrated MLP`) accept raw `X_train` / `X_val` / `X_test` DataFrames directly and beat the Module 3 baseline on `X_val` (`0.5027` and `0.4798` vs. `0.4742` PR-AUC) without touching `X_test` until the final cell.

**Status:** Verified and integrated into `Alternative_Models_Comparison.ipynb`.

---

## Audit 7 - SHAP Explainer Compatibility with Stacking Ensembles & Shared `time_split()` Alignment

### 1. Catch: Identified flaws

1. When `Stacking (4 Tuned Models)` outperformed all single models on `X_val` (`PR-AUC = 0.5038`), `shap.TreeExplainer` could not be applied directly to the heterogeneous `StackingClassifier` meta-estimator, and running `TreeExplainer` on a 200-tree `RandomForestClassifier` caused a severe runtime bottleneck.
2. Section 4 originally implemented a manual chronological slice instead of calling the team's shared `src.data_prep.time_split()` helper.

### 2. Root cause

Generic interpretability blocks assume the winning model is always a single tree estimator, while parallel module updates temporarily decoupled the chronological slicing code from `src/data_prep.py`.

### 3. Technical & statistical consequences

- Calling `shap.TreeExplainer` on a `StackingClassifier` raises a `TypeError` / `InvalidModelError`, breaking notebook execution before error profiling and final holdout evaluation.
- Duplicating chronological split logic risks subtle boundary discrepancies across team notebooks.

### 4. Correct: Human intervention

- Explicitly routed SHAP feature attribution to `LightGBM (Tuned)` (`TreeExplainer` finishes in <5 seconds) while computing the validation confusion matrix and error profile (`TP=537, FP=538, FN=391, TN=6769`) on the winning `Stacking` champion.
- Replaced the manual chronological split with `src.data_prep.time_split()` and enforced strict validation-only model selection (`Stacking` selected on `X_val` PR-AUC = `0.5038`, threshold = `0.2469`) prior to the single evaluation on `X_test`.

### 5. Experimental verification

- Verified exact chronological split counts (`24,705 / 8,235 / 8,236`) and class-prior drift (`4.81% -> 11.07% -> 30.83%`, chronological validation PR-AUC = `0.0930`).
- Verified single-pass holdout generalization on untouched `X_test`: `Stacking (4 Tuned Models)` achieved **PR-AUC = `0.4625`**, **ROC-AUC = `0.7975`**, and **F1 = `0.5086`** (outperforming `Logistic Regression` at `0.4423` PR-AUC, `0.7908` ROC-AUC, and `0.4812` F1).

**Status:** Verified and integrated into `Alternative_Models_Comparison.ipynb`.

### Overall conclusion (Module 4)

Enforcing pipeline encapsulation inside cross-validation, post-hoc probability calibration, SHAP explainer routing, and shared chronological splitting (`src.data_prep.time_split()`) eliminated leakage and runtime failures while establishing `Stacking (4 Tuned Models)` as the top-performing model across both `X_val` and `X_test`.

**Evidence:** `Alternative_Models_Comparison.ipynb`, `src/data_prep.py`
---

## Author: Chaima Garoui

**Module:** Module 1 — Week 3: Model Comparison and Error Analysis  
**Date:** 2026-11-27  
**AI tool:** ChatGPT  
**Related interaction:** PROMPT_LOG.md — Week 3 Model Evaluation and Error Analysis

## Audit 8 - Statistical Reliability of Subgroup Model Performance

### 1. Catch: Identified limitation

The initial AI-assisted error analysis compared model recall across age groups, occupations, education levels and marital status.

However, recall percentages alone do not account for differences in subgroup sample sizes.

For example, customers with an unknown occupation achieved 100% recall, but this result was based on only five actual subscribers.

### 2. Root cause

The initial evaluation focused on point estimates of recall and false positive rates without measuring statistical uncertainty.

Small subgroups can produce extreme performance values that may not generalize to new customers.

### 3. Statistical consequences

- Small subgroups may appear to have exceptionally high or low model performance.
- Recall estimates may be unstable when the number of actual subscribers is limited.
- Differences between demographic groups may be overinterpreted.
- Marketing decisions based only on subgroup recall may lead to unreliable conclusions.
- Undefined recall values may be incorrectly interpreted as zero performance.

### 4. Correct: Human intervention

The subgroup evaluation was extended by calculating 95% Wilson confidence intervals for recall.

The number of actual subscribers was retained alongside each recall estimate to support its interpretation.

Confidence interval visualizations were generated for age, occupation, education and marital status.

Groups containing no actual subscribers were assigned NaN for recall and excluded from recall confidence interval visualizations.

The overall model recall was included as a descriptive reference.

### 5. Experimental verification

Recall was calculated as:

Recall = TP / (TP + FN)

The Wilson confidence interval was calculated using the number of true positives as the number of successes and the number of actual subscribers as the sample size.

Actual validation results:

| Subgroup | Actual subscribers | Recall | 95% Wilson CI | CI width |
|---|---:|---:|---:|---:|
| Unknown occupation | 5 | 100.00% | 56.55%–100.00% | 43.45 pp |
| Student | 57 | 84.21% | 72.64%–91.46% | 18.83 pp |
| Admin. | 299 | 60.54% | 54.89%–65.91% | 11.01 pp |
| Blue-collar | 120 | 25.83% | 18.84%–34.33% | 15.

---

## Module 3 Update — Logistic Regression Hyperparameter Tuning

**Author:** Mariem Dridi  
**Related prompt:** PROMPT_LOG.md — Entry 8  
**Evidence:** Baseline_Model.ipynb

### Issue 1 — Test Set Access Outside the Experimental Scope

**AI-generated suggestion**

The initial AI response included test-set class distributions and a final evaluation section using the held-out random and temporal test sets.

**Root cause**

The initial prompt did not explicitly specify the validation-only boundary required for this stage of the project. The AI therefore proposed a complete machine learning evaluation workflow rather than respecting the narrower scope of Module 3.

**Potential impact**

Premature test-set inspection could influence subsequent methodological decisions and compromise the independence of the final evaluation.

**Human correction**

- Restricted GridSearchCV to training data.
- Selected decision thresholds exclusively on validation data.
- Excluded test-set inspection and evaluation from the tuning experiment.
- Added explicit test-set protection requirements to the refined prompt.

**Verification**

The implemented tuning and evaluation procedures used training and validation variables only. The refined AI response also excluded test-set evaluation.

**Status:** Corrected scope mismatch. No actual test leakage was observed during this task.

### Issue 2 — Incomplete Comparison Table Implementation

**AI-generated suggestion**

The initial response initialized an empty results list and defined the four experiments but subsequently referenced `results_df` without creating it in the supplied implementation.

**Root cause**

The AI omitted the intermediate evaluation and DataFrame construction step.

**Potential impact**

The proposed visualization code was not independently executable and could raise a NameError in a clean notebook environment.

**Human correction**

The project notebook explicitly constructs the comparison DataFrame containing the tuned and untuned Logistic Regression results for both evaluation protocols.

The refined prompt also required all variables and comparison DataFrames to be defined before use.

**Verification**

The comparison table and performance visualization were successfully generated in Baseline_Model.ipynb.

The original AI-generated implementation was not executed in full.

**Status:** Corrected implementation omission.

### Additional Verification 1 — Shared Split Consistency

The initial AI response independently recreated the random and temporal splitting logic instead of directly calling the shared project functions.

This introduced a potential reproducibility and maintenance risk.

The proposed training and validation indices were compared experimentally with the shared implementation.

| Protocol | Identical training indices | Identical validation indices |
|---|---|---|
| Random | True | True |
| Temporal | True | True |

All four comparisons returned True.

No actual experimental inconsistency was detected. The shared `split()` and `time_split()` functions were retained to prevent future divergence between team members.

**Status:** Verified maintenance risk, not a confirmed splitting error.

### Additional Verification 2 — Temporal Cross-Validation

An additional experiment compared StratifiedKFold and TimeSeriesSplit on the temporal training partition.

For each fold, the maximum training position was compared with the minimum validation position.

| Cross-validation strategy | Folds respecting temporal row order |
|---|---|
| StratifiedKFold | 0/5 |
| TimeSeriesSplit | 5/5 |

The experiment confirmed that TimeSeriesSplit preserves temporal row ordering, whereas shuffled StratifiedKFold does not.

The initial AI response already recommended the correct cross-validation strategy. This experiment therefore validates an AI recommendation rather than correcting an AI error.

The chronological interpretation remains conditional on the preservation of the original dataset ordering.

**Status:** Methodological recommendation experimentally verified.

### Final Human Assessment

The refined prompt improved the alignment of the AI response with the project's shared implementation, evaluation boundaries and statistical assumptions.

However, AI-generated code was not accepted automatically. Recommendations were reviewed against the existing notebook and verified where appropriate.

The final experiments demonstrated that hyperparameter tuning produced negligible improvements and did not resolve the weak temporal generalization performance.

No unsupported numerical results, confirmed leakage claims or unverified causal explanations were introduced.

