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
