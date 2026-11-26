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