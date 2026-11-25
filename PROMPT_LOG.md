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