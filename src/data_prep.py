"""Module 2: cleaning, split and (unfitted) preprocessor for Bank Marketing.

Nothing in this file fits anything. The preprocessor is returned unfitted so
the caller can fit it on the TRAIN split only.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# Explicit column lists (no dtype guessing). "duration" is absent on purpose.
NUMERIC_COLS = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "day_of_week", "poutcome",
]
BINARY_COLS = ["was_contacted_before"]  # already 0/1, passed through as is


def load_raw(path):
    """Read the raw semicolon-separated CSV. No cleaning here."""
    # keep_default_na=False: the literal text "unknown" is a category, and we
    # do not want pandas to turn any other string into NaN by surprise.
    return pd.read_csv(path, sep=";", keep_default_na=False)


def clean(df):
    """Row-wise cleaning only. Nothing here learns a statistic from the data."""
    df = df.copy()  # do not modify the caller's DataFrame

    # Exact duplicates are removed BEFORE the split, otherwise a copy could
    # land in train and another in test. The original index is kept on purpose
    # (no reset_index) so overlap between splits can be checked by index.
    df = df.drop_duplicates()

    # 999 means "never contacted", not "999 days ago": keep that as a flag...
    df["was_contacted_before"] = (df["pdays"] != 999).astype(int)
    # ...and replace 999 by 0 so it does not distort the scaler. The flag
    # tells a real pdays == 0 apart from the replaced value.
    df["pdays"] = df["pdays"].where(df["pdays"] != 999, 0)

    # duration is only known after the call ends and reveals the outcome:
    # it cannot be used to predict before the call.
    # KNOWN COST (decision: keep): dedup above ran while duration was still
    # present, so only exact duplicates were removed. Once duration is gone,
    # about 2,000 rows are identical to another row on all remaining features
    # (a few hundred test rows have a twin in train). We keep them, so test
    # scores are slightly optimistic, mainly for models that can memorise rows.
    df = df.drop(columns="duration")

    # Target as integers: 1 = subscribed, 0 = did not.
    df["y"] = (df["y"] == "yes").astype(int)
    return df


def split(df):
    """Stratified 60/20/20 split. Returns X_train, X_val, X_test, y_train, y_val, y_test."""
    X = df.drop(columns="y")
    y = df["y"]

    # Step 1: 60% train, 40% held aside. stratify keeps the yes-share equal.
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    # Step 2: halve the held-aside part into validation and test (20% / 20%).
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor():
    """Return an UNFITTED ColumnTransformer. Fit it on X_train only."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            # "unknown" is just another value, so it gets its own column.
            # handle_unknown="ignore": a category unseen in train becomes all
            # zeros instead of raising an error at val/test time.
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ],
        remainder="drop",  # any column not listed above is dropped, never silently kept
    )
