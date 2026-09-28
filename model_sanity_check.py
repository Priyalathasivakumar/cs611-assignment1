"""
Sanity check — is the feature store actually ML-ready?

NOT a graded deliverable. It answers two questions before any modelling
starts in Assignment 2:
  1. Do the gold stores join cleanly and train a model end to end?
  2. Is there leakage hiding in them?

On (2), a test AUC above ~0.95 would be evidence of leakage, not excellence.
A moderate score is the PASSING outcome here.

All preprocessing is fitted inside a scikit-learn Pipeline on the TRAINING
FOLD ONLY, so no test-set statistic reaches the training data.

Run with:  python model_sanity_check.py
"""

import glob
import os
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import config

# Identifiers and metadata that must never be fed to the model
NON_FEATURE_COLS = {
    "Customer_ID", "loan_id", "loan_start_date", "snapshot_date",
    "feature_store_version", "label", "label_def", "mob", "dpd", "overdue_amt",
}

TEST_FRACTION = 0.25


def load_gold(path: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No partitions under {path}. Run main.py first.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main() -> int:
    print("=" * 78)
    print("FEATURE STORE SANITY CHECK | logistic regression, temporal split")
    print("=" * 78)

    features = load_gold(config.GOLD_FEATURE_STORE)
    labels   = load_gold(config.GOLD_LABEL_STORE)
    print(f"feature store : {features.shape[0]:,} rows x {features.shape[1]} cols")
    print(f"label store   : {labels.shape[0]:,} rows")

    # --- join -------------------------------------------------------------
    data = features.merge(labels[["loan_id", "label"]], on="loan_id", how="inner")
    print(f"joined        : {data.shape[0]:,} rows "
          f"({data.shape[0] / len(features):.1%} of feature store)")
    print(f"default rate  : {data.label.mean():.2%}")

    # --- temporal split ---------------------------------------------------
    data = data.sort_values("loan_start_date").reset_index(drop=True)
    cutoff = data.loc[int(len(data) * (1 - TEST_FRACTION)), "loan_start_date"]

    train = data[data.loan_start_date <  cutoff]
    test  = data[data.loan_start_date >= cutoff]
    print(f"\ntrain         : {len(train):,} loans before {cutoff} "
          f"(default {train.label.mean():.2%})")
    print(f"test          : {len(test):,} loans on/after {cutoff} "
          f"(default {test.label.mean():.2%})")

    # --- feature matrix ---------------------------------------------------
    feature_cols = [c for c in data.columns if c not in NON_FEATURE_COLS]
    numeric     = [c for c in feature_cols if pd.api.types.is_numeric_dtype(data[c])]
    categorical = [c for c in feature_cols if c not in numeric]
    print(f"\nfeatures used : {len(numeric)} numeric + {len(categorical)} categorical")

    preprocessor = ColumnTransformer([
        # median/mode imputation FIT ON TRAIN ONLY — imputing before the split
        # would leak the test distribution into training
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale",  StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]), categorical),
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("clf",  LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    model.fit(train[feature_cols], train["label"])

    train_auc = roc_auc_score(train["label"], model.predict_proba(train[feature_cols])[:, 1])
    test_pred = model.predict_proba(test[feature_cols])[:, 1]
    test_auc  = roc_auc_score(test["label"], test_pred)
    test_ap   = average_precision_score(test["label"], test_pred)

    print("\n" + "-" * 78)
    print("RESULTS")
    print("-" * 78)
    print(f"  train AUC          : {train_auc:.4f}")
    print(f"  test  AUC          : {test_auc:.4f}")
    print(f"  test  avg precision: {test_ap:.4f}  (baseline {test.label.mean():.4f})")
    print(f"  train-test gap     : {train_auc - test_auc:+.4f}")

    # --- verdict ----------------------------------------------------------
    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    ok = True
    if test_auc > 0.95:
        print("  [!] Test AUC above 0.95 - investigate for leakage.")
        ok = False
    elif test_auc < 0.55:
        print("  [!] Test AUC near chance - features carry little signal.")
        ok = False
    else:
        print(f"  [OK] Test AUC {test_auc:.3f} sits in the plausible band for an")
        print("       application scorecard - signal present, no leakage signature.")

    if train_auc - test_auc > 0.10:
        print("  [!] Large train-test gap - overfitting or temporal drift.")
    else:
        print("  [OK] Train and test are close; the model generalises forward in time.")

    # --- what the model leaned on -----------------------------------------
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    top = (pd.DataFrame({"feature": [n.split("__", 1)[-1] for n in names], "coef": coefs})
             .assign(mag=lambda d: d.coef.abs())
             .nlargest(15, "mag"))

    print("\n" + "-" * 78)
    print("TOP 15 COEFFICIENTS (standardised)")
    print("-" * 78)
    for _, row in top.iterrows():
        print(f"  {row.feature:45s} {row.coef:+7.3f}  "
              f"{'raises' if row.coef > 0 else 'lowers'} default risk")

    print("\nSanity check complete.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())