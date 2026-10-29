"""Load the cleaned Adult data written by study/data/fetch_adult.py."""
import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "adult.csv"

CATEGORICAL = ["workclass", "education", "marital_status", "occupation", "relationship",
               "race", "sex", "native_country"]
NUMERIC = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]


def load():
    if not DATA.exists():
        raise SystemExit("run `python study/data/fetch_adult.py` first")
    df = pd.read_csv(DATA, keep_default_na=False)
    y = (df["income"] == ">50K").astype(int).to_numpy()
    return df, y


def categorical_table(df):
    """All-categorical version for the multiway tree: numeric columns are binned.

    education_num duplicates education and fnlwgt is a sampling weight, so both are left out.
    """
    out = df[CATEGORICAL].copy()
    out["age_band"] = pd.cut(df["age"], [0, 24, 34, 44, 54, 64, 200],
                             labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"])
    out["hours_band"] = pd.cut(df["hours_per_week"], [0, 34, 40, 50, 200],
                               labels=["<35", "35-40", "41-50", ">50"])
    out["has_capital_gain"] = np.where(df["capital_gain"] > 0, "yes", "no")
    out["has_capital_loss"] = np.where(df["capital_loss"] > 0, "yes", "no")
    return out.astype(str)


def encode(table):
    """Map each column's categories to 0..k-1 (codes fixed on the full table)."""
    return np.column_stack([pd.Categorical(table[c]).codes for c in table.columns]).astype(np.int64)


def one_hot(df):
    """Numeric columns as they are plus one-hot categoricals, for scikit-learn trees."""
    return pd.get_dummies(df[NUMERIC + CATEGORICAL], columns=CATEGORICAL, dtype=np.uint8)
