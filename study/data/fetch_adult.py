"""Download the UCI Adult (census income) data, verify it, and write a clean copy.

Checks, in order:
1. The downloaded file matches a known SHA-256, so every run uses the same bytes.
2. The schema: 15 columns, numeric columns parse as integers, the label has two values.
3. A data-quality report: row count, missing values ('?') per column, duplicate rows.

Output: study/data/adult.csv (with a header) and study/data/quality_report.json.
Rows with a missing value are kept and the missing value is recorded as the category
"Missing", because dropping them would remove some groups more than others.

Usage:  python study/data/fetch_adult.py
"""
import hashlib
import json
import pathlib
import sys
import urllib.request

import pandas as pd

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
SHA256 = "21e0ea2f925a00338929a8c86c27354c72ac1c79819bcca81c7d91c3d64218c2"
HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "adult-all.csv"
OUT = HERE / "adult.csv"
REPORT = HERE / "quality_report.json"

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
           "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
           "hours_per_week", "native_country", "income"]
NUMERIC = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = [c for c in COLUMNS if c not in NUMERIC + ["income"]]
EXPECTED_ROWS = 48842


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download():
    if RAW.exists() and sha256(RAW) == SHA256:
        print(f"using cached {RAW.name}")
        return
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, RAW)
    got = sha256(RAW)
    if got != SHA256:
        RAW.unlink()
        sys.exit(f"checksum mismatch: expected {SHA256}, got {got}")


def load_and_validate():
    df = pd.read_csv(RAW, header=None, names=COLUMNS, skipinitialspace=True,
                     dtype=str, keep_default_na=False)
    problems = []
    if df.shape != (EXPECTED_ROWS, len(COLUMNS)):
        problems.append(f"shape {df.shape}, expected ({EXPECTED_ROWS}, {len(COLUMNS)})")
    for col in NUMERIC:
        parsed = pd.to_numeric(df[col], errors="coerce")
        if parsed.isna().any():
            problems.append(f"{col}: {int(parsed.isna().sum())} values are not numbers")
        df[col] = parsed
    # some copies of Adult write the test-set labels with a trailing period
    df["income"] = df["income"].str.rstrip(".")
    labels = set(df["income"].unique())
    if labels != {"<=50K", ">50K"}:
        problems.append(f"unexpected labels {labels}")
    if problems:
        sys.exit("schema check failed:\n  " + "\n  ".join(problems))
    return df


def quality_report(df):
    missing = {c: int((df[c] == "?").sum()) for c in CATEGORICAL if (df[c] == "?").any()}
    return {
        "source": URL,
        "sha256": SHA256,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values_marked_?": missing,
        "rows_with_any_missing": int((df[CATEGORICAL] == "?").any(axis=1).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "share_income_over_50k": round(float((df["income"] == ">50K").mean()), 4),
    }


def main():
    download()
    df = load_and_validate()
    report = quality_report(df)
    for c in CATEGORICAL:
        df[c] = df[c].replace("?", "Missing")
    df.to_csv(OUT, index=False)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
