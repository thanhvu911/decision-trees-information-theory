"""E2. Pre-pruning vs post-pruning, measured, plus E3's subgroup measurements on Adult.

The paper says pre-pruning "performs poorly when exposed to new data" and that post-pruning
"demands high-quality processors". This experiment compares, on three datasets:

  unpruned     a full scikit-learn tree
  pre-pruned   max_depth and min_samples_leaf chosen by 5-fold cross-validation
  post-pruned  cost-complexity pruning, ccp_alpha chosen by 5-fold cross-validation

each with entropy (information gain) and Gini impurity, over 10 random stratified 70/30
splits. It records test accuracy, balanced accuracy, tree size and tuning time.

For Adult it also records, for every fitted model, accuracy and error rates for groups by
sex and race (E3): does a choice made on pooled accuracy hide a larger change for one group?

Outputs: study/results/e2_pruning.csv, study/results/e3_subgroups.csv
Usage:   python study/experiments/e2_pruning.py   (about 20-30 minutes on 2 cores)
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adult import load, one_hot  # noqa: E402
from groups import group_metrics  # noqa: E402

RESULTS = ROOT / "results"
SEEDS = range(10)
PRE_GRID = {"max_depth": [3, 5, 8, 12, 16, None], "min_samples_leaf": [1, 5, 20, 50, 100]}
N_ALPHAS = 30


def datasets():
    df, y = load()
    yield "adult", one_hot(df).to_numpy(dtype=float), y, df
    bc = load_breast_cancer()
    yield "breast_cancer", bc.data, bc.target, None
    dg = load_digits()
    yield "digits", dg.data, dg.target, None


def alpha_candidates(X, y, criterion, seed):
    path = DecisionTreeClassifier(criterion=criterion, random_state=seed) \
        .cost_complexity_pruning_path(X, y)
    alphas = np.unique(path.ccp_alphas[:-1])  # the last alpha prunes to the root
    alphas = alphas[alphas > 0]
    if len(alphas) > N_ALPHAS:
        alphas = np.unique(np.quantile(alphas, np.linspace(0, 1, N_ALPHAS)))
    return np.concatenate([[0.0], alphas])


def fit_all(X, y, criterion, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    base = DecisionTreeClassifier(criterion=criterion, random_state=seed)
    models = {}

    t0 = time.time()
    models["unpruned"] = (base.fit(X, y), time.time() - t0, {})

    t0 = time.time()
    pre = GridSearchCV(DecisionTreeClassifier(criterion=criterion, random_state=seed),
                       PRE_GRID, cv=cv, n_jobs=2).fit(X, y)
    models["pre-pruned"] = (pre.best_estimator_, time.time() - t0, pre.best_params_)

    t0 = time.time()
    post = GridSearchCV(DecisionTreeClassifier(criterion=criterion, random_state=seed),
                        {"ccp_alpha": alpha_candidates(X, y, criterion, seed)},
                        cv=cv, n_jobs=2).fit(X, y)
    models["post-pruned"] = (post.best_estimator_, time.time() - t0, post.best_params_)
    return models


def main():
    RESULTS.mkdir(exist_ok=True)
    rows, group_rows = [], []
    for name, X, y, frame in datasets():
        for seed in SEEDS:
            tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y,
                                      random_state=seed)
            for criterion in ("entropy", "gini"):
                for method, (model, secs, params) in fit_all(X[tr], y[tr], criterion, seed).items():
                    pred = model.predict(X[te])
                    rows.append({
                        "dataset": name, "seed": seed, "criterion": criterion, "method": method,
                        "test_acc": float((pred == y[te]).mean()),
                        "balanced_acc": balanced_accuracy_score(y[te], pred),
                        "leaves": model.get_n_leaves(), "depth": model.get_depth(),
                        "tune_seconds": secs, "params": str(params),
                    })
                    if frame is not None:
                        for g in group_metrics(frame.iloc[te], y[te], pred):
                            group_rows.append({"seed": seed, "criterion": criterion,
                                               "method": method, **g})
            print(f"{name} seed {seed} done", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "e2_pruning.csv", index=False)
        pd.DataFrame(group_rows).to_csv(RESULTS / "e3_subgroups.csv", index=False)

    res = pd.DataFrame(rows)
    print(res.groupby(["dataset", "criterion", "method"])
          [["test_acc", "balanced_acc", "leaves", "tune_seconds"]].agg(["mean", "std"]).round(4)
          .to_string())


if __name__ == "__main__":
    main()
