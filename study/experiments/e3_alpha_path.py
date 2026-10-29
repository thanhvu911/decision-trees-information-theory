"""E3. Along the cost-complexity pruning path, how do error rates change for each group?

On one Adult split (seed 0, entropy), grow the full tree, then prune it at a range of
ccp_alpha values from the pruning path. For each alpha, record 5-fold cross-validated
accuracy on the training split (what a pooled model choice looks at) and test-set accuracy,
false negative rate and false positive rate, overall and by sex and race group.

Output: study/results/e3_alpha_path.csv
Usage:  python study/experiments/e3_alpha_path.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adult import load, one_hot  # noqa: E402
from groups import group_metrics  # noqa: E402

SEED = 0
N_ALPHAS = 40


def main():
    df, y = load()
    X = one_hot(df).to_numpy(dtype=float)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=SEED)
    path = DecisionTreeClassifier(criterion="entropy", random_state=SEED) \
        .cost_complexity_pruning_path(X[tr], y[tr])
    alphas = np.unique(path.ccp_alphas[:-1])
    alphas = alphas[alphas > 0]
    grid = np.concatenate([[0.0], np.unique(np.quantile(alphas, np.linspace(0, 1, N_ALPHAS)))])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows = []
    for alpha in grid:
        model = DecisionTreeClassifier(criterion="entropy", random_state=SEED, ccp_alpha=alpha)
        cv_acc = cross_val_score(model, X[tr], y[tr], cv=cv, n_jobs=2).mean()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        for g in group_metrics(df.iloc[te], y[te], pred):
            rows.append({"ccp_alpha": alpha, "leaves": model.get_n_leaves(),
                         "cv_accuracy": cv_acc, **g})
        print(f"alpha={alpha:.2e} leaves={model.get_n_leaves():5d} cv_acc={cv_acc:.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "e3_alpha_path.csv", index=False)


if __name__ == "__main__":
    main()
