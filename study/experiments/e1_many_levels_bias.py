"""E1. Does information gain prefer a meaningless feature with many values, and does gain
ratio fix it?

The paper lists "biased toward features with many levels" as a weakness of decision trees.
This experiment measures that bias on the Adult census data.

Part A (theory check). For a feature with K values that is unrelated to the label, the
information gain measured on n rows is not 0. Its expected value is about
(K - 1) / (2 n ln 2) bits (from the chi-squared approximation to the G-test). We draw random
features with the real Adult labels and compare.

Part B (trees). We add one random "noise" feature with K values to the 12 real categorical
features and grow multiway trees with information gain or gain ratio, at several depths.
For 10 random 70/30 splits we record how often the trees split on the noise feature and
what it costs in test accuracy, compared with the same tree grown without the noise feature.

Outputs: study/results/e1_theory.csv, study/results/e1_trees.csv
Usage:   python study/experiments/e1_many_levels_bias.py
"""
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adult import categorical_table, encode, load  # noqa: E402
from tree import MultiwayTree, split_scores  # noqa: E402

RESULTS = ROOT / "results"
SEEDS = range(10)
LEVELS = [2, 10, 100, 1000]
DEPTHS = [3, 5, 8]


def part_a(y, rng):
    rows = []
    for n in [100, 1000, 10000, len(y)]:
        for k in LEVELS:
            gains, ratios = [], []
            for _ in range(200):
                idx = rng.choice(len(y), n, replace=False)
                x = rng.integers(0, k, n)
                g, s = split_scores(x, y[idx], k, 2)
                gains.append(g)
                ratios.append(g / s if s > 0 else 0.0)
            rows.append({"n_rows": n, "levels": k,
                         "mean_gain": np.mean(gains),
                         "theory_gain": (k - 1) / (2 * n * math.log(2)),
                         "mean_gain_ratio": np.mean(ratios)})
    return pd.DataFrame(rows)


def real_feature_gains(X, y, names):
    out = {}
    for f, name in enumerate(names):
        out[name] = split_scores(X[:, f], y, X[:, f].max() + 1, 2)[0]
    return pd.Series(out).sort_values(ascending=False)


def part_b(X, y):
    rows = []
    noise_col = X.shape[1]
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
        rng = np.random.default_rng(1000 + seed)
        for criterion in ("gain", "gain_ratio"):
            for depth in DEPTHS:
                base = MultiwayTree(criterion, max_depth=depth).fit(X[tr], y[tr])
                base_acc = float((base.predict(X[te]) == y[te]).mean())
                for k in LEVELS:
                    noise = rng.integers(0, k, len(y))
                    Xn = np.column_stack([X, noise])
                    t0 = time.time()
                    tree = MultiwayTree(criterion, max_depth=depth).fit(Xn[tr], y[tr])
                    acc = float((tree.predict(Xn[te]) == y[te]).mean())
                    nodes = tree.internal_nodes()
                    noise_nodes = [d for d, f, _ in nodes if f == noise_col]
                    rows.append({
                        "seed": seed, "criterion": criterion, "max_depth": depth, "levels": k,
                        "test_acc": acc, "test_acc_without_noise": base_acc,
                        "acc_change": acc - base_acc,
                        "internal_nodes": len(nodes), "noise_nodes": len(noise_nodes),
                        "share_noise_nodes": len(noise_nodes) / max(len(nodes), 1),
                        "rows_reaching_noise_split": sum(n for d, f, n in nodes if f == noise_col) / len(tr),
                        "shallowest_noise_depth": min(noise_nodes) if noise_nodes else np.nan,
                        "root_is_noise": nodes[0][1] == noise_col if nodes else False,
                        "leaves": tree.n_leaves(), "seconds": time.time() - t0,
                    })
        print(f"seed {seed} done", flush=True)
    return pd.DataFrame(rows)


def main():
    df, y = load()
    table = categorical_table(df)
    X = encode(table)
    RESULTS.mkdir(exist_ok=True)

    print("Information gain of each real feature on all rows (bits):")
    print(real_feature_gains(X, y, list(table.columns)).round(4).to_string())

    theory = part_a(y, np.random.default_rng(0))
    theory.to_csv(RESULTS / "e1_theory.csv", index=False)
    print("\nPart A: spurious gain of a random feature")
    print(theory.round(5).to_string(index=False))

    trees = part_b(X, y)
    trees.to_csv(RESULTS / "e1_trees.csv", index=False)
    summary = (trees.groupby(["criterion", "max_depth", "levels"])
               [["share_noise_nodes", "rows_reaching_noise_split", "acc_change", "test_acc"]]
               .mean().round(4))
    print("\nPart B: mean over 10 splits")
    print(summary.to_string())


if __name__ == "__main__":
    main()
