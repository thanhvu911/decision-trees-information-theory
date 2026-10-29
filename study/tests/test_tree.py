import math
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from tree import MultiwayTree, entropy_from_counts, split_scores  # noqa: E402
import fetch_adult  # noqa: E402

# The six-row tennis table from the class slides: weather, humidity, wind -> play
TENNIS = np.array([
    [0, 0, 0, 0],  # Sunny, High, Weak, No
    [0, 0, 1, 0],  # Sunny, High, Strong, No
    [1, 0, 0, 1],  # Overcast, High, Weak, Yes
    [2, 0, 0, 1],  # Rain, High, Weak, Yes
    [2, 1, 0, 1],  # Rain, Normal, Weak, Yes
    [2, 1, 1, 0],  # Rain, Normal, Strong, No
])


def test_entropy_matches_paper_example():
    assert entropy_from_counts([6, 4]) == pytest.approx(0.971, abs=1e-3)
    assert entropy_from_counts([3, 1]) == pytest.approx(0.811, abs=1e-3)
    assert entropy_from_counts([5, 0]) == 0.0


def test_gain_and_gain_ratio_on_slides_table():
    X, y = TENNIS[:, :3], TENNIS[:, 3]
    results = [split_scores(X[:, f], y, X[:, f].max() + 1, 2) for f in range(3)]
    gains = [g for g, _ in results]
    ratios = [g / s for g, s in results]
    assert gains == pytest.approx([0.541, 0.0, 0.459], abs=1e-3)
    assert ratios[0] == pytest.approx(0.371, abs=1e-3)
    assert ratios[2] == pytest.approx(0.500, abs=1e-3)


def test_criteria_choose_different_roots():
    X, y = TENNIS[:, :3], TENNIS[:, 3]
    assert MultiwayTree("gain").fit(X, y).root_.feature == 0        # weather
    assert MultiwayTree("gain_ratio").fit(X, y).root_.feature == 2  # wind


def test_unlimited_tree_fits_consistent_data():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(300, 4))
    y = ((X[:, 0] == 1) ^ (X[:, 2] == 2)).astype(int)  # interaction, no single-feature rule
    for criterion in ("gain", "gain_ratio"):
        tree = MultiwayTree(criterion).fit(X, y)
        assert (tree.predict(X) == y).all()


def test_max_depth_is_respected():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 4, size=(500, 5))
    y = rng.integers(0, 2, size=500)
    tree = MultiwayTree("gain", max_depth=2).fit(X, y)
    assert max(d for d, _, _ in tree.internal_nodes()) <= 1


def test_unseen_value_falls_back_to_node_majority():
    X = np.array([[0], [0], [0], [1]])
    y = np.array([1, 1, 1, 0])
    tree = MultiwayTree("gain").fit(X, y)
    tree.n_values_ = np.array([3])
    assert tree.predict(np.array([[2]]))[0] == 1


def test_spurious_gain_of_random_feature_matches_theory():
    """For a feature unrelated to the label, 2 n ln2 * IG is approximately chi-squared with
    (K - 1)(C - 1) degrees of freedom, so E[IG] is about (K - 1) / (2 n ln 2) bits for two
    classes. This is the many-levels bias, measured."""
    rng = np.random.default_rng(2)
    n, k = 4000, 50
    gains = []
    for _ in range(200):
        x = rng.integers(0, k, n)
        y = rng.integers(0, 2, n)
        gains.append(split_scores(x, y, k, 2)[0])
    expected = (k - 1) / (2 * n * math.log(2))
    assert np.mean(gains) == pytest.approx(expected, rel=0.1)


def test_quality_report_counts_missing_and_duplicates():
    rows = [["39", "State-gov", "1", "Bachelors", "13", "Never-married", "?", "Not-in-family",
             "White", "Male", "0", "0", "40", "United-States", "<=50K"]] * 2
    df = pd.DataFrame(rows, columns=fetch_adult.COLUMNS)
    report = fetch_adult.quality_report(df)
    assert report["missing_values_marked_?"] == {"occupation": 2}
    assert report["duplicate_rows"] == 1
    assert report["rows_with_any_missing"] == 2
