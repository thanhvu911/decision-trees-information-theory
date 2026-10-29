"""A small multiway decision tree for categorical features (ID3 / C4.5 style).

Written for the follow-up study, so the splitting criterion is the only thing that changes
between runs. Each internal node splits on one feature and gets one child per value seen in
the training rows at that node. A feature is used at most once on any root-to-leaf path.

Criteria
--------
"gain"        information gain, IG(S, A) = H(Y) - H(Y | A), which equals the mutual
              information I(Y; A) between the feature and the label in the node's rows.
"gain_ratio"  C4.5's correction: IG(S, A) / H(A), where H(A) ("split information") is the
              entropy of the feature's own values. Following Quinlan (1993), only features
              whose gain is at least the average gain at that node are considered, so a
              feature with tiny split information cannot win on a tiny gain.
"""
import numpy as np


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def split_scores(x, y, n_values, n_classes):
    """Information gain and split information of splitting labels y on feature x."""
    table = np.bincount(x * n_classes + y, minlength=n_values * n_classes)
    table = table.reshape(n_values, n_classes)
    sizes = table.sum(axis=1)
    n = sizes.sum()
    h_y = entropy_from_counts(table.sum(axis=0))
    used = sizes > 0
    cond = sum(sizes[v] / n * entropy_from_counts(table[v]) for v in np.flatnonzero(used))
    return h_y - cond, entropy_from_counts(sizes)


class Node:
    __slots__ = ("feature", "children", "prediction", "n", "depth")

    def __init__(self, prediction, n, depth):
        self.feature = None
        self.children = {}
        self.prediction = prediction
        self.n = n
        self.depth = depth


class MultiwayTree:
    def __init__(self, criterion="gain", max_depth=None, min_samples_split=2, min_gain=1e-9):
        if criterion not in ("gain", "gain_ratio"):
            raise ValueError("criterion must be 'gain' or 'gain_ratio'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)
        self.n_values_ = X.max(axis=0) + 1
        self.n_classes_ = int(y.max()) + 1
        self.root_ = self._grow(X, y, np.arange(len(y)), set(range(X.shape[1])), 0)
        return self

    def _choose(self, X, y, idx, features):
        scores = {}
        for f in features:
            gain, split_info = split_scores(X[idx, f], y[idx], self.n_values_[f], self.n_classes_)
            if split_info > 0:
                scores[f] = (gain, split_info)
        if not scores:
            return None, 0.0
        if self.criterion == "gain":
            f = max(scores, key=lambda k: (scores[k][0], -k))
            return f, scores[f][0]
        avg = np.mean([g for g, _ in scores.values()])
        eligible = [k for k, (g, _) in scores.items() if g >= avg - 1e-12]
        f = max(eligible, key=lambda k: (scores[k][0] / scores[k][1], -k))
        return f, scores[f][0]

    def _grow(self, X, y, idx, features, depth):
        counts = np.bincount(y[idx], minlength=self.n_classes_)
        node = Node(int(np.argmax(counts)), len(idx), depth)
        if (counts.max() == len(idx) or len(idx) < self.min_samples_split or not features
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node
        f, gain = self._choose(X, y, idx, features)
        if f is None or gain <= self.min_gain:
            return node
        node.feature = f
        values = X[idx, f]
        for v in np.unique(values):
            node.children[int(v)] = self._grow(X, y, idx[values == v], features - {f}, depth + 1)
        return node

    def _leaf(self, row):
        node = self.root_
        while node.feature is not None:
            child = node.children.get(int(row[node.feature]))
            if child is None:  # value not seen at this node in training
                break
            node = child
        return node.prediction

    def predict(self, X):
        X = np.asarray(X, dtype=np.int64)
        return np.array([self._leaf(row) for row in X])

    def internal_nodes(self):
        """(depth, feature, n_rows) for every internal node."""
        out, stack = [], [self.root_]
        while stack:
            node = stack.pop()
            if node.feature is not None:
                out.append((node.depth, node.feature, node.n))
                stack.extend(node.children.values())
        return out

    def n_leaves(self):
        count, stack = 0, [self.root_]
        while stack:
            node = stack.pop()
            if node.feature is None:
                count += 1
            else:
                stack.extend(node.children.values())
        return count
