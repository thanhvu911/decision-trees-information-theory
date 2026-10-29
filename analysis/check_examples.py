"""
Independent check of the worked examples in the paper and slides.

Recomputes every entropy / information-gain number shown, using only the
counts and the small table printed on the slides, and adds the gain-ratio
correction the paper mentions but does not compute.

Run:  python analysis/check_examples.py
"""
from collections import Counter
from math import log2


def entropy(labels):
    n = len(labels)
    return -sum(c / n * log2(c / n) for c in Counter(labels).values())


def split_entropy(rows, attr):
    n = len(rows)
    groups = {}
    for r in rows:
        groups.setdefault(r[attr], []).append(r["play"])
    return sum(len(g) / n * entropy(g) for g in groups.values()), groups


def gini(labels):
    n = len(labels)
    return 1 - sum((c / n) ** 2 for c in Counter(labels).values())


print("Paper, section 2: worked examples")
s = ["Yes"] * 6 + ["No"] * 4
s1 = ["Yes"] * 3 + ["No"]
s2 = ["Yes"] * 3 + ["No"] * 3
h_split = 4 / 10 * entropy(s1) + 6 / 10 * entropy(s2)
print(f"  H(S), 6 Yes / 4 No        = {entropy(s):.3f}   (paper: 0.97)")
print(f"  H(S1), 3 Yes / 1 No       = {entropy(s1):.3f}   (paper: 0.8)")
print(f"  H(S2), 3 Yes / 3 No       = {entropy(s2):.3f}   (paper: 1.0)")
print(f"  weighted entropy after    = {h_split:.3f}   (paper: 0.92)")
print(f"  information gain          = {entropy(s) - h_split:.3f}   (paper: 0.97 - 0.92 = 0.05)")

print("\nSlides 7-8: the six-row tennis table")
table = [
    ("Sunny", "High", "Weak", "No"),
    ("Sunny", "High", "Strong", "No"),
    ("Overcast", "High", "Weak", "Yes"),
    ("Rain", "High", "Weak", "Yes"),
    ("Rain", "Normal", "Weak", "Yes"),
    ("Rain", "Normal", "Strong", "No"),
]
rows = [dict(zip(["weather", "humidity", "wind", "play"], t)) for t in table]
h = entropy([r["play"] for r in rows])
print(f"  H(S) = {h:.3f}   (slides: 1.0)")
slide_ig = {"weather": 0.541, "humidity": 0.541, "wind": None}
for attr in ["weather", "humidity", "wind"]:
    hs, groups = split_entropy(rows, attr)
    counts = {k: dict(Counter(v)) for k, v in groups.items()}
    split_info = entropy([r[attr] for r in rows])
    ig = h - hs
    shown = "not shown" if slide_ig[attr] is None else f"slides: {slide_ig[attr]}"
    print(f"  {attr:8s} groups={counts}")
    print(f"           IG = {ig:.3f} ({shown})   gain ratio = {ig / split_info:.3f}")

print("\nGini impurity range")
print(f"  2 classes, 50/50  : {gini(['a', 'b']):.3f}   (paper: maximum 0.5)")
print(f"  3 classes, 1/3 each: {gini(['a', 'b', 'c']):.3f}   (maximum is 1 - 1/k for k classes)")
