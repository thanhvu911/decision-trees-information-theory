"""Accuracy and error rates for groups of rows (used by E2 and E3)."""
import numpy as np


def race_group(race):
    return race if race in ("White", "Black") else "Other races"


def rates(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    pos, neg = y == 1, y == 0
    return {
        "n": int(len(y)),
        "share_positive": float(pos.mean()),
        "accuracy": float((pred == y).mean()),
        # false negative rate: people earning >50K whom the model predicts as <=50K
        "fnr": float((pred[pos] == 0).mean()) if pos.any() else np.nan,
        # false positive rate: people earning <=50K whom the model predicts as >50K
        "fpr": float((pred[neg] == 1).mean()) if neg.any() else np.nan,
        "selection_rate": float((pred == 1).mean()),
    }


def group_metrics(frame, y, pred):
    """One dict per group: all rows, each sex, each race group."""
    y, pred = np.asarray(y), np.asarray(pred)
    out = [{"attribute": "all", "group": "all", **rates(y, pred)}]
    for g in ("Female", "Male"):
        m = (frame["sex"] == g).to_numpy()
        out.append({"attribute": "sex", "group": g, **rates(y[m], pred[m])})
    race = frame["race"].map(race_group).to_numpy()
    for g in ("White", "Black", "Other races"):
        m = race == g
        out.append({"attribute": "race", "group": g, **rates(y[m], pred[m])})
    return out
