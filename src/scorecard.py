"""Helpers for a Weight-of-Evidence (WoE) logistic regression scorecard."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def make_bins(x, y, n_bins=6):
    """Quantile bin edges for one numeric feature (fitted on training data only)."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def woe_table(x, y, edges):
    """Weight of Evidence and Information Value for one binned feature."""
    b = pd.cut(x, edges, include_lowest=True)
    t = pd.DataFrame({"bin": b, "y": y}).groupby("bin", observed=False)["y"].agg(["count", "sum"])
    t.columns = ["n", "bads"]
    t["goods"] = t["n"] - t["bads"]
    # 0.5 smoothing avoids log(0) in empty bins
    dist_good = (t["goods"] + 0.5) / (t["goods"].sum() + 0.5)
    dist_bad = (t["bads"] + 0.5) / (t["bads"].sum() + 0.5)
    t["bad_rate"] = t["bads"] / t["n"]
    t["woe"] = np.log(dist_good / dist_bad)
    t["iv"] = (dist_good - dist_bad) * t["woe"]
    return t


class WoEEncoder:
    """Fit bins + WoE on training data, then transform any dataset."""

    def __init__(self, n_bins=6):
        self.n_bins = n_bins
        self.edges, self.tables = {}, {}

    def fit(self, X, y):
        for c in X.columns:
            self.edges[c] = make_bins(X[c], y, self.n_bins)
            self.tables[c] = woe_table(X[c], y, self.edges[c])
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for c in self.edges:
            b = pd.cut(X[c], self.edges[c], include_lowest=True)
            out[c] = b.map(self.tables[c]["woe"]).astype(float)
        return out

    def iv(self):
        return pd.Series({c: t["iv"].sum() for c, t in self.tables.items()}).sort_values(ascending=False)


def gini_ks(y, p):
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": float(np.max(tpr - fpr))}


def to_points(pd_, base_score=600, base_odds=50, pdo=20):
    """Convert PD to a credit score: +20 points doubles the good:bad odds."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    odds = (1 - pd_) / pd_
    return offset + factor * np.log(odds)
