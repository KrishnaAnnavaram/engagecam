"""Frame and clip metrics, per-source slices, and the dataset-source confound checks."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from engagecam.states import STATES


def ece(y: np.ndarray, P: np.ndarray, bins: int = 15) -> float:
    conf, pred = P.max(1), P.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def report(y: np.ndarray, P: np.ndarray) -> dict:
    """Metrics for one prediction set. ``P`` has one column for each state in ``STATES`` order."""
    y = np.asarray(y, int)
    pred = P.argmax(1)
    labels = list(range(len(STATES)))
    present = sorted(set(y.tolist()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
        out = {
            "n": int(len(y)),
            "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "accuracy": float((pred == y).mean()),
            "kappa": float(cohen_kappa_score(y, pred)),
            "log_loss": float(log_loss(y, np.clip(P, 1e-7, 1), labels=labels)),
            "ece": ece(y, P),
        }
    aucs = [roc_auc_score((y == k).astype(int), P[:, k]) for k in present if 0 < (y == k).sum() < len(y)]
    out["auc_ovr_macro"] = float(np.mean(aucs)) if aucs else float("nan")
    out["per_state"] = {STATES[k]: {"precision": float(prec[k]), "recall": float(rec[k]), "f1": float(f1[k]),
                                    "support": int(sup[k])} for k in labels}
    out["confusion"] = {"labels": list(STATES), "matrix": confusion_matrix(y, pred, labels=labels).tolist()}
    return out


def clip_level(frames: pd.DataFrame, P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean frame probabilities for each clip. Returns (clip labels, clip probabilities)."""
    df = pd.DataFrame(P, columns=list(STATES))
    df["clip_id"] = frames["clip_id"].to_numpy()
    df["y"] = y
    g = df.groupby("clip_id", sort=True)
    return g["y"].first().to_numpy(), g[list(STATES)].mean().to_numpy()


def by_source(frames: pd.DataFrame, P: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    for src in sorted(frames["source"].unique()):
        m = (frames["source"] == src).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[src] = {"n": int(m.sum()), "balanced_accuracy": float(balanced_accuracy_score(y[m], P[m].argmax(1))),
                        "states": sorted({STATES[k] for k in y[m]})}
    return out


def state_source_association(manifest: pd.DataFrame) -> dict:
    """Cramér's V between state and source (0 = independent, 1 = the source gives the state)."""
    table = pd.crosstab(manifest["state"], manifest["source"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"cramers_v": float("nan"), "table": table.to_dict()}
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    return {"cramers_v": v, "table": table.to_dict()}


def source_probe(X_train, src_train, X_test, src_test, seed: int = 0) -> dict:
    """Can the features predict the source on unseen subjects? Compare with the majority rate."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed)).fit(X_train, src_train)
    acc = float((clf.predict(X_test) == np.asarray(src_test)).mean())
    majority = float(pd.Series(src_test).value_counts(normalize=True).iloc[0])
    return {"probe_accuracy": acc, "majority_rate": majority, "lift": acc - majority}
