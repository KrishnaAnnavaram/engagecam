"""Baseline classifiers as scikit-learn pipelines.

Each reduction (PCA) and each scaler is a pipeline step, so it is fitted on the training split only.
Class imbalance is handled with class weights, or with random oversampling of the TRAINING rows only
(after the split). No synthetic sample can reach validation or test.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from engagecam.states import STATES

BASELINES = ("pca_logreg", "hog_logreg", "hog_mlp")


def build(name: str, seed: int = 42, n_components: int = 64):
    if name == "pca_logreg":
        return make_pipeline(StandardScaler(), PCA(n_components=n_components, random_state=seed),
                             LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))
    if name == "hog_logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced",
                                                                  random_state=seed))
    if name == "hog_mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, max_iter=400,
                                                             early_stopping=True, random_state=seed))
    raise KeyError(f"unknown baseline {name!r}. Known: {BASELINES}")


def oversample_train(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random oversampling of the minority states to the majority count. Use it on TRAINING rows only."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=len(STATES))
    target = counts.max()
    idx = [np.arange(len(y))]
    for k, n in enumerate(counts):
        if 0 < n < target:
            idx.append(rng.choice(np.flatnonzero(y == k), target - n, replace=True))
    sel = np.concatenate(idx)
    return X[sel], y[sel]


def full_proba(model, X: np.ndarray) -> np.ndarray:
    """predict_proba with one column for each of the 7 states."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(STATES)))
    out[:, model.classes_] = p
    return out
