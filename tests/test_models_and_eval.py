import numpy as np
import pandas as pd
import pytest

from engagecam.baselines import build, full_proba, oversample_train
from engagecam.engagement import frame_index, session_summary, smoothed
from engagecam.evaluate import clip_level, report, source_probe, state_source_association
from engagecam.pipeline import run
from engagecam.states import ENGAGEMENT_WEIGHTS, INDEX, STATES


def test_pca_is_fitted_inside_the_pipeline(feats):
    X = feats.pixels
    clf = build("pca_logreg", n_components=16).fit(X[:100], np.arange(100) % 7)
    pca = clf.named_steps["pca"]
    assert pca.n_samples_ == 100  # only the rows given to fit


def test_oversampling_touches_training_rows_only():
    X = np.arange(20).reshape(10, 2)
    y = np.array([0] * 7 + [1] * 3)
    Xo, yo = oversample_train(X, y, seed=0)
    assert np.bincount(yo).tolist() == [7, 7]
    assert set(map(tuple, Xo)) <= set(map(tuple, X))  # copies of training rows, nothing new


def test_full_proba_has_seven_columns(feats):
    clf = build("hog_logreg").fit(feats.hog[:60], np.array([0, 3] * 30))
    P = full_proba(clf, feats.hog[:5])
    assert P.shape == (5, 7) and np.allclose(P.sum(1), 1)


def test_report_keys_and_label_order():
    y = np.array([0, 1, 2, 3, 4, 5, 6])
    P = np.eye(7)
    rep = report(y, P)
    assert rep["balanced_accuracy"] == 1.0 and rep["confusion"]["labels"] == list(STATES)
    for key in ("macro_f1", "kappa", "log_loss", "ece", "auc_ovr_macro", "per_state"):
        assert key in rep


def test_clip_level_averages_frames():
    frames = pd.DataFrame({"clip_id": ["a", "a", "b"]})
    P = np.zeros((3, 7))
    P[0, 0], P[1, 1], P[2, 2] = 1, 1, 1
    y_clip, P_clip = clip_level(frames, P, np.array([0, 0, 2]))
    assert y_clip.tolist() == [0, 2] and P_clip[0, 0] == 0.5


def test_state_source_association():
    m = pd.DataFrame({"state": ["yawn"] * 10 + ["boredom"] * 10, "source": ["b"] * 10 + ["a"] * 10})
    assert state_source_association(m)["cramers_v"] == pytest.approx(1.0)
    m["source"] = ["a", "b"] * 10
    assert state_source_association(m)["cramers_v"] == pytest.approx(0.0)


def test_source_probe_finds_a_source_shortcut():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    src = np.where(X[:, 0] > 0, "b", "a")
    out = source_probe(X[:100], src[:100], X[100:], src[100:])
    assert out["probe_accuracy"] > 0.9 and out["lift"] > 0.3


def test_engagement_index():
    P = np.zeros((2, 7))
    P[0, INDEX["engagement"]] = 1
    P[1, INDEX["sleep"]] = 1
    assert frame_index(P).tolist() == [1.0, 0.0]
    assert smoothed(np.array([1.0, 0.0]), alpha=0.5).tolist() == [1.0, 0.5]
    assert session_summary(P)["mean_index"] == 0.5
    assert set(ENGAGEMENT_WEIGHTS) == set(STATES)
    with pytest.raises(ValueError):
        frame_index(np.ones((2, 3)))


def test_run_scores_the_untouched_test_split(manifest, feats):
    res = run(manifest, feats, "pca_logreg", "subject", seed=0)
    test_paths = {s["path"] for s in res.samples}
    from engagecam.splits import subject_split
    split = subject_split(manifest, seed=0)
    assert test_paths <= set(split.loc[split["split"] == "test", "path"])  # visual checks from test only
    assert res.test["n"] == int((split["split"] == "test").sum())
    assert res.test["balanced_accuracy"] > 1 / 7
    assert 0 <= res.engagement["mean_index"] <= 1


def test_frame_split_is_more_optimistic_than_subject_split(manifest, feats):
    gap = {}
    for mode in ("frame", "subject"):
        gap[mode] = np.mean([run(manifest, feats, "pca_logreg", mode, seed=s).test["balanced_accuracy"] for s in range(3)])
    assert gap["frame"] > gap["subject"]
