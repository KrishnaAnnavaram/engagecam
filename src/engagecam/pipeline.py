"""Run one baseline experiment: split, features, fit on train, score val and the untouched test split."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engagecam.baselines import build, full_proba, oversample_train
from engagecam.engagement import session_summary
from engagecam.evaluate import by_source, clip_level, report, source_probe
from engagecam.features import pixels_and_hog
from engagecam.preprocess import spec_for
from engagecam.splits import frame_split, subject_split
from engagecam.states import STATES, encode


@dataclass
class Features:
    pixels: np.ndarray
    hog: np.ndarray

    def for_model(self, model: str) -> np.ndarray:
        return self.pixels if model == "pca_logreg" else self.hog


def compute_features(manifest: pd.DataFrame, source) -> Features:
    pix, h = pixels_and_hog(manifest["path"], source, spec_for("baseline"))
    return Features(pix, h)


@dataclass
class Result:
    model: str
    split_mode: str
    seed: int
    val: dict
    test: dict
    test_clip: dict
    test_by_source: dict
    source_probe: dict
    engagement: dict
    samples: list = field(default_factory=list)


def run(manifest: pd.DataFrame, feats: Features, model: str = "hog_logreg", split_mode: str = "subject",
        seed: int = 42, oversample: bool = False, n_samples: int = 8) -> Result:
    split = subject_split(manifest, seed=seed) if split_mode == "subject" else frame_split(manifest, seed=seed)
    X_all = feats.for_model(model)
    y_all = np.asarray(encode(split["state"]))
    parts = {s: (split["split"] == s).to_numpy() for s in ("train", "val", "test")}
    X_tr, y_tr = X_all[parts["train"]], y_all[parts["train"]]
    if oversample:
        X_tr, y_tr = oversample_train(X_tr, y_tr, seed)  # after the split, training rows only
    clf = build(model, seed).fit(X_tr, y_tr)
    P_val = full_proba(clf, X_all[parts["val"]])
    P_te = full_proba(clf, X_all[parts["test"]])
    test_frames = split[parts["test"]].reset_index(drop=True)
    y_te = y_all[parts["test"]]
    y_clip, P_clip = clip_level(test_frames, P_te, y_te)
    # can the model's own features tell the source? (a shortcut risk when states depend on the source)
    probe = source_probe(X_all[parts["train"]], split.loc[parts["train"], "source"],
                         X_all[parts["test"]], split.loc[parts["test"], "source"], seed)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(test_frames), size=min(n_samples, len(test_frames)), replace=False)
    samples = [{"path": test_frames["path"].iloc[i], "true": STATES[y_te[i]], "pred": STATES[int(P_te[i].argmax())],
                "p": float(P_te[i].max())} for i in pick]  # visual checks come from the TEST split only
    return Result(model, split_mode, seed, report(y_all[parts["val"]], P_val), report(y_te, P_te),
                  report(y_clip, P_clip), by_source(test_frames, P_te, y_te), probe, session_summary(P_te), samples)
