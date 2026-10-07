"""Subject-disjoint train / val / test splits.

Video datasets give many near-identical frames of one person. A frame-level split lets the model
learn faces and rooms instead of states. Here each subject (and so each clip) is in one split.
``frame_split`` exists only to measure that leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


class SplitLeakError(AssertionError):
    """A subject or a clip is in more than one split."""


def subject_split(df: pd.DataFrame, test_fraction: float = 0.2, val_fraction: float = 0.15, seed: int = 42) -> pd.DataFrame:
    if not (0.05 <= test_fraction <= 0.4 and 0.05 <= val_fraction <= 0.4):
        raise ValueError("fractions must be between 0.05 and 0.4")
    out = df.reset_index(drop=True).copy()

    def take(idx: np.ndarray, frac: float, rs: int) -> np.ndarray:
        k = max(2, int(round(1 / frac)))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=rs)
        _, held = next(sgkf.split(idx, out["state"].iloc[idx], out["subject_id"].iloc[idx]))
        return idx[held]

    all_idx = np.arange(len(out))
    test = take(all_idx, test_fraction, seed)
    rest = np.setdiff1d(all_idx, test)
    val = take(rest, val_fraction / (1 - test_fraction), seed + 1)
    out["split"] = "train"
    out.loc[test, "split"] = "test"
    out.loc[val, "split"] = "val"
    assert_disjoint(out)
    return out


def frame_split(df: pd.DataFrame, test_fraction: float = 0.2, val_fraction: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Random frame-level split. It leaks subjects. Use it only for the leakage comparison."""
    out = df.reset_index(drop=True).copy()
    idx = np.arange(len(out))
    rest, test = train_test_split(idx, test_size=test_fraction, stratify=out["state"], random_state=seed)
    train, val = train_test_split(rest, test_size=val_fraction / (1 - test_fraction),
                                  stratify=out["state"].iloc[rest], random_state=seed)
    out["split"] = "train"
    out.loc[test, "split"] = "test"
    out.loc[val, "split"] = "val"
    return out


def shared_ids(df: pd.DataFrame, column: str) -> set[str]:
    per = df.groupby(column)["split"].nunique()
    return set(per[per > 1].index)


def assert_disjoint(df: pd.DataFrame) -> None:
    for col in ("subject_id", "clip_id"):
        shared = shared_ids(df, col)
        if shared:
            raise SplitLeakError(f"{len(shared)} {col} values are in more than one split, first {sorted(shared)[0]}")
