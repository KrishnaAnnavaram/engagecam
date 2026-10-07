"""The engagement index: a weighted sum of state probabilities, smoothed over time.

It is an aggregate signal for a session, not a judgement of one person. Do not use it for grades,
discipline or any decision about an individual.
"""

from __future__ import annotations

import numpy as np

from engagecam.states import ENGAGEMENT_WEIGHTS, STATES

WEIGHTS = np.asarray([ENGAGEMENT_WEIGHTS[s] for s in STATES])


def frame_index(P: np.ndarray) -> np.ndarray:
    """Engagement index of each frame in [0, 1]: sum over states of P(state) x weight."""
    P = np.asarray(P, float)
    if P.ndim != 2 or P.shape[1] != len(STATES):
        raise ValueError(f"P must have shape (n, {len(STATES)})")
    return P @ WEIGHTS


def smoothed(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Exponential moving average over time (alpha = weight of the new frame)."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    out = np.empty(len(values))
    acc = None
    for i, v in enumerate(values):
        acc = v if acc is None else alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def session_summary(P: np.ndarray, alpha: float = 0.3) -> dict:
    idx = frame_index(P)
    sm = smoothed(idx, alpha)
    return {"frames": int(len(idx)), "mean_index": float(idx.mean()), "final_smoothed": float(sm[-1]),
            "share_low": float((sm < 0.4).mean())}
