"""Synthetic face-like frames with subjects, clips and two sources. No real person is shown.

Each state changes the eyes, the mouth, the brows and the head tilt. Each subject has its own skin
tone, face width and background. Source ``src_b`` (a car-like camera) holds most ``yawn`` and ``sleep``
clips, as YawDD did in the earlier data, so the source confound is measurable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from engagecam.states import STATES

# eye openness, mouth openness, brow offset (negative = lowered), head tilt in pixels
_CUES = {
    "active": (1.0, 0.30, 0.0, 0.0),
    "boredom": (0.5, 0.05, 0.0, 3.0),
    "confusion": (0.9, 0.10, 2.0, 1.0),
    "engagement": (1.0, 0.05, 0.5, 0.0),
    "frustration": (0.8, 0.12, -2.0, 0.0),
    "sleep": (0.0, 0.05, 0.0, 4.0),
    "yawn": (0.3, 0.90, 1.0, 1.0),
}
SOURCE_B_STATES = ("yawn", "sleep")


def _ellipse(yy, xx, cy, cx, ry, rx):
    return ((yy - cy) / max(ry, 0.3)) ** 2 + ((xx - cx) / max(rx, 0.3)) ** 2 <= 1.0


def render_face(state: str, subject: dict, source: str, rng: np.random.Generator, size: int = 64) -> np.ndarray:
    eye, mouth, brow, tilt = _CUES[state]
    eye = float(np.clip(eye * subject["eye"] + rng.normal(0, 0.15), 0, 1))
    mouth = float(np.clip(mouth + subject["mouth"] + rng.normal(0, 0.08), 0, 1))
    brow += subject["brow"] + rng.normal(0, 0.7)
    tilt += subject["tilt"] + rng.normal(0, 1.0)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    s = size / 64.0
    img = np.empty((size, size, 3), np.float32)
    img[:] = subject["background"]
    if source == "src_b":
        img *= 0.75  # darker, grayer camera
        img += 0.25 * img.mean(-1, keepdims=True)
    cy, cx = 32 * s + rng.normal(0, 1), 32 * s + tilt * s
    face = _ellipse(yy, xx, cy, cx, 22 * s, subject["width"] * s)
    img[face] = subject["skin"]
    dark = np.asarray((40, 30, 30), np.float32)
    for dx in (-8, 8):
        ex = cx + dx * s + tilt * 0.3
        img[_ellipse(yy, xx, cy - 5 * s, ex, max(0.4, 3.0 * eye) * s, 3.5 * s)] = dark
        by = cy - 11 * s - brow * s + (dx > 0) * (brow * 0.5 * s if state == "confusion" else 0)
        img[_ellipse(yy, xx, by, ex, 0.9 * s, 4.0 * s)] = dark * 1.5
    img[_ellipse(yy, xx, cy + 11 * s, cx, max(0.6, 7.0 * mouth) * s, (5 + 2 * mouth) * s)] = (120, 40, 50)
    img += rng.normal(0, 6.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_dataset(n_subjects: int = 60, clips_per_subject: int = 4, frames_per_clip: int = 6, size: int = 64,
                 seed: int = 0):
    """Return ``(manifest, images)``. ``images`` maps the manifest ``path`` to a uint8 RGB array."""
    rng = np.random.default_rng(seed)
    rows, images = [], {}
    states = list(STATES)
    for sid in range(n_subjects):
        subject = {"skin": rng.uniform((150, 100, 80), (240, 200, 170)),
                   "width": rng.uniform(15, 19),
                   "background": rng.uniform(40, 220, 3),
                   # each person shows the states in their own way: this makes subject leakage measurable
                   "eye": rng.uniform(0.6, 1.0), "mouth": rng.normal(0, 0.12),
                   "brow": rng.normal(0, 1.2), "tilt": rng.normal(0, 1.5)}
        own_states = rng.choice(states, size=clips_per_subject, replace=False)
        for cid, state in enumerate(own_states):
            p_b = 0.85 if state in SOURCE_B_STATES else 0.1
            source = "src_b" if rng.random() < p_b else "src_a"
            for f in range(frames_per_clip):
                path = f"synthetic/{source}/{state}/S{sid:03d}__S{sid:03d}C{cid}__{f}.png"
                images[path] = render_face(str(state), subject, source, rng, size)
                rows.append({"path": path, "subject_id": f"S{sid:03d}", "clip_id": f"S{sid:03d}C{cid}",
                             "source": source, "state": str(state), "frame": f})
    return pd.DataFrame(rows), images


def write_dataset(root: str | Path, **kwargs) -> pd.DataFrame:
    """Write PNG files in the ``<root>/<source>/<state>/<subject>__<clip>__<frame>.png`` layout."""
    manifest, images = make_dataset(**kwargs)
    root = Path(root)
    new_paths = []
    for path in manifest["path"]:
        rel = Path(path).relative_to("synthetic")
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(images[path]).save(dest)
        new_paths.append(str(dest))
    manifest = manifest.copy()
    manifest["path"] = new_paths
    return manifest
