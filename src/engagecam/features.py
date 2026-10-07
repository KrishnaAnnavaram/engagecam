"""Image sources and handcrafted features (gradient-orientation histograms) for the baselines."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from engagecam.preprocess import InputSpec, load_rgb, preprocess


class ImageSource(Protocol):
    def get(self, path: str) -> np.ndarray: ...


class FileSource:
    def get(self, path: str) -> np.ndarray:
        return load_rgb(path)


class MemorySource:
    def __init__(self, images: dict[str, np.ndarray]):
        self.images = images

    def get(self, path: str) -> np.ndarray:
        return self.images[path]


def hog(gray: np.ndarray, cells: int = 8, bins: int = 9) -> np.ndarray:
    """Histogram of gradient orientations in a ``cells`` x ``cells`` grid, L2-normalised per cell."""
    gy, gx = np.gradient(gray.astype(np.float32))
    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) % 180.0) / 180.0 * bins
    h, w = gray.shape
    out = []
    for i in range(cells):
        for j in range(cells):
            sl = (slice(i * h // cells, (i + 1) * h // cells), slice(j * w // cells, (j + 1) * w // cells))
            hist = np.bincount(np.minimum(ang[sl].astype(int), bins - 1).ravel(), weights=mag[sl].ravel(), minlength=bins)
            out.append(hist / (np.linalg.norm(hist) + 1e-6))
    return np.concatenate(out).astype(np.float32)


def pixels_and_hog(paths, source: ImageSource, spec: InputSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return (flattened gray pixels, HOG features), both from the shared ``preprocess``."""
    pix, feats = [], []
    for p in paths:
        g = preprocess(source.get(p), spec)
        if g.ndim == 3:
            g = g.mean(0)
        pix.append(g.ravel())
        feats.append(hog(g))
    return np.stack(pix), np.stack(feats)
