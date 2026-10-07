"""One preprocessing function for training AND inference.

Images are always read with Pillow as RGB. If a caller has an OpenCV (BGR) array, ``from_bgr`` changes
it to RGB first. ``preprocess`` then crops the face, resizes to the spec size and scales the values
the way the backbone expects. A parity test checks that the file path and the BGR path give the same
array.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class InputSpec:
    name: str
    size: int
    scaling: str  # "unit" (0-1), "imagenet" (mean/std) or "raw255" (0-255, the model rescales)
    gray: bool = False


SPECS = {
    "baseline": InputSpec("baseline", 64, "unit", gray=True),
    "tiny_cnn": InputSpec("tiny_cnn", 64, "imagenet"),
    "efficientnet_b0": InputSpec("efficientnet_b0", 224, "imagenet"),
    "keras_efficientnet_b0": InputSpec("keras_efficientnet_b0", 224, "raw255"),
}


def spec_for(name: str) -> InputSpec:
    if name not in SPECS:
        raise KeyError(f"unknown input spec {name!r}. Known: {sorted(SPECS)}")
    return SPECS[name]


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def from_bgr(array: np.ndarray) -> np.ndarray:
    """Change an OpenCV BGR array into RGB."""
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected an H x W x 3 array")
    return np.ascontiguousarray(array[:, :, ::-1])


def preprocess(img_rgb: np.ndarray, spec: InputSpec, detector=None) -> np.ndarray:
    """uint8 RGB H x W x 3 -> float32 array. Gray specs give H x W, colour specs give 3 x H x W."""
    if img_rgb.dtype != np.uint8 or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("preprocess needs a uint8 RGB array of shape H x W x 3")
    from engagecam.faces import CenterCropDetector

    face = (detector or CenterCropDetector()).crop(img_rgb)
    im = Image.fromarray(face).resize((spec.size, spec.size), Image.BILINEAR)
    if spec.gray:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    x = np.asarray(im, dtype=np.float32)
    if spec.scaling == "unit":
        x = x / 255.0
    elif spec.scaling == "imagenet":
        x = (x / 255.0 - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
    elif spec.scaling != "raw255":
        raise ValueError(f"unknown scaling {spec.scaling!r}")
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess_path(path: str | Path, spec: InputSpec, detector=None) -> np.ndarray:
    return preprocess(load_rgb(path), spec, detector)
