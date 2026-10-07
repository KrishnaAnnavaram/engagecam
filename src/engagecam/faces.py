"""Face crop before classification, so the background and the camera are less of a shortcut.

``CenterCropDetector`` is the offline default (a square centre crop with a margin). ``MediaPipeDetector``
uses MediaPipe face detection (``pip install engagecam[faces]``) and falls back to the centre crop when
it finds no face.
"""

from __future__ import annotations

import numpy as np


class CenterCropDetector:
    def __init__(self, keep: float = 0.8):
        if not 0.2 <= keep <= 1.0:
            raise ValueError("keep must be between 0.2 and 1.0")
        self.keep = keep

    def crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        side = int(min(h, w) * self.keep)
        top, left = (h - side) // 2, (w - side) // 2
        return img[top:top + side, left:left + side]


class MediaPipeDetector:
    def __init__(self, margin: float = 0.25, min_confidence: float = 0.5):
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError("mediapipe is not installed. Run: pip install 'engagecam[faces]'") from exc
        self.detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_confidence)
        self.margin = margin
        self.fallback = CenterCropDetector()

    def crop(self, img: np.ndarray) -> np.ndarray:  # pragma: no cover - needs the extra
        res = self.detector.process(img)
        if not res.detections:
            return self.fallback.crop(img)
        box = res.detections[0].location_data.relative_bounding_box
        h, w = img.shape[:2]
        x0, y0 = box.xmin - self.margin * box.width, box.ymin - self.margin * box.height
        x1, y1 = box.xmin + (1 + self.margin) * box.width, box.ymin + (1 + self.margin) * box.height
        x0, y0, x1, y1 = (max(0, int(x0 * w)), max(0, int(y0 * h)), min(w, int(x1 * w)), min(h, int(y1 * h)))
        return img[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else self.fallback.crop(img)
