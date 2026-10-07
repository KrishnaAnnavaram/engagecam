"""The frame manifest: one row for each face image, with subject, clip and source ids.

All splits, training runs and reports read the manifest, so there is one data layout only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from engagecam.states import normalise_state

COLUMNS = ("path", "subject_id", "clip_id", "source", "state", "frame")
# file name convention for build_manifest: <subject>__<clip>__<frame>.<ext> inside <root>/<source>/<state>/
NAME = re.compile(r"^(?P<subject>[^_]+(?:_[^_]+)*)__(?P<clip>[^_]+(?:_[^_]+)*)__(?P<frame>\d+)$")


class ManifestError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("manifest validation failed:\n  - " + "\n  - ".join(problems))


def validate_manifest(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ManifestError([f"missing columns: {missing}"])
    out = df[list(COLUMNS)].copy()
    problems = []
    for col in ("path", "subject_id", "clip_id", "source", "state"):
        if out[col].isna().any():
            problems.append(f"{col}: {int(out[col].isna().sum())} missing values")
    try:
        out["state"] = [normalise_state(s) for s in out["state"]]
    except KeyError as exc:
        problems.append(str(exc))
    for col in ("subject_id", "clip_id", "source"):
        out[col] = out[col].astype(str)
    if out["path"].duplicated().any():
        problems.append(f"path: {int(out['path'].duplicated().sum())} duplicates")
    subj = out.groupby("clip_id")["subject_id"].nunique()
    if (subj > 1).any():
        problems.append(f"{int((subj > 1).sum())} clips belong to more than one subject, first {subj[subj > 1].index[0]}")
    if problems:
        raise ManifestError(problems)
    return out.reset_index(drop=True)


def build_manifest(root: str | Path) -> pd.DataFrame:
    """Scan ``<root>/<source>/<state>/<subject>__<clip>__<frame>.(jpg|png)`` into a manifest."""
    rows, bad = [], []
    for path in sorted(Path(root).glob("*/*/*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        m = NAME.match(path.stem)
        if not m:
            bad.append(path.name)
            continue
        rows.append({"path": str(path), "subject_id": m["subject"], "clip_id": m["clip"],
                     "source": path.parent.parent.name, "state": path.parent.name, "frame": int(m["frame"])})
    if bad:
        raise ManifestError([f"{len(bad)} files do not follow <subject>__<clip>__<frame>, first {bad[0]}"])
    if not rows:
        raise ManifestError([f"no images under {root}"])
    return validate_manifest(pd.DataFrame(rows))


def load_manifest(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{p} does not exist. Run `engagecam manifest` or `engagecam synth`.")
    return validate_manifest(pd.read_csv(p))
