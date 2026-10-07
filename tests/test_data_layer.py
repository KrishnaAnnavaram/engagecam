import numpy as np
import pandas as pd
import pytest
from PIL import Image

from engagecam.faces import CenterCropDetector
from engagecam.manifest import ManifestError, build_manifest, validate_manifest
from engagecam.preprocess import from_bgr, load_rgb, preprocess, preprocess_path, spec_for
from engagecam.splits import SplitLeakError, assert_disjoint, frame_split, shared_ids, subject_split
from engagecam.states import STATES, UnknownState, daisee_state, encode, normalise_state, yawdd_state


def test_state_order_and_aliases():
    assert STATES[0] == "active" and len(STATES) == 7
    assert normalise_state("Yawning") == "yawn" and normalise_state(" Engaged ") == "engagement"
    assert encode(["sleep", "Boredom"]) == [5, 1]
    with pytest.raises(UnknownState):
        normalise_state("happy")


@pytest.mark.parametrize("levels,expected", [
    ((3, 1, 0, 0), "boredom"), ((0, 3, 0, 0), "engagement"), ((0, 1, 2, 2), "confusion"),
    ((1, 1, 1, 1), "active"), ((0, 2, 0, 3), "frustration"),
])
def test_daisee_rule(levels, expected):
    assert daisee_state(*levels) == expected


def test_daisee_rule_refuses_bad_levels():
    with pytest.raises(ValueError):
        daisee_state(5, 0, 0, 0)
    assert yawdd_state("talking") == "active" and yawdd_state("yawning") == "yawn"


def test_manifest_rules(manifest):
    bad = manifest.copy()
    bad.loc[0, "clip_id"] = bad.loc[bad["subject_id"] != bad.loc[0, "subject_id"], "clip_id"].iloc[0]
    with pytest.raises(ManifestError, match="more than one subject"):
        validate_manifest(bad)
    with pytest.raises(ManifestError, match="missing columns"):
        validate_manifest(manifest.drop(columns=["subject_id"]))
    worse = manifest.copy()
    worse.loc[1, "state"] = "happy"
    with pytest.raises(ManifestError):
        validate_manifest(worse)


def test_build_manifest_from_folders(tmp_path):
    for src, state, name in [("daisee", "Boredom", "u1__c1__0.png"), ("yawdd", "yawn", "u2__c9__3.png")]:
        (tmp_path / src / state).mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / src / state / name)
    m = build_manifest(tmp_path)
    assert list(m["state"]) == ["boredom", "yawn"] and list(m["subject_id"]) == ["u1", "u2"]
    (tmp_path / "daisee" / "Boredom" / "bad-name.png").write_bytes(b"x")
    with pytest.raises(ManifestError, match="do not follow"):
        build_manifest(tmp_path)


def test_subject_split_is_disjoint(manifest):
    for seed in range(3):
        split = subject_split(manifest, seed=seed)
        assert not shared_ids(split, "subject_id") and not shared_ids(split, "clip_id")
        assert set(split["split"]) == {"train", "val", "test"}


def test_frame_split_leaks_and_is_detected(manifest):
    split = frame_split(manifest, seed=0)
    assert shared_ids(split, "subject_id")
    with pytest.raises(SplitLeakError):
        assert_disjoint(split)


def test_preprocess_parity_between_file_and_bgr_array(tmp_path, source, manifest):
    rgb = source.get(manifest["path"].iloc[0])
    path = tmp_path / "f.png"
    Image.fromarray(rgb).save(path)
    spec = spec_for("tiny_cnn")
    from_file = preprocess_path(path, spec)
    from_cv2 = preprocess(from_bgr(rgb[:, :, ::-1]), spec)  # an OpenCV caller gives BGR
    assert np.array_equal(from_file, from_cv2)
    assert np.array_equal(load_rgb(path), rgb)


def test_specs_match_the_backbones(source, manifest):
    img = source.get(manifest["path"].iloc[0])
    assert preprocess(img, spec_for("efficientnet_b0")).shape == (3, 224, 224)
    raw = preprocess(img, spec_for("keras_efficientnet_b0"))
    assert raw.max() > 1.5  # 0-255 input for a model that rescales inside
    assert preprocess(img, spec_for("baseline")).shape == (64, 64)
    with pytest.raises(ValueError):
        preprocess(img.astype(np.float32), spec_for("baseline"))
    with pytest.raises(KeyError):
        spec_for("resnet")


def test_center_crop():
    out = CenterCropDetector(0.5).crop(np.zeros((100, 60, 3), np.uint8))
    assert out.shape == (30, 30, 3)
    with pytest.raises(ValueError):
        CenterCropDetector(0.1)
