import json

import numpy as np
import pytest

from engagecam.cli import main


def test_cli_round_trip(tmp_path, capsys):
    data = tmp_path / "d"
    assert main(["synth", "--out", str(data), "--subjects", "24"]) == 0
    manifest = data / "manifest.csv"
    assert main(["validate", str(manifest)]) == 0
    assert "Cramer's V" in capsys.readouterr().out
    rebuilt = tmp_path / "m2.csv"
    assert main(["manifest", str(data), "--out", str(rebuilt)]) == 0
    split = tmp_path / "s.csv"
    assert main(["split", str(manifest), "--out", str(split)]) == 0
    assert "shared subjects: 0" in capsys.readouterr().out
    runs = tmp_path / "runs"
    assert main(["train", str(manifest), "--seeds", "0", "--oversample", "--out", str(runs)]) == 0
    rec = json.loads((runs / "hog_logreg_subject_seed0.json").read_text(encoding="utf-8"))
    assert {"val", "test", "test_clip", "source_probe", "samples"} <= set(rec)
    one = sorted(data.glob("*/*/*.png"))[0]
    assert main(["infer", str(manifest), str(one)]) == 0
    out = capsys.readouterr().out
    assert "engagement index" in out and "consent" in out


def test_cli_errors(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "none.csv")]) == 1
    assert main(["manifest", str(tmp_path)]) == 1
    assert "error:" in capsys.readouterr().err


def test_deep_tiny_cnn(manifest, source, tmp_path):
    torch = pytest.importorskip("torch")
    from engagecam.deep import FrameDataset, TinyCNN, train
    from engagecam.splits import subject_split

    ds = FrameDataset(manifest.head(4), source, "tiny_cnn")
    x, y = ds[0]
    assert tuple(x.shape) == (3, 64, 64) and isinstance(y, int)
    assert TinyCNN()(torch.zeros(2, 3, 64, 64)).shape == (2, 7)
    out = train(subject_split(manifest, seed=0), source, "tiny_cnn", epochs=2, out_dir=tmp_path)
    assert len(out["history"]) == 2 and out["best_val"] == max(out["history"])
    assert 0 <= out["test"]["balanced_accuracy"] <= 1 and np.isfinite(out["test"]["log_loss"])
