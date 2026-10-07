"""Command line interface: ``engagecam <command> [options]``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from engagecam.baselines import BASELINES
from engagecam.evaluate import state_source_association
from engagecam.features import FileSource, MemorySource
from engagecam.manifest import ManifestError, build_manifest, load_manifest, validate_manifest
from engagecam.pipeline import compute_features, run
from engagecam.splits import shared_ids, subject_split
from engagecam.synthetic import make_dataset, write_dataset

NOTICE = ("engagecam gives an aggregate signal for research on online sessions. Use it only with consent. "
          "Do not use it to judge, grade or discipline a person.")


def _seed() -> int:
    raw = os.environ.get("ENGAGECAM_SEED", "42").strip() or "42"
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"ENGAGECAM_SEED must be an integer, got {raw!r}") from exc


def _data_dir() -> Path:
    return Path(os.environ.get("ENGAGECAM_DATA_DIR", "").strip() or "data")


def _out_dir() -> Path:
    return Path(os.environ.get("ENGAGECAM_OUTPUT_DIR", "").strip() or "runs")


def _summary(res) -> str:
    t, c = res.test, res.test_clip
    return (f"{res.model:11s} split={res.split_mode:7s} val bal acc {res.val['balanced_accuracy']:.3f} | "
            f"TEST frames: bal acc {t['balanced_accuracy']:.3f}, macro F1 {t['macro_f1']:.3f}, ECE {t['ece']:.3f} | "
            f"TEST clips: bal acc {c['balanced_accuracy']:.3f} | source probe lift {res.source_probe['lift']:+.3f}")


def cmd_synth(args) -> int:
    out = Path(args.out) if args.out else _data_dir() / "synthetic"
    manifest = write_dataset(out, n_subjects=args.subjects, seed=args.seed)
    manifest.to_csv(out / "manifest.csv", index=False)
    print(f"wrote {len(manifest)} SYNTHETIC frames and {out / 'manifest.csv'}")
    return 0


def cmd_manifest(args) -> int:
    m = build_manifest(args.root)
    m.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(m)} frames, {m['subject_id'].nunique()} subjects, {m['clip_id'].nunique()} clips")
    return 0


def cmd_validate(args) -> int:
    m = load_manifest(args.manifest)
    assoc = state_source_association(m)
    print(f"OK: {len(m)} frames, {m['subject_id'].nunique()} subjects, {m['clip_id'].nunique()} clips")
    print("frames per state: " + json.dumps(m["state"].value_counts().to_dict()))
    print(f"state-source association (Cramer's V): {assoc['cramers_v']:.3f}")
    return 0


def cmd_split(args) -> int:
    split = subject_split(load_manifest(args.manifest), seed=args.seed if args.seed is not None else _seed())
    split.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {split['split'].value_counts().to_dict()}, shared subjects: {len(shared_ids(split, 'subject_id'))}")
    return 0


def cmd_train(args) -> int:
    m = load_manifest(args.manifest)
    feats = compute_features(m, FileSource())
    results = [run(m, feats, args.model, args.split_mode, s, oversample=args.oversample)
               for s in [int(x) for x in args.seeds.split(",")]]
    out = Path(args.out) if args.out else _out_dir()
    out.mkdir(parents=True, exist_ok=True)
    for r in results:
        (out / f"{r.model}_{r.split_mode}_seed{r.seed}.json").write_text(json.dumps(r.__dict__, indent=1, default=float),
                                                                           encoding="utf-8")
        print(_summary(r))
    print(NOTICE)
    return 0


def cmd_infer(args) -> int:
    from engagecam.baselines import build, full_proba
    from engagecam.engagement import frame_index
    from engagecam.features import hog
    from engagecam.preprocess import preprocess_path, spec_for
    from engagecam.states import STATES, encode

    m = load_manifest(args.manifest)
    train_rows = subject_split(m, seed=_seed())
    train_rows = train_rows[train_rows["split"] == "train"]
    feats = compute_features(train_rows, FileSource())
    clf = build("hog_logreg", _seed()).fit(feats.hog, np.asarray(encode(train_rows["state"])))
    for path in args.images:
        P = full_proba(clf, hog(preprocess_path(path, spec_for("baseline")))[None, :])
        top = np.argsort(-P[0])[:3]
        print(f"{path}: " + ", ".join(f"{STATES[k]} {P[0, k]:.2f}" for k in top) + f" | engagement index {frame_index(P)[0]:.2f}")
    print(NOTICE)
    return 0


def cmd_demo(args) -> int:
    manifest, images = make_dataset(n_subjects=60, seed=0)
    manifest = validate_manifest(manifest)
    feats = compute_features(manifest, MemorySource(images))
    assoc = state_source_association(manifest)
    print(f"SYNTHETIC data: {len(manifest)} frames, {manifest['subject_id'].nunique()} subjects, "
          f"{manifest['clip_id'].nunique()} clips, state-source Cramer's V {assoc['cramers_v']:.3f}\n")
    for split_mode in ("frame", "subject"):
        for model in ("pca_logreg", "hog_logreg"):
            rs = [run(manifest, feats, model, split_mode, s) for s in range(3)]
            val = np.mean([r.val["balanced_accuracy"] for r in rs])
            test = np.mean([r.test["balanced_accuracy"] for r in rs])
            clip = np.mean([r.test_clip["balanced_accuracy"] for r in rs])
            lift = np.mean([r.source_probe["lift"] for r in rs])
            print(f"{model:11s} split={split_mode:7s} 3 seeds: val bal acc {val:.3f}, test bal acc {test:.3f}, "
                  f"test clip bal acc {clip:.3f}, source probe lift {lift:+.3f}")
    print("\nWith a frame split, the same subjects are in train and test, so the scores are too high.")
    print(NOTICE)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="engagecam", description="Subject-disjoint engagement-state benchmark.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="write SYNTHETIC face-like frames and a manifest")
    s.add_argument("--out", help="output folder (default: $ENGAGECAM_DATA_DIR/synthetic)")
    s.add_argument("--subjects", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("manifest", help="scan <root>/<source>/<state>/<subject>__<clip>__<frame>.jpg")
    s.add_argument("root")
    s.add_argument("--out", default="data/manifest.csv")
    s.set_defaults(func=cmd_manifest)

    s = sub.add_parser("validate", help="check a manifest and the state-source association")
    s.add_argument("manifest")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("split", help="write a subject-disjoint train/val/test split")
    s.add_argument("manifest")
    s.add_argument("--out", default="data/split.csv")
    s.add_argument("--seed", type=int, default=None)
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("train", help="train and score a baseline over seeds")
    s.add_argument("manifest")
    s.add_argument("--model", default="hog_logreg", choices=list(BASELINES))
    s.add_argument("--split-mode", default="subject", choices=["subject", "frame"])
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--oversample", action="store_true", help="random oversampling of training rows only")
    s.add_argument("--out", help="report folder (default: $ENGAGECAM_OUTPUT_DIR or runs)")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("infer", help="score images with a baseline fitted on the training split")
    s.add_argument("manifest")
    s.add_argument("images", nargs="+")
    s.set_defaults(func=cmd_infer)

    s = sub.add_parser("demo", help="offline demo: frame split against subject split on SYNTHETIC data")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ManifestError, FileNotFoundError, KeyError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
