#!/usr/bin/env python3
"""Face Asset Extractor — anime video → labelled transparent PNG assets.

Usage:
  python extractor.py extract  --videos <dir> --workdir <dir>
  python extractor.py detect   --workdir <dir>
  python extractor.py cluster  --workdir <dir>
  python extractor.py annotate --workdir <dir> [--budget 100]
  python extractor.py fit      --workdir <dir>
  python extractor.py classify --workdir <dir>
  python extractor.py export   --workdir <dir> --out <dir>
  python extractor.py run      --videos <dir> --workdir <dir> --out <dir>  (all-in-one)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import classify as classify_mod
import clustering
import exporter as exporter_mod
import features as features_mod
import video_loader as loader_mod
from annotator import Annotator
from face_detect import FaceDetector
from matting import white_bg_matting


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

def load_meta(workdir: Path) -> list[loader_mod.FrameInfo]:
    data = json.loads((workdir / "meta.json").read_text())
    return [
        loader_mod.FrameInfo(
            frame_id=d["frame_id"], video=d["video"],
            video_frame=d["video_frame"], timestamp=d["timestamp"],
            path=d["path"],
        )
        for d in data
    ]


def load_frames(workdir: Path, meta: list[loader_mod.FrameInfo]) -> list[np.ndarray]:
    return [cv2.imread(str(workdir / m.path)) for m in meta]


def load_alpha_masks(workdir: Path, meta: list[loader_mod.FrameInfo]) -> list[np.ndarray]:
    masks = []
    alpha_dir = workdir / "alpha"
    for m in meta:
        if alpha_dir.exists():
            a = cv2.imread(str(alpha_dir / f"{m.frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
            masks.append(a if a is not None else np.zeros((480, 640), np.uint8))
        else:
            masks.append(np.zeros((480, 640), np.uint8))
    return masks


# ----------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------

def cmd_extract(args) -> None:
    workdir = Path(args.workdir)
    videos = sorted(Path(args.videos).glob("*.mp4")) + sorted(Path(args.videos).glob("*.mov")) + sorted(Path(args.videos).glob("*.webm"))
    if not videos:
        print(f"No videos found in {args.videos}")
        return

    print(f"Extracting {len(videos)} videos @ {args.fps}fps…")
    ld = loader_mod.VideoLoader(workdir, sample_fps=args.fps)
    frames = ld.extract(videos)
    ld.save_meta(frames)


def cmd_detect(args) -> None:
    workdir = Path(args.workdir)
    meta = load_meta(workdir)
    frames = load_frames(workdir, meta)
    det = FaceDetector()

    alpha_dir = workdir / "alpha"
    alpha_dir.mkdir(exist_ok=True)

    detections: list[dict | None] = []
    for i, (m, frame) in enumerate(zip(meta, frames)):
        rgba = white_bg_matting(frame)
        cv2.imwrite(str(alpha_dir / f"{m.frame_id:06d}.png"), rgba[:, :, 3])
        d = det.detect(frame, rgba[:, :, 3])
        if d:
            detections.append({
                "bbox": list(d.bbox),
                "left_eye": list(d.left_eye) if d.left_eye else None,
                "right_eye": list(d.right_eye) if d.right_eye else None,
                "nose": list(d.nose) if d.nose else None,
                "mouth": list(d.mouth) if d.mouth else None,
                "confidence": d.confidence,
            })
        else:
            detections.append(None)
        if (i + 1) % 200 == 0:
            print(f"  … {i + 1}/{len(meta)}")

    found = sum(1 for d in detections if d)
    print(f"Faces found: {found}/{len(meta)}")
    (workdir / "detections.json").write_text(json.dumps(detections))


def cmd_cluster(args) -> None:
    workdir = Path(args.workdir)
    meta = load_meta(workdir)
    frames = load_frames(workdir, meta)
    masks = load_alpha_masks(workdir, meta)
    detections = json.loads((workdir / "detections.json").read_text())
    det_objs = [_det_from_dict(d) for d in detections]

    print("Extracting features…")
    feats: list[dict[str, float] | None] = []
    for frame, mask, d in zip(frames, masks, det_objs):
        feats.append(features_mod.extract_features(frame, mask, d))

    clusters = clustering.dedup_frames(feats)
    reps = clustering.pick_representatives(clusters, frames)
    print(f"Clusters: {len(clusters)} → candidates: {len(reps)}")

    (workdir / "features.json").write_text(json.dumps(feats))
    (workdir / "candidates.json").write_text(json.dumps(reps))


def cmd_annotate(args) -> None:
    workdir = Path(args.workdir)
    meta = load_meta(workdir)
    frames = load_frames(workdir, meta)
    candidates = json.loads((workdir / "candidates.json").read_text())
    detections = [_det_from_dict(d) for d in json.loads((workdir / "detections.json").read_text())]

    annotations = {}
    ann_path = workdir / "annotations.json"
    if ann_path.exists():
        annotations = json.loads(ann_path.read_text())

    prefills = {}
    cls_path = workdir / "classifier.json"
    if cls_path.exists():
        clf = classify_mod.StateClassifier()
        if clf.load(cls_path):
            feats = json.loads((workdir / "features.json").read_text())
            for fid in candidates:
                f = feats[fid]
                if f:
                    pred = clf.predict(f)
                    prefills[str(fid)] = {k: v[0] for k, v in pred.items()}

    print(f"Annotate {len(candidates)} candidates (budget {args.budget})")
    ann = Annotator(
        workdir, candidates, frames, detections,
        annotations=annotations, prefills=prefills, budget=args.budget,
    )
    ann.run()


def cmd_fit(args) -> None:
    workdir = Path(args.workdir)
    ann_path = workdir / "annotations.json"
    if not ann_path.exists():
        print("No annotations yet — run annotate first")
        return
    annotations = json.loads(ann_path.read_text())
    feats = json.loads((workdir / "features.json").read_text())

    clf = classify_mod.StateClassifier()
    clf.fit(
        {str(i): f for i, f in enumerate(feats) if f},
        annotations,
    )
    clf.save(workdir / "classifier.json")
    print(f"Fitted on {len(annotations)} examples → classifier.json")


def cmd_classify(args) -> None:
    workdir = Path(args.workdir)
    clf = classify_mod.StateClassifier()
    if not clf.load(workdir / "classifier.json"):
        print("No classifier — run fit first")
        return
    feats = json.loads((workdir / "features.json").read_text())

    results: dict[str, dict] = {}
    for i, f in enumerate(feats):
        if not f:
            continue
        pred = clf.predict(f)
        results[str(i)] = {
            "head": pred["head"][0],
            "eye": pred["eye"][0],
            "mouth": pred["mouth"][0],
            "conf": min(pred["head"][1], pred["eye"][1], pred["mouth"][1]),
        }

    # merge manual annotations (they win)
    ann_path = workdir / "annotations.json"
    if ann_path.exists():
        annotations = json.loads(ann_path.read_text())
        for fid, ann in annotations.items():
            results[fid] = {
                "head": ann.get("head") or results.get(fid, {}).get("head", "center"),
                "eye": ann.get("eye") or results.get(fid, {}).get("eye", "open"),
                "mouth": ann.get("mouth") or results.get(fid, {}).get("mouth", "closed"),
                "conf": 1.0,
            }

    (workdir / "classifications.json").write_text(json.dumps(results, indent=1))
    low_conf = sum(1 for r in results.values() if r["conf"] < 0.5)
    print(f"Classified {len(results)} frames ({low_conf} low-confidence)")


def cmd_export(args) -> None:
    workdir = Path(args.workdir)
    meta = load_meta(workdir)
    frames = load_frames(workdir, meta)
    classifications = json.loads((workdir / "classifications.json").read_text())

    counts = exporter_mod.export(workdir, Path(args.out), frames, classifications)
    total = sum(counts.values())
    print(f"Exported {total} PNGs → {args.out}")
    for theme, n in counts.items():
        print(f"  {theme}: {n}")


# ----------------------------------------------------------------------

def _det_from_dict(d: dict | None):
    if not d:
        return None
    from face_detect import FaceDetection
    return FaceDetection(
        bbox=tuple(d["bbox"]),
        left_eye=tuple(d["left_eye"]) if d.get("left_eye") else None,
        right_eye=tuple(d["right_eye"]) if d.get("right_eye") else None,
        nose=tuple(d["nose"]) if d.get("nose") else None,
        mouth=tuple(d["mouth"]) if d.get("mouth") else None,
        confidence=d.get("confidence", 0.0),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Face Asset Extractor")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("extract")
    p.add_argument("--videos", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--fps", type=float, default=10.0)
    p.set_defaults(func=cmd_extract)

    for name in ["detect", "cluster", "annotate", "fit", "classify"]:
        p = sub.add_parser(name)
        p.add_argument("--workdir", required=True)
        if name == "annotate":
            p.add_argument("--budget", type=int, default=100)
        p.set_defaults(func=globals()[f"cmd_{name}"])

    p = sub.add_parser("export")
    p.add_argument("--workdir", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export)

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
