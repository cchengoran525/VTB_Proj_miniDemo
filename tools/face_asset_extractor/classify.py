"""Label learning — fit centroids from annotations, then auto-classify.

The classifier is a simple standardised nearest-centroid model.  It is
trained ONLY from the user's annotations, so "what counts as L1 vs L2"
is defined by the user's own labels, not hard-coded thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from features import FEATURE_NAMES


class StateClassifier:
    def __init__(self) -> None:
        self.centroids: dict[str, dict[str, np.ndarray]] = {}
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None

    # ------------------------------------------------------------------

    def fit(
        self,
        features: dict[str, dict[str, float]],
        annotations: dict[str, dict],
    ) -> None:
        """Learn per-class centroids from annotated examples."""
        x: list[np.ndarray] = []
        y_head: list[str] = []
        y_eye: list[str] = []
        y_mouth: list[str] = []

        for fid, ann in annotations.items():
            feat = features.get(fid)
            if feat is None:
                continue
            x.append(np.array([feat[n] for n in FEATURE_NAMES]))
            y_head.append(ann.get("head", ""))
            y_eye.append(ann.get("eye", ""))
            y_mouth.append(ann.get("mouth", ""))

        if not x:
            raise ValueError("No annotated examples with features found")

        X = np.vstack(x)
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0) + 1e-6
        Xz = (X - self.means) / self.stds

        for field, labels in [
            ("head", y_head), ("eye", y_eye), ("mouth", y_mouth),
        ]:
            centroids: dict[str, np.ndarray] = {}
            for lab in set(labels):
                if not lab:
                    continue
                centroids[lab] = Xz[[i for i, y in enumerate(labels) if y == lab]].mean(axis=0)
            self.centroids[field] = centroids

    # ------------------------------------------------------------------

    def predict(
        self, feat: dict[str, float]
    ) -> dict[str, tuple[str, float]]:
        """Predict (label, confidence) for head / eye / mouth."""
        if self.means is None:
            raise RuntimeError("Classifier not fitted yet")
        x = np.array([feat[n] for n in FEATURE_NAMES])
        z = (x - self.means) / self.stds

        out: dict[str, tuple[str, float]] = {}
        for field, cent in self.centroids.items():
            best_lab, best_d = "", float("inf")
            second_d = float("inf")
            for lab, c in cent.items():
                d = float(np.linalg.norm(z - c))
                if d < best_d:
                    second_d = best_d
                    best_d, best_lab = d, lab
                elif d < second_d:
                    second_d = d
            # confidence = margin ratio (1 = far from other classes)
            conf = 1.0 - best_d / (best_d + second_d + 1e-9)
            out[field] = (best_lab, conf)
        return out

    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "centroids": {
                f: {lab: c.tolist() for lab, c in cs.items()}
                for f, cs in self.centroids.items()
            },
            "means": self.means.tolist() if self.means is not None else None,
            "stds": self.stds.tolist() if self.stds is not None else None,
        }, indent=1))

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        self.centroids = {
            f: {lab: np.array(c) for lab, c in cs.items()}
            for f, cs in data["centroids"].items()
        }
        self.means = np.array(data["means"]) if data["means"] else None
        self.stds = np.array(data["stds"]) if data["stds"] else None
        return True
