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

# Single standardized feature space shared by all three fields.
# Ablated against per-field subsets (LOO on the seed annotations):
# subsets dropped useful cross-signal; the shared space scored best overall.
FIELD_FEATURES = {f: FEATURE_NAMES for f in ("head", "eye", "mouth")}


class StateClassifier:
    def __init__(self) -> None:
        self.centroids: dict[str, dict[str, np.ndarray]] = {}
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # field → (mean, std)

    # ------------------------------------------------------------------

    def fit(
        self,
        features: dict[str, dict[str, float]],
        annotations: dict[str, dict],
    ) -> None:
        """Learn per-class centroids from annotated examples."""
        x_all: list[np.ndarray] = []
        ids: list[str] = []
        y: dict[str, list[str]] = {"head": [], "eye": [], "mouth": []}

        for fid, ann in annotations.items():
            feat = features.get(fid)
            if feat is None:
                continue
            x_all.append(np.array([feat[n] for n in FEATURE_NAMES]))
            ids.append(fid)
            for field in y:
                y[field].append(ann.get(field, ""))

        if not x_all:
            raise ValueError("No annotated examples with features found")

        X_all = np.vstack(x_all)
        col = {name: i for i, name in enumerate(FEATURE_NAMES)}

        for field, labels in y.items():
            names = FIELD_FEATURES[field]
            X = X_all[:, [col[n] for n in names]]
            mean = X.mean(axis=0)
            std = X.std(axis=0) + 1e-6
            Xz = (X - mean) / std
            self.stats[field] = (mean, std)

            centroids: dict[str, np.ndarray] = {}
            for lab in set(labels):
                if not lab:
                    continue
                centroids[lab] = Xz[[i for i, v in enumerate(labels) if v == lab]].mean(axis=0)
            self.centroids[field] = centroids

    # ------------------------------------------------------------------

    def predict(self, feat: dict[str, float]) -> dict[str, tuple[str, float]]:
        """Predict (label, confidence) for head / eye / mouth."""
        if not self.stats:
            raise RuntimeError("Classifier not fitted yet")
        vec = {n: feat.get(n, 0.0) for n in FEATURE_NAMES}

        out: dict[str, tuple[str, float]] = {}
        for field, cent in self.centroids.items():
            mean, std = self.stats[field]
            names = FIELD_FEATURES[field]
            z = (np.array([vec[n] for n in names]) - mean) / std

            best_lab, best_d = "", float("inf")
            second_d = float("inf")
            for lab, c in cent.items():
                d = float(np.linalg.norm(z - c))
                if d < best_d:
                    second_d = best_d
                    best_d, best_lab = d, lab
                elif d < second_d:
                    second_d = d
            # Margin between best and second-best class distance.
            # 0 = ambiguous (both classes equally near), →1 = clear winner.
            conf = (second_d - best_d) / (second_d + best_d + 1e-9)
            out[field] = (best_lab, conf)
        return out

    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "centroids": {
                f: {lab: c.tolist() for lab, c in cs.items()}
                for f, cs in self.centroids.items()
            },
            "stats": {
                f: (m.tolist(), s.tolist()) for f, (m, s) in self.stats.items()
            },
        }, indent=1))

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        self.centroids = {
            f: {lab: np.array(c) for lab, c in cs.items()}
            for f, cs in data["centroids"].items()
        }
        self.stats = {
            f: (np.array(m), np.array(s))
            for f, (m, s) in data["stats"].items()
        }
        return True
