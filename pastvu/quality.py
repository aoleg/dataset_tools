"""Low-quality photos: a second logistic probe on the category embedding,
trained on two folders of the samples folder, _quality_good and _quality_bad.

The probe needs no model of its own: it reads the SigLIP 2 embedding the
category classifier computes anyway, so it costs one dot product per photo.
A photo with P(bad) at or above the threshold is skipped and never saved,
like a photo smaller than --min-side; reclassify deletes the saved ones.

Calibrated on 28,620 downloaded photos with 94 good and 131 bad samples: at
0.88 about 3 % of the photos go, nearly all of them film and TV frame grabs
and soft, foggy or washed-out scans. The probability shifts whenever the two
folders change, so check the threshold again after editing them. C is fixed:
a looser fit spreads the scores towards 0 and 1 and moves every threshold.

Like every folder whose name starts with "_", the two folders (and any other,
such as a folder of photos not sorted yet) are ignored by the category
classifier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

GOOD, BAD = "_quality_good", "_quality_bad"
THRESHOLD = 0.88
C = 8
MIN_EACH = 20
REASON = "low quality"


def sample_items(samples: Path, scan) -> dict[int, list]:
    """The images of the two folders, as classify ImageItems relative to the
    samples folder. scan: classify's scan_root. -> {0: good, 1: bad}"""
    out = {0: [], 1: []}
    for label, folder in ((0, GOOD), (1, BAD)):
        root = Path(samples) / folder
        if root.is_dir():
            for it in scan(root):
                it.root, it.rel = Path(samples).resolve(), f"{folder}/{it.rel}"
                out[label].append(it)
    return out


class Probe:
    """P(bad) of L2-normalised embeddings."""

    def __init__(self, model, threshold: float, n_good: int, n_bad: int):
        self.model, self.threshold, self.n_good, self.n_bad = model, threshold, n_good, n_bad

    def p_bad(self, X) -> np.ndarray:
        X = np.asarray(X, np.float32).reshape(-1, self.model.coef_.shape[1])
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        return self.model.predict_proba(X)[:, 1]

    def reason(self, p: float) -> str:
        return f"{REASON} (P(bad) {p:.2f})"


def probe_from_samples(items: dict[int, list], vectors: dict, threshold: float = THRESHOLD, log=print,
                       min_each: int = MIN_EACH) -> Probe | None:
    """Train the probe on the two folders. vectors: rel -> embedding. threshold 0: off."""
    if threshold <= 0:
        log("  quality filter off (quality_threshold 0)")
        return None
    X, y = [], []
    for label, its in items.items():
        for it in its:
            v = vectors.get(it.rel)
            if v is not None:
                X.append(np.asarray(v, np.float32))
                y.append(label)
    n_bad = int(sum(y))
    n_good = len(y) - n_bad
    if n_good < min_each or n_bad < min_each:
        if n_good or n_bad:
            log(f"  quality filter off: {GOOD} holds {n_good} and {BAD} {n_bad} images "
                f"(at least {min_each} in each needed)")
        return None
    from sklearn.linear_model import LogisticRegression
    X = np.stack(X)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    model = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X, np.array(y))
    log(f"  quality filter: trained on {n_good:,} good and {n_bad:,} bad sample images; "
        f"P(bad) >= {threshold} is skipped")
    return Probe(model, threshold, n_good, n_bad)
