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

A photo that is one of the samples (its embedding at cosine MATCH_COS or
closer to a sample's) takes the sample's label whatever the probe says: a
copy of a _quality_bad sample is skipped or deleted, a copy of a
_quality_good sample is never sent away for its quality. The probe catches
only about half of its own bad samples (on 428 bad and 270 good), and a
sample copied out of the base stayed there. The cosine finds a copy that was
re-encoded or downscaled since; the sha256 does not.

Like every folder whose name starts with "_", the two folders (and any other,
such as a folder of photos not sorted yet) are ignored by the category
classifier.

NIQE beside the probe (quality_threshold_niqe of the conf; 0 turns it off): a
second logistic regression stacks the probe's logit and NIQE above NIQE_FROM,
the photo's NIQE at the training resolution (niqe.py, stored per photo).
Film grain and print texture give NIQE 6 to 12 and are fine for training; a
halftone screen that survives the 1024² downscale gives more. Under NIQE_FROM
the NIQE side adds nothing, so a grainy photo is judged by its probe logit
alone. The stack is fitted on the samples: the probe's logit out of fold (10
folds) and NIQE of each sample at 1024² (cached in the samples folder).
Chosen on the base (2026-11-06) with 134 verdicts on 500 random saved photos:
the full NIQE score or its 72 block features in the stack flagged film grain,
which no _quality_good sample had; at 0.80 this rule lost 2 of 93 good
verdicts and caught 13 of 41 bad ones, 4.7 % of the base. A review of what it
flagged showed that a fine halftone screen (about 1.65 px at 1024²) is fine for
training too, with NIQE 27 to 32, above the coarse screens of _quality_bad
(13 to 19). With three such photos in _quality_good the fitted NIQE weight
fell to 0.011 against 1.09 for the logit: the weights come from the folders.
That base downloads and reviews at 0.85. A photo without a
NIQE score is judged by the probe alone at quality_threshold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

GOOD, BAD = "_quality_good", "_quality_bad"
THRESHOLD = 0.88
C = 8
MIN_EACH = 20
REASON = "low quality"
# A saved photo and the sample copied from it measured 0.998 or more (167 of them, 34 rewritten since);
# two different photos of the same scene stay well under.
MATCH_COS = 0.995
NONE = -1
# the NIQE stack: NIQE counts only above NIQE_FROM; the stack's C; the out-of-fold split of the probe
NIQE_FROM = 12.0
THRESHOLD_NIQE = 0.80
STACK_C = 1.0
FOLDS = 10
NIQE_CACHE = "niqe.json"            # in classify's _embeddings folder of the samples


def reason(p: float, with_niqe: bool = False) -> str:
    return f"{REASON} (P(bad) {p:.2f}{' with NIQE' if with_niqe else ''})"


def sample_reason(cos: float) -> str:
    return f"{REASON} (a {BAD} sample, cos {cos:.3f})"


def _unit(X) -> np.ndarray:
    X = np.asarray(X, np.float32)
    X = X.reshape(len(X), -1) if X.ndim > 1 else X.reshape(1, -1)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


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
    """P(bad) of L2-normalised embeddings, and which embeddings are copies of the samples."""

    def __init__(self, model, threshold: float, n_good: int, n_bad: int, samples=None, labels=None, rels=None):
        """samples: the unit embeddings of both folders; labels: 0 good, 1 bad for each; rels: their paths
        relative to the samples folder."""
        self.model, self.threshold, self.n_good, self.n_bad = model, threshold, n_good, n_bad
        dim = model.coef_.shape[1]
        self.samples = np.zeros((0, dim), np.float32) if samples is None else np.asarray(samples, np.float32)
        self.labels = np.zeros(0, int) if labels is None else np.asarray(labels, int)
        self.rels = list(rels or [])
        self.stack = None                  # (w_logit, w_niqe, intercept, mu, sd) once fit_stack has run
        self.threshold_niqe = 0.0

    def logit(self, X) -> np.ndarray:
        return self.model.decision_function(_unit(X).reshape(-1, self.model.coef_.shape[1]))

    def p_bad(self, X) -> np.ndarray:
        """The probe alone (cherrypick, and the photos without a NIQE score)."""
        return 1.0 / (1.0 + np.exp(-self.logit(X)))

    def score(self, X, niqe=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """P(bad) of each embedding, the threshold it is held to, and whether NIQE was used: the stack
        at threshold_niqe where the stack is on and a NIQE score is given (finite), else the probe
        alone at threshold."""
        lg = self.logit(X)
        p, thr = 1.0 / (1.0 + np.exp(-lg)), np.full(len(lg), float(self.threshold))
        with_niqe = np.zeros(len(lg), bool)
        if self.stack is not None and niqe is not None:
            s = np.asarray(niqe, float).reshape(-1)
            with_niqe = np.isfinite(s)
            w1, w2, b, mu, sd = self.stack
            z = (np.maximum(0.0, np.nan_to_num(s) - NIQE_FROM) - mu) / sd
            p = np.where(with_niqe, 1.0 / (1.0 + np.exp(-(w1 * lg + w2 * z + b))), p)
            thr = np.where(with_niqe, self.threshold_niqe, thr)
        return p, thr, with_niqe

    def reason(self, p: float, with_niqe: bool = False) -> str:
        return reason(p, with_niqe)

    def fit_stack(self, niqe, threshold_niqe: float, log=print) -> bool:
        """The NIQE stack from the NIQE score of each sample (in the order of self.rels; NaN for none):
        the probe's out-of-fold logit and NIQE above NIQE_FROM, z-scored. False (the stack stays off)
        with threshold_niqe 0, or when no sample has NIQE above NIQE_FROM to learn the NIQE side from."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        if threshold_niqe <= 0:
            return False
        s = np.asarray(niqe, float)
        ok = np.isfinite(s)
        y, X = self.labels[ok], self.samples[ok]
        h = np.maximum(0.0, s[ok] - NIQE_FROM)
        if min(int((y == 0).sum()), int((y == 1).sum())) < FOLDS:
            log("  quality filter: NIQE side off (too few sample images with a NIQE score)")
            return False
        if h.std() == 0:
            log(f"  quality filter: NIQE side off (no sample image with NIQE above {NIQE_FROM:g})")
            return False
        lp = np.zeros(len(y))
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(X, y):
            m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X[tr], y[tr])
            lp[te] = m.decision_function(X[te])
        mu, sd = float(h.mean()), float(h.std())
        st = LogisticRegression(C=STACK_C, max_iter=5000, class_weight="balanced").fit(
            np.c_[lp, (h - mu) / sd], y)
        self.stack = (float(st.coef_[0, 0]), float(st.coef_[0, 1]), float(st.intercept_[0]), mu, sd)
        self.threshold_niqe = float(threshold_niqe)
        log(f"  quality filter: + NIQE above {NIQE_FROM:g} ({int(ok.sum()):,} sample images scored, "
            f"{int((h > 0).sum())} above); P(bad) >= {threshold_niqe:g} with NIQE is skipped")
        return True

    def match(self, X) -> tuple[np.ndarray, np.ndarray]:
        """For each embedding, the label of the sample it is a copy of (0 good, 1 bad, NONE) and the
        cosine to that sample (to the nearest one when NONE)."""
        X = _unit(X).reshape(-1, self.model.coef_.shape[1])
        if not len(X) or not len(self.samples):
            return np.full(len(X), NONE), np.zeros(len(X))
        sims = X @ self.samples.T
        near = sims.argmax(axis=1)
        cos = sims[np.arange(len(X)), near]
        return np.where(cos >= MATCH_COS, self.labels[near], NONE), cos.astype(float)


def probe_from_samples(items: dict[int, list], vectors: dict, threshold: float = THRESHOLD, log=print,
                       min_each: int = MIN_EACH) -> Probe | None:
    """Train the probe on the two folders. vectors: rel -> embedding. threshold 0: off."""
    if threshold <= 0:
        log("  quality filter off (quality_threshold 0)")
        return None
    X, y, rels = [], [], []
    for label, its in items.items():
        for it in its:
            v = vectors.get(it.rel)
            if v is not None:
                X.append(np.asarray(v, np.float32))
                y.append(label)
                rels.append(it.rel)
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
    return Probe(model, threshold, n_good, n_bad, X, np.array(y), rels)


def sample_niqe(samples: Path, rels: list[str], scorer, threads: int = 8, log=print) -> np.ndarray:
    """NIQE of each sample image (NaN for a picture without one), cached in <samples>/_embeddings/niqe.json
    by path, size and modification time under niqe.VERSION: only new or changed images are scored."""
    import json
    from concurrent.futures import ThreadPoolExecutor

    import niqe
    import process as pr
    root = Path(samples)
    path = root / "_embeddings" / NIQE_CACHE
    cache = {}
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("version") == niqe.VERSION:
            cache = meta.get("entries", {})
    except (OSError, ValueError):
        pass
    out = np.full(len(rels), np.nan)
    keys, todo = {}, []
    for i, rel in enumerate(rels):
        st = (root / rel).stat()
        keys[rel] = [st.st_size, st.st_mtime_ns]
        e = cache.get(rel)
        if e and list(e[:2]) == keys[rel]:
            out[i] = np.nan if e[2] is None else e[2]
        else:
            todo.append(i)
    if todo:
        def load(i):
            data = (root / rels[i]).read_bytes()
            return pr.decode(data, pr.head_of(data))

        with ThreadPoolExecutor(threads) as pool:
            for i, a in zip(todo, pool.map(load, todo)):
                r = scorer(a)
                out[i] = np.nan if r.score is None else r.score
        log(f"  quality filter: NIQE of {len(todo):,} sample images")
    entries = {rel: keys[rel] + [None if np.isnan(out[i]) else float(out[i])] for i, rel in enumerate(rels)}
    if todo or entries != cache:
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": niqe.VERSION, "entries": entries}), encoding="utf-8")
        tmp.replace(path)
    return out
