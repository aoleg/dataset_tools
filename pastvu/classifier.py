"""classify's sorter, one image at a time or in batches: the samples folder
is embedded (classify's cache in the samples folder), the probe is trained
once, and each image gets classify's decision: the probe's class, the share
of its 10 nearest examples, the confidence gate, and the byte-identical
example rule. The embeddings of the saved images go into the store, keyed by
PastVu number, so reclassify needs the GPU only for images without one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import process as pr  # puts the tool folders on sys.path
import classify as cl


class LazyEncoder:
    """classify's Encoder, loaded on the first image it embeds: reclassify
    with every embedding in the store never loads it."""

    def __init__(self, patches: int):
        self.patches = patches
        self.model_id = cl.MODEL_REPO
        self._enc = None

    @property
    def loaded(self) -> bool:
        return self._enc is not None

    def embed_arrays(self, arrays):
        if self._enc is None:
            self._enc = cl.Encoder(patches=self.patches)
        return self._enc.embed_arrays(arrays)


DEFAULT_PRIORITY_MARGIN = 0.15


def choose(p: np.ndarray, share: np.ndarray, margin: float, min_confidence: float,
           vote_min_share: float) -> tuple[int, int, float, float, str]:
    """The class of one image under the priority rule. p: the probe's class
    probabilities, share: the share of the nearest examples per class, both
    in the order of the samples folder (lowest number first).

    The near classes are those within `margin` of the top probability; the
    image goes to the lowest-numbered of them. The gates judge the near group
    together: its summed probability against min_confidence, its summed share
    of the nearest examples against vote_min_share. With margin 0 the near
    group is the top class alone, which is classify's own rule.
    -> (chosen class, top class, group probability, group share, reason)"""
    top = int(p.argmax())
    near = np.flatnonzero(p >= p[top] - margin) if margin > 0 else np.array([top])
    chosen = int(near.min())
    conf = float(p[near].sum())
    group_share = float(share[near].sum())
    reason = ""
    if group_share < vote_min_share:
        reason = cl.REASON_DISAGREE
    elif conf < min_confidence:
        reason = cl.REASON_CONFIDENCE
    return chosen, top, conf, group_share, reason


@dataclass
class Verdict:
    folder: str          # the output folder relative to the scope: class, or _unsure/<class>
    cls: str
    sub: str
    confidence: float    # the probability of the chosen class
    share: float
    vote: str
    reason: str
    top: str = ""        # the class the probe scored highest, when the priority rule chose another
    group: float = 0.0   # the summed probability of the near classes, which the gate judged

    def as_dict(self) -> dict:
        d = {"folder": self.folder, "class": self.cls, "subclass": self.sub, "confidence": round(self.confidence, 4),
             "share": round(self.share, 2), "vote": self.vote, "reason": self.reason}
        if self.top:
            d.update(priority_over=self.top, group_confidence=round(self.group, 4))
        return d


class Classifier:
    def __init__(self, samples: Path, patches: int, min_confidence: float, encoder=None,
                 priority_margin: float = DEFAULT_PRIORITY_MARGIN):
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=getattr(cl, "BLAS_THREADS", 8))
        if not samples or not Path(samples).is_dir():
            raise SystemExit(f"samples folder {samples} does not exist; set [paths] samples in the conf or give --samples")
        self.samples = Path(samples)
        self.patches = patches
        self.min_confidence = min_confidence
        self.priority_margin = max(0.0, float(priority_margin))
        self.encoder = encoder or LazyEncoder(patches)
        classes = cl.discover_samples(self.samples, cl.sidecar_set(cl.DEFAULT_SIDECARS))
        cl.check_samples(classes)
        args = SimpleNamespace(patches=patches, threads=8, batch=64, reembed=False)
        res = cl.embed_samples(self.samples, classes, self.encoder, args)
        rep = cl.Report()
        self.ts, self.cv, self.probe, self.example_hashes = cl.fit_probe(classes, res, rep)
        self.n_classes = len(self.ts.class_names)

    @property
    def model_id(self) -> str:
        return cl.MODEL_REPO

    @property
    def class_names(self) -> list[str]:
        return list(self.ts.class_names)

    def embed(self, array: np.ndarray) -> np.ndarray:
        return self.encoder.embed_arrays([pr.naflex_resize(array, self.patches)])[0]

    def embed_file(self, path: Path) -> np.ndarray:
        data = Path(path).read_bytes()
        head = pr.head_of(data)
        return self.embed(pr.decode(data, head))

    def decide_many(self, X: np.ndarray) -> list[Verdict]:
        X = np.asarray(X, np.float32)
        p_sub = self.probe.proba_sub(X)
        p_cls = cl.class_proba(p_sub, self.ts.sub_to_class, self.n_classes)
        shares = cl.knn_shares(self.ts.X, self.ts.y_class, X, self.n_classes)
        out = []
        sub_to_class = np.asarray(self.ts.sub_to_class)
        for i in range(X.shape[0]):
            ci, top, group, group_share, reason = choose(p_cls[i], shares[i], self.priority_margin,
                                                         self.min_confidence, cl.VOTE_MIN_SHARE)
            subs = np.flatnonzero(sub_to_class == ci)
            si = int(subs[p_sub[i, subs].argmax()])
            name = self.ts.class_names[ci]
            folder = f"{cl.UNSURE_DIRNAME}/{name}" if reason else name
            out.append(Verdict(folder, name, self.ts.sub_names[si], float(p_cls[i, ci]), group_share,
                               self.ts.class_names[int(shares[i].argmax())], reason,
                               top=self.ts.class_names[top] if top != ci else "", group=group))
        return out

    def decide(self, vec: np.ndarray) -> Verdict:
        return self.decide_many(vec[None])[0]

    def example_class(self, sha256: str | None) -> str | None:
        """The class of a sample that is a byte-identical copy (by sha256), or None."""
        ci = self.example_hashes.get(sha256) if sha256 else None
        return self.ts.class_names[ci] if ci is not None else None

    def example_override(self, data: bytes, verdict: Verdict) -> Verdict:
        name = self.example_class(hashlib.sha256(data).hexdigest())
        if name is None:
            return verdict
        return Verdict(name, name, name, 1.0, verdict.share, verdict.vote, cl.REASON_EXAMPLE)
