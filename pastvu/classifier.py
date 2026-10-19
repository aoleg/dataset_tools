"""classify's sorter, one image at a time: the samples folder is embedded
(classify's cache), the probe is trained once at start, and each processed
image gets classify's decision: the probe's class, the share of its 10 nearest
examples, the confidence gate, and the byte-identical example rule.

The output folder mirrors a classify output: <out>/<class>/,
<out>/_unsure/<predicted class>/, <out>/_classify/plan.csv and the embedding
cache <out>/_embeddings, so classify's own --extract and re-sorting work on it.
"""

from __future__ import annotations

import csv
import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import process  # noqa: F401 - puts the tool folders on sys.path
import classify as cl

PLAN_HEADER = ["source", "destination", "class", "subclass", "confidence", "share", "vote", "isolation", "reason",
               "pass", "sidecars"]


@dataclass
class Verdict:
    folder: str          # the output folder relative to <out>: class, or _unsure/<class>
    cls: str
    sub: str
    confidence: float
    share: float
    vote: str
    reason: str


class Classifier:
    def __init__(self, samples: Path, patches: int, min_confidence: float, encoder=None, log=print):
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=cl.BLAS_THREADS if hasattr(cl, "BLAS_THREADS") else 8)
        self.patches = patches
        self.min_confidence = min_confidence
        self.encoder = encoder or cl.Encoder(patches=patches)
        classes = cl.discover_samples(samples, cl.sidecar_set(cl.DEFAULT_SIDECARS))
        cl.check_samples(classes)
        args = SimpleNamespace(patches=patches, threads=8, batch=64, reembed=False)
        res = cl.embed_samples(samples, classes, self.encoder, args)
        rep = cl.Report()
        self.ts, self.cv, self.probe, self.example_hashes = cl.fit_probe(classes, res, rep)
        self.report_lines = rep.lines
        self.n_classes = len(self.ts.class_names)

    @property
    def class_names(self) -> list[str]:
        return list(self.ts.class_names)

    def embed(self, array: np.ndarray) -> np.ndarray:
        import process as pr
        return self.encoder.embed_arrays([pr.naflex_resize(array, self.patches)])[0]

    def decide(self, vec: np.ndarray) -> Verdict:
        X = vec[None].astype(np.float32)
        p_sub = self.probe.proba_sub(X)
        p_cls = cl.class_proba(p_sub, self.ts.sub_to_class, self.n_classes)
        ci = int(p_cls[0].argmax())
        si = int(p_sub[0].argmax())
        conf = float(p_cls[0, ci])
        shares = cl.knn_shares(self.ts.X, self.ts.y_class, X, self.n_classes)
        vote = self.ts.class_names[int(shares[0].argmax())]
        share = float(shares[0, ci])
        name = self.ts.class_names[ci]
        reason = ""
        if share < cl.VOTE_MIN_SHARE:
            reason = cl.REASON_DISAGREE
        elif conf < self.min_confidence:
            reason = cl.REASON_CONFIDENCE
        folder = f"{cl.UNSURE_DIRNAME}/{name}" if reason else name
        return Verdict(folder, name, self.ts.sub_names[si], conf, share, vote, reason)

    def example_override(self, data: bytes, verdict: Verdict) -> Verdict:
        """A byte-identical copy of a sample takes the sample's class, as in classify."""
        h = hashlib.sha256(data).hexdigest()
        ci = self.example_hashes.get(h)
        if ci is None:
            return verdict
        name = self.ts.class_names[ci]
        return Verdict(name, name, name, 1.0, verdict.share, verdict.vote, cl.REASON_EXAMPLE)


class OutputIndex:
    """The embedding cache and plan.csv of the output folder, written as the
    stream goes. Thread-safe."""

    FLUSH_EVERY = 200

    def __init__(self, out: Path, model_id: str, patches: int):
        self.out = out
        self.cache = cl.EmbeddingCache(out, model_id, patches)
        self.plan = out / cl.RUN_DIRNAME / "plan.csv"
        self.lock = threading.Lock()
        self.pending = 0

    def add(self, rel: str, path: Path, vec: np.ndarray, source: str, verdict: Verdict | None) -> None:
        st = path.stat()
        item = SimpleNamespace(rel=rel, size=st.st_size, mtime_ns=st.st_mtime_ns)
        with self.lock:
            self.cache.put([item], vec[None].astype(np.float16))
            self.plan.parent.mkdir(parents=True, exist_ok=True)
            new = not self.plan.is_file()
            with open(self.plan, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(PLAN_HEADER)
                if verdict:
                    w.writerow([source, str(path), verdict.cls, verdict.sub, f"{verdict.confidence:.4f}",
                                f"{verdict.share:.2f}", verdict.vote, "0.0000", verdict.reason, 1,
                                os.path.splitext(rel)[0] + ".txt"])
                else:
                    w.writerow([source, str(path), "", "", "", "", "", "", "", 1, os.path.splitext(rel)[0] + ".txt"])
            self.pending += 1
            if self.pending >= self.FLUSH_EVERY:
                self.cache.save()
                self.pending = 0

    def flush(self) -> None:
        with self.lock:
            if self.pending:
                self.cache.save()
                self.pending = 0
