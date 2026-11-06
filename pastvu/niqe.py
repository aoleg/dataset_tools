"""NIQE, the Natural Image Quality Evaluator (Mittal, Soundararajan and Bovik,
"Making a 'Completely Blind' Image Quality Analyzer", IEEE SPL 2013), on the
picture as the trainer sees it.

The picture is area-downscaled to at most PIXELS (1024² by default, never
upscaled): raster dots, grain and print texture count only where they survive
the downscale to the training resolution. Then, as published: the luma,
rounded; mean-subtracted contrast-normalised coefficients (MSCN, 7x7 Gaussian,
sigma 7/6, replicate border); 96 px blocks without overlap, the right and the
bottom remainder dropped. Per block an asymmetric generalised Gaussian fit of
the coefficients and of their products with the four neighbours (18 numbers),
and the same again on the half-size image (MATLAB bicubic with antialiasing,
48 px blocks): 36 features per block.

What comes out per photo (Result):
- score: the published NIQE, the distance of the photo's feature Gaussian to
  the Gaussian of the pristine photo collection of the LIVE release. Higher
  is worse.
- feat: the mean and the standard deviation (ddof 0) of the 36 features over
  the blocks with no undefined feature, 72 float32. The quality rule may use
  these instead of the score; they are what it learns from the samples.
Both are None for a picture smaller than two blocks a side.

The picture goes to the device as it is (a view of a bigger array too) and is
downscaled there, equal to Pillow's Image.resize(..., BOX) value for value:
running sums over Pillow's pixel runs, rounded half up as its fixed point does.
All blocks of a picture are then fitted at once. The image steps run in
float64: in float32 the rounding noise of a flat area (a blown-out sky) makes
blocks that are undefined in the reference look fitted, and the score of such
a photo moved by up to 9. The half-size resize is a strided 8-tap convolution
(the same as MATLAB's matrix, to 1e-13). The block fits run in float32. The
36x36 algebra of the score runs on the CPU in float64. About 5.5 ms per photo
on an RTX 5090, 3 ms of it on the GPU, the rest kernel launches.

The niqe command (run() at the end) scores the saved photos that have no row of
VERSION in the store: threads read and decode the files, the calling thread
scores them one at a time, and the rows go in BATCH per transaction.

The algorithm follows BasicSR's numpy NIQE (basicsr/metrics/niqe.py,
https://github.com/XPixelGroup/BasicSR, Apache License 2.0), which reproduces
the official MATLAB release to four decimals. niqe_pris_params.npz is BasicSR's
file: the pristine model and the Gaussian window of the LIVE MATLAB release
(http://live.ece.utexas.edu/research/quality/niqe_release.zip), whose values
are those of its modelparameters.mat. Copyright (c) 2011 The University of
Texas at Austin; the release's notice, which must go with every copy, and
BasicSR's licence are in NIQE_NOTICE.txt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARAMS = HERE / "niqe_pris_params.npz"
PIXELS = 1024 * 1024
BLOCK = 96
N_FEAT = 36
VERSION = "niqe1-1024"          # implementation, parameters and pixel budget; a change makes stored rows stale
LUMA = {"yiq": (0.299, 0.587, 0.114, 0.0),                       # pyiqa's default, the research numbers
        "ycbcr": (65.481 / 255, 128.553 / 255, 24.966 / 255, 16.0)}   # BasicSR's and MATLAB's Y


@dataclass
class Result:
    score: float | None
    feat: np.ndarray | None          # 2 * N_FEAT float32: means, then standard deviations


def target_size(w: int, h: int, pixels: int = PIXELS) -> tuple[int, int]:
    """(w, h) area-downscaled to at most pixels, never upscaled."""
    if w * h <= pixels:
        return w, h
    k = math.sqrt(pixels / (w * h))
    return max(1, round(w * k)), max(1, round(h * k))


def downscale(rgb: np.ndarray, pixels: int = PIXELS) -> np.ndarray:
    """An HxWx3 uint8 array downscaled with Pillow's BOX filter to at most pixels, never upscaled.
    Niqe does the same on the GPU; this is the reference."""
    h, w = rgb.shape[:2]
    size = target_size(w, h, pixels)
    if size == (w, h):
        return rgb
    from PIL import Image
    return np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize(size, Image.BOX))


def _box_ranges(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Pillow's BOX filter along one axis, with Pillow's own double arithmetic (precompute_coeffs
    in Resample.c): output pixel i is the mean of the input pixels j in [xmin, xmax) whose
    ((j - centre) + 0.5) * (1 / scale) lies in (-0.5, 0.5], centre (i + 0.5) * scale. Always a run of
    pixels: (start, end), end exclusive."""
    scale = n_in / n_out
    ss = 1.0 / scale
    centre = (np.arange(n_out) + 0.5) * scale
    support = 0.5 * scale
    xmin = np.maximum((centre - support + 0.5).astype(np.int64), 0)
    xmax = np.minimum((centre + support + 0.5).astype(np.int64), n_in)
    j = xmin[:, None] + np.arange(int((xmax - xmin).max()))[None, :]
    x = ((j - centre[:, None]) + 0.5) * ss
    inside = (j < xmax[:, None]) & (x > -0.5) & (x <= 0.5)
    start = j[np.arange(n_out), inside.argmax(axis=1)]
    end = j[np.arange(n_out), inside.shape[1] - 1 - inside[:, ::-1].argmax(axis=1)] + 1
    return start, end


def _resize_half(n_in: int) -> np.ndarray:
    """The (ceil(n_in/2), n_in) matrix of MATLAB's imresize(x, 0.5, 'bicubic') with antialiasing
    along one axis: Keys cubic (a = -0.5) stretched to 8 taps, rows summing to 1, the border by
    half-sample symmetric reflection. For an even n_in, row i is the 8 taps of HALF_TAPS on input
    2i-3 .. 2i+4 with the border folded back, which Niqe computes as a strided convolution."""
    scale = 0.5
    n_out = math.ceil(n_in * scale)
    width = 4 / scale
    x = np.arange(1, n_out + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - width / 2)
    p = math.ceil(width) + 2
    idx = left[:, None] + np.arange(p)[None, :]
    d = np.abs((u[:, None] - idx) * scale)
    w = np.where(d <= 1, 1.5 * d**3 - 2.5 * d**2 + 1, np.where(d <= 2, -0.5 * d**3 + 2.5 * d**2 - 4 * d + 2, 0.0))
    w = scale * w
    w /= w.sum(axis=1, keepdims=True)
    src = idx.astype(np.int64) - 1                      # 0-based input index
    src = np.where(src < 0, -src - 1, src)              # half-sample symmetric: -1 -> 0, -2 -> 1
    src = np.where(src >= n_in, 2 * n_in - src - 1, src)
    m = np.zeros((n_out, n_in))
    np.add.at(m, (np.repeat(np.arange(n_out), p), src.ravel()), w.ravel())
    return m


class Niqe:
    """NIQE on one device. device None: CUDA when available, else the CPU."""

    def __init__(self, device=None, luma: str = "yiq", pixels: int = PIXELS, dtype=None, fit_dtype=None):
        import torch
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or torch.float64            # the image steps
        self.fit_dtype = fit_dtype or torch.float32    # the block fits
        self.pixels = pixels
        self.luma = LUMA[luma]
        prm = np.load(PARAMS)
        self.mu_p = prm["mu_pris_param"].ravel().astype(np.float64)
        self.cov_p = prm["cov_pris_param"].astype(np.float64)
        self.window = torch.as_tensor(prm["gaussian_window"], dtype=self.dtype, device=self.device)[None, None]
        gam = np.arange(0.2, 10.001, 0.001)
        lg = torch.lgamma
        g = torch.as_tensor(gam, dtype=torch.float64)
        self.gam = g.to(self.device, self.fit_dtype)
        # r(gamma) rises strictly over the grid, so the nearest entry is a binary search
        self.r_gam = (2 * lg(2 / g) - lg(1 / g) - lg(3 / g)).exp().to(self.device, self.fit_dtype).contiguous()
        self._boxes = {}
        taps = _resize_half(96)[10, 17:25]                         # an interior row: input 17 .. 24
        self.half_taps = torch.as_tensor(taps, dtype=self.dtype, device=self.device).view(1, 1, 8)

    # --- the steps -----------------------------------------------------------------------

    def _upload(self, rgb: np.ndarray):
        """The uint8 array on the device as it is (a view of a bigger array is fine; nothing writes to it)."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)          # a read-only array: never written
            return self.torch.from_numpy(rgb).to(self.device)

    def _box_axis(self, x, dim: int, n_out: int):
        """Pillow's BOX downscale of an int32 (H, W, 3) tensor along dim, rounded half up as Pillow's
        fixed point does, so the result equals Image.resize(..., BOX) of the uint8 array."""
        torch = self.torch
        n_in = x.shape[dim]
        key = (n_in, n_out)
        if key not in self._boxes:
            st, en = _box_ranges(n_in, n_out)
            self._boxes[key] = tuple(torch.as_tensor(v, device=self.device) for v in (st, en, en - st))
        st, en, cnt = self._boxes[key]
        shape = [1, 1, 1]
        shape[dim] = n_in
        c = torch.cat([torch.zeros_like(x.narrow(dim, 0, 1)), x.cumsum(dim, dtype=torch.int32)], dim)
        s = c.index_select(dim, en) - c.index_select(dim, st)
        shape[dim] = n_out
        cnt = cnt.view(shape).to(torch.int32)
        return (2 * s + cnt).div(2 * cnt, rounding_mode="floor")

    def _picture(self, rgb: np.ndarray):
        """The picture on the device after the downscale, int32 (h, w, 3)."""
        t = self._upload(rgb).to(self.torch.int32)
        if t.ndim == 2:                                           # grey: the same value in R, G and B
            t = t[..., None].expand(-1, -1, 3)
        t = t[..., :3]                                            # RGBA: the alpha left out
        h, w = t.shape[:2]
        w2, h2 = target_size(w, h, self.pixels)
        if (w2, h2) != (w, h):
            t = self._box_axis(t, 1, w2)                          # Pillow: the horizontal pass first
            t = self._box_axis(t, 0, h2)
        return t

    def _gray(self, t):
        a, b, c, off = self.luma
        x = t.to(self.dtype)
        y = x[..., 0] * a + x[..., 1] * b + x[..., 2] * c + off
        return y.round()

    def _mscn(self, img):
        f = self.torch.nn.functional
        x = f.pad(img[None, None], (3, 3, 3, 3), mode="replicate")
        mu = f.conv2d(x, self.window)
        sq = f.conv2d(x * x, self.window)
        sigma = (sq - mu * mu).abs().sqrt()
        return ((img[None, None] - mu) / (sigma + 1))[0, 0]

    def _half_axis(self, x):
        """MATLAB's bicubic half size along the last axis of (n, L), L even: the border folded by
        half-sample symmetry (3 pixels a side), then the 8 taps at stride 2."""
        torch = self.torch
        n, length = x.shape
        xp = torch.cat([x[:, :3].flip(1), x, x[:, -3:].flip(1)], 1)
        return torch.nn.functional.conv1d(xp[:, None], self.half_taps, stride=2)[:, 0]

    def _half_size(self, img):
        """img (H, W), both even -> (H/2, W/2): rows first, then columns, as the matrix product
        M_h @ img @ M_w.T of _resize_half."""
        x = self._half_axis((img / 255.0).T.contiguous()).T          # along H
        return self._half_axis(x.contiguous()) * 255.0                # along W

    def _aggd(self, x):
        """Asymmetric generalised Gaussian fit of each row: (alpha, beta_l, beta_r). A row with no
        negative or no positive value gives NaN betas and alpha 0.2, as the reference argmin does."""
        torch = self.torch
        neg, pos = x < 0, x > 0
        x2 = x * x
        ls = ((x2 * neg).sum(1) / neg.sum(1)).sqrt()
        rs = ((x2 * pos).sum(1) / pos.sum(1)).sqrt()
        g = ls / rs
        rhat = x.abs().mean(1) ** 2 / x2.mean(1)
        rn = rhat * (g ** 3 + 1) * (g + 1) / (g ** 2 + 1) ** 2
        r = self.r_gam
        nan = torch.isnan(rn)
        rn0 = torch.where(nan, r[0], rn)
        i = torch.searchsorted(r, rn0.contiguous()).clamp(1, len(r) - 1)
        lower = (rn0 - r[i - 1]).abs() <= (r[i] - rn0).abs()          # a tie goes to the lower index, as argmin
        i = torch.where(lower, i - 1, i)
        i = torch.where(nan, torch.zeros_like(i), i)
        alpha = self.gam[i]
        k = (0.5 * (torch.lgamma(1 / alpha) - torch.lgamma(3 / alpha))).exp()
        return alpha, ls * k, rs * k

    def _block_features(self, img, bs: int):
        """img (H, W), H and W multiples of bs -> (blocks, 18)."""
        torch = self.torch
        h, w = img.shape
        blocks = img.reshape(h // bs, bs, w // bs, bs).permute(0, 2, 1, 3).reshape(-1, bs, bs).to(self.fit_dtype)
        n = blocks.shape[0]
        # the block and its products with the four neighbours (wrapping inside the block), one fit
        sets = [blocks] + [blocks * torch.roll(blocks, shift, dims=(1, 2))
                           for shift in ((0, 1), (1, 0), (1, 1), (1, -1))]
        alpha, bl, br = (v.reshape(5, n) for v in self._aggd(torch.cat(sets).reshape(5 * n, -1)))
        mean = (br - bl) * (torch.lgamma(2 / alpha) - torch.lgamma(1 / alpha)).exp()
        feats = [alpha[0], (bl[0] + br[0]) / 2]
        for k in range(1, 5):
            feats += [alpha[k], mean[k], bl[k], br[k]]
        return torch.stack(feats, 1)

    # --- the photo -------------------------------------------------------------------------

    def block_features(self, rgb: np.ndarray) -> np.ndarray | None:
        """(blocks, 36) float64 of an HxWx3 uint8 picture after the downscale, or None when it is
        smaller than two blocks a side."""
        w2, h2 = target_size(rgb.shape[1], rgb.shape[0], self.pixels)
        nh, nw = h2 // BLOCK, w2 // BLOCK
        if nh < 2 or nw < 2:
            return None
        with self.torch.inference_mode():
            img = self._gray(self._picture(rgb)[:nh * BLOCK, :nw * BLOCK])
            f1 = self._block_features(self._mscn(img), BLOCK)
            half = self._half_size(img)
            f2 = self._block_features(self._mscn(half), BLOCK // 2)
            return self.torch.cat([f1, f2], 1).double().cpu().numpy()

    def __call__(self, rgb: np.ndarray) -> Result:
        F = self.block_features(rgb)
        if F is None:
            return Result(None, None)
        ok = np.isfinite(F).all(axis=1)
        if ok.sum() < 2:
            return Result(None, None)
        Fo = F[ok]
        feat = np.concatenate([Fo.mean(0), Fo.std(0)]).astype(np.float32)
        mu = np.nanmean(F, axis=0)
        cov = np.cov(Fo, rowvar=False)
        d = self.mu_p - mu
        q = float(d @ np.linalg.pinv((self.cov_p + cov) / 2) @ d)
        return Result(math.sqrt(max(q, 0.0)), feat)


# --- the niqe command: the saved photos without a row --------------------------------------

THREADS = 12          # decode threads: a 2.5 MP JPEG takes about 40 ms of CPU, the GPU about 5 ms
AHEAD = 4             # decoded photos per thread waiting for the GPU
BATCH = 64            # rows per transaction
FOLLOW_EVERY = 60     # --follow: seconds between two looks for new photos


def read_photo(store, base: Path, cid: int, path: str) -> tuple:
    """(cid, rel, upright RGB array or None, why None). The path is asked from the store at read time
    (a reclassify may have moved the photo); a photo no longer saved (forgotten, in a review) is
    "gone". The stream's decoder, so the pixels are the ones the stream scores."""
    import process as pr
    from store import DONE
    st = store.status(cid)
    if not st or st.get("state") != DONE or not st.get("path"):
        return cid, path, None, "gone"
    rel = st["path"]
    try:
        data = (base / rel).read_bytes()
    except OSError:
        return cid, rel, None, "missing"
    try:
        return cid, rel, pr.decode(data, pr.head_of(data)), None
    except Exception:  # noqa: BLE001 - a damaged file is counted, not fatal
        return cid, rel, None, "unreadable"


def save_rows(store, rows: list[tuple], tries: int = 10) -> None:
    """save_niqe, waiting out a writer that holds the database longer than the busy timeout."""
    import sqlite3
    import time
    for attempt in range(tries):
        try:
            store.save_niqe(rows)
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or attempt == tries - 1:
                raise
            time.sleep(2)


def report(stats: dict, total: int, seconds: float, vram=None, final: bool = False) -> str:
    done = stats["scored"] + stats["no_niqe"]
    rate = done / seconds if seconds > 0 else 0.0
    left = max(0, total - done - stats["missing"] - stats["unreadable"] - stats["gone"] - stats["failed"])
    eta = f", about {left / rate / 60:,.0f} min left" if rate > 0 and left and not final else ""
    text = (f"{'Finished: ' if final else ''}{done:,} of {total:,} photos scored in {seconds / 60:.1f} min "
            f"({rate:.1f}/s{eta}); {stats['no_niqe']:,} have no NIQE (too small or flat)")
    rest = [f"{stats[k]:,} {k}" for k in ("missing", "unreadable", "gone", "failed") if stats[k]]
    if rest:
        text += "; " + ", ".join(rest)
    if vram is not None and getattr(vram, "ok", False):
        text += f"\n  {vram.text()}"
    return text


def run(store, base: Path, targets: list[tuple[int, str]], scorer, display, report_every: int = 1000,
        threads: int = THREADS, follow=None, vram=None, stop=None) -> dict:
    """Score the targets ((cid, path), in order) and write their rows, BATCH per transaction.
    The threads read and decode at most threads * AHEAD photos ahead; the calling thread scores
    them one at a time. follow: a function giving the targets to add when the list is done, asked
    every FOLLOW_EVERY s until Ctrl+C (None: stop when the list is done); a photo tried once in this
    run is not tried again. A photo missing, unreadable, gone or failed gets no row, so the next run
    lists it again; a picture without NIQE gets a row with no score, so it is not listed again.
    -> the counts, and "stopped" ("Ctrl+C" or None)."""
    import time
    from collections import Counter, deque
    from concurrent.futures import ThreadPoolExecutor
    stats = Counter()
    rows: list[tuple] = []
    tried = {cid for cid, _ in targets}
    queue_ = deque(targets)
    total = len(targets)
    t0 = time.time()
    stopped = None
    window = deque()

    def flush():
        if rows:
            save_rows(store, list(rows))
            rows.clear()

    def note_progress():
        done = sum(stats[k] for k in ("scored", "no_niqe", "missing", "unreadable", "gone", "failed"))
        if report_every and done % report_every == 0:
            display.print(report(stats, total, time.time() - t0, vram))

    pool = ThreadPoolExecutor(max(1, threads))
    try:
        while True:
            while queue_ and len(window) < max(1, threads) * AHEAD:
                window.append(pool.submit(read_photo, store, base, *queue_.popleft()))
            if not window:
                flush()
                if follow is None:
                    break
                display.set(f"{stats['scored'] + stats['no_niqe']:,} scored; waiting for new photos")
                deadline = time.time() + FOLLOW_EVERY
                while time.time() < deadline and not (stop is not None and stop.is_set()):
                    time.sleep(0.5)
                if stop is not None and stop.is_set():
                    stopped = "Ctrl+C"
                    break
                new = [(c, p) for c, p in follow() if c not in tried]
                tried.update(c for c, _ in new)
                queue_.extend(new)
                total += len(new)
                if new:
                    display.add_total(len(new))
                continue
            cid, rel, array, why = window.popleft().result()
            if array is None:
                stats[why] += 1
                display.line({"cid": cid, "state": why, "path": rel})
            else:
                display.set(f"{cid}: {rel}")
                try:
                    r = scorer(array)
                except Exception as e:  # noqa: BLE001 - one photo's error is counted; the next run tries again
                    stats["failed"] += 1
                    if stats["failed"] <= 3:
                        display.print(f"[niqe] {cid}: {type(e).__name__}: {e}")
                else:
                    rows.append((cid, VERSION, r.score, r.feat))
                    stats["scored" if r.score is not None else "no_niqe"] += 1
                    display.line({"cid": cid, "state": "scored", "niqe": None if r.score is None else round(r.score, 4)})
                if len(rows) >= BATCH:
                    flush()
            display.advance(saved=array is not None)
            note_progress()
            if stop is not None and stop.is_set():
                stopped = "Ctrl+C"
                break
    except KeyboardInterrupt:
        stopped = "Ctrl+C"
    finally:
        for f in window:
            f.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        flush()
    display.print(report(stats, total, time.time() - t0, vram, final=True))
    if stopped:
        display.print(f"Stopped: {stopped}. Run the same command again to continue.")
    out = dict(stats)
    out.update(stopped=stopped, total=total)
    return out
