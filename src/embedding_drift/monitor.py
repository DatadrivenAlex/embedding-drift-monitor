"""Compare normalized embeddings against a fixed baseline."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Report:
    reference_count: int
    current_count: int
    dimensions: int
    reference_mean_similarity: float
    current_mean_similarity: float
    js_divergence: float
    threshold: float
    alert: bool

    def to_dict(self):
        return asdict(self)


def _normalize(vectors):
    try:
        data = np.asarray(vectors, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("vectors must be a rectangular numeric matrix") from exc
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1:
        raise ValueError("each batch needs at least two non-empty vectors")
    if not np.isfinite(data).all():
        raise ValueError("vectors must contain only finite numbers")
    # Stable norm avoids overflow for large but finite coordinates.
    scale = np.max(np.abs(data), axis=1)
    if np.any(scale == 0):
        raise ValueError("zero vectors cannot be normalized")
    scaled = data / scale[:, None]
    return scaled / np.linalg.norm(scaled, axis=1)[:, None]


def compare(reference, current, *, threshold=0.1, bins=10):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between 0 and 1")
    if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 100:
        raise ValueError("bins must be an integer from 2 to 100")
    base, sample = _normalize(reference), _normalize(current)
    if base.shape[1] != sample.shape[1]:
        raise ValueError("embedding dimensions must match")
    center = base.mean(axis=0)
    length = np.linalg.norm(center)
    if length < 1e-12:
        raise ValueError("reference centroid has no stable direction")
    center = center / length
    ref_sim = np.clip(base @ center, -1, 1)
    cur_sim = np.clip(sample @ center, -1, 1)
    edges = np.linspace(-1, 1, bins + 1)
    ref_hist = np.histogram(ref_sim, bins=edges)[0].astype(float)
    cur_hist = np.histogram(cur_sim, bins=edges)[0].astype(float)
    # Jensen-Shannon divergence, log base 2; zero-probability bins contribute zero.
    p, q = ref_hist / ref_hist.sum(), cur_hist / cur_hist.sum()
    mid = (p + q) / 2
    def kl(a):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / mid[mask])))
    divergence = max(0.0, min(1.0, (kl(p) + kl(q)) / 2))
    return Report(len(base), len(sample), base.shape[1], float(ref_sim.mean()),
                  float(cur_sim.mean()), divergence, float(threshold), divergence > threshold)
