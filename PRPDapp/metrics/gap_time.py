# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np


def circular_hist(angles_deg: np.ndarray, weights: np.ndarray | None = None, nbins: int = 360) -> Tuple[np.ndarray, np.ndarray]:
    ang = np.asarray(angles_deg, dtype=float) % 360.0
    w = np.ones_like(ang) if weights is None else np.asarray(weights, dtype=float)
    bins = np.linspace(0.0, 360.0, nbins + 1)
    hist, edges = np.histogram(ang, bins=bins, weights=w)
    centers = (edges[:-1] + edges[1:]) * 0.5
    return hist.astype(float), centers


def _largest_low_activity_gap(hist: np.ndarray, thr: float) -> float:
    # Busca el tramo continuo más largo con hist < thr (considerando circularidad)
    n = int(hist.size)
    if n == 0:
        return 0.0
    mask = (hist < thr).astype(int)
    m2 = np.concatenate([mask, mask])
    best = 0; cur = 0
    for val in m2:
        if val == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(min(best, n))  # en bins


def _gap_ms_from_angles(angles_deg: np.ndarray, weights: np.ndarray | None, mains_hz: float, nbins: int) -> float | None:
    if angles_deg.size == 0:
        return None
    hist, _ = circular_hist(angles_deg, weights=None if weights is None else weights, nbins=nbins)
    if hist.max() <= 0:
        return None
    thr = max(1.0, 0.2 * float(hist.max()))  # 20% del máximo como umbral bajo
    gap_bins = _largest_low_activity_gap(hist, thr)
    gap_deg = gap_bins * (360.0 / nbins)
    T_ms = 1000.0 / float(mains_hz)
    val = float((gap_deg / 360.0) * T_ms)
    # clamp 0..16.6667
    if not np.isfinite(val):
        return None
    if val < 0.0:
        val = 0.0
    if val > 16.6668:
        val = 16.6668
    return float(val)


def _bootstrap_gap_ms(angles_deg: np.ndarray, weights: np.ndarray | None, mains_hz: float, nbins: int, n_boot: int) -> Tuple[float | None, float | None]:
    if angles_deg.size == 0:
        return None, None
    ang = np.asarray(angles_deg, dtype=float)
    w = None if weights is None else np.asarray(weights, dtype=float)
    N = ang.size
    # Probabilidades proporcionales a pesos si se dan; si no, uniforme
    if w is not None:
        ww = w.astype(float)
        s = ww.sum()
        if s <= 0:
            probs = np.ones(N, dtype=float) / float(N)
        else:
            probs = ww / s
    else:
        probs = np.ones(N, dtype=float) / float(N)
    samples: list[float] = []
    for _ in range(int(max(1, n_boot))):
        idx = np.random.choice(N, size=N, replace=True, p=probs)
        gm = _gap_ms_from_angles(ang[idx], None if w is None else w[idx], mains_hz, nbins)
        if gm is not None:
            samples.append(gm)
    if not samples:
        return None, None
    P5 = float(np.percentile(samples, 5))
    P50 = float(np.percentile(samples, 50))
    return P5, P50


def gap_stats_from_angles(angles_deg: np.ndarray, weights: np.ndarray | None = None, mains_hz: float = 60.0, nbins: int = 360, n_boot: int = 100) -> Dict[str, float | None]:
    ang = np.asarray(angles_deg, dtype=float) % 360.0
    w = None if weights is None else np.asarray(weights, dtype=float)

    L_mask = (ang < 180.0)
    H_mask = ~L_mask

    out: Dict[str, float | None] = {
        'gap_P5_ms_L': None, 'gap_P50_ms_L': None,
        'gap_P5_ms_H': None, 'gap_P50_ms_H': None,
    }

    def _compute(mask):
        if not np.any(mask):
            return None, None
        return _bootstrap_gap_ms(ang[mask], (None if w is None else w[mask]), mains_hz, nbins, n_boot)

    out['gap_P5_ms_L'], out['gap_P50_ms_L'] = _compute(L_mask)
    out['gap_P5_ms_H'], out['gap_P50_ms_H'] = _compute(H_mask)
    return out


