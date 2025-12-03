# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, List
import numpy as np


def _circ_mean_deg(ang_deg: np.ndarray, w: np.ndarray | None = None) -> float:
    ang = np.deg2rad(np.asarray(ang_deg, dtype=float))
    if w is None:
        w = np.ones_like(ang)
    else:
        w = np.asarray(w, dtype=float)
    sx = float((np.cos(ang) * w).sum())
    sy = float((np.sin(ang) * w).sum())
    return float((np.rad2deg(np.arctan2(sy, sx)) + 360.0) % 360.0)


def _fwhm_deg(hist: np.ndarray, centers: np.ndarray, peak_idx: int) -> float:
    if hist.size == 0:
        return 0.0
    n = hist.size
    half = 0.5 * float(hist[peak_idx])
    # expand left
    i = peak_idx
    while True:
        j = (i - 1) % n
        if hist[j] >= half and j != peak_idx:
            i = j
        else:
            break
    left = i
    # expand right
    i = peak_idx
    while True:
        j = (i + 1) % n
        if hist[j] >= half and j != peak_idx:
            i = j
        else:
            break
    right = i
    # span in bins
    span_bins = (right - left) % n
    bin_deg = float(centers[1] - centers[0]) if n > 1 else 360.0
    return float(span_bins) * bin_deg


def fit_clustering(angles_deg: np.ndarray, weights: np.ndarray | None, method: str = 'hist', eps_deg: float = 8.0, min_samples: int = 20, k: int = 2, smooth_deg: float = 5.0) -> Tuple[np.ndarray, List[float], List[float], int | None]:
    """
    MVP sin dependencias externas. method='hist' agrupa picos sobre umbral.
    Retorna: labels[N], centers_deg[list], widths_deg[list], dominant_idx
    """
    ang = np.asarray(angles_deg, dtype=float) % 360.0
    if ang.size == 0:
        return np.asarray([], dtype=int), [], [], None
    w = np.ones_like(ang) if weights is None else np.asarray(weights, dtype=float)
    # Histograma de 72 bins (~5°)
    nbins = max(36, int(round(360.0 / max(1.0, smooth_deg))))
    hist, edges = np.histogram(ang, bins=np.linspace(0, 360, nbins+1), weights=w)
    centers = (edges[:-1] + edges[1:]) * 0.5
    if hist.max() <= 0:
        return np.zeros_like(ang, dtype=int), [], [], None
    thr = 0.25 * float(hist.max())
    # detectar picos simples
    peaks: List[int] = []
    for i in range(hist.size):
        if hist[i] >= thr and hist[i] >= hist[i-1] and hist[i] >= hist[(i+1) % hist.size]:
            peaks.append(i)
    if not peaks:
        peaks = [int(hist.argmax())]
    centers_deg = [float(centers[i]) for i in peaks]
    widths_deg = [_fwhm_deg(hist, centers, i) for i in peaks]
    # dominio: índice por mayor peso
    peak_weights = [float(hist[i]) for i in peaks]
    dominant_idx = int(np.argmax(peak_weights)) if peaks else None
    # Etiquetado por cercanía angular al centro más cercano
    labels = np.zeros(ang.shape[0], dtype=int)
    for idx in range(ang.shape[0]):
        diffs = [abs(((ang[idx] - c + 180.0) % 360.0) - 180.0) for c in centers_deg]
        labels[idx] = int(np.argmin(diffs))
    return labels, centers_deg, widths_deg, dominant_idx
