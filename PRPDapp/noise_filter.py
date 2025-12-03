# noise_filter.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np

def by_cluster(phi_deg: np.ndarray,
               q: np.ndarray,
               labels: np.ndarray,
               min_cluster_frac: float = 0.08,
               min_cluster_size: int = 40):
    """
    Mantiene clusters dominantes: tamaño >= max(min_cluster_size, min_cluster_frac*N).
    labels: -1 para ruido si viene de DBSCAN/HDBSCAN.
    """
    n = len(q)
    keep = np.zeros(n, dtype=bool)
    if labels is None or labels.size != n:
        # fallback: densidad por ventana de fase
        return by_density(phi_deg, q, win_deg=30.0, min_rel_energy=0.02)

    vals, counts = np.unique(labels, return_counts=True)
    thr = max(min_cluster_size, int(min_cluster_frac * n))
    dom_labels = set(v for v, c in zip(vals, counts) if v >= 0 and c >= thr)

    if not dom_labels:
        return by_density(phi_deg, q, win_deg=30.0, min_rel_energy=0.02)

    for dl in dom_labels:
        keep |= (labels == dl)
    return phi_deg[keep], q[keep], keep

def by_density(phi_deg: np.ndarray, q: np.ndarray,
               win_deg: float = 30.0,
               min_rel_energy: float = 0.02):
    """
    Mantiene regiones de fase con energía local suficiente.
    """
    N = len(q)
    if N == 0:
        return phi_deg, q, np.zeros(0, dtype=bool)
    bins = int(round(360.0 / max(5.0, win_deg)))
    h, edges = np.histogram((phi_deg % 360.0), bins=bins, range=(0.0, 360.0), weights=np.abs(q))
    tot = h.sum() + 1e-12
    rel = h / tot
    hot = (rel >= min_rel_energy)
    # marca puntos cuyo bin es “hot”
    idx = np.floor((phi_deg % 360.0) / (360.0 / bins)).astype(int)
    keep = hot[np.clip(idx, 0, bins - 1)]
    return phi_deg[keep], q[keep], keep