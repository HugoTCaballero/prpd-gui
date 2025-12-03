# phase_align.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np

def _wrap_deg(phi):
    phi = np.mod(phi, 360.0)
    phi[phi < 0] += 360.0
    return phi

def _phase_energy(phi_deg: np.ndarray, q: np.ndarray, bins=72) -> float:
    """Energía por histograma de fase (proxy de alineación)."""
    h, _ = np.histogram(_wrap_deg(phi_deg), bins=bins, range=(0.0, 360.0), weights=np.abs(q))
    # concentración = energía en los k bins más altos vs total
    k = max(3, bins // 24)  # ~45° en total
    topk = np.sort(h)[-k:].sum()
    tot = h.sum() + 1e-12
    return float(topk / tot)

def coarse_align(phi_deg: np.ndarray, q: np.ndarray, candidates=(0.0, 120.0, 240.0)) -> tuple[np.ndarray, float]:
    """Devuelve (phi_aligned, best_shift)."""
    best_s, best_e = 0.0, -1.0
    for s in candidates:
        e = _phase_energy(phi_deg + s, q)
        if e > best_e:
            best_e = e
            best_s = s
    return _wrap_deg(phi_deg + best_s), best_s

def fine_refine(phi_deg: np.ndarray, q: np.ndarray, around: float, span=20.0, step=2.0) -> tuple[np.ndarray, float]:
    """Refina ±span° alrededor del desplazamiento 'around'."""
    best_s, best_e = around, -1.0
    for ds in np.arange(-span, span + 1e-9, step):
        s = around + ds
        e = _phase_energy(phi_deg + s, q)
        if e > best_e:
            best_e = e
            best_s = s
    return _wrap_deg(phi_deg + best_s), best_s

def auto_align(phi_deg: np.ndarray, q: np.ndarray, refine=True) -> tuple[np.ndarray, float]:
    """Alineación automática: {0,120,240} y refinamiento fino opcional."""
    phi1, s1 = coarse_align(phi_deg, q)
    if refine:
        phi2, s2 = fine_refine(phi_deg, q, s1, span=20.0, step=2.0)
        return phi2, s2
    return phi1, s1
