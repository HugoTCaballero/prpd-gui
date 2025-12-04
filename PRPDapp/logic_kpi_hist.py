from __future__ import annotations

import numpy as np
from typing import Dict, Any


def _count_active_bins(h: np.ndarray, rel_thr: float = 0.2) -> int:
    """
    Cuenta cuántos bins superan un umbral relativo (20 % del máximo por defecto).
    Trabaja sobre histogramas ya en log (H = log10(1+conteos)).
    """
    h = np.asarray(h, dtype=float)
    if h.size == 0:
        return 0

    h = h - np.min(h)
    max_val = float(np.max(h))
    if max_val <= 0:
        return 0

    thr = max_val * rel_thr
    return int(np.sum(h >= thr))


def _dominant_bin_index(h: np.ndarray) -> int:
    """
    Devuelve el índice (0..N-1) del bin con valor máximo.
    Si el histograma está vacío, regresa -1.
    """
    h = np.asarray(h, dtype=float)
    if h.size == 0:
        return -1
    return int(np.argmax(h))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Correlación de Pearson simple entre dos histogramas.
    Útil como indicador de simetría entre semiciclos.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0

    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def kpi_from_histograms(hist: Dict[str, Any]) -> Dict[str, float]:
    """
    Calcula los KPIs basados en histogramas H_amp/H_ph.

    Devuelve dict con:
        - hist_amp_active_bins_pos/neg
        - hist_ph_active_bins_pos/neg
        - hist_amp_peak_bin_pos/neg
        - hist_ph_peak_bin_pos/neg
        - hist_amp_pos_neg_corr
        - hist_ph_pos_neg_corr
    """
    H_amp_pos = np.asarray(hist.get("H_amp_pos", []), dtype=float)
    H_amp_neg = np.asarray(hist.get("H_amp_neg", []), dtype=float)
    H_ph_pos = np.asarray(hist.get("H_ph_pos", []), dtype=float)
    H_ph_neg = np.asarray(hist.get("H_ph_neg", []), dtype=float)

    kpi: Dict[str, float] = {}

    kpi["hist_amp_active_bins_pos"] = _count_active_bins(H_amp_pos)
    kpi["hist_amp_active_bins_neg"] = _count_active_bins(H_amp_neg)
    kpi["hist_ph_active_bins_pos"] = _count_active_bins(H_ph_pos)
    kpi["hist_ph_active_bins_neg"] = _count_active_bins(H_ph_neg)

    kpi["hist_amp_peak_bin_pos"] = _dominant_bin_index(H_amp_pos)
    kpi["hist_amp_peak_bin_neg"] = _dominant_bin_index(H_amp_neg)
    kpi["hist_ph_peak_bin_pos"] = _dominant_bin_index(H_ph_pos)
    kpi["hist_ph_peak_bin_neg"] = _dominant_bin_index(H_ph_neg)

    kpi["hist_amp_pos_neg_corr"] = _corr(H_amp_pos, H_amp_neg)
    kpi["hist_ph_pos_neg_corr"] = _corr(H_ph_pos, H_ph_neg)

    return kpi
