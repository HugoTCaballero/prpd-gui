"""Helpers para histogramas H_amp / H_ph reutilizables en UI y KPIs."""

from __future__ import annotations

import numpy as np
from typing import Dict, Any

def _count_peaks(arr: np.ndarray) -> int:
    arr = np.asarray(arr, dtype=float)
    if arr.size < 3:
        return 0
    return int(np.sum((arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:])))


def compute_semicycle_histograms_from_aligned(
    aligned: Dict[str, Any],
    N: int = 16,
    amp_range: tuple[float, float] = (0.0, 100.0),
) -> Dict[str, np.ndarray]:
    """
    Calcula H_amp_pos, H_amp_neg, H_ph_pos y H_ph_neg a partir de los datos
    alineados (phase_deg, amplitude), usando EXACTAMENTE la lógica actual
    de _draw_histograms_semiciclo.
    """
    phase_deg = np.asarray(aligned.get("phase_deg", []), dtype=float)
    amplitude = np.asarray(aligned.get("amplitude", []), dtype=float)

    if phase_deg.size == 0 or amplitude.size == 0:
        return {
            "H_amp_pos": np.zeros(N, dtype=float),
            "H_amp_neg": np.zeros(N, dtype=float),
            "H_ph_pos": np.zeros(N, dtype=float),
            "H_ph_neg": np.zeros(N, dtype=float),
        }

    phi = phase_deg % 360.0
    pos = (phi < 180.0)
    neg = ~pos

    a_pos, _ = np.histogram(amplitude[pos], bins=N, range=amp_range)
    a_neg, _ = np.histogram(amplitude[neg], bins=N, range=amp_range)
    H_amp_pos = np.log10(1.0 + a_pos.astype(float))
    H_amp_neg = np.log10(1.0 + a_neg.astype(float))

    phi_pos = phi[pos]
    phi_neg = (phi[neg] - 180.0)
    p_pos, _ = np.histogram(phi_pos, bins=N, range=(0.0, 180.0))
    p_neg, _ = np.histogram(phi_neg, bins=N, range=(0.0, 180.0))
    H_ph_pos = np.log10(1.0 + p_pos.astype(float))
    H_ph_neg = np.log10(1.0 + p_neg.astype(float))

    return {
        "H_amp_pos": H_amp_pos,
        "H_amp_neg": H_amp_neg,
        "H_ph_pos": H_ph_pos,
        "H_ph_neg": H_ph_neg,
    }


def compute_hist_kpis(hist: Dict[str, np.ndarray]) -> Dict[str, float | int]:
    """KPIs numéricos a partir de H_amp/H_ph (N=16)."""
    h_amp_pos = np.asarray(hist.get("H_amp_pos", []), dtype=float)
    h_amp_neg = np.asarray(hist.get("H_amp_neg", []), dtype=float)
    h_ph_pos = np.asarray(hist.get("H_ph_pos", []), dtype=float)
    h_ph_neg = np.asarray(hist.get("H_ph_neg", []), dtype=float)

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.size == 0 or b.size == 0 or a.size != b.size:
            return np.nan
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            return np.nan
        a = a[mask]
        b = b[mask]
        if a.size < 2:
            return np.nan
        a_center = a - np.mean(a)
        b_center = b - np.mean(b)
        a_std = np.std(a_center)
        b_std = np.std(b_center)
        if a_std == 0 or b_std == 0:
            return np.nan
        try:
            return float(np.mean(a_center * b_center) / (a_std * b_std))
        except Exception:
            return np.nan

    def _safe_ratio(a: float, b: float) -> float | float:
        try:
            if b == 0:
                return np.inf if a > 0 else 0.0
            return a / b
        except Exception:
            return np.nan

    amp_sum_pos = float(np.sum(h_amp_pos)) if h_amp_pos.size else 0.0
    amp_sum_neg = float(np.sum(h_amp_neg)) if h_amp_neg.size else 0.0
    ph_sum_pos = float(np.sum(h_ph_pos)) if h_ph_pos.size else 0.0
    ph_sum_neg = float(np.sum(h_ph_neg)) if h_ph_neg.size else 0.0

    # Umbral relativo para contar bins activos (10% del máximo de cada vector)
    def _active_bins(arr: np.ndarray) -> int:
        if arr.size == 0:
            return 0
        mx = float(np.max(arr))
        if mx <= 0:
            return 0
        thr = mx * 0.10
        return int(np.sum(arr >= thr))

    # Bin dominante (índice 1..N); si no hay datos -> 0
    def _peak_bin(arr: np.ndarray) -> int:
        if arr.size == 0:
            return 0
        return int(np.argmax(arr) + 1)

    kpis = {
        "amp_max_pos": float(np.max(h_amp_pos)) if h_amp_pos.size else 0.0,
        "amp_max_neg": float(np.max(h_amp_neg)) if h_amp_neg.size else 0.0,
        "amp_sum_pos": amp_sum_pos,
        "amp_sum_neg": amp_sum_neg,
        "amp_ratio_pos_neg": _safe_ratio(amp_sum_pos, amp_sum_neg),
        "amp_peaks_pos": _count_peaks(h_amp_pos),
        "amp_peaks_neg": _count_peaks(h_amp_neg),
        "ph_max_pos": float(np.max(h_ph_pos)) if h_ph_pos.size else 0.0,
        "ph_max_neg": float(np.max(h_ph_neg)) if h_ph_neg.size else 0.0,
        "ph_sum_pos": ph_sum_pos,
        "ph_sum_neg": ph_sum_neg,
        "ph_ratio_pos_neg": _safe_ratio(ph_sum_pos, ph_sum_neg),
        "ph_peaks_pos": _count_peaks(h_ph_pos),
        "ph_peaks_neg": _count_peaks(h_ph_neg),
        "ph_corr": _safe_corr(h_ph_pos, h_ph_neg),
        # bins activos y bin dominante
        "hist_amp_active_bins_pos": _active_bins(h_amp_pos),
        "hist_amp_active_bins_neg": _active_bins(h_amp_neg),
        "hist_ph_active_bins_pos": _active_bins(h_ph_pos),
        "hist_ph_active_bins_neg": _active_bins(h_ph_neg),
        "hist_amp_peak_bin_pos": _peak_bin(h_amp_pos),
        "hist_amp_peak_bin_neg": _peak_bin(h_amp_neg),
        "hist_ph_peak_bin_pos": _peak_bin(h_ph_pos),
        "hist_ph_peak_bin_neg": _peak_bin(h_ph_neg),
        "hist_amp_pos_neg_corr": _safe_corr(h_amp_pos, h_amp_neg),
        "hist_ph_pos_neg_corr": _safe_corr(h_ph_pos, h_ph_neg),
    }
    return kpis
