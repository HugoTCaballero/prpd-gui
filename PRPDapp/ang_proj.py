from __future__ import annotations

import numpy as np
from typing import Dict, Any


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Normaliza un vector a [0,1] usando su máximo; si es plano, devuelve el mismo arreglo."""
    if vec.size == 0:
        return vec
    vmax = float(np.max(vec))
    if vmax <= 0:
        return vec
    return vec / vmax


def _count_peaks(vec: np.ndarray, thr_rel: float = 0.1) -> int:
    """
    Cuenta picos simples (vec[i] > vecinos) por encima de un umbral relativo.
    Se evita depender de scipy.signal para mantener ligereza.
    """
    v = np.asarray(vec, dtype=float)
    if v.size < 3:
        return 0
    thr = float(np.max(v)) * thr_rel
    peaks = (v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]) & (v[1:-1] >= thr)
    return int(np.sum(peaks))


def compute_ang_proj(
    aligned: Dict[str, Any],
    n_phase_bins: int = 32,
    n_amp_bins: int = 16,
    n_points: int = 64,
) -> Dict[str, Any]:
    """
    Construye proyecciones fase/amplitud (ANGPD avanzado) separadas por polaridad.

    Retorna curvas normalizadas en [0,1] para cada polaridad y ejes target fijos.
    No reemplaza el ANGPD clásico; se agrega como bloque nuevo en `result`.
    """
    phase = np.asarray(aligned.get("phase_deg", []), dtype=float)
    amp = np.asarray(aligned.get("amplitude", []), dtype=float)
    if not phase.size or not amp.size:
        return {
            "n_points": n_points,
            "phase_pos": np.zeros(n_points, dtype=float),
            "phase_neg": np.zeros(n_points, dtype=float),
            "amp_pos": np.zeros(n_points, dtype=float),
            "amp_neg": np.zeros(n_points, dtype=float),
            "amp_min": 0.0,
            "amp_max": 0.0,
        }

    # Polarity/sign: usa campo si existe; si no, deduce por fase
    pol = aligned.get("polarity")
    if pol is None:
        pol = aligned.get("sign")
    pol = np.asarray(pol, dtype=float) if pol is not None else None
    if pol is None or pol.size != phase.size:
        phi = np.mod(phase, 360.0)
        pol = np.where((phi >= 0.0) & (phi < 180.0), 1.0, -1.0)

    mask_pos = pol > 0
    mask_neg = pol < 0

    # Bins de fase y amplitud
    phase_edges = np.linspace(0.0, 360.0, n_phase_bins + 1)
    amp_min = float(np.min(amp)) if amp.size else 0.0
    amp_max = float(np.max(amp)) if amp.size else 1.0
    if amp_max <= amp_min:
        amp_max = amp_min + 1.0
    amp_edges = np.linspace(amp_min, amp_max, n_amp_bins + 1)

    # Histogramas 2D
    H_pos, _, _ = np.histogram2d(
        amp[mask_pos], phase[mask_pos],
        bins=[amp_edges, phase_edges],
    )
    H_neg, _, _ = np.histogram2d(
        amp[mask_neg], phase[mask_neg],
        bins=[amp_edges, phase_edges],
    )

    # Proyección fase: sum sobre amplitud (eje 0)
    phase_pos = H_pos.sum(axis=0)
    phase_neg = H_neg.sum(axis=0)

    # Proyección amplitud: sum sobre fase (eje 1)
    amp_pos = H_pos.sum(axis=1)
    amp_neg = H_neg.sum(axis=1)

    # Interpolar a n_points con normalización
    phase_centers = 0.5 * (phase_edges[:-1] + phase_edges[1:])
    target_phase = np.linspace(0.0, 360.0, n_points)
    phase_pos_i = np.interp(target_phase, phase_centers, _normalize(phase_pos))
    phase_neg_i = np.interp(target_phase, phase_centers, _normalize(phase_neg))

    amp_centers = 0.5 * (amp_edges[:-1] + amp_edges[1:])
    target_amp = np.linspace(amp_min, amp_max, n_points)
    amp_pos_i = np.interp(target_amp, amp_centers, _normalize(amp_pos))
    amp_neg_i = np.interp(target_amp, amp_centers, _normalize(amp_neg))

    return {
        "n_points": n_points,
        "phase_pos": phase_pos_i,
        "phase_neg": phase_neg_i,
        "amp_pos": amp_pos_i,
        "amp_neg": amp_neg_i,
        "amp_min": amp_min,
        "amp_max": amp_max,
    }


def compute_ang_proj_kpis(
    ang_proj: Dict[str, Any],
    phase_threshold: float = 0.1,
    amp_threshold: float = 0.1,
) -> Dict[str, float]:
    """
    KPIs básicos a partir de las proyecciones suavizadas ANGPD 2.0.
    """
    n_points = int(ang_proj.get("n_points", 64))
    phase_pos = np.asarray(ang_proj.get("phase_pos", []), dtype=float)
    phase_neg = np.asarray(ang_proj.get("phase_neg", []), dtype=float)
    amp_pos = np.asarray(ang_proj.get("amp_pos", []), dtype=float)
    amp_neg = np.asarray(ang_proj.get("amp_neg", []), dtype=float)

    phase_total = phase_pos + phase_neg
    amp_total = amp_pos + amp_neg

    # Anchura efectiva de fase: tramo donde supera umbral relativo
    max_phase = float(np.max(phase_total)) if phase_total.size else 0.0
    mask_phase = phase_total > (phase_threshold * max_phase if max_phase > 0 else 0.0)
    if mask_phase.any():
        idx = np.where(mask_phase)[0]
        width_rel = (idx[-1] - idx[0]) / max(n_points - 1, 1)
        phase_width_deg = width_rel * 360.0
    else:
        phase_width_deg = 0.0

    # Simetría entre mitades de fase
    half = n_points // 2
    energy_first = float(phase_total[:half].sum()) if phase_total.size else 0.0
    energy_second = float(phase_total[half:].sum()) if phase_total.size else 0.0
    den = energy_first + energy_second
    phase_sym = 1.0 - abs(energy_first - energy_second) / den if den > 0 else 0.0

    # Picos
    n_peaks_phase = _count_peaks(phase_total, thr_rel=phase_threshold)
    n_peaks_amp = _count_peaks(amp_total, thr_rel=amp_threshold)

    # Concentración de amplitud
    mean_amp = float(np.mean(amp_total)) if amp_total.size else 0.0
    max_amp = float(np.max(amp_total)) if amp_total.size else 0.0
    amp_conc = (max_amp / (mean_amp + 1e-12)) if mean_amp > 0 else 0.0

    return {
        "phase_width_deg": float(phase_width_deg),
        "phase_symmetry": float(phase_sym),
        "phase_peaks": int(n_peaks_phase),
        "amp_concentration": float(amp_conc),
        "amp_peaks": int(n_peaks_amp),
    }
