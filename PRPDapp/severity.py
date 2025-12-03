# severity.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np

def _safe(v, eps=1e-12):
    return float(np.clip(v, eps, np.inf))

def physical_block(q_pc: np.ndarray, sample_seconds: float | None = None) -> float:
    """
    Métrica física [0..1]: combina q_mean, q_peak y (opcional) tasa de pulsos.
    Asume q en pC (IEC 60270).
    """
    if len(q_pc) == 0:
        return 0.0
    q_abs = np.abs(q_pc)
    q_mean = np.mean(q_abs)
    q_peak = np.max(q_abs)
    s_q = float(q_mean / (_safe(q_peak)))
    # normalizadores gruesos (ajústalos con tus datos)
    q_peak_norm = np.tanh(q_peak / 500.0)     # satura ~500 pC
    q_mean_norm = np.tanh(q_mean / 200.0)     # satura ~200 pC
    rate_norm = 0.0
    if sample_seconds and sample_seconds > 0:
        rate = len(q_pc) / sample_seconds
        rate_norm = np.tanh(rate / 50.0)      # satura ~50 cps
    # mezcla ponderada
    return float(0.5 * q_peak_norm + 0.35 * q_mean_norm + 0.15 * rate_norm)

def phase_coherence(phi_deg: np.ndarray, q_pc: np.ndarray, bins=72) -> float:
    """Concentración de energía por fase (0..1)."""
    if len(q_pc) == 0:
        return 0.0
    h, _ = np.histogram((phi_deg % 360.0), bins=bins, range=(0.0, 360.0), weights=np.abs(q_pc))
    k = max(3, bins // 24)
    topk = np.sort(h)[-k:].sum()
    tot = h.sum() + 1e-12
    return float(np.clip(topk / tot, 0.0, 1.0))

def class_confidence(probs: dict[str, float]) -> float:
    """
    Confianza (0..1) priorizando cavidad/superficial.
    probs: {'cavidad':p1, 'superficial':p2, 'corona':p3, 'flotante':p4}
    """
    p_cav = float(probs.get('cavidad', 0.0))
    p_sup = float(probs.get('superficial', 0.0))
    p_cor = float(probs.get('corona', 0.0))
    p_flo = float(probs.get('flotante', 0.0))
    # cavidad/superficial pesan más; flotante resta un poco
    score = 0.45 * p_cav + 0.35 * p_sup + 0.20 * p_cor - 0.10 * p_flo
    return float(np.clip(score, 0.0, 1.0))

def severity_index(phi_deg: np.ndarray,
                   q_pc: np.ndarray,
                   probs: dict[str, float],
                   sample_seconds: float | None = None,
                   noise_ratio: float | None = None) -> float:
    """
    Índice 0..100.
    """
    phys = physical_block(q_pc, sample_seconds=sample_seconds)      # 0..1
    coh = phase_coherence(phi_deg, q_pc)                            # 0..1
    cls = class_confidence(probs)                                   # 0..1

    raw = 0.55 * phys + 0.25 * coh + 0.20 * cls
    if noise_ratio is not None:
        # penaliza ruido elevado
        raw *= float(np.clip(1.0 - 0.6 * noise_ratio, 0.2, 1.0))

    return float(np.clip(100.0 * raw, 0.0, 100.0))

def bucket(sev: float) -> str:
    if sev >= 75.0:
        return "ALTA"
    if sev >= 45.0:
        return "MEDIA"
    if sev >= 20.0:
        return "BAJA"
    return "MUY BAJA"
