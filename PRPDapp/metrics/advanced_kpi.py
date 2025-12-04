import numpy as np

import numpy as np
from PRPDapp.config_pd import CLASS_NAMES, CLASS_INFO
from PRPDapp.logic_hist import compute_semicycle_histograms_from_aligned


def _weighted_percentile(data: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    data = np.asarray(data, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if data.size == 0 or weights.size != data.size:
        return np.nan
    sorter = np.argsort(data)
    data = data[sorter]
    weights = weights[sorter]
    cumsum = np.cumsum(weights)
    if cumsum[-1] <= 0:
        return np.nan
    cutoff = percentile / 100.0 * cumsum[-1]
    idx = np.searchsorted(cumsum, cutoff)
    idx = min(idx, data.size - 1)
    return data[idx]


def _weighted_moments(data: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    data = np.asarray(data, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if data.size == 0 or weights.size != data.size:
        return (np.nan, np.nan, np.nan)
    w_sum = np.sum(weights)
    if w_sum == 0:
        return (np.nan, np.nan, np.nan)
    mean = np.sum(weights * data) / w_sum
    diff = data - mean
    m2 = np.sum(weights * diff**2) / w_sum
    m3 = np.sum(weights * diff**3) / w_sum
    m4 = np.sum(weights * diff**4) / w_sum
    return mean, m3, m4


def _weighted_skew_kurt(data: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean, m3, m4 = _weighted_moments(data, weights)
    data = np.asarray(data, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if data.size == 0 or weights.size != data.size:
        return (np.nan, np.nan)
    w_sum = np.sum(weights)
    if w_sum == 0:
        return (np.nan, np.nan)
    diff = data - mean
    m2 = np.sum(weights * diff**2) / w_sum
    if m2 <= 0:
        return (np.nan, np.nan)
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2**2) - 3.0
    return (skew, kurt)


def _count_peaks_1d(arr: np.ndarray) -> int:
    arr = np.asarray(arr, dtype=float)
    if arr.size < 3:
        return 0
    count = 0
    for i in range(1, arr.size - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            count += 1
    return count


def compute_advanced_metrics(result: dict, bins_amp: int = 32, bins_phase: int = 32) -> dict:
    """KPIs avanzados.

    - Histogramas H_amp/H_ph: bins N=16 (o configurables) basados en amplitud/fase.
    - ANGPD/N-ANGPD: usar SIEMPRE las curvas 1D ya calculadas en result["angpd"].
    """
    aligned = result.get("aligned", {}) if isinstance(result, dict) else {}
    ang = result.get("angpd", {}) if isinstance(result, dict) else {}

    ph = np.asarray(aligned.get("phase_deg", []), dtype=float)
    amp = np.asarray(aligned.get("amplitude", []), dtype=float)
    qty = np.asarray(aligned.get("quantity", []), dtype=float)

    phi_curve = np.asarray(ang.get("phi_centers", []), dtype=float)
    ang_amp = np.asarray(ang.get("angpd", []), dtype=float)
    nang_amp = np.asarray(ang.get("n_angpd", []), dtype=float)
    ang_qty = np.asarray(ang.get("angpd_qty", []), dtype=float)
    nang_qty = np.asarray(ang.get("n_angpd_qty", []), dtype=float)

    has_core = ph.size and amp.size and qty.size
    if not has_core and not phi_curve.size:
        return {}

    # Histogramas H_amp/H_ph (lógica centralizada; N=16 por defecto)
    hist_h = compute_semicycle_histograms_from_aligned(aligned, N=16)
    phi_mod = ph % 360.0 if ph.size else np.asarray([], dtype=float)
    pos_mask = (phi_mod < 180.0) if phi_mod.size else np.zeros(0, dtype=bool)
    neg_mask = ~pos_mask if phi_mod.size else np.zeros(0, dtype=bool)

    # Skew/Kurt/medianas con pesos qty
    pos_skew, pos_kurt = _weighted_skew_kurt(ph[pos_mask], qty[pos_mask]) if pos_mask.any() else (np.nan, np.nan)
    neg_skew, neg_kurt = _weighted_skew_kurt(ph[neg_mask], qty[neg_mask]) if neg_mask.any() else (np.nan, np.nan)

    median_pos = _weighted_percentile(ph[pos_mask], qty[pos_mask], 50) if pos_mask.any() else np.nan
    median_neg = _weighted_percentile(ph[neg_mask], qty[neg_mask], 50) if neg_mask.any() else np.nan
    p95_amp = float(np.percentile(amp, 95)) if amp.size else np.nan

    qty_pos = float(np.sum(qty[pos_mask])) if pos_mask.any() else 0.0
    qty_neg = float(np.sum(qty[neg_mask])) if neg_mask.any() else 0.0
    pulses_ratio = qty_pos / qty_neg if qty_neg else np.nan

    # KPIs basados en curvas ANGPD ya calculadas (phi_curve vs ang/nang qty)
    peaks_pos = peaks_neg = np.nan
    phase_corr = np.nan
    if phi_curve.size:
        # usar n_angpd_qty para morfología de cantidad
        pos_curve = nang_qty[(phi_curve % 360.0) < 180.0] if nang_qty.size else np.asarray([], dtype=float)
        neg_curve = nang_qty[(phi_curve % 360.0) >= 180.0] if nang_qty.size else np.asarray([], dtype=float)
        if pos_curve.size:
            peaks_pos = _count_peaks_1d(pos_curve)
        if neg_curve.size:
            peaks_neg = _count_peaks_1d(neg_curve)
        if pos_curve.size and neg_curve.size and pos_curve.size == neg_curve.size:
            try:
                phase_corr = float(np.corrcoef(pos_curve, neg_curve)[0, 1])
            except Exception:
                phase_corr = np.nan

    hist_block = {
        # Curvas ANGPD tal cual vienen en result["angpd"]
        "phi_centers": phi_curve.tolist() if phi_curve.size else [],
        "angpd": ang_amp.tolist() if ang_amp.size else [],
        "n_angpd": nang_amp.tolist() if nang_amp.size else [],
        "angpd_qty": ang_qty.tolist() if ang_qty.size else [],
        "n_angpd_qty": nang_qty.tolist() if nang_qty.size else [],
        # Histogramas H_amp/H_ph (bins configurables)
        "amp_hist_pos": amp_hist_pos.tolist(),
        "amp_edges_pos": amp_edges_pos.tolist(),
        "amp_hist_neg": amp_hist_neg.tolist(),
        "amp_edges_neg": amp_edges_neg.tolist(),
        "phase_hist_pos": phase_hist_pos.tolist(),
        "phase_hist_neg": phase_hist_neg.tolist(),
        "ph_edges": ph_edges,
    }

    metrics = {
        "hist": {**hist_block, **hist_h},
        "skewness": {"pos_skew": pos_skew, "neg_skew": neg_skew},
        "kurtosis": {"pos_kurt": pos_kurt, "neg_kurt": neg_kurt},
        "num_peaks": {"pos": peaks_pos, "neg": peaks_neg},
        "phase_corr": phase_corr,
        "phase_medians_p95": {
            "median_pos_phase": median_pos,
            "median_neg_phase": median_neg,
            "p95_amp": p95_amp,
        },
        "pulses_ratio": pulses_ratio,
    }

    scores = {"cavidad": 0.0, "superficial": 0.0, "corona": 0.0, "flotante": 0.0, "ruido": 0.0, "suspendida": 0.0}
    if np.isfinite(median_pos) and np.isfinite(median_neg):
        if median_pos < 120 and median_neg > 240:
            scores["cavidad"] += 1
    if np.isfinite(phase_corr) and phase_corr > 0.4:
        scores["superficial"] += 1
    if np.isfinite(peaks_pos) and np.isfinite(peaks_neg) and peaks_pos <= 1 and peaks_neg <= 1:
        scores["corona"] += 1
    if np.isfinite(pos_kurt) and np.isfinite(neg_kurt) and pos_kurt < 0 and neg_kurt < 0:
        scores["flotante"] += 1
    if np.isnan(phase_corr) or phase_corr < 0.1:
        scores["ruido"] += 0.5
        scores["suspendida"] += 0.5

    total = sum(scores.values()) or 1.0
    probs = {k: v / total for k, v in scores.items()}
    metrics["heuristic_probs"] = probs
    metrics["heuristic_top"] = max(probs, key=probs.get)
    return metrics
