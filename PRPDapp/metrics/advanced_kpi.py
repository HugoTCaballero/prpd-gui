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


def _weighted_moments(data: np.ndarray, weights: np.ndarray) -> tuple[float, float, float, float]:
    data = np.asarray(data, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if data.size == 0 or weights.size != data.size:
        return (np.nan, np.nan, np.nan, np.nan)
    w_sum = np.sum(weights)
    if w_sum == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    mean = np.sum(weights * data) / w_sum
    diff = data - mean
    m2 = np.sum(weights * diff**2) / w_sum
    m3 = np.sum(weights * diff**3) / w_sum
    m4 = np.sum(weights * diff**4) / w_sum
    return mean, m2, m3, m4


def _weighted_skew_kurt(data: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean, m2, m3, m4 = _weighted_moments(data, weights)
    data = np.asarray(data, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if data.size == 0 or weights.size != data.size:
        return (np.nan, np.nan)
    w_sum = np.sum(weights)
    if w_sum == 0:
        return (np.nan, np.nan)
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


def compute_advanced_metrics(result: dict, bins_amp: int = 32, bins_phase: int = 32) -> dict:
    """KPIs avanzados (ANGPD 2.0 + histogramas)."""
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

    # Si no hay quantity, usar pesos unitarios para no bloquear los KPIs
    if not qty.size and ph.size == amp.size and ph.size > 0:
        qty = np.ones_like(ph, dtype=float)

    has_core = ph.size and amp.size and qty.size
    if not has_core and not phi_curve.size:
        return {}

    # Histogramas H_amp/H_ph (logica centralizada; N=16 por defecto)
    hist_h = compute_semicycle_histograms_from_aligned(aligned, N=16)
    phi_mod = ph % 360.0 if ph.size else np.asarray([], dtype=float)
    pos_mask = (phi_mod < 180.0) if phi_mod.size else np.zeros(0, dtype=bool)
    neg_mask = ~pos_mask if phi_mod.size else np.zeros(0, dtype=bool)

    nd_reasons: dict[str, str] = {}
    n_total = int(phi_mod.size) if phi_mod.size else 0
    n_pos = int(pos_mask.sum()) if pos_mask.size else 0
    n_neg = int(neg_mask.sum()) if neg_mask.size else 0

    # Skew/Kurt/medianas con pesos qty
    min_points_moments = 3
    pos_skew = pos_kurt = np.nan
    neg_skew = neg_kurt = np.nan
    if n_pos >= min_points_moments:
        pos_skew, pos_kurt = _weighted_skew_kurt(ph[pos_mask], qty[pos_mask])
        if not np.isfinite(pos_skew) or not np.isfinite(pos_kurt):
            nd_reasons["skewness.pos_skew"] = "Skew/Kurt (semiciclo +): varianza cero o pesos nulos."
            nd_reasons["kurtosis.pos_kurt"] = nd_reasons["skewness.pos_skew"]
    else:
        nd_reasons["skewness.pos_skew"] = f"Skew/Kurt (semiciclo +): insuficientes pulsos (n={n_pos})."
        nd_reasons["kurtosis.pos_kurt"] = nd_reasons["skewness.pos_skew"]

    if n_neg >= min_points_moments:
        neg_skew, neg_kurt = _weighted_skew_kurt(ph[neg_mask], qty[neg_mask])
        if not np.isfinite(neg_skew) or not np.isfinite(neg_kurt):
            nd_reasons["skewness.neg_skew"] = "Skew/Kurt (semiciclo -): varianza cero o pesos nulos."
            nd_reasons["kurtosis.neg_kurt"] = nd_reasons["skewness.neg_skew"]
    else:
        nd_reasons["skewness.neg_skew"] = f"Skew/Kurt (semiciclo -): insuficientes pulsos (n={n_neg})."
        nd_reasons["kurtosis.neg_kurt"] = nd_reasons["skewness.neg_skew"]

    median_pos = _weighted_percentile(ph[pos_mask], qty[pos_mask], 50) if n_pos else np.nan
    median_neg = _weighted_percentile(ph[neg_mask], qty[neg_mask], 50) if n_neg else np.nan
    p95_amp = float(np.percentile(amp, 95)) if amp.size else np.nan

    qty_pos = float(np.sum(qty[pos_mask])) if n_pos else 0.0
    qty_neg = float(np.sum(qty[neg_mask])) if n_neg else 0.0
    pulses_ratio = qty_pos / qty_neg if qty_neg else np.nan

    # Histogramas por semiciclo con pesos (para overlay y KPIs faltantes)
    amp_hist_pos = np.asarray([], dtype=float)
    amp_hist_neg = np.asarray([], dtype=float)
    amp_edges_pos = np.asarray([], dtype=float)
    amp_edges_neg = np.asarray([], dtype=float)
    phase_hist_pos = np.asarray([], dtype=float)
    phase_hist_neg = np.asarray([], dtype=float)
    ph_edges = np.asarray([], dtype=float)
    try:
        amp_max_range = float(np.nanmax(amp)) if amp.size else 100.0
        if not np.isfinite(amp_max_range) or amp_max_range <= 0:
            amp_max_range = 100.0
        amp_range = (0.0, max(amp_max_range, 1.0))
        if pos_mask.any():
            amp_hist_pos, amp_edges_pos = np.histogram(
                amp[pos_mask], bins=bins_amp, range=amp_range, weights=qty[pos_mask]
            )
        if neg_mask.any():
            amp_hist_neg, amp_edges_neg = np.histogram(
                amp[neg_mask], bins=bins_amp, range=amp_range, weights=qty[neg_mask]
            )
        if phi_mod.size:
            phase_hist, ph_edges = np.histogram(
                phi_mod, bins=bins_phase * 2, range=(0.0, 360.0), weights=qty if qty.size else None
            )
            phase_hist_pos = phase_hist[:bins_phase]
            phase_hist_neg = phase_hist[bins_phase:]
    except Exception:
        pass

    # KPIs basados en curvas ANGPD ya calculadas (phi_curve vs ang/nang qty) + fallback por histograma
    peaks_pos = peaks_neg = np.nan
    phase_corr = np.nan
    pos_curve = np.asarray([], dtype=float)
    neg_curve = np.asarray([], dtype=float)
    if phi_curve.size and (nang_qty.size or ang_qty.size):
        pos_curve = nang_qty[(phi_curve % 360.0) < 180.0] if nang_qty.size else np.asarray([], dtype=float)
        neg_curve = nang_qty[(phi_curve % 360.0) >= 180.0] if nang_qty.size else np.asarray([], dtype=float)
    if not pos_curve.size or not neg_curve.size:
        try:
            pos_curve, _ = np.histogram(
                ph[pos_mask], bins=bins_phase, range=(0.0, 180.0), weights=qty[pos_mask] if qty.size else None
            ) if pos_mask.any() else (np.asarray([], dtype=float), None)
            neg_curve, _ = np.histogram(
                ph[neg_mask], bins=bins_phase, range=(0.0, 180.0), weights=qty[neg_mask] if qty.size else None
            ) if neg_mask.any() else (np.asarray([], dtype=float), None)
        except Exception:
            pos_curve = np.asarray([], dtype=float)
            neg_curve = np.asarray([], dtype=float)
    if pos_curve.size:
        peaks_pos = _count_peaks_1d(pos_curve)
    if neg_curve.size:
        peaks_neg = _count_peaks_1d(neg_curve)
    if pos_curve.size and neg_curve.size and pos_curve.size == neg_curve.size:
        phase_corr = _safe_corr(pos_curve, neg_curve)
        if not np.isfinite(phase_corr):
            nd_reasons.setdefault("phase_corr", "Correlación de fases: datos insuficientes o varianza cero.")
    else:
        nd_reasons.setdefault("phase_corr", "Correlación de fases: curvas pos/neg no disponibles.")

    hist_block = {
        "phi_centers": phi_curve.tolist() if phi_curve.size else [],
        "angpd": ang_amp.tolist() if ang_amp.size else [],
        "n_angpd": nang_amp.tolist() if nang_amp.size else [],
        "angpd_qty": ang_qty.tolist() if ang_qty.size else [],
        "n_angpd_qty": nang_qty.tolist() if nang_qty.size else [],
        "amp_hist_pos": amp_hist_pos.tolist(),
        "amp_edges_pos": amp_edges_pos.tolist(),
        "amp_hist_neg": amp_hist_neg.tolist(),
        "amp_edges_neg": amp_edges_neg.tolist(),
        "phase_hist_pos": phase_hist_pos.tolist(),
        "phase_hist_neg": phase_hist_neg.tolist(),
        "ph_edges": ph_edges.tolist(),
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
        "counts": {"total": n_total, "pos": n_pos, "neg": n_neg},
        "nd_reasons": nd_reasons,
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
