import numpy as np


def _circ_mean_deg_array(ph: np.ndarray) -> float | None:
    try:
        if ph.size == 0:
            return None
        ph_rad = np.deg2rad(ph)
        mean_angle = np.rad2deg(np.arctan2(np.sin(ph_rad).mean(), np.cos(ph_rad).mean())) % 360.0
        return float(mean_angle)
    except Exception:
        return None


def _circ_width_deg_array(ph: np.ndarray) -> float | None:
    try:
        if ph.size == 0:
            return None
        ph_rad = np.deg2rad(ph)
        R = np.sqrt(np.square(np.sin(ph_rad).mean()) + np.square(np.cos(ph_rad).mean()))
        width = np.rad2deg(np.sqrt(2 * (1 - R)))
        return float(width)
    except Exception:
        return None


def compute_pd_metrics(result: dict, gap_stats: dict | None = None) -> dict:
    aligned = result.get("aligned", {}) or {}
    ph = np.asarray(aligned.get("phase_deg", []), dtype=float)
    amp = np.asarray(aligned.get("amplitude", []), dtype=float)
    sem = np.asarray(aligned.get("semicycle", []), dtype=int)
    if ph.size and amp.size and ph.size != amp.size:
        n = min(ph.size, amp.size)
        ph = ph[:n]
        amp = amp[:n]
        if sem.size:
            sem = sem[:n]
    if sem.size != amp.size and ph.size == amp.size:
        sem = (ph % 360.0 < 180.0).astype(int)
    phi_mod = ph % 360.0 if ph.size else np.asarray([], dtype=float)
    if amp.size and sem.size == amp.size:
        mask_pos = (sem == 1)
    elif amp.size and phi_mod.size == amp.size:
        mask_pos = (phi_mod < 180.0) if phi_mod.size == amp.size else np.zeros_like(amp, dtype=bool)
    else:
        mask_pos = np.zeros_like(amp, dtype=bool)
    mask_neg = np.logical_not(mask_pos)
    amp_pos = amp[mask_pos] if amp.size else np.asarray([], dtype=float)
    amp_neg = amp[mask_neg] if amp.size else np.asarray([], dtype=float)
    if phi_mod.size and phi_mod.size == mask_pos.size:
        ph_pos = phi_mod[mask_pos]
        ph_neg = (phi_mod[mask_neg] - 180.0)
    else:
        ph_pos = phi_mod[phi_mod < 180.0]
        ph_neg = (phi_mod[phi_mod >= 180.0] - 180.0)

    def _safe_p95(arr):
        return float(np.percentile(arr, 95)) if arr.size else 0.0

    def _count_peaks(ph_vals, span_deg):
        if ph_vals is None or not getattr(ph_vals, "size", 0):
            return 0
        hist, _ = np.histogram(ph_vals, bins=16, range=(0.0, span_deg))
        if hist.size < 3:
            return 0
        peaks_mask = (hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:])
        return int(np.sum(peaks_mask))

    p95_pos = _safe_p95(np.abs(amp_pos)) if amp_pos.size else 0.0
    p95_neg = _safe_p95(np.abs(amp_neg)) if amp_neg.size else 0.0
    amp_ratio = float(p95_pos / p95_neg) if p95_neg > 1e-6 else (float("inf") if p95_pos > 0 else 0.0)
    p95_mean = 0.5 * (p95_pos + p95_neg)
    ang = result.get("angpd", {}) or {}
    angpd = np.asarray(ang.get("angpd", []), dtype=float)
    n_angpd = np.asarray(ang.get("n_angpd", []), dtype=float)
    total_ang = float(np.sum(angpd)) if angpd.size else 0.0
    n_ang_ratio = float(np.sum(n_angpd) / total_ang) if total_ang > 0 else 0.0
    phase_center = _circ_mean_deg_array(ph)
    phase_width = _circ_width_deg_array(ph)
    phase_center_pos = _circ_mean_deg_array(ph_pos)
    phase_center_neg = _circ_mean_deg_array(ph_neg)
    if phase_center_neg is not None:
        phase_center_neg = (phase_center_neg + 180.0) % 360.0
    phase_width_pos = _circ_width_deg_array(ph_pos)
    phase_width_neg = _circ_width_deg_array(ph_neg)
    peaks = _count_peaks(ph, 360.0)
    raw_peaks_pos = _count_peaks(ph_pos, 180.0)
    raw_peaks_neg = _count_peaks(ph_neg, 180.0)
    if peaks > 0:
        denom = max(raw_peaks_pos + raw_peaks_neg, 1)
        frac_pos = raw_peaks_pos / denom if denom else 0.5
        if raw_peaks_pos == 0 and raw_peaks_neg == 0 and ph.size > 0:
            frac_pos = (ph < 180.0).sum() / float(ph.size)
        peaks_pos = int(round(peaks * frac_pos))
        peaks_pos = max(0, min(peaks, peaks_pos))
        peaks_neg = max(0, peaks - peaks_pos)
    else:
        peaks_pos = 0
        peaks_neg = 0
    total_count = int(ph.size)
    count_pos = int(amp_pos.size)
    count_neg = int(amp_neg.size)
    balance = None
    if total_count > 0:
        balance = (count_pos - count_neg) / float(total_count)
    snr = float(p95_mean / (float(np.mean(np.abs(amp))) + 1e-6)) if amp.size else 0.0
    gap_p50 = gap_stats.get("p50_ms") if isinstance(gap_stats, dict) else None
    gap_p5 = gap_stats.get("p5_ms") if isinstance(gap_stats, dict) else None
    gap_class_p50 = gap_stats.get("classification") if isinstance(gap_stats, dict) else {}
    gap_class_p5 = gap_stats.get("classification_p5") if isinstance(gap_stats, dict) else {}
    return {
        "amp_p95_pos": round(p95_pos, 2),
        "amp_p95_neg": round(p95_neg, 2),
        "amp_ratio": round(amp_ratio, 2) if np.isfinite(amp_ratio) else None,
        "p95_mean": round(p95_mean, 2),
        "n_ang_ratio": round(n_ang_ratio, 3),
        "phase_center": None if phase_center is None else round(phase_center, 1),
        "phase_width": None if phase_width is None else round(phase_width, 1),
        "phase_center_pos": None if phase_center_pos is None else round(phase_center_pos, 1),
        "phase_center_neg": None if phase_center_neg is None else round(phase_center_neg, 1),
        "phase_width_pos": None if phase_width_pos is None else round(phase_width_pos, 1),
        "phase_width_neg": None if phase_width_neg is None else round(phase_width_neg, 1),
        "n_peaks": peaks,
        "n_peaks_pos": peaks_pos,
        "n_peaks_neg": peaks_neg,
        "total_count": total_count,
        "count_pos": count_pos,
        "count_neg": count_neg,
        "pulse_balance": None if balance is None else round(balance, 3),
        "snr": round(snr, 3),
        "gap_p50": gap_p50,
        "gap_p5": gap_p5,
        "gap_class_p50": gap_class_p50,
        "gap_class_p5": gap_class_p5,
    }


def classify_pd(metrics: dict) -> dict:
    p95_mean = float(metrics.get("p95_mean", 0.0) or 0.0)
    amp_ratio = float(metrics.get("amp_ratio", 0.0) or 0.0)
    n_ang = float(metrics.get("n_ang_ratio", 0.0) or 0.0)
    n_peaks = int(metrics.get("n_peaks", 0) or 0)
    gap_p50 = metrics.get("gap_p50")
    gap_p5 = metrics.get("gap_p5")
    gap_class_p50 = metrics.get("gap_class_p50") or {}
    no_dp = ((gap_p50 is not None and gap_p50 >= 500.0) or (gap_p5 is not None and gap_p5 >= 500.0))
    total_count = int(metrics.get("total_count") or 0)
    count_pos = int(metrics.get("count_pos") or 0)
    count_neg = int(metrics.get("count_neg") or 0)
    pulse_imbalance = abs(count_pos - count_neg) / max(1, total_count) if total_count else 0.0

    def _gap_weight(value, strong=False):
        if value is None:
            return 0.0
        if value < 3.0:
            return 1.5 if strong else 1.0
        if value < 7.0:
            return 0.75 if strong else 0.5
        return 0.0

    p50_weight = _gap_weight(gap_p50, strong=True)
    p5_weight = _gap_weight(gap_p5, strong=False)
    if gap_p50 is not None and gap_p50 > 20.0:
        p5_weight *= 0.4
    gap_penalty = p50_weight + p5_weight

    if n_peaks >= 3:
        pd_type = "Superficial / Tracking"
        location = "Superficie de aislamiento e interfaces. En aceite: posibles streamers en cumbre/paredes."
    elif (n_peaks >= 2 and n_ang > 1.5) or n_ang > 2.0:
        pd_type = "Superficial / Tracking"
        location = "Superficie de aislamiento e interfaces. En aceite: posibles streamers en cumbre/paredes."
    elif amp_ratio > 1.2 and n_peaks <= 1:
        pd_type = "Corona"
        location = "Bordes con punta expuestos. En aceite: lodos/partículas metálicas formando canales."
    else:
        if pulse_imbalance > 0.20 and n_peaks <= 2:
            pd_type = "Superficial / Tracking"
            location = "Superficie de aislamiento e interfaces"
        else:
            pd_type = "Cavidad interna"
            location = "Volumen interno del aislamiento (aceite o sólido)"
    if no_dp:
        pd_type = "Sin descargas"
        location = "N/D"
        risk = "Sin descargas"
        stage = "Sin descargas"
        life_score = None
        life = None
        actions = "No se detectaron descargas parciales."
        return {
            "pd_type": pd_type,
            "location": location,
            "risk": risk,
            "stage": stage,
            "life_years": life,
            "life_score": life_score,
            "actions": actions,
        }

    def _norm(val, lo, hi):
        if val <= lo:
            return 0.0
        if val >= hi:
            return 100.0
        return (val - lo) / max(hi - lo, 1e-6) * 100.0

    mag_norm = _norm(p95_mean, 20.0, 80.0)
    act_norm = _norm(total_count, 2000.0, 12000.0)
    penal = 0.6 * act_norm + 0.4 * mag_norm
    life_score = max(0.0, (100.0 - penal) - gap_penalty * 7.0)

    def _interp(ls, lo, hi, y0, y1):
        if ls <= lo:
            return y0
        if ls >= hi:
            return y1
        return y0 + (y1 - y0) * (ls - lo) / max(hi - lo, 1e-6)

    if life_score < 15:
        risk = "Crítico"; stage = "Cercana a ruptura de aislamiento"
        life = _interp(life_score, 0, 15, 0.9, 1.5)
    elif life_score < 30:
        risk = "Grave"; stage = "Etapa de aceleración"
        life = _interp(life_score, 15, 30, 1.5, 2.5)
    elif life_score < 40:
        risk = "Moderado"; stage = "Etapa de estabilización"
        life = _interp(life_score, 30, 40, 2.5, 5.0)
    elif life_score < 60:
        risk = "Bajo"; stage = "Etapa de aceleración"
        life = _interp(life_score, 40, 60, 5.0, 8.0)
    elif life_score < 75:
        risk = "Incipiente"; stage = "Etapa incipiente"
        life = _interp(life_score, 60, 75, 8.0, 12.0)
    else:
        risk = "Descargas parciales no detectadas"; stage = "Se necesitan más monitoreos"
        life = _interp(min(life_score, 100.0), 75, 100, 12.0, 21.4)

    if pd_type.startswith("Corona"):
        life *= 1.05
    if pd_type.startswith("Superficial") and n_ang > 1.5:
        life *= 0.9
    life = max(0.5, life)

    if risk.startswith("Descargas parciales no detectadas") and (
        gap_penalty > 0.0 or gap_p50 is not None or gap_p5 is not None
    ):
        risk = "Incipiente"
        stage = "Etapa incipiente"
        life = min(life, 12.0)

    gap_level = str(gap_class_p50.get("level_name", "")).lower()
    if "crític" in gap_level or "crit" in gap_level:
        risk = "Crítico"
        stage = "Cercana a ruptura de aislamiento"
        life_score = min(life_score, 15.0)
        life = min(life, 1.0)
    else:
        if gap_penalty >= 2.0:
            risk = "Crítico"; stage = "Cercana a ruptura de aislamiento"; life = min(life, 1.5)
        elif gap_penalty >= 1.0:
            if risk in ("Crítico", "Grave"):
                pass
            elif life_score < 40:
                risk = "Grave"; stage = "Etapa de aceleración"; life = min(life, 2.5)

    if risk == "Crítico":
        actions = "Detener equipo y planear reparación/sustitución inmediata."
        if gap_penalty >= 2.0 or "crit" in gap_level:
            actions += " Considerar ventana de reemplazo: valores críticos indican riesgo alto de ruptura. Monitoreo semanal."
        elif gap_penalty >= 1.0:
            actions += " Monitoreo semanal."
    elif risk == "Grave":
        actions = "Planear sustitución (si la tendencia es creciente) y realizar evaluación especializada; monitoreo cada 3 meses."
        if gap_penalty >= 1.0 or "grave" in gap_level:
            actions += " Considerar ventana de reemplazo si el gap-time sigue crítico."
    elif risk == "Moderado":
        actions = "Monitorear cada 6 meses y revisar tendencias."
    elif risk == "Bajo":
        actions = "Vigilancia rutinaria; monitoreo cada 12 meses."
    elif risk.startswith("Incipiente"):
        actions = "Seguimiento anual y confirmar con mediciones adicionales."
    else:
        actions = "Sin descargas parciales detectadas; confirmar con monitoreo regular."
    if risk in ("Crítico", "Grave") and gap_penalty >= 2.0:
        actions += " Monitoreo semanal o continuo."

    return {
        "pd_type": pd_type,
        "location": location,
        "risk": risk,
        "stage": stage,
        "life_years": round(life, 1),
        "life_score": round(life_score, 1),
        "life_interval": _life_interval(risk, life),
        "actions": actions,
    }


def _life_interval(risk: str, life: float | None) -> str:
    """Devuelve un intervalo legible de vida remanente en años."""
    r = (risk or "").lower()
    if "crític" in r:
        return "≈ 0.9–1.5 años"
    if "grave" in r:
        return "≈ 1.5–2.5 años"
    if "moderado" in r:
        return "≈ 2.5–5 años"
    if "bajo" in r:
        return "≈ 5–8 años"
    if "incipiente" in r:
        return "≈ 8–12 años"
    if "descargas parciales no detectadas" in r or "sin descargas" in r:
        return "≈ 12–21.4 años"
    if life is None:
        return "N/D"
    lo = max(life * 0.85, 0.5)
    hi = life * 1.15
    return f"≈ {lo:.1f}–{hi:.1f} años"
