from __future__ import annotations

import numpy as np

PD_CLASSES = [
    "corona",
    "superficial_tracking",
    "cavidad_interna",
    "flotante",
    "ruido_baja",
]

PD_LABELS = {
    "corona": "Corona +/-",
    "superficial_tracking": "Superficial / Tracking",
    "cavidad_interna": "Cavidad interna",
    "flotante": "Descarga flotante",
    "ruido_baja": "Ruido / baja severidad",
}


def build_rule_features(result: dict) -> dict:
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    fa = result.get("fa_kpis", {}) if isinstance(result, dict) else {}
    kpis = result.get("kpi", {}) if isinstance(result, dict) else {}
    gap = result.get("gap_stats", {}) or result.get("gap_summary", {}) or {}
    hist = kpis.get("hist", {}) if isinstance(kpis, dict) else {}
    skew = metrics.get("skewness", {})
    kurt = metrics.get("kurtosis", {})
    num_peaks = metrics.get("num_peaks", {}) if isinstance(metrics, dict) else {}
    med_p95 = metrics.get("phase_medians_p95", {}) if isinstance(metrics, dict) else {}

    features = {
        # FA-KPIs
        "fa_phase_width": fa.get("phase_width_deg"),
        "fa_phase_center": fa.get("phase_center_deg"),
        "fa_symmetry": fa.get("symmetry_index"),
        "fa_max_amp": fa.get("max_amplitude"),
        "fa_p95_amp": fa.get("p95_amplitude") or fa.get("p95_amp"),
        "fa_concentration": fa.get("ang_amp_concentration_index"),

        # Hist/metrics avanzados
        "skew_pos": skew.get("pos_skew") if isinstance(skew, dict) else skew if isinstance(skew, (int, float)) else None,
        "skew_neg": skew.get("neg_skew") if isinstance(skew, dict) else None,
        "kurt_pos": kurt.get("pos_kurt") if isinstance(kurt, dict) else None,
        "kurt_neg": kurt.get("neg_kurt") if isinstance(kurt, dict) else None,
        "num_peaks_pos": num_peaks.get("pos") if isinstance(num_peaks, dict) else metrics.get("n_peaks_pos"),
        "num_peaks_neg": num_peaks.get("neg") if isinstance(num_peaks, dict) else metrics.get("n_peaks_neg"),
        "phase_corr": metrics.get("phase_corr"),
        "median_pos_phase": med_p95.get("median_pos_phase"),
        "median_neg_phase": med_p95.get("median_neg_phase"),
        "p95_global_amp": med_p95.get("p95_amp"),

        # Severidad base
        "n_angpd_angpd_ratio": kpis.get("n_ang_ratio"),
        "total_pulses": metrics.get("total_count") or fa.get("total_pulses") or kpis.get("total_pulses"),

        # Gap-time (según nombres disponibles)
        "gap_p50_ms": gap.get("p50_ms") or gap.get("P50_ms"),
        "gap_p5_ms": gap.get("p5_ms") or gap.get("P5_ms"),
        "gap_p50_level": gap.get("P50_level") or gap.get("level_name"),
        "gap_p5_level": gap.get("P5_level"),
    }
    return features


def _safe(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def rule_based_scores(features: dict) -> dict:
    scores = {cls: 0.0 for cls in PD_CLASSES}

    fw = _safe(features.get("fa_phase_width"))
    sym = _safe(features.get("fa_symmetry"))
    conc = _safe(features.get("fa_concentration"))
    max_amp = _safe(features.get("fa_max_amp"))
    p95_amp = _safe(features.get("fa_p95_amp")) or _safe(features.get("p95_global_amp"))
    ratio = _safe(features.get("n_angpd_angpd_ratio"))
    n_peaks_pos = _safe(features.get("num_peaks_pos"))
    n_peaks_neg = _safe(features.get("num_peaks_neg"))
    skew_pos = _safe(features.get("skew_pos"))
    skew_neg = _safe(features.get("skew_neg"))
    kurt_pos = _safe(features.get("kurt_pos"))
    kurt_neg = _safe(features.get("kurt_neg"))
    corr = _safe(features.get("phase_corr"))
    pulses = _safe(features.get("total_pulses"))
    gap_p50 = _safe(features.get("gap_p50_ms"))
    gap_p5 = _safe(features.get("gap_p5_ms"))

    # Cavidad interna
    if fw and fw < 90:
        scores["cavidad_interna"] += 1.0
    if sym > 0.85 and corr > 0.8:
        scores["cavidad_interna"] += 1.0
    if conc > 1.3:
        scores["cavidad_interna"] += 0.5

    # Superficial / tracking
    if 90 <= fw <= 180:
        scores["superficial_tracking"] += 1.0
    if sym > 0.75 and corr > 0.6:
        scores["superficial_tracking"] += 0.5
    if p95_amp and max_amp and p95_amp > 0.3 * max_amp:
        scores["superficial_tracking"] += 0.5

    # Corona (unipolaridad, pocos picos, amplitud moderada)
    peaks_total = (n_peaks_pos or 0) + (n_peaks_neg or 0)
    # usar skew para detectar asimetría fuerte
    if abs(skew_pos) > 1.0 or abs(skew_neg) > 1.0:
        scores["corona"] += 0.5
    if peaks_total <= 2 and (p95_amp < 0.4 * max_amp if max_amp else True):
        scores["corona"] += 1.0
    if fw < 140 and sym < 0.7:
        scores["corona"] += 0.5

    # Flotante (muy disperso, varios picos, kurtosis baja)
    if fw > 180:
        scores["flotante"] += 1.0
    if peaks_total >= 3:
        scores["flotante"] += 0.5
    if (kurt_pos and kurt_pos < 2.5) or (kurt_neg and kurt_neg < 2.5):
        scores["flotante"] += 0.5

    # Ruido / baja severidad
    if pulses < 200 and (p95_amp < 0.2 * max_amp if max_amp else True):
        scores["ruido_baja"] += 1.0
    if ratio < 3.0 and conc < 1.1 and fw > 120:
        scores["ruido_baja"] += 0.5
    if gap_p50 > 20 and gap_p5 > 10:
        scores["ruido_baja"] += 0.5

    total = sum(v for v in scores.values() if v > 0)
    if total <= 0:
        scores["ruido_baja"] = 1.0
        total = 1.0
    probs = {cls: scores[cls] / total for cls in PD_CLASSES}
    return probs


def infer_pd_summary(features: dict, probs: dict) -> dict:
    class_id = max(probs, key=probs.get)
    class_label = PD_LABELS.get(class_id, class_id)

    gap_p50 = _safe(features.get("gap_p50_ms"))
    gap_p5 = _safe(features.get("gap_p5_ms"))
    ratio = _safe(features.get("n_angpd_angpd_ratio"))
    pulses = _safe(features.get("total_pulses"))

    if gap_p5 < 3.0:
        stage = "avanzada"
        risk = "alto"
    elif 3.0 <= gap_p5 < 7.0:
        stage = "en desarrollo"
        risk = "medio"
    else:
        stage = "incipiente"
        risk = "bajo"
    if ratio > 15 or pulses > 2000:
        if risk == "medio":
            risk = "alto"
        elif risk == "bajo":
            risk = "medio"

    if class_id == "superficial_tracking":
        location = "Superficie de aislamiento e interfaces"
    elif class_id == "cavidad_interna":
        location = "Interior de devanados y canales de aceite"
    elif class_id == "corona":
        location = "Puntas y aristas de conductores / boquillas"
    elif class_id == "flotante":
        location = "Conexiones flojas / elementos flotantes"
    else:
        location = "Sin indicios claros de defecto localizado"

    explanation = [
        f"Clase dominante: {class_label} (prob={probs.get(class_id,0):.2f}).",
        f"Etapa estimada: {stage}. Riesgo: {risk}.",
        f"Gap-time P50={gap_p50:.1f} ms, P5={gap_p5:.1f} ms.",
        f"Relación N-ANGPD/ANGPD≈{ratio:.1f}, pulsos útiles≈{pulses}.",
        f"Ubicación probable: {location}.",
    ]

    return {
        "class_id": class_id,
        "class_label": class_label,
        "class_probs": probs,
        "stage": stage,
        "risk_level": risk,
        "location_hint": location,
        "explanation": explanation,
    }

