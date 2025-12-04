from __future__ import annotations

import numpy as np

from PRPDapp.pd_rules_config import (
    RULESET_VERSION,
    TH_FA_WIDTH_CAVIDAD_MAX,
    TH_SYM_CAVIDAD_MIN,
    TH_CORR_CAVIDAD_MIN,
    TH_CONC_CAVIDAD_MIN,
    TH_FA_WIDTH_SUP_MIN,
    TH_FA_WIDTH_SUP_MAX,
    TH_SYM_SUP_MIN,
    TH_CORR_SUP_MIN,
    TH_P95_REL_SUP_MIN,
    TH_PULSES_RATIO_CORONA,
    TH_P95_REL_CORONA_MAX,
    TH_PEAKS_TOTAL_CORONA,
    TH_SKEW_CORONA_MIN,
    TH_FA_WIDTH_CORONA_MAX,
    TH_SYM_CORONA_MAX,
    TH_FA_WIDTH_FLOTANTE_MIN,
    TH_PEAKS_FLOTANTE_MIN,
    TH_KURT_FLOTANTE_MAX,
    TH_PULSES_RUIDO_MAX,
    TH_P95_REL_RUIDO_MAX,
    TH_RATIO_RUIDO_MAX,
    TH_CONC_RUIDO_MAX,
    TH_GAP_P50_RUIDO_MIN,
    TH_GAP_P5_RUIDO_MIN,
    TH_GAP_P5_AVANZADA_MAX,
    TH_GAP_P5_DESARROLLO_MAX,
    TH_RATIO_NANGPD_RIESGO_ALTO,
    TH_PULSES_RIESGO_ALTO,
)

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
        "pulses_ratio": metrics.get("pulses_ratio"),

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
    pulses_ratio = _safe(features.get("pulses_ratio"), default=1.0)
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
    if fw and fw < TH_FA_WIDTH_CAVIDAD_MAX:
        scores["cavidad_interna"] += 1.0
    if sym > TH_SYM_CAVIDAD_MIN and corr > TH_CORR_CAVIDAD_MIN:
        scores["cavidad_interna"] += 1.0
    if conc > TH_CONC_CAVIDAD_MIN:
        scores["cavidad_interna"] += 0.5

    # Superficial / tracking
    if TH_FA_WIDTH_SUP_MIN <= fw <= TH_FA_WIDTH_SUP_MAX:
        scores["superficial_tracking"] += 1.0
    if sym > TH_SYM_SUP_MIN and corr > TH_CORR_SUP_MIN:
        scores["superficial_tracking"] += 0.5
    if p95_amp and max_amp and p95_amp > TH_P95_REL_SUP_MIN * max_amp:
        scores["superficial_tracking"] += 0.5

    # Corona (unipolaridad, pocos picos, amplitud moderada)
    peaks_total = (n_peaks_pos or 0) + (n_peaks_neg or 0)
    if pulses_ratio > TH_PULSES_RATIO_CORONA or pulses_ratio < 1.0 / max(TH_PULSES_RATIO_CORONA, 1e-6):
        scores["corona"] += 1.0
    if abs(skew_pos) > TH_SKEW_CORONA_MIN or abs(skew_neg) > TH_SKEW_CORONA_MIN:
        scores["corona"] += 0.5
    if peaks_total <= TH_PEAKS_TOTAL_CORONA and (p95_amp < TH_P95_REL_CORONA_MAX * max_amp if max_amp else True):
        scores["corona"] += 1.0
    if fw < TH_FA_WIDTH_CORONA_MAX and sym < TH_SYM_CORONA_MAX:
        scores["corona"] += 0.5

    # Flotante (muy disperso, varios picos, kurtosis baja)
    if fw > TH_FA_WIDTH_FLOTANTE_MIN:
        scores["flotante"] += 1.0
    if peaks_total >= TH_PEAKS_FLOTANTE_MIN:
        scores["flotante"] += 0.5
    if (kurt_pos and kurt_pos < TH_KURT_FLOTANTE_MAX) or (kurt_neg and kurt_neg < TH_KURT_FLOTANTE_MAX):
        scores["flotante"] += 0.5

    # Ruido / baja severidad
    if pulses < TH_PULSES_RUIDO_MAX and (p95_amp < TH_P95_REL_RUIDO_MAX * max_amp if max_amp else True):
        scores["ruido_baja"] += 1.0
    if ratio < TH_RATIO_RUIDO_MAX and conc < TH_CONC_RUIDO_MAX and fw > TH_FA_WIDTH_SUP_MAX:
        scores["ruido_baja"] += 0.5
    if gap_p50 > TH_GAP_P50_RUIDO_MIN and gap_p5 > TH_GAP_P5_RUIDO_MIN:
        scores["ruido_baja"] += 0.5

    total = sum(v for v in scores.values() if v > 0)
    if total <= 0:
        scores["ruido_baja"] = 1.0
        total = 1.0
    probs = {cls: scores[cls] / total for cls in PD_CLASSES}
    return probs


def infer_pd_summary(features: dict, probs: dict) -> dict:
    """Construye el resumen legible para la GUI a partir de las probabilidades."""
    class_id = max(probs, key=probs.get)
    class_label = PD_LABELS.get(class_id, class_id)

    gap_p50 = _safe(features.get("gap_p50_ms"))
    gap_p5 = _safe(features.get("gap_p5_ms"))
    ratio = _safe(features.get("n_angpd_angpd_ratio"))
    pulses = _safe(features.get("total_pulses"))

    if gap_p5 < TH_GAP_P5_AVANZADA_MAX:
        stage = "avanzada"
        risk = "alto"
    elif TH_GAP_P5_AVANZADA_MAX <= gap_p5 < TH_GAP_P5_DESARROLLO_MAX:
        stage = "en desarrollo"
        risk = "medio"
    else:
        stage = "incipiente"
        risk = "bajo"

    if ratio > TH_RATIO_NANGPD_RIESGO_ALTO or pulses > TH_PULSES_RIESGO_ALTO:
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
        f"Clase dominante: {class_label} (prob={probs.get(class_id, 0.0):.2f}).",
        f"Etapa estimada: {stage}. Riesgo: {risk}.",
        f"Gap-time P50={gap_p50:.1f} ms, P5={gap_p5:.1f} ms.",
        f"Relación N-ANGPD/ANGPD≈{ratio:.1f}, pulsos útiles≈{pulses:.0f}.",
        f"Ubicación probable: {location}.",
    ]

    return {
        "class_id": class_id,
        "class_label": class_label,
        "class_probs": probs,
        "classes": list(probs.keys()),
        "ruleset_version": RULESET_VERSION,
        "stage": stage,
        "risk_level": risk,
        "location_hint": location,
        "explanation": explanation,
    }
