from __future__ import annotations

import numpy as np

from PRPDapp.pd_rules_config import (
    RULESET_VERSION,
    TH_RATIO_RIESGO_ALTO,
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

LOCATION_HINT = {
    "superficial_tracking": "Superficie de aislamiento e interfaces (pantallas, soportes, conexiones superficiales).",
    "cavidad_interna": "Interior de devanados, canales de aceite y zonas encapsuladas del aislamiento.",
    "corona": "Puntas, aristas de conductores, boquillas y elementos expuestos a campo intenso en aire.",
    "flotante": "Conexiones flojas, partes no aterrizadas o elementos metálicos flotantes dentro del tanque.",
    "ruido_baja": "Sin indicios claros de defecto localizado (ruido / actividad de muy baja severidad).",
}


def build_rule_features(result: dict) -> dict:
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    fa = result.get("fa_kpis", {}) if isinstance(result, dict) else {}
    # Fuente preferida: kpis consolidados si existen
    kpis_consolidated = result.get("kpis", {}) if isinstance(result, dict) else {}
    kpis = kpis_consolidated if kpis_consolidated else result.get("kpi", {}) if isinstance(result, dict) else {}
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


def _infer_stage_from_gap_and_energy(features: dict) -> str:
    gap_p5 = _safe(features.get("gap_p5_ms"))
    ratio = _safe(features.get("n_angpd_angpd_ratio"))
    pulses = _safe(features.get("total_pulses"))
    if gap_p5 <= 0:
        stage = "no_evaluada"
    elif gap_p5 < TH_GAP_P5_AVANZADA_MAX:
        stage = "avanzada"
    elif gap_p5 < TH_GAP_P5_DESARROLLO_MAX:
        stage = "en_desarrollo"
    else:
        stage = "incipiente"
    if stage == "incipiente" and ratio > TH_RATIO_RIESGO_ALTO and pulses > TH_PULSES_RIESGO_ALTO:
        stage = "en_desarrollo"
    elif stage == "en_desarrollo" and ratio > (TH_RATIO_RIESGO_ALTO * 1.5) and pulses > (TH_PULSES_RIESGO_ALTO * 2):
        stage = "avanzada"
    return stage


def _compute_severity_index(features: dict) -> float:
    gap_p5 = _safe(features.get("gap_p5_ms"))
    ratio = _safe(features.get("n_angpd_angpd_ratio"))
    pulses = _safe(features.get("total_pulses"))
    p95 = _safe(features.get("fa_p95_amp") or features.get("p95_global_amp"))
    # Normalizaciones
    s_gap = 0.0
    if gap_p5 > 0:
        s_gap = max(0.0, min(1.0, (15.0 - gap_p5) / 15.0))
    s_ratio = max(0.0, min(1.0, ratio / 30.0))
    s_pulses = max(0.0, min(1.0, pulses / 5000.0))
    s_p95 = max(0.0, min(1.0, p95))
    w_gap, w_ratio, w_pulses, w_p95 = 0.4, 0.3, 0.2, 0.1
    s = (w_gap * s_gap + w_ratio * s_ratio + w_pulses * s_pulses + w_p95 * s_p95) / (w_gap + w_ratio + w_pulses + w_p95)
    return float(10.0 * s)


def _severity_level_from_index(sev_idx: float) -> str:
    if sev_idx < 3.0:
        return "bajo"
    elif sev_idx < 6.0:
        return "medio"
    return "alto"


def _combine_stage_and_severity(stage: str, severity: str) -> str:
    if stage == "incipiente":
        return "bajo" if severity == "bajo" else "medio"
    if stage == "en_desarrollo":
        return "alto" if severity == "alto" else "medio"
    if stage == "avanzada":
        return "medio" if severity == "bajo" else "alto"
    return "medio"


def _compute_lifetime_score(stage: str, severity_index: float) -> int:
    stage_score = {"incipiente": 1.0, "en_desarrollo": 2.0, "avanzada": 3.0}.get(stage, 2.0)
    s_stage = (stage_score - 1.0) / 2.0
    s_sev = max(0.0, min(1.0, severity_index / 10.0))
    w_sev, w_stage = 0.7, 0.3
    damage = w_sev * s_sev + w_stage * s_stage
    lt = int(round(100.0 * (1.0 - damage)))
    return max(0, min(100, lt))


def _map_lifetime_band(lifetime_score: int) -> tuple[str, str]:
    if lifetime_score >= 80:
        return "L1", "Condición buena. Riesgo bajo en los próximos 5 años (si no cambian las condiciones)."
    if lifetime_score >= 60:
        return "L2", "Riesgo bajo a moderado en 1–5 años. Recomendable monitoreo periódico."
    if lifetime_score >= 40:
        return "L3", "Riesgo moderado. Conviene planear acciones en 1–3 años."
    if lifetime_score >= 20:
        return "L4", "Riesgo alto. Recomendada intervención en meses–1 año según criticidad."
    return "L5", "Riesgo muy alto. Posible evolución en meses o menos; revisar de forma prioritaria."


def _recommend_actions(class_id: str, stage: str, severity: str, risk: str) -> list[str]:
    actions: list[str] = []
    if risk == "bajo":
        actions.append("Mantener el monitoreo de DP en paradas y revisiones periódicas.")
    elif risk == "medio":
        actions.append("Aumentar frecuencia de monitoreo de DP y correlacionar con DGA y pruebas eléctricas.")
        actions.append("Planear inspección focalizada de la zona sospechosa en la siguiente ventana de mantenimiento.")
    elif risk == "alto":
        actions.append("Evaluar reducción de carga o restricciones mientras se investiga la causa.")
        actions.append("Programar inspección y pruebas complementarias de forma prioritaria.")
        actions.append("Considerar estrategias de mitigación o reparación según hallazgos.")

    if class_id == "superficial_tracking":
        actions.append("Revisar limpieza, humedad y posibles caminos de tracking en superficies y soportes.")
    elif class_id == "cavidad_interna":
        actions.append("Correlacionar con DGA, SFRA y pruebas de aislamiento para confirmar defectos internos.")
        actions.append("Analizar criticidad del activo ante una posible falla interna del aislamiento.")
    elif class_id == "corona":
        actions.append("Inspeccionar puntas, aristas y boquillas; revisar distancias de aislamiento y condiciones del aire.")
    elif class_id == "flotante":
        actions.append("Verificar conexiones, bornes y partes metálicas flojas o no aterrizadas.")
    elif class_id == "ruido_baja":
        actions.append("Confirmar que la señal corresponde a ruido; mantener vigilancia sin sobrerreaccionar.")
    return actions


def infer_pd_summary(features: dict, probs: dict) -> dict:
    """Construye el resumen legible para la GUI a partir de las probabilidades."""
    # Normalizar/proteger probs
    if not probs:
        probs = {cls: (1.0 if cls == "ruido_baja" else 0.0) for cls in PD_CLASSES}
    values = np.asarray([probs.get(cls, 0.0) for cls in PD_CLASSES], dtype=float)
    total = float(values.sum())
    if total <= 0:
        values = np.asarray([1.0 if cls == "ruido_baja" else 0.0 for cls in PD_CLASSES], dtype=float)
        total = float(values.sum())
    probs_norm = {cls: float(values[i] / total) for i, cls in enumerate(PD_CLASSES)}

    class_id = max(probs_norm, key=probs_norm.get)
    class_label = PD_LABELS.get(class_id, class_id)

    gap_p50 = features.get("gap_p50_ms")
    gap_p5 = features.get("gap_p5_ms")
    ratio = features.get("n_angpd_angpd_ratio")
    pulses = features.get("total_pulses")

    # Etapa y severidad
    stage = _infer_stage_from_gap_and_energy(features)
    sev_idx = _compute_severity_index(features)
    sev_level = _severity_level_from_index(sev_idx)
    risk = _combine_stage_and_severity(stage, sev_level)

    # LifeTime
    lifetime_score = _compute_lifetime_score(stage, sev_idx)
    lifetime_band, lifetime_text = _map_lifetime_band(lifetime_score)

    # Ubicación
    location = LOCATION_HINT.get(class_id, "Ubicación no determinada")

    explanation = [
        f"Clase dominante: {class_label} (prob={probs_norm.get(class_id, 0.0):.2f}).",
        f"Etapa estimada: {stage}. Severidad: {sev_level} (índice {sev_idx:.1f}/10).",
    ]
    if gap_p50 is not None and gap_p5 is not None:
        explanation.append(f"Gap-time: P50≈{gap_p50:.2f} ms, P5≈{gap_p5:.2f} ms.")
    elif gap_p5 is not None:
        explanation.append(f"Gap-time: P5≈{gap_p5:.2f} ms.")
    if ratio is not None:
        explanation.append(f"Relación N-ANGPD/ANGPD≈{ratio:.1f}.")
    if pulses is not None:
        explanation.append(f"Número de pulsos útiles≈{int(pulses)}.")
    explanation.append(f"LifeTime score: {lifetime_score}/100 ({lifetime_band}).")
    explanation.append(f"Ubicación probable: {location}.")

    actions = _recommend_actions(class_id, stage, sev_level, risk)

    return {
        "class_id": class_id,
        "class_label": class_label,
        "class_probs": probs_norm,
        "dominant_pd": class_label,
        "location_hint": location,
        "stage": stage,
        "severity_level": sev_level,
        "severity_index": float(sev_idx),
        "risk_level": risk,
        "lifetime_score": lifetime_score,
        "lifetime_band": lifetime_band,
        "lifetime_text": lifetime_text,
        "actions": actions,
        "explanation": explanation,
        "ruleset_version": RULESET_VERSION,
    }
