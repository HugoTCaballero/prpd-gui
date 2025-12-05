from __future__ import annotations

from typing import Any, Dict, Optional

# Matriz de decision centrada en transformadores sumergidos (aceite mineral o vegetal)
REGLAS_CONCLUSION: Dict[str, Dict[str, str]] = {
    "Descarga Superficial": {
        "stage": "Etapa 1",
        "fa_range": "FA < 5",
        "severity": "Baja",
        "lifetime_band": ">85%",
        "actions": "Monitoreo semestral. Revisar puntos accesibles y aisladores externos.",
    },
    "Descarga Corona": {
        "stage": "Etapa 2",
        "fa_range": "5 < FA < 15",
        "severity": "Media",
        "lifetime_band": "60-85%",
        "actions": "Medicion mensual. Inspeccion visual de bornes, ventilacion y descarga externa.",
    },
    "Descarga Interna": {
        "stage": "Etapa 3",
        "fa_range": "FA > 15",
        "severity": "Alta",
        "lifetime_band": "<60%",
        "actions": "Inspeccion urgente interna. Corroborar con DGA, termografia y ultrasonido. Considerar paro programado.",
    },
    "Descarga Cavidad": {
        "stage": "Etapa 2-3",
        "fa_range": "10 < FA < 20",
        "severity": "Alta",
        "lifetime_band": "50-70%",
        "actions": "Confirmar con nube S3. Analizar historico de gas y humedad. Revisar sellos y conexion de potencia.",
    },
    "Descarga Compleja": {
        "stage": "Etapas mixtas",
        "fa_range": "FA > 25",
        "severity": "Critica",
        "lifetime_band": "<45%",
        "actions": "Priorizar estudio completo. Definir ventana de reemplazo o reconexion parcial.",
    },
}


def interpreta_lifetime_score(score: Any) -> str:
    """Convierte el score numerico a una frase autoexplicativa."""
    try:
        val = float(score)
    except Exception:
        return "Vida util no disponible."
    if val > 85:
        return "El activo se encuentra en condiciones optimas de operacion."
    if val > 60:
        return "El activo muestra senales de envejecimiento leve."
    if val > 45:
        return "El activo presenta desgaste importante, con riesgo operativo."
    return "El activo se encuentra en una condicion critica, con alto riesgo de falla."


def _normalize_final_class(value: Any) -> Optional[str]:
    """Mapea etiquetas libres a las claves de REGLAS_CONCLUSION."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if "complej" in text or "mixt" in text:
        return "Descarga Compleja"
    if "corona" in text:
        return "Descarga Corona"
    if "superfic" in text or "track" in text:
        return "Descarga Superficial"
    if "cav" in text or "void" in text:
        return "Descarga Cavidad"
    if "intern" in text:
        return "Descarga Interna"
    return None


def _band_from_score(score: Any) -> Optional[str]:
    try:
        val = float(score)
    except Exception:
        return None
    if val > 85:
        return ">85%"
    if val > 60:
        return "60-85%"
    if val > 45:
        return "50-70%"
    return "<45%"


def _pick_first(mapping: Dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return None


def build_conclusion_block(
    result: Dict[str, Any] | None,
    summary: Dict[str, Any] | None = None,
    *,
    visual_extended: bool = False,
) -> Dict[str, Any]:
    """Construye el bloque de conclusiones para activos sumergidos en aceite."""
    res = result or {}
    summary = summary or {}
    rule_pd = res.get("rule_pd", {}) if isinstance(res, dict) else {}
    analysis = res.get("analysis", {}) if isinstance(res, dict) and isinstance(res.get("analysis"), dict) else {}
    kpis = res.get("kpis", {}) if isinstance(res, dict) else {}
    fa_kpis = res.get("fa_kpis", {}) if isinstance(res, dict) else {}
    metrics_adv = res.get("metrics_advanced", {}) if isinstance(res, dict) else {}

    final_raw = analysis.get("final_class") or res.get("final_class") or rule_pd.get("class_label") or summary.get("pd_type") or res.get("predicted")
    dominant = _normalize_final_class(final_raw) or _normalize_final_class(rule_pd.get("class_id")) or _normalize_final_class(summary.get("pd_type"))
    if not dominant:
        dominant = final_raw or "No clasificada"

    reglas = REGLAS_CONCLUSION.get(dominant) if isinstance(dominant, str) else None

    lifetime_score = _pick_first(res, ["lifetime_score"]) or rule_pd.get("lifetime_score") or summary.get("life_score")
    risk_level = (reglas or {}).get("severity") or rule_pd.get("risk_level") or summary.get("risk") or "No definido"
    rule_stage = (reglas or {}).get("stage") or rule_pd.get("stage") or summary.get("stage") or "No definida"
    lifetime_band = (reglas or {}).get("lifetime_band") or rule_pd.get("lifetime_band") or _band_from_score(lifetime_score) or "No definida"
    actions = (reglas or {}).get("actions") or rule_pd.get("actions") or summary.get("actions")
    if isinstance(actions, list):
        actions = " ".join(str(a) for a in actions if a)
    if not actions:
        actions = "No definido"

    location_hint = res.get("location_hint") or rule_pd.get("location_hint") or summary.get("location") or "No disponible"
    evolution_stage = res.get("cluster_stage") or res.get("prpd_stage") or summary.get("stage") or rule_stage

    fa_value = None
    if isinstance(kpis, dict):
        fa_value = _pick_first(kpis, ["fa", "fa_concentration_index", "fa_phase_width_deg", "fa_p95_amplitude"])
    if fa_value is None and isinstance(fa_kpis, dict):
        fa_value = _pick_first(fa_kpis, ["ang_amp_concentration_index", "phase_width_deg", "p95_amplitude"])

    conclusion = {
        "asset_type": "oil_immersed",
        "dominant_discharge": dominant,
        "risk_level": risk_level,
        "rule_pd_stage": rule_stage,
        "lifetime_score_band": lifetime_band,
        "lifetime_score_text": interpreta_lifetime_score(lifetime_score),
        "actions": actions,
        "location_hint": location_hint or "No disponible",
        "fa_value": fa_value,
        "evolution_stage": evolution_stage,
        "lifetime_score": lifetime_score,
    }

    if visual_extended:
        images = {
            "s3": res.get("s3") or res.get("clouds_s3"),
            "angpd": res.get("angpd"),
            "histograms": metrics_adv.get("hist") if isinstance(metrics_adv, dict) else None,
        }
        if any(v is not None for v in images.values()):
            conclusion["images"] = images

    return conclusion
