from __future__ import annotations

from typing import Any

from PRPDapp import pd_rules


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _pick_first(mapping: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return None


def _band_from_score(score: float | int | None) -> str:
    if score is None:
        return "N/D"
    try:
        val = float(score)
    except Exception:
        return "N/D"
    if val >= 80:
        return "5+ anos"
    if val >= 60:
        return "1-5 anos"
    if val >= 40:
        return "1-3 anos"
    if val >= 20:
        return "meses-1 ano"
    return "<6 meses"


def _severity_level_display(severity_index_0_10: float | None) -> str:
    if severity_index_0_10 is None:
        return "N/D"
    try:
        sev = float(severity_index_0_10)
    except Exception:
        return "N/D"
    if sev >= 8.0:
        return "Crítica"
    if sev >= 6.0:
        return "Alta"
    if sev >= 3.0:
        return "Media"
    return "Baja"


def _risk_display(risk_level: str | None) -> str:
    if not isinstance(risk_level, str) or not risk_level.strip():
        return "N/D"
    r = risk_level.strip().lower()
    if r.startswith("baj"):
        return "Bajo"
    if r.startswith("med"):
        return "Medio"
    if r.startswith("alt"):
        return "Alto"
    return risk_level


def _gap_score(value_ms: float | None) -> float | None:
    if value_ms is None:
        return None
    try:
        val = float(value_ms)
    except Exception:
        return None
    if val >= 500.0:
        return 100.0
    if val > 7.0:
        return 85.0
    if val > 3.0:
        return 60.0
    return 30.0


def _stage_display(stage_code: str | None, *, has_gap: bool) -> str:
    if not stage_code or stage_code == "no_evaluada" or not has_gap:
        return "No evaluada"
    mapping = {"incipiente": "Etapa 1", "en_desarrollo": "Etapa 2", "avanzada": "Etapa 3"}
    return mapping.get(stage_code, "No evaluada")


def _map_ann_to_rule_class(class_name: str | None) -> str:
    if not isinstance(class_name, str):
        return "ruido_baja"
    name = class_name.strip().lower()
    if not name:
        return "ruido_baja"
    if "super" in name or "track" in name:
        return "superficial_tracking"
    if "cav" in name or "void" in name or "intern" in name:
        return "cavidad_interna"
    if "corona" in name:
        return "corona"
    if "flot" in name:
        return "flotante"
    if "ruido" in name or "noise" in name or "indeterm" in name:
        return "ruido_baja"
    if "suspend" in name:
        return "ruido_baja"
    return "ruido_baja"


def compute_severity_oil(
    kpis: dict[str, Any] | None,
    gap: dict[str, Any] | None,
    ann: dict[str, Any] | dict[str, float] | None,
    ruleset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Severidad por KPI para transformadores sumergidos en aceite (mineral/vegetal).

    Devuelve una estructura estable para la GUI:
    - severity_index: 0..10
    - severity_level: Baja/Media/Alta/Crítica
    - risk_level: Bajo/Medio/Alto
    - lifetime_score: 0..100
    - lifetime_band: ">85%", "60-85%", ...
    - stage: Etapa 1/2/3 o No evaluada
    - actions: lista de acciones recomendadas
    """
    kpis = kpis or {}
    gap = gap or {}
    ruleset = ruleset or {}

    metrics = kpis.get("metrics") if isinstance(kpis.get("metrics"), dict) else {}
    metrics_adv = kpis.get("metrics_advanced") if isinstance(kpis.get("metrics_advanced"), dict) else {}
    fa_kpis = kpis.get("fa_kpis") if isinstance(kpis.get("fa_kpis"), dict) else {}
    kpis_block = kpis.get("kpis") if isinstance(kpis.get("kpis"), dict) else {}

    gap_p50 = _pick_first(gap, ["p50_ms", "P50_ms"]) or _pick_first(metrics, ["gap_p50"])
    gap_p5 = _pick_first(gap, ["p5_ms", "P5_ms"]) or _pick_first(metrics, ["gap_p5"])
    gap_p50 = _to_float(gap_p50)
    gap_p5 = _to_float(gap_p5)
    gap_primary = gap_p50 if (gap_p50 is not None and gap_p50 > 0) else gap_p5
    no_gap = gap_primary is None or gap_primary >= 500.0
    has_gap = (gap_primary is not None and gap_primary > 0 and gap_primary < 500.0)

    ratio = _pick_first(kpis_block, ["n_angpd_angpd_ratio", "n_ang_ratio"])
    if ratio is None:
        ratio = _pick_first(metrics, ["n_ang_ratio"])
    ratio = _to_float(ratio)

    total_pulses = _pick_first(metrics, ["total_count"])
    if total_pulses is None:
        total_pulses = _pick_first(fa_kpis, ["total_pulses"])
    total_pulses = _to_float(total_pulses)

    p95_amp = _pick_first(fa_kpis, ["p95_amplitude", "p95_amp"])
    if p95_amp is None:
        p95_amp = _pick_first(metrics_adv, ["phase_medians_p95"])
        if isinstance(p95_amp, dict):
            p95_amp = p95_amp.get("p95_amp")
    if p95_amp is None:
        p95_amp = _pick_first(metrics, ["p95_mean", "amp_p95_pos", "amp_p95_neg"])
    p95_amp = _to_float(p95_amp)

    features = {
        "gap_p50_ms": gap_p50,
        "gap_p5_ms": gap_p5,
        "n_angpd_angpd_ratio": ratio,
        "total_pulses": total_pulses,
        "fa_p95_amp": p95_amp,
    }

    # ----- PD dominante (ANN preferido; fallback reglas) -----
    ann_probs: dict[str, float] = {}
    ann_source = "none"
    if isinstance(ann, dict):
        if isinstance(ann.get("probs"), dict):
            try:
                ann_probs = {str(k): float(v) for k, v in ann.get("probs", {}).items() if v is not None}
                ann_source = str(ann.get("source") or "ann")
            except Exception:
                ann_probs = {}
        else:
            try:
                ann_probs = {str(k): float(v) for k, v in ann.items() if v is not None}
                ann_source = "ann"
            except Exception:
                ann_probs = {}

    if ann_probs:
        dom_ann = max(ann_probs, key=ann_probs.get)
        class_id = _map_ann_to_rule_class(dom_ann)
        class_prob = float(ann_probs.get(dom_ann, 0.0))
    else:
        ann_source = "rules"
        rule_probs = pd_rules.rule_based_scores(features)
        class_id = max(rule_probs, key=rule_probs.get)
        class_prob = float(rule_probs.get(class_id, 0.0))

    class_label = pd_rules.PD_LABELS.get(class_id, class_id)
    location_hint = pd_rules.LOCATION_HINT.get(class_id, "Ubicación no determinada")

    # ----- Etapa / severidad / riesgo -----
    stage_code = pd_rules._infer_stage_from_gap_and_energy(features)
    severity_index = pd_rules._compute_severity_index(features)
    sev_level_internal = pd_rules._severity_level_from_index(severity_index)
    risk_internal = (
        pd_rules._combine_stage_and_severity(stage_code, sev_level_internal)
        if has_gap
        else sev_level_internal
    )
    if no_gap:
        stage_code = "no_evaluada"
        severity_index = 0.0
        risk_internal = "sin descargas"
        sev_level_internal = pd_rules._severity_level_from_index(severity_index)
    elif gap_primary is not None:
        if gap_primary > 7.0:
            severity_index = min(severity_index, 2.9)
            risk_internal = "bajo"
        elif gap_primary > 3.0:
            severity_index = max(3.0, min(severity_index, 5.9))
            risk_internal = "medio"
        else:
            severity_index = max(severity_index, 6.0)
            risk_internal = "alto"
        sev_level_internal = pd_rules._severity_level_from_index(severity_index)
        if gap_p5 is not None and gap_p5 > 0 and gap_p5 <= 3.0 and risk_internal == "bajo":
            risk_internal = "medio"

    kpi_score = max(0.0, min(100.0, 100.0 - float(severity_index or 0.0) * 10.0))
    p50_score = _gap_score(gap_p50)
    p5_score = _gap_score(gap_p5)
    if no_gap:
        lifetime_score = kpi_score
    else:
        base_p50 = p50_score if p50_score is not None else kpi_score
        base_p5 = p5_score if p5_score is not None else kpi_score
        lifetime_score = 0.55 * base_p50 + 0.15 * base_p5 + 0.30 * kpi_score

    ann_is_ruido = bool(ann_probs) and class_id == "ruido_baja"
    if ann_is_ruido:
        lifetime_score = 100.0
        severity_index = 0.0
        risk_internal = "sin descargas"
        stage_code = "no_evaluada"
        sev_level_internal = pd_rules._severity_level_from_index(severity_index)

    lifetime_score = max(0.0, min(100.0, float(lifetime_score)))
    lifetime_band = _band_from_score(lifetime_score)
    lifetime_text = pd_rules._map_lifetime_band(int(round(lifetime_score)))[1]

    actions = pd_rules._recommend_actions(class_id, stage_code, sev_level_internal, risk_internal)
    stage = _stage_display(stage_code, has_gap=has_gap)
    severity_level = _severity_level_display(severity_index)
    risk_level = _risk_display(risk_internal)

    notes: list[str] = []
    if not has_gap:
        notes.append("Sin gap-time: etapa no evaluada; riesgo/vida basados solo en KPIs disponibles.")
    if ratio is None:
        notes.append("Relación N-ANGPD/ANGPD no disponible en este flujo.")
    if total_pulses is None:
        notes.append("Total de pulsos no disponible en este flujo.")
    if p95_amp is None:
        notes.append("P95 de amplitud no disponible en este flujo.")

    return {
        "class_id": class_id,
        "class_label": class_label,
        "class_prob": class_prob,
        "location_hint": location_hint,
        "stage_code": stage_code,
        "stage": stage,
        "severity_index": float(severity_index) if severity_index is not None else None,
        "severity_level": severity_level,
        "risk_level": risk_level,
        "lifetime_score": int(lifetime_score) if lifetime_score is not None else None,
        "lifetime_band": lifetime_band,
        "lifetime_text": lifetime_text,
        "actions": actions,
        "notes": notes,
        "inputs": {
            "gap_p50_ms": gap_p50,
            "gap_p5_ms": gap_p5,
            "n_ang_ratio": ratio,
            "total_pulses": total_pulses,
            "p95_amp": p95_amp,
        },
        "ruleset": {"version": getattr(pd_rules, "RULESET_VERSION", None), **ruleset},
        "ann_source": ann_source,
    }
