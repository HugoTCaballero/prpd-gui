# Auditoría rápida — ANN / KPI / Conclusiones (2025-12-19)

## 1) ¿De dónde vienen los KPIs de **Conclusiones**?

- `PRPDapp/main.py::_get_conclusion_insight` calcula:
  - KPIs base desde `PRPDapp/logic.py::compute_pd_metrics` → `payload["metrics"]` y `result["metrics"]`.
  - KPIs avanzados (skew/kurt/medianas/corr/peaks + histogram blocks) desde `PRPDapp/metrics/advanced_kpi.py::compute_advanced_metrics` → `payload["metrics_advanced"]` y `result["metrics_advanced"]`.
  - Bloque de conclusiones desde `PRPDapp/conclusion_rules.py::build_conclusion_block`.
- Render: `PRPDapp/ui_render.py::render_conclusions` consume `payload["metrics"]`, `payload["metrics_advanced"]` y `payload["conclusion_block"]`.

## 2) ¿Por qué salen muchos “N/D”?

Principales causas detectadas:

1) **Inconsistencia de llaves/fuentes**
   - `PRPDapp/pd_rules.py::build_rule_features` leía `skewness/kurtosis/phase_corr/...` desde `result["metrics"]`, pero esas llaves viven en `result["metrics_advanced"]` → las reglas pierden información.
   - `result["kpis"]` (consolidado en `PRPDapp/prpd_core.py`) no incluía algunas llaves esperadas (`n_ang_ratio`, gap-time), por lo que severidad/etapa quedaban incompletas.

2) **Flujos sin gap-time**
   - Si no se carga XML gap-time, `gap_p50_ms/gap_p5_ms` no existen → “No evaluada” o se baja confianza en etapa/riesgo.

3) **Muestras insuficientes / varianza cero**
   - Skew/kurt/corr requieren suficientes puntos y variabilidad por semiciclo; tras filtros/máscaras puede quedar muy poco → `NaN` y la UI lo muestra como “N/D”.

4) **Filtros dejan arrays vacíos**
   - Máscaras/filtros (pixel/qty/fase) pueden dejar `aligned.phase_deg`/`aligned.amplitude` vacíos o muy pequeños → métricas parciales.

## 3) ¿Por qué no aparecía kurtosis en Conclusiones?

- `compute_advanced_metrics` sí calcula `kurtosis` (`pos_kurt/neg_kurt`), pero:
  - no se renderizaba en la tabla (filas omitidas en la compactación), o
  - `metrics_advanced` no estaba presente en ese flujo (no calculado/guardado por excepción o por no pasar aligned).

## 4) ¿Por qué “se cortaba” la tabla?

- Había espaciado vertical fijo por fila en resoluciones bajas → algunas filas quedaban fuera del área visible.
- Solución: espaciado adaptativo al alto disponible (sin depender de un número fijo de filas).
