# Auditoría rápida — ANN / KPI / Conclusiones (2025-12-19)

## 0) Cambio clave: pipeline unico (compute_all) + ANN 4 clases + layout alineado

- Fuente unica de verdad: `PRPDapp/pipeline.py:23` centraliza `compute_all` y reemplaza computos en vistas; GUI orquesta en `PRPDapp/main.py:2256`.
- KPI avanzados con mini-hist ampliado y layout mas limpio: `PRPDapp/main.py:3110`.
- Acciones en Conclusiones con chips y mejor legibilidad: `PRPDapp/ui_render.py:560` + `PRPDapp/main.py:4950`.
- Exportar todo incluye KPI avanzados, FA profile, ANGPD avanzado y ANN+Gap: `PRPDapp/main.py:4770`.
- PDF export actualizado a informe multipagina con Nubes y KPI avanzados: `PRPDapp/report.py:1`.
- Ayuda/README abre PDF y README PDF regenerado desde ayuda completa: `PRPDapp/main.py:2245` + `PRPDapp/PRPD ? GUI Unificada (README).pdf`.
- Export conclusiones JSON serializable: `PRPDapp/main.py:5225` usa default para numpy arrays.
- Mascara aplicada antes del procesamiento: `PRPDapp/pipeline.py:95` filtra `raw_data` y todo el flujo usa PRPD enmascarado (plots/KPI/ANN/severidad).
- ANN a 5 clases (Corona/Superficial/Tracking/Cavidad/Flotante/Ruido) + mixto/empates + validacion: `PRPDapp/pipeline.py:16` (probs filtradas/renormalizadas, `ann.valid`, regla Mixto 2%).
- ANN evita empates 3-vias: `PRPDapp/pipeline.py:226` marca invalid si top1-top3 <= 2% y cae a heuristic_top (sin 33.3% ficticio).
- Display ANN y texto dominante desde resultado: `PRPDapp/main.py:3833` y `PRPDapp/main.py:3846` consumen `ann.display` (sin recalcular).
- KPI consolidado con n_ang_ratio + gap: `PRPDapp/pipeline.py:368`.
- Layout consistente y sin empujes: reanclaje GridSpec `PRPDapp/main.py:3564`, colorbar con Ax dedicado `PRPDapp/main.py:783` y `PRPDapp/main.py:1025`, reset en ANN/GAP `PRPDapp/ui_render.py:661`, mini-hist legible en KPI avanzados `PRPDapp/main.py:3112`.
- Mascara fija por grados con offset: `PRPDapp/pipeline.py:31` aplica mascara en fase alineada (no se desplaza al cambiar 0/120/240).
- Mascara Void mantiene rangos y no corta lineas: `PRPDapp/pipeline.py:34` aplica la mascara solo como vista y conserva clustering; `PRPDapp/main.py:602` restaura rangos.
- ANN UI fuerza fallback si empates o ann.valid=False: `PRPDapp/main.py:3896` evita 33.33/33.33/33.33 en pantalla.
- Severidad basada en P50/P5 y vida en tiempo: `PRPDapp/pd_rules.py:171` y `PRPDapp/severity_oil.py:19` corrigen coherencia con gap-time.
- Acciones y textos sin truncar; se elimina badge LifeTime redundante: `PRPDapp/ui_render.py:560`.
- S2 Strong con fallback si recorta demasiado: `PRPDapp/prpd_core.py:950`.
- ANN incluye Ruido en display (toggle oculta clases en UI): `PRPDapp/pipeline.py:15` + `PRPDapp/main.py:3890`.
- Gap-time extenso total se usa en conclusiones: `PRPDapp/main.py:2297` + `PRPDapp/conclusion_rules.py:96`.
- Cola critica (P5) visible en Conclusiones y texto de vida remanente restaurado: `PRPDapp/ui_render.py:640`.
- Mascara Corona fuerza ANN=100% Corona: `PRPDapp/pipeline.py:352` + `PRPDapp/main.py:2235`.
- ANGPD combinado/nubes responden a Pixel/Qty: recálculo de ANGPD/ANGPD qty + ang_proj + FA profile en `PRPDapp/pipeline.py:271`.

## 1) ?De donde vienen los KPIs de **Conclusiones**?

- `PRPDapp/pipeline.py:23` (compute_all) calcula:
  - KPIs base desde `PRPDapp/logic.py::compute_pd_metrics` -> `result["metrics"]`.
  - KPIs avanzados desde `PRPDapp/metrics/advanced_kpi.py::compute_advanced_metrics` -> `result["metrics_advanced"]`.
  - Bloque de conclusiones desde `PRPDapp/conclusion_rules.py::build_conclusion_block` -> `result["conclusion_block"]`.
- Render: `PRPDapp/ui_render.py::render_conclusions` consume `payload["metrics"]`, `payload["metrics_advanced"]` y `payload["conclusion_block"]`.
- `PRPDapp/main.py::_get_conclusion_insight` solo formatea texto con datos ya calculados.

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
