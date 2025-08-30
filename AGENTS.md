1) AGENTS.md (adaptado a PRPD con flags exactas y rutas out)
# AGENTS.md — Proyecto PRPD (Win11 + Python 3.13)

## Overview
Pipeline B3–B6 + UI (`interface.py`) para análisis PRPD.  
**Estado de verdad (único):** tras subclustering → pairing semicíclico (hard-thresholds) → política “Otras” → consolidación post-pair ⇒ `items_final`.  
Con `items_final` se recalcula **p_global** y se reescriben **todas** las salidas (CSV/figuras/HTML).  
**Compatibilidad estricta con B6:** no cambiar el **esquema** de `summary.csv`.

## Estructura esperada
bloque1.py
bloque2.py
bloque3.py
bloque4.py
bloque5.py
bloque6.py
interface.py
pairing_utils.py
Run_all.ps1
data\ # XML (opcional)
out\ # salidas (se crea en runtime)
AGENTS.md
README.md
tests\run_profiles.ps1
tools\validate_outputs.py


## Entorno
- Windows 11 + **Python 3.13**.
- Sin dependencias nuevas obligatorias. **SciPy opcional** (si está, usar Hungarian; si no, fallback **greedy**).
- Soportar rutas con `\` y comillas dobles.

## Flags obligatorios (CLI/UI)
**Pairing**
- `--pair-max-phase-deg`
- `--pair-max-y-ks`
- `--pair-min-weight-ratio`
- `--pair-miss-penalty`
- `--pair-y-mode {abs|scaled|auto}` (default `abs`)
- `--pair-enforce-same-k` (efectivo solo si ambos items traen `k_label`)
- `--pair-hard-thresholds` (pre-filtrado de aristas fuera de tolerancia)
- `--pairs-show-lines` (solo visual; no altera cálculo ni CSV)

**Subclustering / Otras**
- `--sub-min-pct`
- `--allow-otras`
- `--otras-min-score`
- `--otras-cap`
- `--otras-to-noise` (si existe en el repo; si no, ignorar)

**Summary**
- `--summary-mode {pre|postpair|auto}` (default `auto`)

**B4 / UI**
- `bloque4.py` acepta **XML posicional**.
- `interface.py` y `Run_all.ps1` deben **propagar** estos flags B3–B6.

## Contratos de I/O (NO romper)
- `summary.csv` (esquema **inmutable**; B6 lo consume tal cual).
- `p_multiplicity.csv`: `type,count_sources,p_sources` (+ opcionales `weight_sum,p_weighted`).
- `paired_sources.csv`: `pair_id,id_pos,phi_pos,y_pos,w_pos,id_neg,phi_neg,y_neg,w_neg,dphi,dy,w_ratio,type` (+ opcional `cost`).
- Si agregas nuevos archivos, usa sufijos (`*_final.csv`, `*_paired_2d.png`) sin reemplazar nombres de contrato.

**Regla de oro:** si hay pares, `summary.csv` refleja **post-pair** (en `auto|postpair`); si NO hay pares, se mantiene **pre-pair**.

## Algoritmo de pairing (duro)
- `dphi`: diferencia **semicíclica** (mínimo vs ±180°).
- `dy` según `--pair-y-mode`:
  - `abs` (default): diferencia absoluta (sin reescalado oculto)
  - `scaled`: `dy/100` (histórico, mantener si ya existe)
  - `auto`: igual a `abs` salvo que el repo defina otra lógica explícita
- `w_ratio = min/max`.
- `miss_penalty`: aplicar **una sola vez** por nodo no emparejado.
- `--pair-hard-thresholds`: aristas que violen umbrales se excluyen (o coste ∞) **antes** de resolver.
- Resolver con **Hungarian** si SciPy está disponible; en caso contrario **greedy**.
- Ejecutar pairing **una sola vez** y **después** de `subclusters_valid`.

## Consolidación post-pair
- Re-etiquetar o reclusterizar sobre representantes/centroides para obtener `k_final`.
- Reporte debe mostrar **antes/después**: `k_inicial` (elbow/silhouette) → `k_final`, con métricas.

## Reporte (HTML)
- Tablas: clusters, subclusters, pares dobles (conteos/porcentajes).
- Panel resumen por tipo con **bandas ±20%** en UI (render, **no** CSV):
  - `p_lo = max(0, p*0.8)`, `p_hi = min(1, p*1.2)`
- Figuras PRPD 2D/3D **post-pair**.
- `--pairs-show-lines` solo dibuja líneas de unión (no altera datos).

## Logging (orden en B5)
1) Parámetros de pairing + `y_mode`  
2) `Subclusters válidos: N`  
3) `Pares formados: M`  
4) `Multiplicidad post-fusión: …`  
Evitar doble bloque; mensajes en español consistentes.

## Robustez (Windows)
- Validar entradas y rutas (`out_prefix` normalizado).
- Escrituras **atómicas**; si `PermissionError` (archivo abierto en Excel), escribir `*_safe` y loggear.
- Guard clause: si `len(subclusters_valid)==0` ⇒ NO escribir `paired_*` ni `p_multiplicity`, solo log.

## QA reproducible (obligatorio)
**Datasets:** `1superficial.xml`, `1superficial2coronas.xml`, `2cavidades.xml`, `corona.xml`, `doblecavidad.xml`.  
**Perfiles:** `strict` y `lax`, con `out-prefix` distintos.

**Criterios:**
- Mismo XML strict vs lax → diferencias en `paired_sources.csv`, `p_multiplicity.csv` y, si hay pares, `summary.csv`.
- `doblecavidad` → pares bipolares y consolidación visible (`k_final ≤ k_inicial`).
- `corona/superficial` → no generar cavidades espurias.

## Comandos de referencia (Windows, rutas **out** explícitas)
**Strict (doblecavidad):**
python bloque5.py data\doblecavidad.xml --sensor auto --tipo-tx seco --subclusters --sub-min-pct 0.02 ^
--pair-max-phase-deg 5 --pair-max-y-ks 0.05 --pair-y-mode abs --pair-min-weight-ratio 0.9 ^
--pair-miss-penalty 0.8 --pair-hard-thresholds --summary-mode auto ^
--out-prefix out\doblecavidad_strict


**Lax (doblecavidad):**
python bloque5.py data\doblecavidad.xml --sensor auto --tipo-tx seco --subclusters --sub-min-pct 0.02 ^
--pair-max-phase-deg 90 --pair-max-y-ks 0.5 --pair-y-mode abs --pair-min-weight-ratio 0.1 ^
--pair-miss-penalty 0.0 --pair-hard-thresholds --summary-mode auto ^
--out-prefix out\doblecavidad_lax


(Repetir strict/lax para `data\2cavidades.xml`, `data\corona.xml`, `data\1superficial.xml` con `out\*_strict` y `out\*_lax`).

