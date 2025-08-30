#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bloque 6 — Consola y Reporte Final (robusto y compatible).
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse
import json
import base64
import warnings
import csv

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def first_existing_path(*cands: str | Path) -> Path | None:
    for c in cands:
        if not c:
            continue
        p = Path(c)
        if p.exists():
            return p
    return None


def read_b4_metrics(metrics_path: Path | None) -> dict:
    """
    Lee métricas de B4 si existen. Es tolerante a nombres de columnas.
    Devuelve dict con: k_use, n_clusters_hdb, frac_ruido, silhouette.
    """
    out = {'k_use': 0, 'n_clusters_hdb': 0, 'frac_ruido': 0.0, 'silhouette': 0.0}
    if not metrics_path or not Path(metrics_path).exists():
        return out
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return out

    row = df.iloc[0]

    k_candidates = ['k', 'k_use', 'k_auto', 'k_opt', 'k_opt_auto']
    for kcol in k_candidates:
        if kcol in df.columns:
            try:
                out['k_use'] = int(row.get(kcol, 0))
                break
            except Exception:
                pass

    n_candidates = ['n_clusters_hdb_sin_ruido', 'n_clusters_hdb', 'hdbscan_clusters']
    for ncol in n_candidates:
        if ncol in df.columns:
            try:
                out['n_clusters_hdb'] = int(row.get(ncol, 0))
                break
            except Exception:
                pass

    if 'frac_ruido' in df.columns:
        try: out['frac_ruido'] = float(row.get('frac_ruido', 0.0))
        except Exception: pass

    for scol in ['silhouette', 'silhouette_k']:
        if scol in df.columns:
            try:
                out['silhouette'] = float(row.get(scol, 0.0))
                break
            except Exception:
                pass
    return out


def read_b5_summary(summary_path: Path | None) -> dict:
    """
    Lee el archivo *_summary.csv generado por Bloque 5 y extrae únicamente
    la sección de probabilidades globales (p_global) junto con algunas
    métricas adicionales.

    El archivo consiste en secciones separadas por líneas en blanco.
    La primera sección tiene encabezado ``tipo,p_global`` (o con punto y coma
    como separador) y contiene una fila por cada tipo de descarga.  No se
    intentan reconstruir valores faltantes ni promediar.

    También se extraen las probabilidades de múltiples fuentes (p_?_ge2) y la
    política de "otras" si están presentes en secciones posteriores.

    Parameters
    ----------
    summary_path : Path or None
        Ruta al archivo summary.csv.  Si es None o no existe se devuelve un
        diccionario con campos vacíos.

    Returns
    -------
    dict
        Diccionario con las claves 'p_global', 'p_ge2', 'otras_policy',
        'otras_raw_score', 'otras_min_score' y 'otras_cap'.  Las claves
        'p_global' y 'p_ge2' son diccionarios por tipo.
    """
    result = {
        'p_global': {},
        'p_ge2': {},
        'otras_policy': None,
        'otras_raw_score': None,
        'otras_min_score': None,
        'otras_cap': None,
    }
    if not summary_path or not Path(summary_path).exists():
        return result
    try:
        # Leer el archivo completo para análisis manual.  Detectar separador (coma o punto y coma).
        text = Path(summary_path).read_text(encoding='utf-8', errors='ignore').strip().splitlines()
        if not text:
            return result
        # Detectar delimitador en la primera línea no vacía
        delim = ','
        for line in text:
            if line.strip():
                if ';' in line and ',' not in line:
                    delim = ';'
                break
        # Iterar sobre líneas con csv.reader
        import csv as _csv
        # Convertir todas las líneas a listas usando el delimitador detectado
        rows: list[list[str]] = list(_csv.reader(text, delimiter=delim))
        # Extraer p_global de la primera sección
        in_global = False
        for row in rows:
            # Saltar filas vacías
            if not row or all(not c.strip() for c in row):
                # Sección p_global termina al encontrar línea en blanco
                if in_global:
                    break
                continue
            # Determinar encabezado para iniciar la sección
            # Sección p_global comienza cuando se encuentra header 'tipo' y 'p_global'
            if row[0].strip().lower() == 'tipo':
                # Normalizar encabezado
                header = [c.strip().lower() for c in row]
                # Si la segunda columna es p_global, activamos la sección
                if len(header) >= 2 and 'p_global' in header[1]:
                    in_global = True
                    continue
                else:
                    # Otro encabezado, ignorar
                    in_global = False
                    continue
            if in_global:
                # Esperamos pares tipo, valor
                if len(row) >= 2:
                    tipo = row[0].strip().lower()
                    val_str = row[1].strip()
                    if tipo:
                        try:
                            result['p_global'][tipo] = float(val_str)
                        except Exception:
                            # Si no es un número, dejar sin registrar
                            pass
        # Segundo pase: capturar p_ge2 y política de otras
        for row in rows:
            if not row or len(row) < 2:
                continue
            key = row[0].strip().lower()
            val = row[1].strip()
            # p_ge2_<tipo>
            if key.startswith('p_cavidad_ge2'):
                try:
                    result['p_ge2']['cavidad'] = float(val)
                except Exception:
                    pass
            elif key.startswith('p_superficial_ge2'):
                try:
                    result['p_ge2']['superficial'] = float(val)
                except Exception:
                    pass
            elif key.startswith('p_corona_ge2'):
                try:
                    result['p_ge2']['corona'] = float(val)
                except Exception:
                    pass
            elif key.startswith('p_flotante_ge2'):
                try:
                    result['p_ge2']['flotante'] = float(val)
                except Exception:
                    pass
            # Otras policy y valores
            elif key == 'otras_policy':
                result['otras_policy'] = val
            elif key == 'otras_raw_score':
                try:
                    result['otras_raw_score'] = float(val)
                except Exception:
                    pass
            elif key == 'otras_min_score':
                try:
                    result['otras_min_score'] = float(val)
                except Exception:
                    pass
            elif key == 'otras_cap':
                try:
                    result['otras_cap'] = float(val)
                except Exception:
                    pass
        return result
    except Exception:
        return result


def embed_image_if_needed(path: Path, embed_assets: bool) -> str:
    """Devuelve data URI si embed_assets, si no ruta de archivo (string)."""
    if not embed_assets:
        return str(path)
    try:
        data = path.read_bytes()
        ext = path.suffix.lower()
        mime = 'image/png' if ext in ('.png',) else ('image/jpeg' if ext in ('.jpg', '.jpeg') else 'application/octet-stream')
        b64 = base64.b64encode(data).decode('utf-8')
        return f"data:{mime};base64,{b64}"
    except Exception:
        return str(path.name)


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_report_html(
    xml_path: str,
    sensor: str,
    tipo_tx: str,
    phase_shift: float,
    recluster: bool,
    k_use: int,
    k_manual: int | None,
    metrics: dict,
    summary: dict,
    fig_paths: dict,
    subclusters: bool,
    multiplicity_counts: dict | None,
    out_prefix: str,
    *,
    p_ge2: dict | None = None,
    subclusters_info: list | None = None,
) -> str:
    """Construye el HTML (simple, auto-contenido)."""
    reclust_str = 'sí' if recluster else 'no'
    date_str = datetime.now().strftime('%d/%m/%Y')
    # Preparar cadena de desfase.  Si se especifica un valor manual (phase_shift
    # proviene de args.phase_align numérico), se indica "(manual)".  El
    # parámetro phase_shift puede ser cualquier float; se redondea al entero
    # más cercano para presentación.  Se utiliza la palabra clave "Delta="
    # en lugar del símbolo griego para asegurar compatibilidad en Windows.
    if isinstance(phase_shift, (int, float)):
        delta_value = int(round(phase_shift))
    else:
        try:
            delta_value = int(round(float(phase_shift)))
        except Exception:
            delta_value = 0
    manual_flag = False
    # Si se pasa una cadena o float a phase_shift con el atributo 'manual'
    # externo (establecido en process_single_xml), se agrega el sufijo
    # '(manual)'.  Para mantener retrocompatibilidad se considera que
    # cualquier valor negativo de phase_shift no es manual.
    try:
        # build_report_html puede recibir phase_shift como tuple (value, manual_flag)
        # para distinguir manual.  Si es así, desempaquetar.
        if isinstance(phase_shift, tuple) and len(phase_shift) == 2:
            delta_value, manual_flag = phase_shift
    except Exception:
        pass
    delta_display = f"Delta={delta_value} deg" + (" (manual)" if manual_flag else "")
    p_global = summary.get('p_global', {})
    otras_policy = summary.get('otras_policy')

    H = []
    H += [
        '<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Reporte PRPD</title>',
        '<style>body{font-family:Arial,sans-serif;color:#333;margin:0;padding:0}h1,h2,h3{color:#003366}',
        'table{border-collapse:collapse;width:100%;margin-bottom:1em}table,th,td{border:1px solid #ccc}',
        'th,td{padding:4px 8px;text-align:left}.fig-container{text-align:center;margin:1em 0}.fig-container img{max-width:100%;height:auto;border:1px solid #ddd}.small{font-size:.9em}</style>',
        '</head><body>'
    ]
    # Portada
    H += [
        '<div style="text-align:center;padding:40px 20px;border-bottom:2px solid #003366;">',
        f'<h1>Reporte PRPD – {Path(xml_path).name}</h1>',
        f'<p class="small">Fecha: {date_str}</p>',
        f'<p class="small">Sensor: {sensor.upper()} &nbsp;&nbsp; Tipo TX: {tipo_tx.capitalize()}</p>',
        # Mostrar desfase aplicando formato Delta=XX deg y, si es manual, añadir sufijo.
        f'<p class="small">{delta_display} &nbsp;&nbsp; Reclustering: {reclust_str}</p>',
        f'<p class="small">K_auto: {k_use}' + (f' &nbsp;&nbsp; K_manual: {k_manual}' if k_manual else '') + '</p>',
        '</div>',
    ]
    # Resumen
    H += [
        '<h2>Resumen ejecutivo</h2>',
        # Tarjetas con resumen de clusters
        '<div class="fig-container">',
        (f'<img src="{fig_paths.get("cards")}" alt="Tarjetas">' if fig_paths.get("cards") else '<em>(Sin tarjetas)</em>'),
        '</div>',
        # Mezcla global por tipo
        '<div class="fig-container">',
        (f'<img src="{fig_paths.get("global_mix")}" alt="Mezcla global">' if fig_paths.get("global_mix") else '<em>(Sin mezcla global)</em>'),
        '</div>',
        # Distribución por cluster (stack_by_cluster) si existe
        '<div class="fig-container">',
        (f'<img src="{fig_paths.get("stack")}" alt="Distribución por cluster">' if fig_paths.get("stack") else ''),
        '</div>',
        # Tabla de probabilidades globales
        '<table><tr><th>Tipo</th><th>Probabilidad</th></tr>'
    ]
    for t in ['cavidad', 'superficial', 'corona', 'flotante', 'otras', 'ruido']:
        if t == 'otras' and otras_policy == 'suppressed':
            continue
        # Mostrar "—" cuando no haya información (None).  Si el valor existe, mostrar porcentaje.
        if t in p_global:
            val = p_global.get(t)
            try:
                # Considerar NaN o infinito como valor faltante
                import math as _math
                if val is None or (_math.isnan(val) if isinstance(val, float) else False) or (_math.isinf(val) if isinstance(val, float) else False):
                    raise ValueError
                pct = val * 100
                H.append(f'<tr><td>{t.capitalize()}</td><td>{pct:.1f}%</td></tr>')
            except Exception:
                H.append(f'<tr><td>{t.capitalize()}</td><td>—</td></tr>')
        else:
            H.append(f'<tr><td>{t.capitalize()}</td><td>—</td></tr>')
    H += ['</table><hr>']

    # Diagnóstico / top clusters
    H += ['<h2>Diagnóstico detallado</h2>']
    try:
        top_csv = fig_paths.get('top_clusters_csv')
        df_top = pd.read_csv(top_csv) if top_csv else pd.DataFrame()
    except Exception:
        df_top = pd.DataFrame()
    if not df_top.empty:
        H.append('<table><tr><th>Cluster</th><th>Tipo</th><th>Prob.</th><th>Tamaño (%)</th><th>Estabilidad</th><th>Frac. ruido</th></tr>')
        for _, r in df_top.iterrows():
            H.append(f'<tr><td>{r.get("cluster_id","")}</td><td>{str(r.get("top_tipo","")).capitalize()}</td>'
                     f'<td>{float(r.get("top_prob",0))*100:.1f}%</td><td>{r.get("size_pct","")}</td>'
                     f'<td>{r.get("stability_hdb","")}</td><td>{r.get("frac_ruido_hdb","")}</td></tr>')
        H.append('</table>')

    # Multiplicidad / subclusters
    if subclusters:
        H += ['<h3>Multiplicidad de descargas</h3>']
        if multiplicity_counts:
            cav = multiplicity_counts.get('cavidad', 0)
            cor = multiplicity_counts.get('corona', 0)
            sup = multiplicity_counts.get('superficial', 0)
            H.append(f'<p>Se detectan {cav} lóbulos de cavidad, {cor} grupos de corona y {sup} bandas superficiales.</p>')
        if p_ge2:
            H.append('<p>Probabilidad de múltiples fuentes (≥2) por tipo:</p><ul>')
            for t in ['cavidad','superficial','corona','flotante']:
                if t in p_ge2 and p_ge2[t] is not None:
                    H.append(f'<li>{t.capitalize()}: {p_ge2[t]:.2f}</li>')
            H.append('</ul>')
        if fig_paths.get('multiplicity_fig'):
            H += ['<div class="fig-container">', f'<img src="{fig_paths["multiplicity_fig"]}" alt="Multiplicidad">', '</div>']

    # Validación B4
    H += ['<h2>Validación de clustering</h2>']
    if fig_paths.get('curvas_combinadas'):
        H += ['<div class="fig-container">', f'<img src="{fig_paths["curvas_combinadas"]}" alt="Curvas combinadas">', '</div>']
    else:
        H += ['<em>(Sin curvas combinadas)</em>']

    # Comparativa PRPD 2D auto/manual
    H += ['<h2>Comparativa K_auto vs K_manual</h2>']
    if fig_paths.get('prpd_auto_2d'):
        H += ['<div class="fig-container">', f'<img src="{fig_paths["prpd_auto_2d"]}" alt="PRPD auto 2D">', '</div>']
    if k_manual is not None and fig_paths.get('prpd_manual_2d'):
        H += ['<div class="fig-container">', f'<img src="{fig_paths["prpd_manual_2d"]}" alt="PRPD manual 2D">', '</div>']

    # Sección de visualizaciones 3D
    H += ['<h2>Visualizaciones 3D</h2>']
    # Enlaces a archivos interactivos Plotly. No se incrustan directamente para mantener el tamaño del reporte.
    if fig_paths.get('natural_3d_html'):
        H.append(f'<p><a href="{fig_paths["natural_3d_html"]}">PRPD Natural 3D</a></p>')
    if fig_paths.get('aligned_3d_html'):
        H.append(f'<p><a href="{fig_paths["aligned_3d_html"]}">PRPD Alineado 3D</a></p>')
    if fig_paths.get('paired_html'):
        H.append(f'<p><a href="{fig_paths["paired_html"]}">PRPD Emparejado 3D</a></p>')

    # Apéndice
    H += [
        '<h2>Apéndice técnico</h2><ul>',
        f'<li>Sensor: {sensor.upper()}</li>',
        f'<li>Tipo de transformador: {tipo_tx}</li>',
        f'<li>Fase alineada (modo visual): {delta_display}</li>',
        f'<li>K automático (B4): {k_use}</li>',
        (f'<li>K manual: {k_manual}</li>' if k_manual is not None else ''),
        f'<li># clusters HDBSCAN (sin ruido): {metrics.get("n_clusters_hdb","N/A")}</li>',
        f'<li>Fracción de ruido: {metrics.get("frac_ruido",0.0)*100:.1f}%</li>',
        f'<li>Silhouette (k_use): {metrics.get("silhouette",0.0):.3f}</li>',
        '</ul>',
        '</body></html>'
    ]
    return '\n'.join(H)

# ---------------------------------------------------------------------------
# Normalización de prefijo
# ---------------------------------------------------------------------------

def normalize_prefix(out_prefix: str, xml_file: str) -> Path:
    p = Path(out_prefix)
    xml_stem = Path(xml_file).stem
    name = p.name
    if xml_stem not in name:
        name = f"{name}_{xml_stem}"
    return p.parent / name

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Bloque 6: reporte final.")
    ap.add_argument('--xml', required=True)
    ap.add_argument('--phase-align', type=str, default='auto')
    ap.add_argument('--recluster-after-align', action='store_true')
    ap.add_argument('--sensor', type=str, default='auto')
    ap.add_argument('--tipo-tx', type=str, default='seco')
    ap.add_argument('--k-manual', type=int, default=None)
    ap.add_argument('--allow-otras', type=str, default='true')
    ap.add_argument('--otras-min-score', type=float, default=0.12)
    ap.add_argument('--otras-cap', type=float, default=0.25)
    ap.add_argument('--subclusters', action='store_true')
    ap.add_argument('--sub-min-pct', type=float, default=0.02)
    ap.add_argument('--out-prefix', type=str, default='reporte')
    ap.add_argument('--embed-assets', action='store_true')
    ap.add_argument('--batch-dir', type=str, default=None)
    ap.add_argument('--no-pdf', action='store_true')
    args = ap.parse_args()

    def process_single_xml(xml_path: str, report_out_prefix: str) -> None:
        report_prefix_path = Path(report_out_prefix)
        # Asegurar sufijo '_reporte' para el archivo HTML final. Si ya termina en '_reporte' no se duplica.
        if not report_prefix_path.name.endswith('_reporte'):
            report_prefix_path = report_prefix_path.with_name(f"{report_prefix_path.name}_reporte")
        # Directorio de salida local.  Si el prefijo no tiene directorio, usar carpeta actual.
        out_dir_local = report_prefix_path.parent if report_prefix_path.parent != Path('') else Path('.')
        # Base se define estrictamente como el nombre del prefijo de salida (sin añadir ni quitar
        # partes).  Anteriormente se truncaba el nombre en el último guion bajo, lo cual
        # producía "time_20000101" a partir de "time_20000101_025330" y provocaba que
        # no se encontrasen los archivos generados por los bloques anteriores.  Se
        # mantiene el nombre completo para que coincida con los prefijos producidos por
        # B3–B5 (p.ej., "time_20000101_025330").
        base = out_dir_local / Path(report_out_prefix).name
        # También conservar el nombre base para búsquedas alternativas y el stem del XML.
        # base_name corresponde al nombre completo del prefijo de salida (por ejemplo,
        # "time_20000101_025330_testfinal").  xml_stem es el nombre del archivo XML
        # sin extensión (por ejemplo, "time_20000101_025330").  Se usarán ambos para
        # buscar archivos de B4/B5 generados con distintos criterios de nomenclatura.
        base_name = Path(report_out_prefix).name
        xml_stem = Path(xml_path).stem

        # ------ B4: aceptar con/sin prefijo ------
        # Para las rutas de B4, intentar con 'base' (raíz sin sufijo) y con el nombre
        # completo del out_prefix. Esto aumenta la robustez ante prefijos con guiones
        # o sufijos adicionales.  Se agregan patrones alternativos que incluyen
        # Path(report_out_prefix).name para encontrar archivos generados con el
        # nombre completo del prefijo de salida.
        base_name = Path(report_out_prefix).name
        # Métricas y curvas de B4 pueden estar nombradas con el prefijo de salida completo
        # (base/base_name) o únicamente con el stem del XML (xml_stem).  Construir
        # múltiples candidatos para cada archivo.
        metrics_path      = first_existing_path(
            f"{base}_b4_metrics.csv", f"{base}_metrics.csv",
            f"{base_name}_b4_metrics.csv", f"{base_name}_metrics.csv",
            f"{xml_stem}_b4_metrics.csv", f"{xml_stem}_metrics.csv"
        )
        curvas_png        = first_existing_path(
            f"{base}_b4_curvas.png", f"{base}_curvas.png",
            f"{base_name}_b4_curvas.png", f"{base_name}_curvas.png",
            f"{xml_stem}_b4_curvas.png", f"{xml_stem}_curvas.png"
        )
        curvas_comb_png   = first_existing_path(
            f"{base}_b4_curvas_combinadas.png", f"{base}_curvas_combinadas.png",
            f"{base_name}_b4_curvas_combinadas.png", f"{base_name}_curvas_combinadas.png",
            f"{xml_stem}_b4_curvas_combinadas.png", f"{xml_stem}_curvas_combinadas.png"
        )
        prpd_auto_2d      = first_existing_path(
            f"{base}_b4_prpd_k_auto_2d.png", f"{base}_prpd_k_auto_2d.png",
            f"{base_name}_b4_prpd_k_auto_2d.png", f"{base_name}_prpd_k_auto_2d.png",
            f"{xml_stem}_b4_prpd_k_auto_2d.png", f"{xml_stem}_prpd_k_auto_2d.png"
        )
        prpd_auto_3d      = first_existing_path(
            f"{base}_b4_prpd_k_auto_3d.png", f"{base}_prpd_k_auto_3d.png",
            f"{base_name}_b4_prpd_k_auto_3d.png", f"{base_name}_prpd_k_auto_3d.png",
            f"{xml_stem}_b4_prpd_k_auto_3d.png", f"{xml_stem}_prpd_k_auto_3d.png"
        )
        prpd_manual_2d    = first_existing_path(
            f"{base}_b4_prpd_k_manual_2d.png", f"{base}_prpd_k_manual_2d.png",
            f"{base_name}_b4_prpd_k_manual_2d.png", f"{base_name}_prpd_k_manual_2d.png",
            f"{xml_stem}_b4_prpd_k_manual_2d.png", f"{xml_stem}_prpd_k_manual_2d.png"
        )
        prpd_manual_3d    = first_existing_path(
            f"{base}_b4_prpd_k_manual_3d.png", f"{base}_prpd_k_manual_3d.png",
            f"{base_name}_b4_prpd_k_manual_3d.png", f"{base_name}_prpd_k_manual_3d.png",
            f"{xml_stem}_b4_prpd_k_manual_3d.png", f"{xml_stem}_prpd_k_manual_3d.png"
        )

        # ------ B5: aceptar con/sin prefijo ------
        probs_csv_path    = first_existing_path(f"{base}_b5_probabilities.csv",      f"{base}_probabilities.csv")
        summary_csv_path  = first_existing_path(f"{base}_b5_summary.csv",            f"{base}_summary.csv")
        top_clusters_csv  = first_existing_path(f"{base}_b5_top_clusters_table.csv", f"{base}_top_clusters_table.csv")
        cards_png         = first_existing_path(f"{base}_b5_cards.png",              f"{base}_cards.png")
        global_mix_png    = first_existing_path(f"{base}_b5_global_mix.png",         f"{base}_global_mix.png")
        stack_png         = first_existing_path(f"{base}_b5_stack_by_cluster.png",   f"{base}_stack_by_cluster.png")
        multiplicity_csv  = first_existing_path(f"{base}_b5_multiplicity.csv",       f"{base}_multiplicity.csv")
        multiplicity_fig  = first_existing_path(f"{base}_b5_multiplicity.png",       f"{base}_multiplicity.png")
        subclusters_csv   = first_existing_path(f"{base}_b5_subclusters.csv",        f"{base}_subclusters.csv")

        # ------ Extras de pares ------
        paired_html       = first_existing_path(f"{base}_paired_3d.html")
        paired_sources    = first_existing_path(f"{base}_paired_sources.csv")
        paired_map_json   = first_existing_path(f"{base}_paired_map.json")
        p_mult_csv        = first_existing_path(f"{base}_p_multiplicity.csv")
        top_sources_csv   = first_existing_path(f"{base}_top_sources_table.csv")

        # ------ Natural 3D (B3) ------
        # 3D interactivas: aceptar nombres nuevos (_natural_plotly3d.html / _aligned_plotly3d.html) y antiguos si existen
        nat_3d_html       = first_existing_path(
            f"{base}_natural_plotly3d.html",
            f"{base}_plotly3d.html"
        )
        aligned_3d_html   = first_existing_path(
            f"{base}_aligned_plotly3d.html"
        )

        # Leer métricas y resumen
        metrics = read_b4_metrics(metrics_path)
        summary = read_b5_summary(summary_csv_path)

        # Determinar fase aplicada: preferir el valor manual si se pasó como argumento.
        # Si args.phase_align es numérico, utilizarlo y marcar manual_flag=True.
        # De lo contrario, intentar leer el best_shift_deg del archivo probabilities
        # como referencia automática.  Si no se puede leer, se deja en 0.
        manual_flag_local = False
        phase_shift_local = 0.0
        pa_spec = (args.phase_align or '').strip().lower()
        if pa_spec not in ('auto', 'none', ''):
            try:
                phase_shift_local = float(pa_spec)
                manual_flag_local = True
            except Exception:
                # No se pudo interpretar como número; se mantiene el valor automático
                manual_flag_local = False
        if not manual_flag_local:
            # Intentar leer best_shift_deg del CSV de probabilidades
            if probs_csv_path and Path(probs_csv_path).exists():
                try:
                    with open(probs_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                        reader = csv.DictReader(f)
                        row = next(reader)
                        phase_shift_local = float(row.get('best_shift_deg', '0.0'))
                except Exception:
                    phase_shift_local = 0.0

        # Conteos de multiplicidad global (si hay CSV)
        multiplicity_counts_local = None
        if args.subclusters and multiplicity_csv and multiplicity_csv.exists():
            try:
                df_mul = pd.read_csv(multiplicity_csv)
                global_row = df_mul[df_mul['cluster_id'] == 'GLOBAL']
                if not global_row.empty:
                    cav = int(global_row['n_cavity_lobes'].values[0])
                    cor = int(global_row['n_corona_groups'].values[0])
                    sup = int(global_row['n_superficial_bands'].values[0])
                    multiplicity_counts_local = {'cavidad': cav, 'corona': cor, 'superficial': sup}
            except Exception:
                multiplicity_counts_local = None

        # Probabilidades p_ge2
        p_ge2_local = summary.get('p_ge2', {}) if isinstance(summary.get('p_ge2', {}), dict) else {}

        # Subclusters influyentes (si CSV existe)
        subclusters_info_local = None
        if args.subclusters and subclusters_csv and subclusters_csv.exists():
            try:
                rows = []
                with open(subclusters_csv, 'r', encoding='utf-8', errors='ignore') as f_sub:
                    reader = csv.DictReader(f_sub)
                    for r in reader:
                        for k in ('frac','stability','weight'):
                            try:
                                r[k] = float(r.get(k, 0.0))
                            except Exception:
                                r[k] = 0.0
                        rows.append(r)
                rows.sort(key=lambda x: -x.get('weight', 0.0))
                subclusters_info_local = rows
            except Exception:
                subclusters_info_local = None

        # Intentar leer k_opt_auto desde JSON de B4 (si existe)
        # Buscar JSON de k_auto tanto por base (prefijo completo) como por xml_stem
        kauto_json = first_existing_path(f"{base}_b4_kauto.json", f"{xml_stem}_b4_kauto.json")
        if kauto_json and Path(kauto_json).exists():
            try:
                data = json.loads(Path(kauto_json).read_text(encoding='utf-8', errors='ignore'))
                k_auto_from_json = int(data.get('k_opt_auto', 0))
                if k_auto_from_json > 0:
                    metrics['k_use'] = k_auto_from_json
            except Exception:
                pass

        # Resolver rutas para HTML (base64 si embed_assets)
        def rel_or_embed(p: Path | None) -> str | None:
            if not p or not p.exists():
                return None
            if args.embed_assets:
                return embed_image_if_needed(p, True)
            try:
                return str(p.relative_to(out_dir_local))
            except Exception:
                return str(p.name)

        fig_paths = {
            'curvas_combinadas': rel_or_embed(curvas_comb_png),
            'prpd_auto_2d': rel_or_embed(prpd_auto_2d),
            'prpd_manual_2d': rel_or_embed(prpd_manual_2d) if args.k_manual is not None else None,
            'cards': rel_or_embed(cards_png),
            'global_mix': rel_or_embed(global_mix_png),
            'stack': rel_or_embed(stack_png),
            'top_clusters_csv': str(top_clusters_csv) if top_clusters_csv and top_clusters_csv.exists() else None,
            'multiplicity_fig': rel_or_embed(multiplicity_fig) if args.subclusters else None,
            # extras de pares
            'paired_html': str(paired_html) if paired_html and paired_html.exists() else None,
            'paired_sources_csv': str(paired_sources) if paired_sources and paired_sources.exists() else None,
            'paired_map_json': str(paired_map_json) if paired_map_json and paired_map_json.exists() else None,
            # natural y aligned 3D (no incrustado; solo para enlaces)
            'natural_3d_html': str(nat_3d_html) if nat_3d_html and nat_3d_html.exists() else None,
            'aligned_3d_html': str(aligned_3d_html) if aligned_3d_html and aligned_3d_html.exists() else None,
        }

        # Generar HTML
        html_content = build_report_html(
            xml_path=xml_path,
            sensor=args.sensor,
            tipo_tx=args.tipo_tx,
            # Pasar phase_shift como tupla (valor, manual_flag) para distinguir manual/auto
            phase_shift=(phase_shift_local, manual_flag_local),
            recluster=args.recluster_after_align,
            k_use=int(metrics.get('k_use', 0)),
            k_manual=args.k_manual,
            metrics=metrics,
            summary=summary,
            fig_paths=fig_paths,
            subclusters=args.subclusters,
            multiplicity_counts=multiplicity_counts_local,
            out_prefix=str(report_prefix_path),
            p_ge2=p_ge2_local,
            subclusters_info=subclusters_info_local,
        )

        # Guardar HTML
        html_out_path = report_prefix_path.with_suffix('.html')
        html_out_path.parent.mkdir(parents=True, exist_ok=True)
        html_out_path.write_text(html_content, encoding='utf-8')

        # PDF opcional
        pdf_out_path = None
        if not args.no_pdf:
            try:
                from weasyprint import HTML  # opcional
                pdf_out_path = report_prefix_path.with_suffix('.pdf')
                HTML(string=html_content, base_url=str(out_dir_local.resolve())).write_pdf(pdf_out_path)
            except Exception:
                print("WARNING: No se pudo generar PDF (WeasyPrint ausente o error).")

        # Índice JSON
        index_data = {
            'source_xml': xml_path,
            'b4_metrics': str(metrics_path) if metrics_path else None,
            'b4_kauto_json': str(kauto_json) if kauto_json and kauto_json.exists() else None,
            'b5_probabilities': str(probs_csv_path) if probs_csv_path else None,
            'b5_summary': str(summary_csv_path) if summary_csv_path else None,
            'b5_top_clusters_table': str(top_clusters_csv) if top_clusters_csv else None,
            'b5_subclusters_csv': str(subclusters_csv) if subclusters_csv else None,
            'natural_3d_html': str(nat_3d_html) if nat_3d_html else None,
            'aligned_3d_html': str(aligned_3d_html) if aligned_3d_html else None,
            'paired_html': str(paired_html) if paired_html else None,
            'paired_sources_csv': str(paired_sources) if paired_sources else None,
            'paired_map_json': str(paired_map_json) if paired_map_json else None,
            'p_multiplicity_csv': str(p_mult_csv) if p_mult_csv else None,
            'top_sources_csv': str(top_sources_csv) if top_sources_csv else None,
            'figures': {k: v for k, v in fig_paths.items() if v},
            'report_html': str(html_out_path),
            'report_pdf': str(pdf_out_path) if pdf_out_path else None,
        }
        json_out = report_prefix_path.parent / f"{report_prefix_path.stem}_index.json"
        json_out.write_text(json.dumps(index_data, indent=2, ensure_ascii=False), encoding='utf-8')

        # Consola: compatible Windows (ASCII)
        reclust_str_local = 'sí' if args.recluster_after_align else 'no'
        # Incluir sufijo (manual) en el resumen de consola si el desfase fue manual
        delta_console = f"Delta={int(round(phase_shift_local))} deg" + (" (manual)" if manual_flag_local else "")
        print(f"{Path(xml_path).name}, {args.sensor}, {args.tipo_tx}, {delta_console}, recluster={reclust_str_local}")
        parts = []
        import math as _math
        for t in ['cavidad', 'superficial', 'corona', 'flotante', 'otras', 'ruido']:
            if t == 'otras' and summary.get('otras_policy') == 'suppressed':
                continue
            val = summary.get('p_global', {}).get(t)
            if val is None or (isinstance(val, float) and (_math.isnan(val) or _math.isinf(val))):
                parts.append(f"{t.capitalize()} —")
            else:
                try:
                    pct = float(val) * 100.0
                    parts.append(f"{t.capitalize()} {pct:.0f}%")
                except Exception:
                    parts.append(f"{t.capitalize()} —")
        print(' | '.join(parts))
        print(f"KMeans={int(metrics.get('k_use',0))}, HDBSCAN={int(metrics.get('n_clusters_hdb',0))}, Ruido={metrics.get('frac_ruido',0.0)*100:.1f}%, Silhouette={metrics.get('silhouette',0.0):.3f}")

    # ---- batch vs single ----
    if args.batch_dir:
        batch = Path(args.batch_dir)
        xmls = sorted(batch.glob("*.xml"))
        base_out_dir = Path(args.out_prefix).parent if Path(args.out_prefix).parent != Path('') else Path('.')
        for x in xmls:
            report_base = base_out_dir / f"{x.stem}_reporte"
            report_norm = normalize_prefix(str(report_base), str(x))
            process_single_xml(str(x), str(report_norm))
        return

    single_prefix = normalize_prefix(args.out_prefix, args.xml)
    process_single_xml(args.xml, str(single_prefix))


if __name__ == "__main__":
    main()
