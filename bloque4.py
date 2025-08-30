#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bloque 4 — Validador/Comparador de clusters PRPD.

Este módulo se encarga de comparar la segmentación automática (k_auto) obtenida
a partir del método del codo (Kneedle) con una segmentación manual
(k_manual).  Trabaja sobre datos PRPD alineados en fase y calcula
métricas de calidad de clustering (pureza, ARI, NMI, silhouette) en
comparación con las etiquetas naturales de HDBSCAN.  Además genera
figuras de las curvas de Silhouette y Elbow, y representaciones 2D/3D
de los mapas PRPD para k_auto y k_manual.

Salida:
  - <out_prefix>_curvas.png: gráfico con curvas de Silhouette y SSE vs k.
  - <out_prefix>_prpd_k_auto_2d.png y 3d: mapas PRPD segmentados con k_auto.
  - <out_prefix>_prpd_k_manual_2d.png y 3d: mapas PRPD segmentados con k_manual.
  - <out_prefix>_metrics.csv: métricas de pureza, ARI, NMI, etc. para ambos k.

El script no imprime texto adicional; sólo muestra los nombres de los
archivos generados al finalizar.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Silenciar advertencias de sklearn deprecations (FutureWarning)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Importar utilidades de prpd_mega y bloque5 para alineación
try:
    from prpd_mega import (
        parse_xml_points,
        normalize_y,
        phase_from_times,
        prpd_hist2d,
        kmeans_over_bins,
        centers_from_edges,
        kneedle_k_from_elbow,
    )
except ImportError as e:
    raise ImportError(
        "No se pudo importar prpd_mega. Asegúrate de que prpd_mega.py está en el mismo directorio."
    ) from e

try:
    # Podemos reutilizar la estimación de corrimiento automático de bloque5
    from bloque5 import estimate_phase_shift_auto  # type: ignore
except Exception:
    estimate_phase_shift_auto = None  # se definirá más abajo si no está disponible

try:
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ---------------------------------------------------------------------------
# Helper para normalizar out-prefix
# ---------------------------------------------------------------------------
def normalize_prefix(out_prefix: str, xml_file: str) -> Path:
    """Normaliza el prefijo de salida para que incluya el stem del XML.

    Si el nombre base de out_prefix no contiene el stem del archivo XML,
    inserta "_<stem>" al final.  Devuelve una ruta Path resultante.

    Parameters
    ----------
    out_prefix : str
        Prefijo de salida proporcionado por el usuario.
    xml_file : str
        Ruta al archivo XML procesado.

    Returns
    -------
    Path
        Ruta normalizada que incluye el stem del XML.
    """
    p = Path(out_prefix)
    xml_stem = Path(xml_file).stem
    name = p.name
    if xml_stem not in name:
        name = f"{name}_{xml_stem}"
    return p.parent / name


def apply_phase_shift(phase_deg: np.ndarray, shift_deg: float) -> np.ndarray:
    """Aplica un corrimiento circular a la fase en grados."""
    return (phase_deg + shift_deg) % 360.0


def estimate_shift_auto_local(phase_deg: np.ndarray, y_norm: np.ndarray, qty: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Estima un desplazamiento de fase utilizando la heurística de bloque5 si
    existe.  Si no se puede importar, devuelve cero.
    """
    if estimate_phase_shift_auto is not None:
        return estimate_phase_shift_auto(phase_deg, y_norm, qty)
    # Fallback: no corrimiento
    return 0.0, {'rho_sym_best': 0.0, 'corona_score_best': 0.0}


def compute_sse(H: np.ndarray, labels_grid: np.ndarray, xedges: np.ndarray, yedges: np.ndarray) -> float:
    """Calcula la suma de cuadrados intra‑cluster (SSE) ponderada por peso.

    Parameters
    ----------
    H : array (ny_bins, nx_bins)
        Histograma de pesos por bin (traspuesto respecto a la convención de
        prpd_hist2d).  Utilícese H.T para indexar como [y,x].
    labels_grid : array (ny_bins, nx_bins)
        Etiquetas de cluster por bin devueltas por kmeans_over_bins.
    xedges, yedges : arrays
        Límites de los bins en fase y amplitud.

    Returns
    -------
    sse : float
        Suma de pesos * distancia euclídea cuadrada al centroide del cluster.
    """
    # Centros de cada bin
    Xc, Yc = centers_from_edges(xedges, yedges)
    ny, nx = labels_grid.shape
    # Acumular sumas ponderadas por cluster
    sums: Dict[int, Tuple[float, float, float]] = {}
    for iy in range(ny):
        for ix in range(nx):
            lab = int(labels_grid[iy, ix])
            w = float(H.T[iy, ix])
            if w <= 0.0:
                continue
            cx = float(Xc[iy, ix])
            cy = float(Yc[iy, ix])
            if lab not in sums:
                sums[lab] = (w * cx, w * cy, w)
            else:
                sx, sy, sw = sums[lab]
                sums[lab] = (sx + w * cx, sy + w * cy, sw + w)
    centroids: Dict[int, Tuple[float, float]] = {}
    for lab, (sx, sy, sw) in sums.items():
        if sw > 0.0:
            centroids[lab] = (sx / sw, sy / sw)
        else:
            centroids[lab] = (0.0, 0.0)
    # Calcular SSE
    sse = 0.0
    for iy in range(ny):
        for ix in range(nx):
            w = float(H.T[iy, ix])
            if w <= 0.0:
                continue
            lab = int(labels_grid[iy, ix])
            cx, cy = centroids.get(lab, (0.0, 0.0))
            dx = float(Xc[iy, ix]) - cx
            dy = float(Yc[iy, ix]) - cy
            sse += w * (dx * dx + dy * dy)
    return sse


def compute_purity(labels_pred: np.ndarray, labels_true: np.ndarray, weights: np.ndarray) -> float:
    """Calcula la pureza ponderada entre dos asignaciones de etiquetas.

    Pureza = sum_i max_j |C_i ∩ T_j| / N, con pesos.

    Parameters
    ----------
    labels_pred : array
        Etiquetas predichas por el clustering.
    labels_true : array
        Etiquetas de referencia (HDBSCAN).  Se permiten etiquetas negativas
        para ruido.
    weights : array
        Pesos asociados a cada evento (qty).

    Returns
    -------
    purity : float
        Valor en [0,1].
    """
    unique_pred = np.unique(labels_pred)
    total_weight = float(np.sum(weights)) if np.sum(weights) > 0 else 0.0
    if total_weight == 0.0:
        return 0.0
    purity_sum = 0.0
    for cl in unique_pred:
        mask = labels_pred == cl
        if not np.any(mask):
            continue
        sub_weights = weights[mask]
        sub_true = labels_true[mask]
        # Suma de pesos por etiqueta de referencia
        counts: Dict[int, float] = {}
        for t, w in zip(sub_true, sub_weights):
            counts[int(t)] = counts.get(int(t), 0.0) + float(w)
        # Seleccionar la etiqueta con más peso
        max_w = max(counts.values()) if counts else 0.0
        purity_sum += max_w
    purity = purity_sum / total_weight
    return purity


def determine_knee(ks: List[int], sses: List[float]) -> int:
    """Detecta el punto de inflexión (knee) en la curva SSE vs k.

    Se aplica el método de la distancia máxima a la recta entre los
    extremos.  Si no se detecta un codo pronunciado, devuelve el
    número de clusters con menor SSE (máximo k).
    """
    if not ks:
        return 1
    # Usar la recta entre el primer y último punto
    x = np.array(ks, dtype=float)
    y = np.array(sses, dtype=float)
    # Normalizar x para una comparación razonable
    x_norm = (x - x[0]) / (x[-1] - x[0] + 1e-12)
    y_norm = (y - y[0]) / (y[-1] - y[0] + 1e-12)
    # Recta entre extremos
    line_start = np.array([x_norm[0], y_norm[0]])
    line_end = np.array([x_norm[-1], y_norm[-1]])
    line_vec = line_end - line_start
    line_vec /= np.linalg.norm(line_vec) + 1e-12
    # Distancias perpendiculares
    max_dist = -1.0
    best_k = ks[0]
    for idx, (xn, yn) in enumerate(zip(x_norm, y_norm)):
        point = np.array([xn, yn])
        # Proyección sobre la recta
        proj_len = np.dot(point - line_start, line_vec)
        proj_point = line_start + proj_len * line_vec
        dist = np.linalg.norm(point - proj_point)
        if dist > max_dist:
            max_dist = dist
            best_k = ks[idx]
    return int(best_k)


def plot_prpd_2d(phase_deg: np.ndarray, y_norm: np.ndarray, qty: np.ndarray,
                 labels_evt: np.ndarray, k_used: int, out_path: Path,
                 palette: Tuple[Tuple[float, float, float], ...]) -> None:
    """Genera un mapa PRPD 2D coloreado por cluster y lo guarda.

    Parameters
    ----------
    phase_deg : array
        Fase alineada de cada evento (0–360).
    y_norm : array
        Amplitud normalizada (0–100).
    qty : array
        Cantidades de cada evento (para ajustar tamaño del punto).
    labels_evt : array
        Etiquetas de cluster por evento (−1 indica ruido).
    k_used : int
        Número de clusters usados en K‑Means.
    out_path : Path
        Ruta del archivo de salida.
    palette : tuple
        Paleta de colores fija de seis colores.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    # Dibujar fondo en escala de grises como densidad aproximada
    # Estimación de densidad simple para visualización
    # No usamos qty para color del fondo por eficiencia; la densidad gris
    # proviene del recuento por pixel.
    try:
        # Construir un histograma grueso para el fondo
        H_bg, xedges_bg, yedges_bg = np.histogram2d(
            phase_deg, y_norm, bins=[100, 50], range=[[0, 360], [0, 100]], weights=qty
        )
        H_bg = H_bg.T
        # Normalizar para tener valores en [0,1]
        if np.max(H_bg) > 0:
            H_bg_norm = H_bg / np.max(H_bg)
        else:
            H_bg_norm = H_bg
        # Dibujar como imshow en escala de grises
        ax.imshow(
            H_bg_norm,
            extent=[0, 360, 0, 100],
            origin='lower',
            cmap='gray',
            aspect='auto',
            alpha=0.3,
        )
    except Exception:
        pass
    # Scatter por clusters
    uniq = np.unique(labels_evt)
    for cl in uniq:
        mask = labels_evt == cl
        if not np.any(mask):
            continue
        if cl < 0:
            color = (0.7, 0.7, 0.7, 0.5)  # ruido
            label = 'Ruido'
        else:
            color = palette[int(cl) % len(palette)]
            label = f'C{cl}'
        ax.scatter(
            phase_deg[mask],
            y_norm[mask],
            s=2,
            c=[color],
            label=label,
            alpha=0.8,
            linewidths=0,
        )
    ax.set_xlabel('Fase (°)')
    ax.set_ylabel('Amplitud normalizada')
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_title(f'PRPD 2D (k={k_used})')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=6)
    # Ajustar el diseño para no recortar títulos y leyendas
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    # Guardar con bbox_inches='tight' para que el título no se corte
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_prpd_3d(phase_deg: np.ndarray, y_norm: np.ndarray, qty: np.ndarray,
                 labels_evt: np.ndarray, k_used: int, out_path: Path,
                 palette: Tuple[Tuple[float, float, float], ...]) -> None:
    """Genera un mapa PRPD 3D coloreado por cluster y lo guarda.

    Ejes: X = Fase (°), Y = amplitud normalizada (0–100), Z = log10(cantidad).
    """
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    # Log de cantidad para z
    z_vals = np.log10(np.array(qty, dtype=float) + 1.0)
    uniq = np.unique(labels_evt)
    for cl in uniq:
        mask = labels_evt == cl
        if not np.any(mask):
            continue
        if cl < 0:
            color = (0.7, 0.7, 0.7, 0.5)
            label = 'Ruido'
        else:
            color = palette[int(cl) % len(palette)]
            label = f'C{cl}'
        ax.scatter(
            phase_deg[mask],
            y_norm[mask],
            z_vals[mask],
            s=3,
            c=[color],
            label=label,
            alpha=0.8,
            linewidths=0,
        )
    ax.set_xlabel('Fase (°)')
    ax.set_ylabel('Amplitud normalizada')
    ax.set_zlabel('log10(Cantidad)')
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 100)
    # Ajustar límites Z automáticamente
    # Título
    ax.set_title(f'PRPD 3D (k={k_used})')
    # Leyenda externa
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=6)
    # Ajustar el diseño y guardar con bbox_inches
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Bloque 4: comparación de clustering K‑Means automático versus manual ' \
            'en mapas PRPD alineados.'
        )
    )
    parser.add_argument('--xml', dest='xml_file', required=True, help='Archivo XML con datos PRPD')
    parser.add_argument('--phase-align', type=str, default='auto', help="Alineación de fase: 'auto', 'none' o un valor numérico en grados")
    parser.add_argument('--recluster-after-align', action='store_true', help='Recalcular K‑Means y HDBSCAN con la fase corregida')
    parser.add_argument('--k-manual', type=int, default=3, help='Número de clusters manual a comparar')
    parser.add_argument('--out-prefix', type=str, default='b4', help='Prefijo para los archivos de salida. Utilice un nombre que incluya la carpeta deseada, por ejemplo "out\\t1_b4".')
    args = parser.parse_args()

    xml_path = args.xml_file
    k_manual = args.k_manual
    # Normalizar el prefijo de salida para incluir el stem del XML
    # Si el usuario no especifica el stem del archivo XML en el nombre base,
    # se insertará automáticamente.  Esto evita nombres ambiguos cuando
    # se ejecuta sobre múltiples archivos.
    out_path = normalize_prefix(args.out_prefix, xml_path)
    out_dir = out_path.parent
    # Crear la carpeta de salida si no existe
    if str(out_dir) != '' and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    # Nombre base para archivos (sin directorio)
    out_prefix_name = out_path.name

    # Cargar datos crudos
    data = parse_xml_points(xml_path)
    raw_y = data.get('raw_y') if 'raw_y' in data else data.get('pixel')
    times = data.get('times') if 'times' in data else data.get('ms')
    qty = data.get('quantity') if 'quantity' in data else data.get('count')
    sample_name = data.get('sample_name')
    # Convertir a arrays numpy
    raw_y = np.array(raw_y, dtype=float)
    times = np.array(times, dtype=float)
    qty = np.array(qty, dtype=float)
    # Limpieza básica: filtrar NaN/inf
    mask_valid = np.isfinite(raw_y) & np.isfinite(times) & np.isfinite(qty)
    raw_y = raw_y[mask_valid]
    times = times[mask_valid]
    qty = qty[mask_valid]
    # Reemplazar valores no positivos de qty por 1
    qty[qty <= 0] = 1.0
    # Recorte de outliers por percentiles 0.5 y 99.5
    def clip_percentiles(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        low = np.percentile(arr, 0.5)
        high = np.percentile(arr, 99.5)
        return np.clip(arr, low, high)
    raw_y = clip_percentiles(raw_y)
    times = clip_percentiles(times)
    qty = clip_percentiles(qty)
    # Normalizar amplitud (invertida) a 0–100
    y_norm, _ = normalize_y(raw_y, sample_name)
    # Fase inicial a partir del tiempo (sin corrimiento inicial)
    phase_raw = phase_from_times(times, 0.0)
    # Alineación de fase
    align_spec = (args.phase_align or 'auto').lower()
    if align_spec == 'auto':
        shift_deg, shift_metrics = estimate_shift_auto_local(phase_raw, y_norm, qty)
        align_mode = 'auto'
    elif align_spec == 'none':
        shift_deg = 0.0
        shift_metrics = {'rho_sym_best': 0.0, 'corona_score_best': 0.0}
        align_mode = 'none'
    else:
        try:
            shift_deg = float(align_spec)
        except Exception:
            shift_deg = 0.0
        shift_metrics = {'rho_sym_best': 0.0, 'corona_score_best': 0.0}
        align_mode = 'manual'
    # Aplicar corrimiento
    phase_corr = apply_phase_shift(phase_raw, shift_deg)
    # Seleccionar fase para clustering
    phase_for_clustering = phase_corr if args.recluster_after_align else phase_raw
    # Construir histograma en bins para clustering (adaptativo hasta 240x120)
    H, xedges, yedges = prpd_hist2d(phase_for_clustering, y_norm, qty, bins_phase=240, bins_y=120)
    ny, nx = H.T.shape
    # HDBSCAN natural sobre bins para evaluar pureza y ruido
    Xc, Yc = centers_from_edges(xedges, yedges)
    mask_bins = H.T > 0
    P = np.c_[Xc[mask_bins], Yc[mask_bins]]
    weights_bins = H.T[mask_bins]
    scaler = MinMaxScaler(); Pn = scaler.fit_transform(P)
    labels_h_bins = None
    bin_stability = None
    if hdbscan is not None:
        # Replicar puntos según peso (redondeando al entero más cercano)
        w_int = np.round(weights_bins).astype(int)
        w_int[w_int < 1] = 1
        idx_map = np.repeat(np.arange(len(Pn)), w_int)
        Pn_rep = np.repeat(Pn, w_int, axis=0)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(50, int(0.005 * len(Pn))), cluster_selection_method='eom')
        labels_rep = clusterer.fit_predict(Pn_rep)
        try:
            probs_rep = clusterer.probabilities_
        except Exception:
            probs_rep = np.ones_like(labels_rep, dtype=float)
        # Voto mayoritario y estabilidad media por bin
        labels_h_bins = np.full(len(Pn), fill_value=-1, dtype=int)
        bin_stability = np.zeros(len(Pn), dtype=float)
        for i in range(len(Pn)):
            maski = idx_map == i
            if not np.any(maski):
                continue
            labs, cnts = np.unique(labels_rep[maski], return_counts=True)
            labels_h_bins[i] = labs[np.argmax(cnts)]
            bin_stability[i] = float(np.mean(probs_rep[maski])) if np.any(maski) else 0.7
        noise_frac = float(np.mean(labels_rep < 0))
    else:
        # Fallback a DBSCAN si no hay HDBSCAN
        db = DBSCAN(eps=0.06, min_samples=5).fit(Pn, sample_weight=weights_bins)
        labels_h_bins = db.labels_
        bin_stability = np.full(len(Pn), 0.7, dtype=float)
        noise_frac = float(np.sum(weights_bins[labels_h_bins < 0])) / float(np.sum(weights_bins)) if np.sum(weights_bins) > 0 else 0.0
    # Reconstruir etiquetas naturales por evento y fracción de ruido
    labels_h_evt = np.full_like(phase_for_clustering, fill_value=-1, dtype=int)
    # Mapas de bin a etiqueta y a estabilidad
    idx_bins = np.argwhere(mask_bins)
    bin_to_lab: Dict[Tuple[int, int], int] = {}
    for idx, (row, col) in enumerate(idx_bins):
        bin_to_lab[(row, col)] = int(labels_h_bins[idx])
    # Asignar etiquetas naturales a cada evento
    x_bins_evt = np.digitize(phase_for_clustering, bins=xedges) - 1
    y_bins_evt = np.digitize(y_norm, bins=yedges) - 1
    for i, (r, c) in enumerate(zip(y_bins_evt, x_bins_evt)):
        if (r, c) in bin_to_lab:
            labels_h_evt[i] = bin_to_lab[(r, c)]
    # Porcentaje de ruido natural
    noise_mask = labels_h_evt < 0
    total_weight = float(np.sum(qty)) if np.sum(qty) > 0 else 0.0
    frac_noise_hdb = float(np.sum(qty[noise_mask])) / total_weight if total_weight > 0 else 0.0
    # Número de clusters naturales sin contar ruido
    n_clusters_hdb = len([lab for lab in np.unique(labels_h_evt) if lab >= 0])

    # Calcular curvas silhouette y SSE para varios valores de k
    # Construir rango de k (evitar k=1 para silhouette)
    k_values: List[int] = []
    silhouette_vals: List[float] = []
    sse_vals: List[float] = []
    # Incluimos k desde 2 hasta un máximo razonable basado en el manual
    max_k = max(8, k_manual + 3)
    ks_to_eval = sorted(set([k_manual] + list(range(2, max_k + 1))))
    # Bucle por cada k
    for k in ks_to_eval:
        if k < 1:
            continue
        try:
            labels_grid, sil = kmeans_over_bins(H, xedges, yedges, k)
        except Exception:
            # Si K‑Means falla, continuar
            continue
        # Silhouette: si es iterable, tomar media; si es escalar, usarlo tal cual
        try:
            sil_val = float(np.mean(sil)) if hasattr(sil, '__iter__') else float(sil)
        except Exception:
            sil_val = 0.0
        # SSE ponderado
        try:
            sse_val = compute_sse(H, labels_grid, xedges, yedges)
        except Exception:
            sse_val = 0.0
        k_values.append(int(k))
        silhouette_vals.append(sil_val)
        sse_vals.append(sse_val)
    # Determinar k_auto por método del codo con Kneedle
    try:
        k_elbow, _knee_idx, _diff = kneedle_k_from_elbow(k_values, sse_vals)
    except Exception:
        # Fallback a nuestro método anterior si falla
        k_elbow = determine_knee(k_values, sse_vals)
    k_auto = k_elbow
    # Preparar mapa entre k y silhouette promedio
    sil_by_k: Dict[int, float] = {k: s for k, s in zip(k_values, silhouette_vals)}
    # Función para obtener etiquetas predichas de K‑Means por evento
    def assign_event_labels(labels_grid: np.ndarray) -> np.ndarray:
        labels_evt = np.full_like(phase_for_clustering, fill_value=-1, dtype=int)
        x_bins_evt_local = np.digitize(phase_for_clustering, bins=xedges) - 1
        y_bins_evt_local = np.digitize(y_norm, bins=yedges) - 1
        nyb, nxb = labels_grid.shape
        for idx_event, (r, c) in enumerate(zip(y_bins_evt_local, x_bins_evt_local)):
            if 0 <= r < nyb and 0 <= c < nxb:
                labels_evt[idx_event] = int(labels_grid[r, c])
        return labels_evt
    # Obtener etiquetas para k_auto y k_manual
    # Para k_auto, recalcular clustering si no está en cache
    labels_grid_auto, _ = kmeans_over_bins(H, xedges, yedges, k_auto)
    labels_evt_auto = assign_event_labels(labels_grid_auto)
    silhouette_auto = sil_by_k.get(k_auto, 0.0)
    labels_grid_manual, _ = kmeans_over_bins(H, xedges, yedges, k_manual)
    labels_evt_manual = assign_event_labels(labels_grid_manual)
    silhouette_manual = sil_by_k.get(k_manual, 0.0)
    # Calcular métricas para auto y manual
    metrics_rows: List[List[object]] = []
    for method, k_used, labels_evt, sil_val in [
        ('auto', k_auto, labels_evt_auto, silhouette_auto),
        ('manual', k_manual, labels_evt_manual, silhouette_manual),
    ]:
        purity = compute_purity(labels_evt, labels_h_evt, qty)
        # ARI y NMI consideran ruido como una categoría más
        try:
            ari = adjusted_rand_score(labels_h_evt, labels_evt)
        except Exception:
            ari = 0.0
        try:
            nmi = normalized_mutual_info_score(labels_h_evt, labels_evt)
        except Exception:
            nmi = 0.0
        metrics_rows.append([
            method,
            int(k_used),
            purity,
            ari,
            nmi,
            n_clusters_hdb,
            frac_noise_hdb,
            sil_val,
            k_elbow,
            shift_deg,
        ])
    # Guardar métricas en CSV
    metrics_path = out_dir / f"{out_prefix_name}_metrics.csv"
    with open(metrics_path, 'w') as f:
        # Ajustar encabezados según lo requerido para el Bloque 6.  La columna
        # n_clusters_hdb_sin_ruido corresponde al número de clusters de HDBSCAN
        # excluyendo el ruido.
        f.write('metodo,k,pureza,ari,nmi,n_clusters_hdb_sin_ruido,frac_ruido,silhouette,k_elbow,delta\n')
        for row in metrics_rows:
            out = []
            for val in row:
                if isinstance(val, float):
                    out.append(f'{val:.5f}')
                else:
                    out.append(str(val))
            f.write(','.join(out) + '\n')
    # Generar figuras de curvas (separadas y combinada)
    # Nombre de archivo para las curvas individuales: utilizar siempre
    # <stem>_b4_curvas.png para distinguir este bloque.  El stem se
    # extrae del archivo XML de entrada.
    try:
        xml_stem = Path(xml_path).stem
    except Exception:
        xml_stem = out_prefix_name
    curvas_path = out_dir / f"{xml_stem}_b4_curvas.png"
    fig_sep, axs_sep = plt.subplots(1, 2, figsize=(8, 4))
    fig_sep.patch.set_facecolor('white')
    for ax in axs_sep:
        ax.set_facecolor('white')
    # Silhouette individual
    axs_sep[0].plot(k_values, silhouette_vals, marker='o', color='C0')
    axs_sep[0].set_title('Curva Silhouette')
    axs_sep[0].set_xlabel('Número de clusters (k)')
    axs_sep[0].set_ylabel('Silhouette promedio')
    axs_sep[0].grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    # SSE individual
    axs_sep[1].plot(k_values, sse_vals, marker='o', color='C1')
    axs_sep[1].axvline(k_elbow, color='C3', linestyle='--', label=f'k_elbow={k_elbow}')
    axs_sep[1].set_title('Curva Elbow (SSE)')
    axs_sep[1].set_xlabel('Número de clusters (k)')
    axs_sep[1].set_ylabel('Suma de cuadrados (SSE)')
    axs_sep[1].grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    axs_sep[1].legend(loc='best', fontsize=7)
    fig_sep.tight_layout()
    # Guardar con bbox_inches para evitar cortes
    fig_sep.savefig(curvas_path, dpi=300, bbox_inches='tight')
    plt.close(fig_sep)
    # Calcular k óptimos según silhouette
    if silhouette_vals:
        # k con máximo silhouette
        sil_arr = np.array(silhouette_vals)
        max_idx = int(np.argmax(sil_arr))
        k_opt_auto = int(k_values[max_idx])
    else:
        k_opt_auto = k_elbow
    # Para manual definimos k_opt_manual como k_manual
    k_opt_manual = k_manual
    # Guardar k_auto y k_elbow en un JSON para uso por la interfaz.  Se
    # utiliza siempre el nombre <stem>_b4_kauto.json para garantizar
    # compatibilidad con la interfaz gráfica.  El stem se extrae del
    # nombre del archivo XML original.  Si la ruta no se puede
    # determinar, se mantiene el nombre basado en out_prefix.
    try:
        import json as _json
        # Determinar el stem del XML
        try:
            xml_stem = Path(xml_path).stem
        except Exception:
            xml_stem = out_prefix_name
        json_out = out_dir / f"{xml_stem}_b4_kauto.json"
        with open(json_out, 'w', encoding='utf-8') as jf:
            _json.dump({"k_elbow_auto": int(k_elbow), "k_opt_auto": int(k_opt_auto)}, jf)
    except Exception:
        pass
    # Imprimir k_elbow_auto y k_opt_auto en consola para referencia
    try:
        print(f"k_elbow_auto={int(k_elbow)}")
        print(f"k_opt_auto={int(k_opt_auto)}")
    except Exception:
        pass
    # Generar figura combinada con eje dual
    # Nombre de archivo para la figura combinada: <stem>_b4_curvas_combinadas.png
    comb_path = out_dir / f"{xml_stem}_b4_curvas_combinadas.png"
    fig_comb, ax1 = plt.subplots(figsize=(6, 4))
    fig_comb.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ln1 = ax1.plot(k_values, silhouette_vals, color='C0', marker='o', label='Silhouette')
    ax1.set_xlabel('Número de clusters (k)')
    ax1.set_ylabel('Silhouette')
    # Segundo eje para SSE
    ax2 = ax1.twinx()
    ln2 = ax2.plot(k_values, sse_vals, color='C3', linestyle='--', label='SSE/Inercia')
    ax2.set_ylabel('SSE/Inercia')
    # Líneas verticales
    ax1.axvline(k_elbow, color='gray', linestyle=':', label='Elbow auto')
    ax1.axvline(k_opt_auto, color='green', linestyle='-', label='k_opt auto')
    # Manual lines if k_manual provided and >0
    if k_manual is not None and k_manual > 0:
        ax1.axvline(k_manual, color='purple', linestyle='--', label='k_manual')
        ax1.axvline(k_opt_manual, color='orange', linestyle='-', label='k_opt manual')
    # Estilo de grilla
    ax1.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    # Leyenda combinada
    lines_comb = ln1 + ln2
    labels_comb = [l.get_label() for l in lines_comb]
    # Añadir vertical lines to legend (avoid duplicates) by plotting dummy lines
    # Recoger handles y labels existentes y añadir
    handles, labels_ = ax1.get_legend_handles_labels()
    # Para que las líneas de SSE también estén en leyenda
    handles2, labels2 = ax2.get_legend_handles_labels()
    # Crear lista final sin duplicados
    combined_handles = handles + handles2
    combined_labels = labels_ + labels2
    ax1.legend(combined_handles, combined_labels, loc='upper right', fontsize=7)
    # Título y subtítulo
    ax1.set_title('Curvas Silhouette y SSE (combinadas)')
    fig_comb.suptitle('Codo detectado automáticamente con Kneedle (Elbow)', fontsize=9, y=1.02)
    # Pie de figura con reglas (tres líneas)
    rules_text = (
        'El método del codo (Kneedle) identifica el k donde la SSE deja de disminuir\n'
        'significativamente. La curva Silhouette mide la coherencia interna: k_opt\n'
        'se toma como el que maximiza la Silhouette. k_manual se compara con estos.'
    )
    fig_comb.text(0.5, -0.15, rules_text, ha='center', va='top', fontsize=7, wrap=True)
    fig_comb.tight_layout(rect=[0, 0, 1, 0.95])
    fig_comb.savefig(comb_path, dpi=300, bbox_inches='tight')
    plt.close(fig_comb)
    # Paleta fija para clusters (misma que bloque5)
    palette = (
        (1.0, 0.0, 1.0),  # magenta
        (0.0, 1.0, 1.0),  # cian
        (0.0, 0.8, 0.0),  # verde
        (1.0, 0.5, 0.0),  # naranja
        (0.0, 0.5, 1.0),  # azul
        (0.0, 0.0, 0.0),  # negro
    )
    # Generar PRPD para k_auto
    prpd_auto_2d_path = out_dir / f"{out_prefix_name}_prpd_k_auto_2d.png"
    plot_prpd_2d(phase_corr, y_norm, qty, labels_evt_auto, k_auto, prpd_auto_2d_path, palette)
    prpd_auto_3d_path = out_dir / f"{out_prefix_name}_prpd_k_auto_3d.png"
    plot_prpd_3d(phase_corr, y_norm, qty, labels_evt_auto, k_auto, prpd_auto_3d_path, palette)
    # Generar PRPD para k_manual
    prpd_manual_2d_path = out_dir / f"{out_prefix_name}_prpd_k_manual_2d.png"
    plot_prpd_2d(phase_corr, y_norm, qty, labels_evt_manual, k_manual, prpd_manual_2d_path, palette)
    prpd_manual_3d_path = out_dir / f"{out_prefix_name}_prpd_k_manual_3d.png"
    plot_prpd_3d(phase_corr, y_norm, qty, labels_evt_manual, k_manual, prpd_manual_3d_path, palette)
    # Imprimir desplazamiento de fase para referencia y rutas generadas.
    # El valor de delta se imprime en la primera línea sin etiqueta adicional,
    # seguido de las rutas de salida.  De este modo se satisface el
    # requisito de mostrar Δ en consola sin añadir texto superfluo.
    # Imprimir la desviación de fase (delta) en la primera línea
    print(f"{shift_deg:.1f}")
    # Imprimir rutas de salida generadas.  Se emplean rutas relativas a la carpeta actual
    # para evitar que contendientes de líneas puedan confundirse.  En lugar de
    # concatenar strings directamente, se convierte cada Path en str.
    print(str(curvas_path))
    print(str(comb_path))
    print(str(prpd_auto_2d_path))
    print(str(prpd_auto_3d_path))
    print(str(prpd_manual_2d_path))
    print(str(prpd_manual_3d_path))
    print(str(metrics_path))


if __name__ == '__main__':
    main()