#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bloque 5 — Tipificación probabilística de patrones PRPD.

Este módulo implementa un análisis heurístico para asociar los
clusters obtenidos a partir de K‑Means (auto) y HDBSCAN (natural) con
categorías de defecto (cavidad, superficial, corona, flotante, otras,
ruido).  Se calculan características estadísticas por cluster a partir
de los puntos crudos del mapa PRPD, se asignan probabilidades por
tipo mediante reglas heurísticas inspiradas en documentos de referencia
(no incluidos aquí) y se fusionan los resultados de ambas
segmentaciones para ofrecer un diagnóstico global de las posibles
fuentes de descargas parciales.

Funciones:

1. `compute_prpd_features(X_phase, Y_amp, qty, labels)`
   Calcula, para cada cluster de `labels`, diversas características
   (fase central, anchura, bimodalidad, simetría, percentiles de
   amplitud, densidad pico, fracción de aislados, etc.).  Devuelve un
   diccionario por cluster con dichos rasgos y métricas de tamaño y
   densidad.

2. `map_scores_to_probs(features, sensor, tipo_tx)`
   Asigna puntuaciones heurísticas por tipo de defecto a partir de las
   características de cada cluster y las normaliza mediante softmax.
   Ajusta las puntuaciones según el sensor y el tipo de
   transformador.  Devuelve un diccionario con las probabilidades por
   tipo para cada cluster.

3. `ensemble_global(probs_auto, probs_nat)`
   Fusiona las asignaciones de las segmentaciones automática
   (K‑Means) y natural (HDBSCAN) para calcular probabilidades globales
   por tipo y estimar cuántas fuentes (clusters) hay por tipo en los
   dos escenarios.  También genera un ranking de clusters por
   confianza.

4. `summarize_dx(global_probs, cluster_probs, features)`
   Construye un informe resumido con los tres defectos más probables,
   mostrando sus características principales, y redacta un texto
   explicativo de hasta tres líneas.  Avisa si las segmentaciones
   discrepan en el tipo predominante.

El módulo incluye una interfaz de línea de comandos para procesar un
archivo XML, realizar la segmentación K‑Means y HDBSCAN (con los
parámetros del Bloque 2 y Bloque 3), calcular las probabilidades y
generar los CSV y figuras especificados.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# Importar utilidades de prpd_mega y bloques anteriores
try:
    # Todas las utilidades necesarias se importan dentro del bloque try.  Si
    # falla la importación, se lanza un error informativo.  La indentación
    # correcta dentro del bloque try evita errores de sintaxis.
    from prpd_mega import (
        parse_xml_points,
        normalize_y,
        phase_from_times,
        prpd_hist2d,
        kmeans_over_bins,
        centers_from_edges,
        hdbscan_on_bins,
        identify_sensor_from_data,
    )
except ImportError as e:
    raise ImportError(
        "No se pudo importar prpd_mega. Asegúrate de que prpd_mega.py está en el mismo directorio."
    ) from e

try:
    # Utilizar HDBSCAN si está disponible para las etiquetas naturales
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# Silenciar advertencias de sklearn deprecations
import warnings

# Importar utilidades de emparejamiento de semiciclos.  El módulo pairing_utils
# implementa el algoritmo de emparejamiento a partir de subclusters con
# distancias de fase cíclicas y diferencias de amplitud, usando el algoritmo
# húngaro cuando SciPy está disponible.  Si el módulo no se encuentra o
# produce un error, la variable se deja en None y el emparejamiento se
# desactivará silenciosamente.
try:
    from pairing_utils import pair_subclusters  # type: ignore[attr-defined]
except Exception:
    pair_subclusters = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Helper para normalizar out-prefix
# ---------------------------------------------------------------------------
def normalize_prefix(out_prefix: str, xml_file: str) -> Path:
    """Normaliza el prefijo de salida para que incluya el stem del XML.

    Si el nombre base de out_prefix no contiene el stem del archivo XML,
    inserta "_<stem>" al final. Devuelve una ruta Path resultante.

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
warnings.filterwarnings("ignore", category=FutureWarning)

# Paleta fija de colores utilizada en los bloques anteriores
DEFAULT_PALETTE = (
    (1.0, 0.0, 1.0),  # magenta
    (0.0, 1.0, 1.0),  # cian
    (0.0, 0.8, 0.0),  # verde
    (1.0, 0.5, 0.0),  # naranja
    (0.0, 0.5, 1.0),  # azul
    (0.0, 0.0, 0.0),  # negro
)

TYPE_LIST = ['cavidad', 'superficial', 'corona', 'flotante', 'otras', 'ruido']

# ---------------------------------------------------------------------------
# Utilities for phase alignment
# ---------------------------------------------------------------------------

def estimate_phase_shift_auto(
    phase_deg: np.ndarray,
    y_norm: np.ndarray,
    qty: np.ndarray,
    *,
    dphi: int = 5,
) -> Tuple[float, Dict[str, float]]:
    """Estima el corrimiento de fase óptimo para alinear cavidad/corona.

    Se construye un histograma ponderado de fase y amplitud (bins de
    360/dphi) y se calcula un puntaje en función de la simetría a
    180°, la prominencia de picos en 90°/270° y la concentración de
    la distribución.  Se devuelve el ángulo de rotación (en grados)
    que maximiza el puntaje.

    Parameters
    ----------
    phase_deg : array
        Fase original de cada evento (0–360).
    y_norm : array
        Amplitud normalizada (0–100).
    qty : array
        Cantidad de cada evento (peso).
    dphi : int, optional
        Resolución angular en grados para evaluar corrimientos.

    Returns
    -------
    best_shift : float
        Desfase en grados que maximiza el puntaje.
    metrics : dict
        Diccionario con métricas del mejor corrimiento (rho_sym, corona_score).
    """
    # Histograma con bins de 0–360 y 0–100
    bins_phi = int(360 / dphi)
    H, xedges, yedges = prpd_hist2d(phase_deg, y_norm, qty)
    # Marginal M(phi) = sum_y H.T (ny,nx) sum across y
    M = np.sum(H.T, axis=0)
    # Normalizar M
    if np.sum(M) > 0:
        M = M / np.sum(M)
    # Precomputar M para rotaciones
    M_ext = np.concatenate((M, M))  # para indexing circular
    phi_vals = np.arange(0, 360, dphi)
    best_shift = 0.0
    best_score = -np.inf
    best_rho = 0.0
    best_corona = 0.0
    # ventanas de corona ±20° alrededor de 90 y 270 en unidades de bins
    win_width = int(20 / dphi)
    idx90 = int(90 / dphi)
    idx270 = int(270 / dphi)
    for shift in phi_vals:
        s = int(shift / dphi)
        # M rotado
        M_shift = M_ext[s:s + len(M)]
        # M rotado 180°
        M_shift180 = M_ext[s + int(180 / dphi):s + int(180 / dphi) + len(M)]
        # Correlación
        rho = 0.0
        # Corr de Pearson; evitar nan
        if np.std(M_shift) > 0 and np.std(M_shift180) > 0:
            rho = float(np.corrcoef(M_shift, M_shift180)[0, 1])
        # Corona score: suma en ventanas alrededor de 90° y 270°
        corona = 0.0
        # indices modulo len(M)
        idx_range90 = [(idx90 - win_width + i) % len(M) for i in range(2 * win_width + 1)]
        idx_range270 = [(idx270 - win_width + i) % len(M) for i in range(2 * win_width + 1)]
        corona = float(np.sum(M_shift[idx_range90]) + np.sum(M_shift[idx_range270]))
        # Concentración (varianza)
        conc = float(np.var(M_shift))
        # Normalizar conc a [0,1] dividiendo por var máxima posible (~0.25)
        norm_conc = conc / 0.25
        score = max(rho, corona) + 0.1 * norm_conc
        if score > best_score:
            best_score = score
            best_shift = float(shift)
            best_rho = rho
            best_corona = corona
    return best_shift, {'rho_sym_best': best_rho, 'corona_score_best': best_corona}


def apply_phase_shift(phase_deg: np.ndarray, shift_deg: float) -> np.ndarray:
    """Aplica un corrimiento circular a la fase en grados."""
    return (phase_deg + shift_deg) % 360.0


def compute_prpd_features(
    X_phase: np.ndarray,
    Y_amp: np.ndarray,
    qty: np.ndarray,
    labels: np.ndarray,
    labels_hdb: np.ndarray | None = None,
    stability_vals: np.ndarray | None = None,
) -> Dict[Any, Dict[str, Any]]:
    """Calcula características del PRPD por cluster.

    Se agrupan los eventos según `labels` (por ejemplo K‑Means) y se
    calculan diversas métricas basadas en las fases, amplitudes y
    pesos `qty`.  Si se proporcionan etiquetas de HDBSCAN y
    probabilidades de estabilidad, se incorporan en las métricas de
    ruido y estabilidad por cluster.

    Parameters
    ----------
    X_phase : array, shape (n_events,)
        Fase de cada evento en grados (0–360).
    Y_amp : array, shape (n_events,)
        Amplitud normalizada (0–100) de cada evento.  Se asume que ya
        ha sido invertida y escalada.
    qty : array, shape (n_events,)
        Recuento de cada evento (peso).  Se utiliza para ponderar las
        estadísticas.  Valores <=0 se reemplazan por 1.
    labels : array, shape (n_events,)
        Etiqueta de cluster por evento.  Un valor negativo (–1) indica
        ruido en la segmentación correspondiente (por ejemplo HDBSCAN).
    labels_hdb : array, optional
        Etiquetas de HDBSCAN por evento.  Si se proporcionan,
        permiten calcular la fracción de puntos etiquetados como
        ruido dentro de cada cluster.
    stability_vals : array, optional
        Estabilidad de HDBSCAN por evento (valores en [0,1]).  Si se
        proporciona, se calcula la media por cluster; en caso
        contrario se utilizará un valor por defecto (0.7).

    Returns
    -------
    features : dict
        Diccionario con clave = cluster_id y valor = dict con los
        campos calculados:

        - n_pts: número de eventos en el cluster.
        - n_noise_hdb: número de eventos que HDBSCAN etiquetó como ruido.
        - frac_noise_hdb: proporción de ruido dentro del cluster.
        - weight_qty_rel: fracción de qty del cluster respecto al total.
        - phase_central: media circular de la fase (grados).
        - phase_width: desviación circular (aprox) de la fase.
        - phase_iqr: rango intercuartílico de fase.
        - band_ratio: proporción de peso en bandas de cavidad (60–120 y 240–300°).
        - semicircle_sym: diferencia normalizada entre semiciclos.
        - peaks_90_270: proporción de peso en ventanas ±20° alrededor de 90 y 270°.
        - quad_hist: densidad relativa por cuadrante (4 valores).
        - amp_p10, amp_p50, amp_p90: percentiles 10, 50 y 90 de amplitud.
        - density_peak: altura relativa del pico de densidad (fase).
        - isolated_ratio: fracción de eventos aislados (kNN) dentro del cluster.
        - stability_hdb: media de estabilidad de HDBSCAN en el cluster (o valor
          por defecto).
        - size_frac: fracción del total de eventos (por conteo).
        - density: fracción del total de qty.
    """
    features: Dict[Any, Dict[str, Any]] = {}
    n_events = len(labels)
    if n_events == 0:
        return features
    # Pesos y normalización
    weights = qty.astype(float).copy()
    weights[weights <= 0] = 1.0
    total_qty = float(np.sum(weights)) if np.sum(weights) > 0 else 1.0
    # Default stability values
    if stability_vals is None:
        stability_vals = np.full(n_events, 0.7, dtype=float)
    # Compute unique clusters
    unique_labels = np.unique(labels)
    for cl in unique_labels:
        mask = labels == cl
        n_pts = int(np.sum(mask))
        if n_pts == 0:
            continue
        phases = X_phase[mask]
        amps = Y_amp[mask]
        w = weights[mask]
        cluster_qty = float(np.sum(w))
        size_frac = float(n_pts) / float(n_events)
        weight_qty_rel = cluster_qty / total_qty if total_qty > 0 else 0.0
        # Noise counts from HDBSCAN if provided
        if labels_hdb is not None:
            hdb_labels_cluster = labels_hdb[mask]
            n_noise_hdb = int(np.sum(hdb_labels_cluster < 0))
        else:
            n_noise_hdb = 0
        frac_noise_hdb = float(n_noise_hdb) / float(n_pts) if n_pts > 0 else 0.0
        # Stability: mean of stability values if provided
        if stability_vals is not None:
            stability_cluster = stability_vals[mask]
            # clamp stability to reasonable range
            stability_h = float(np.clip(np.mean(stability_cluster), 0.3, 0.95))
        else:
            stability_h = 0.7
        # Circular statistics
        angles_rad = np.deg2rad(phases)
        C = np.sum(w * np.cos(angles_rad)) / np.sum(w)
        S = np.sum(w * np.sin(angles_rad)) / np.sum(w)
        mean_angle = np.arctan2(S, C)
        phase_central = (np.rad2deg(mean_angle) + 360.0) % 360.0
        R = np.sqrt(C**2 + S**2)
        circ_std = np.sqrt(-2.0 * np.log(max(R, 1e-8))) * (180.0 / np.pi)
        # IQR of phase (circular approximate)
        p25 = float(np.percentile(phases, 25))
        p50 = float(np.percentile(phases, 50))
        p75 = float(np.percentile(phases, 75))
        width_iqr = p75 - p25
        if width_iqr < 0:
            width_iqr += 360.0
        # Band ratio: cavity bands
        band1 = ((phases >= 60.0) & (phases <= 120.0))
        band2 = ((phases >= 240.0) & (phases <= 300.0))
        band_weight = float(np.sum(w[band1]) + np.sum(w[band2]))
        band_ratio = band_weight / cluster_qty if cluster_qty > 0 else 0.0
        # Densidad individual en bandas de cavidad
        dens_60_120 = float(np.sum(w[band1])) / cluster_qty if cluster_qty > 0 else 0.0
        dens_240_300 = float(np.sum(w[band2])) / cluster_qty if cluster_qty > 0 else 0.0
        # Semicircle symmetry (0–1): smaller means symmetric
        left = ((phases >= 0.0) & (phases < 180.0))
        right = ((phases >= 180.0) & (phases < 360.0))
        wl = float(np.sum(w[left])); wr = float(np.sum(w[right]))
        if wl + wr > 0:
            semicircle_sym = abs(wl - wr) / (wl + wr)
        else:
            semicircle_sym = 0.0
        # Balance entre semiciclos: 1 - asimetría
        balance_semiciclos = 1.0 - semicircle_sym
        # Peaks around 90 and 270 (±20°)
        win90 = ((phases >= 70.0) & (phases <= 110.0))
        win270 = ((phases >= 250.0) & (phases <= 290.0))
        peaks_90_270 = float(np.sum(w[win90]) + np.sum(w[win270])) / cluster_qty if cluster_qty > 0 else 0.0
        # Quadrant histogram (weights per quadrant normalized)
        quads = [((phases >= i) & (phases < i + 90)) for i in [0, 90, 180, 270]]
        quad_weights = [float(np.sum(w[q])) for q in quads]
        quad_sum = sum(quad_weights) + 1e-12
        quad_hist = [qw / quad_sum for qw in quad_weights]
        # Amplitude percentiles
        amp_p10 = float(np.percentile(amps, 10))
        amp_p50 = float(np.percentile(amps, 50))
        amp_p90 = float(np.percentile(amps, 90))
        # Density peak: relative max of phase histogram
        hist, _ = np.histogram(phases, bins=36, range=(0, 360), weights=w)
        density_peak = float(np.max(hist)) / (cluster_qty + 1e-8)
        # Isolated ratio using kNN within cluster (k=15)
        if n_pts >= 5:
            coords = np.column_stack((phases, amps)).astype(float)
            coords[:, 0] /= 360.0
            coords[:, 1] /= 100.0
            k = min(15, coords.shape[0])
            nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
            dists, _ = nbrs.kneighbors(coords)
            avg_dists = np.mean(dists[:, 1:], axis=1)
            thr = float(np.percentile(avg_dists, 75))
            isolated_ratio = float(np.mean(avg_dists > thr))
        else:
            isolated_ratio = 0.0
        # Lobe width: approximate half of interquartile range
        lobe_width = width_iqr / 2.0
        # Simetría 180°: correlación entre histograma de fase y su rotado 180°
        # Histograma de 36 bins
        hist_phase, _ = np.histogram(phases, bins=36, range=(0, 360), weights=w)
        if np.sum(hist_phase) > 0:
            hist_phase_norm = hist_phase / float(np.sum(hist_phase))
        else:
            hist_phase_norm = hist_phase.astype(float)
        # Rotar 180° (18 bins)
        if hist_phase_norm.size > 0:
            hist_shift = np.roll(hist_phase_norm, 18)
            # Correlación de Pearson; controlar varianza cero
            if np.std(hist_phase_norm) > 0 and np.std(hist_shift) > 0:
                simetria180 = float(np.corrcoef(hist_phase_norm, hist_shift)[0, 1])
            else:
                simetria180 = 0.0
        else:
            simetria180 = 0.0
        features[cl] = {
            'n_pts': n_pts,
            'n_noise_hdb': n_noise_hdb,
            'frac_noise_hdb': frac_noise_hdb,
            'weight_qty_rel': weight_qty_rel,
            'phase_central': phase_central,
            'phase_width': circ_std,
            'phase_iqr': width_iqr,
            'band_ratio': band_ratio,
            'semicircle_sym': semicircle_sym,
            'peaks_90_270': peaks_90_270,
            'quad_hist': quad_hist,
            'amp_p10': amp_p10,
            'amp_p50': amp_p50,
            'amp_p90': amp_p90,
            'density_peak': density_peak,
            'isolated_ratio': isolated_ratio,
            'stability_hdb': stability_h,
            'size_frac': size_frac,
            'density': weight_qty_rel,
            # Nuevos rasgos para cavidad
            'dens_60_120': dens_60_120,
            'dens_240_300': dens_240_300,
            'balance_semiciclos': balance_semiciclos,
            'lobe_width': lobe_width,
            'simetria180': simetria180,
        }
    return features


def softmax(x: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Aplica la función softmax a un vector 1D."""
    x_adj = x - np.max(x)
    exps = np.exp(x_adj / float(T))
    return exps / np.sum(exps)


def map_scores_to_probs(
    features: Dict[Any, Dict[str, Any]],
    sensor: str,
    tipo_tx: str,
    *,
    allow_otras: bool = True,
    otras_min_score: float = 0.12,
    otras_cap: float = 0.25,
    alpha: float = 0.02,
    T: float = 1.0,
) -> Dict[Any, Dict[str, float]]:
    """
    Asigna probabilidades de cada tipo de defecto a cada cluster
    utilizando reglas heurísticas y un esquema de ponderación.

    Las probabilidades se calculan a partir de puntuaciones "positivas"
    para cada tipo (cavidad, superficial, corona, flotante, otras) y
    un puntaje de ruido.  Estas puntuaciones se combinan mediante
    softmax, y luego se multiplican por el peso relativo del cluster
    (`weight_qty_rel`) y su estabilidad.  Para los clusters cuyo
    identificador es negativo (ruido de HDBSCAN), se asigna 100 %
    ruido.

    Parameters
    ----------
    features : dict
        Salida de `compute_prpd_features`, contiene los rasgos y
        pesos de cada cluster.
    sensor : str
        Tipo de sensor detectado (UHF, TEV, HFCT).  Ajusta los pesos.
    tipo_tx : str
        Tipo de transformador ('seco' o 'en aceite').  Ajusta los pesos
        para TEV en aceite.
    T : float, optional
        Temperatura para la función softmax.

    Returns
    -------
    probs : dict
        Diccionario por cluster con probabilidades de cada tipo de defecto.
    """
    sensor_norm = (sensor or '').strip().lower()
    tipo_norm = (tipo_tx or '').strip().lower()
    # Pesos por sensor
    base_weights = {'uhf': 1.0, 'tev': 1.0, 'hfct': 1.0}
    sensor_weight = base_weights.get(sensor_norm, 1.0)
    if sensor_norm == 'tev' and 'aceite' in tipo_norm:
        sensor_weight *= 0.85
    probs: Dict[Any, Dict[str, float]] = {}
    # Primera pasada: calcular scores y asignar
    raw_results: Dict[Any, np.ndarray] = {}
    for cl, feat in features.items():
        # Si cluster representa ruido directo (label negativo)
        if int(cl) < 0:
            p_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            probs[cl] = {t: float(p_vec[i]) for i, t in enumerate(TYPE_LIST)}
            continue
        # Extraer rasgos
        phase_width = feat.get('phase_width', 0.0)
        phase_iqr = feat.get('phase_iqr', 0.0)
        band_ratio = feat.get('band_ratio', 0.0)
        semicircle_sym = feat.get('semicircle_sym', 0.0)
        peaks_90_270 = feat.get('peaks_90_270', 0.0)
        density_peak = feat.get('density_peak', 0.0)
        isolated_ratio = feat.get('isolated_ratio', 0.0)
        stability = feat.get('stability_hdb', 0.7)
        frac_noise_hdb = feat.get('frac_noise_hdb', 0.0)
        weight_rel = feat.get('weight_qty_rel', 0.0)
        # Nuevos rasgos para cavidad
        dens60 = feat.get('dens_60_120', 0.0)
        dens240 = feat.get('dens_240_300', 0.0)
        balance_semiciclos = feat.get('balance_semiciclos', 0.0)
        lobe_width = feat.get('lobe_width', 0.0)
        sim180 = feat.get('simetria180', 0.0)
        # Puntuaciones "positivas" (0..1) para cada tipo
        # Cavidad reforzada: combina simetría 180°, densidad en bandas, balance, penalización por ancho y resta picos 90/270
        # Penalización del ancho (óptimo 25–40°/lóbulo). La métrica lobe_width se define como IQR/2.
        # Utilizamos un valor máximo centrado en 32.5° con un rango de 17.5°.
        pen_width = 1.0 - abs(lobe_width - 32.5) / 17.5 if lobe_width is not None else 0.0
        if pen_width < 0.0:
            pen_width = 0.0
        cavity_score = (
            0.35 * sim180 +
            0.35 * (dens60 + dens240) +
            0.15 * balance_semiciclos +
            0.10 * pen_width -
            0.05 * peaks_90_270
        )
        # Ajuste por sensor y tipo de transformador: cavidad más probable en UHF y transformadores secos
        if sensor_norm == 'uhf' and 'seco' in tipo_norm:
            cavity_score *= 1.15
        # Superficial: semiciclo dominante, anchura ancha
        superficial_score = (
            semicircle_sym * 0.5 +
            (phase_iqr / 180.0) * 0.5
        )
        # Corona: picos cercanos a 90/270 y anchura estrecha
        corona_score = (
            peaks_90_270 * 0.6 +
            (1.0 - phase_width / 180.0) * 0.4
        )
        # Flotante: anchura muy alta, densidad de pico baja, distribución homogénea
        quad_var = np.var(feat.get('quad_hist', [0.25, 0.25, 0.25, 0.25]))
        floating_score = (
            (phase_iqr / 180.0) * 0.4 +
            (1.0 - density_peak) * 0.4 +
            (0.25 - quad_var) * 0.2
        )
        # Otras: puntaje bruto basado en ausencia de patrones de otras categorías.  Se
        # calcula a partir de densidades de cavidad, picos de corona y anchuras
        # de fase.  Cuanto mayor la penalización, menor la evidencia de "Otras".
        penalty = (dens60 + dens240) + peaks_90_270 + (phase_iqr / 180.0) + (phase_width / 180.0)
        score_raw_oth = max(0.0, 1.0 - penalty)
        other_score = score_raw_oth
        # Ruido: basado en fracción de ruido y aislados y estabilidad
        if frac_noise_hdb >= 0.6:
            noise_score = 1.0
        else:
            # clamp stability to [0.3,0.95]
            stability_cl = float(np.clip(stability, 0.3, 0.95))
            noise_score = 0.2 * isolated_ratio * (1.0 - stability_cl)
        # Construir vector de scores para todas las categorías
        scores = np.array([
            cavity_score,
            superficial_score,
            corona_score,
            floating_score,
            other_score,
            noise_score,
        ], dtype=float)
        # Suavizado tipo Dirichlet: sumar alpha a cada score
        scores = scores + float(alpha)
        # Añadir epsilon para estabilidad numérica y calcular softmax
        scores = scores + 1e-6
        probs_vec = softmax(scores, T=T)
        # Bloque cavidad: si se cumplen condiciones fuertes, asegurar prob >=0.70
        if (sim180 >= 0.6 and dens60 >= 0.12 and dens240 >= 0.12 and balance_semiciclos >= 0.35):
            cav_idx = TYPE_LIST.index('cavidad')
            if probs_vec[cav_idx] < 0.7:
                remaining = 1.0 - probs_vec[cav_idx]
                if remaining > 0:
                    ratio = (1.0 - 0.7) / remaining
                    for i in range(len(probs_vec)):
                        if i != cav_idx:
                            probs_vec[i] *= ratio
                probs_vec[cav_idx] = 0.7
        # Ajustes para la categoría "Otras"
        otras_policy = 'kept'
        # Suprimir Otras si no se permite o si la evidencia bruta es menor al umbral
        if (not allow_otras) or (score_raw_oth < otras_min_score):
            otras_policy = 'suppressed'
            p_oth = float(probs_vec[4])
            if p_oth > 0:
                p_noise = float(probs_vec[5])
                four_sum = float(np.sum(probs_vec[:4]))
                if four_sum > 0:
                    probs_vec[:4] = probs_vec[:4] + p_oth * (probs_vec[:4] / four_sum)
                probs_vec[4] = 0.0
        else:
            # Mantener Otras pero caparla según otras_cap y 1 - sum(top3)
            p_oth = float(probs_vec[4])
            p_noise = float(probs_vec[5])
            four = probs_vec[:4]
            sorted_four = np.sort(four)[::-1]
            sum_top3 = float(np.sum(sorted_four[:3]))
            max_oth_allowed = min(float(otras_cap), max(0.0, 1.0 - sum_top3 - p_noise))
            if p_oth > max_oth_allowed:
                delta = p_oth - max_oth_allowed
                probs_vec[4] = max_oth_allowed
                four_sum = float(np.sum(four))
                if four_sum > 0:
                    probs_vec[:4] = probs_vec[:4] + delta * (probs_vec[:4] / four_sum)
        # Normalizar nuevamente las categorías no ruido para que sumen 1 - p_ruido
        total_non_noise = float(np.sum(probs_vec[:5]))
        if total_non_noise > 0:
            probs_vec[:5] = probs_vec[:5] / total_non_noise * (1.0 - probs_vec[5])
        # Ponderar por peso relativo y estabilidad
        probs_vec = probs_vec * weight_rel * stability
        # Ajustar sensor
        probs_vec = probs_vec * sensor_weight
        raw_results[cl] = probs_vec
    # Detectar colapso a ruido: todos los clusters no negativos predicen ruido
    non_noise_clusters = [cl for cl in raw_results if int(cl) >= 0]
    all_top_ruido = True
    for cl in non_noise_clusters:
        vec = raw_results[cl]
        top_idx = int(np.argmax(vec))
        if TYPE_LIST[top_idx] != 'ruido':
            all_top_ruido = False
            break
    if all_top_ruido and len(non_noise_clusters) > 0:
        # Reducir ruido en todos los no negativos y renormalizar
        for cl in non_noise_clusters:
            vec = raw_results[cl]
            noise_idx = TYPE_LIST.index('ruido')
            vec[noise_idx] *= 0.3
            vec = vec + 1e-12
            vec = vec / np.sum(vec)
            raw_results[cl] = vec
    # Convertir a diccionario final
    for cl in features.keys():
        if int(cl) < 0:
            # ya se asignó en la primera pasada
            continue
        vec = raw_results.get(cl, np.zeros(len(TYPE_LIST)))
        # Normalizar por suma total para asegurarse
        total = float(np.sum(vec))
        if total > 0:
            vec = vec / total
        probs[cl] = {t: float(vec[i]) for i, t in enumerate(TYPE_LIST)}
    return probs


def ensemble_global(
    probs_auto: Dict[Any, Dict[str, float]],
    probs_nat: Dict[Any, Dict[str, float]],
    features_auto: Dict[Any, Dict[str, Any]],
    features_nat: Dict[Any, Dict[str, Any]],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[int, float]], List[Tuple[str, float, str]]]:
    """
    Fusiona las asignaciones de K‑Means y HDBSCAN para obtener
    probabilidades globales y una estimación del número de fuentes por
    tipo.

    Parameters
    ----------
    probs_auto : dict
        Probabilidades por cluster K‑Means.
    probs_nat : dict
        Probabilidades por cluster HDBSCAN.
    features_auto : dict
        Características por cluster K‑Means (contiene 'density').
    features_nat : dict
        Características por cluster HDBSCAN.

    Returns
    -------
    p_global : dict
        Probabilidad global por tipo de defecto (promedio entre los
        clustering auto y natural, ponderado por densidad de cada cluster).
    prob_n_por_tipo : dict
        Para cada tipo, un diccionario {n: p(n fuentes)} donde n ∈ {0,1,2}.
    ranking : list
        Lista de tuplas (cluster_name, confianza, tipo_predicho) ordenada
        de mayor a menor confianza.  El nombre del cluster indica si es
        automático ('K{id}') o natural ('H{id}').
    """
    # Densidades por cluster para peso
    def global_probs(probs: Dict[Any, Dict[str, float]], feats: Dict[Any, Dict[str, Any]]) -> Dict[str, float]:
        """Suma ponderada de probabilidades por tipo.

        Se utiliza `weight_qty_rel` y `stability_hdb` como peso para cada
        cluster.  Si el peso total es cero, se devuelve un vector uniforme.
        """
        result = {t: 0.0 for t in TYPE_LIST}
        total_weight = 0.0
        for cl, p_dict in probs.items():
            feat = feats.get(cl, {})
            weight_rel = feat.get('weight_qty_rel', 0.0)
            stability = feat.get('stability_hdb', 0.7)
            weight = weight_rel * stability
            total_weight += weight
            for t in TYPE_LIST:
                result[t] += p_dict.get(t, 0.0) * weight
        if total_weight > 0.0:
            for t in result:
                result[t] /= total_weight
        return result
    # Probabilidades globales auto y nat
    p_auto = global_probs(probs_auto, features_auto)
    p_nat = global_probs(probs_nat, features_nat)
    # Promedio simple
    p_global = {t: (p_auto[t] + p_nat[t]) / 2.0 for t in TYPE_LIST}
    # Contar presencia de tipos en cada clustering (top type por cluster)
    def counts_per_type(probs: Dict[Any, Dict[str, float]], feats: Dict[Any, Dict[str, Any]]) -> Dict[str, int]:
        """Cuenta cuántos clusters tienen prob(tipo) > 0.5 (no ruido).
        Sólo considera clusters con identificador no negativo.
        """
        counts = {t: 0 for t in TYPE_LIST}
        for cl, p_dict in probs.items():
            if int(cl) < 0:
                continue
            for t in TYPE_LIST[:-1]:  # excepto ruido
                if p_dict.get(t, 0.0) > 0.5:
                    counts[t] += 1
        return counts
    counts_auto = counts_per_type(probs_auto, features_auto)
    counts_nat = counts_per_type(probs_nat, features_nat)
    # Prob_n: n = 0,1,2 (fuentes por tipo en dos clusterings)
    prob_n_por_tipo: Dict[str, Dict[int, float]] = {}
    for t in TYPE_LIST:
        pres_auto = 1 if counts_auto.get(t, 0) > 0 else 0
        pres_nat = 1 if counts_nat.get(t, 0) > 0 else 0
        n = pres_auto + pres_nat
        # Distribución degenerada: 100 % en el n observado
        prob_dict = {0: 0.0, 1: 0.0, 2: 0.0}
        prob_dict[n] = 1.0
        prob_n_por_tipo[t] = prob_dict
    # Ranking de clusters por confianza (max_prob * densidad)
    ranking: List[Tuple[str, float, str]] = []
    # Ranking con confianza = p_max * peso (weight_rel * stability)
    for cl, p in probs_auto.items():
        feat = features_auto.get(cl, {})
        weight = feat.get('weight_qty_rel', 0.0) * feat.get('stability_hdb', 0.7)
        top_type = max(TYPE_LIST, key=lambda t: p.get(t, 0.0))
        conf = p.get(top_type, 0.0) * weight
        ranking.append((f"K{cl}", conf, top_type))
    for cl, p in probs_nat.items():
        feat = features_nat.get(cl, {})
        weight = feat.get('weight_qty_rel', 0.0) * feat.get('stability_hdb', 0.7)
        top_type = max(TYPE_LIST, key=lambda t: p.get(t, 0.0))
        conf = p.get(top_type, 0.0) * weight
        ranking.append((f"H{cl}", conf, top_type))
    ranking.sort(key=lambda x: x[1], reverse=True)
    return p_global, prob_n_por_tipo, ranking


def summarize_dx(
    p_global: Dict[str, float],
    cluster_probs: Dict[str, Tuple[str, float]],
    features: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, List[str]]:
    """
    Genera un resumen textual de diagnóstico y lista los tres defectos
    más probables.

    Parameters
    ----------
    p_global : dict
        Probabilidad global por tipo.
    cluster_probs : dict
        Diccionario con claves de cluster (prefijo 'K' o 'H') y valor
        (tipo_predicho, confianza).
    features : dict
        Características por cluster (tanto auto como nat).  Las claves
        deben coincidir con las de cluster_probs.

    Returns
    -------
    resumen_text : str
        Texto resumido del diagnóstico global (hasta tres líneas).
    top_lines : list
        Líneas descriptivas de los tres clusters con mayor confianza.
    """
    # Ordenar tipos globales por probabilidad
    tipos_sorted = sorted(p_global.items(), key=lambda x: x[1], reverse=True)
    # Seleccionar top 3 clusters por confianza
    top_clusters = sorted(cluster_probs.items(), key=lambda x: x[1][1], reverse=True)[:3]
    lines: List[str] = []
    for name, (tipo, conf) in top_clusters:
        feat = features.get(name, {})
        phase_c = feat.get('phase_central', np.nan)
        width = feat.get('phase_width', np.nan)
        amp10 = feat.get('amp_p10', np.nan)
        amp90 = feat.get('amp_p90', np.nan)
        size_pct = feat.get('size_frac', 0.0) * 100.0
        lines.append(
            f"{name}: {tipo.capitalize()}, Conf={conf*100:.1f}%",  # tipo y confianza
        )
    # Generar texto global
    top_types = [f"{t.capitalize()} ({p*100:.0f}%)" for t, p in tipos_sorted[:3]]
    resumen = (
        f"Patrones predominantes: {', '.join(top_types)}. "
        f"Se detectan {len(cluster_probs)} clusters en total. "
        "Considere que K‑Means y HDBSCAN pueden discrepar en el número de defectos.")
    return resumen, lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bloque 5: tipificación probabilística de clusters PRPD y "
            "fusión de resultados K‑Means vs. HDBSCAN."
        )
    )
    parser.add_argument('xml_file', help='Archivo XML con datos PRPD')
    parser.add_argument('--phase-shift', type=float, default=0.0, help='Corrimiento de fase inicial (grados)')
    parser.add_argument('--k-use', type=int, default=None, help='Número de clusters K‑Means a utilizar')
    parser.add_argument('--sensor', type=str, default='UHF', help='Tipo de sensor detectado (UHF, TEV, HFCT)')
    parser.add_argument('--tipo-tx', type=str, default='seco', help='Tipo de transformador (seco o en aceite)')
    parser.add_argument('--phase-align', type=str, default='auto', help="Alineación de fase: 'auto', 'none' o un número de grados")
    # Flag opcional para solicitar reclustering después de la alineación de fase.
    # Esta línea solo debe aparecer una vez; la duplicación anterior causaba un error de argparse.
    parser.add_argument(
        '--recluster-after-align',
        action='store_true',
        help='Recalcular K‑Means y HDBSCAN después de alinear la fase'
    )
    parser.add_argument('--phase-period-ms', type=float, default=16.6667, help='Periodo en ms para el cálculo de fase a partir del tiempo')
    parser.add_argument('--sub-min-pct', type=float, default=0.02, help='Umbral mínimo de contribución (fracción) para subclusters')
    parser.add_argument(
        '--subclusters',
        action='store_true',
        help='Detectar subclusters dentro de cada cluster K‑Means y generar métricas de multiplicidad'
    )
    parser.add_argument('--allow-otras', type=str, default='true', help="Permitir categoría 'otras' (true/false)")
    parser.add_argument('--otras-min-score', type=float, default=0.12, help='Umbral mínimo de evidencia para mantener "Otras"')
    parser.add_argument('--otras-cap', type=float, default=0.25, help='Límite superior de contribución de "Otras"')
    parser.add_argument('--out-prefix', type=str, default='b5', help='Prefijo para los archivos de salida (incluya carpeta si es necesario)')
    # Paleta y transparencias para coherencia con Bloque 3
    parser.add_argument('--palette', type=str, default='paper', help="Paleta base para los gráficos (paper, tab20, pastel, viridis, warm, cool)")
    parser.add_argument('--alpha-base', type=float, default=0.25, help='Transparencia de la capa base (no aplicable en B5)')
    parser.add_argument('--alpha-clusters', type=float, default=0.85, help='Transparencia de los colores de clusters en gráficos')
    parser.add_argument('--stack-labels', type=str, choices=['none', 'top', 'all'], default='top',
                        help="Modo de etiquetado en la gráfica apilada: 'none' no muestra etiquetas, 'top' muestra sólo los clusters más grandes y 'all' muestra todas")
    parser.add_argument('--stack-top-n', type=int, default=30,
                        help='Número máximo de clusters a etiquetar cuando stack-labels=top')
    parser.add_argument('--no-annot', action='store_true',
                        help='Desactivar por completo la anotación de porcentajes sobre las barras apiladas')

    # Emparejamiento de subclusters (Corrección 2)
    # Parámetros para el algoritmo de emparejamiento.  Estos sólo aplican cuando
    # se activan los subclusters.  Ver pairing_utils.py para más detalles.
    parser.add_argument('--pair-max-phase-deg', type=float, default=25.0,
                        help='Máximo desfase angular (en grados) para considerar dos lóbulos como parte de la misma fuente')
    parser.add_argument('--pair-max-y-ks', type=float, default=0.25,
                        help='Máxima distancia relativa en amplitud (en unidades de desviación estándar) para emparejar lóbulos')
    parser.add_argument('--pair-min-weight-ratio', type=float, default=0.4,
                        help='Relación mínima de pesos (lóbulo minoritario / mayoritario) para permitir emparejar')
    parser.add_argument('--pair-miss-penalty', type=float, default=0.15,
                        help='Penalización por ausencia de pareja (en la métrica de coste)')
    parser.add_argument(
    	'--pair-enforce-same-k',
    	action='store_true',
    	help='Solo emparejar subclusters del mismo cluster KMeans (avanzado)',
           )
    parser.add_argument(
    	'--pairs-show-lines',
    	action='store_true',
    	help='Mostrar líneas de unión entre los dos semiciclos en el 3D de pares',
    )
    # Flags adicionales expuestos para pairing y resumen (si faltan, añadir)
    try:
        _ = next(a for a in parser._actions if '--pair-y-mode' in getattr(a, 'option_strings', []))
    except StopIteration:
        parser.add_argument('--pair-y-mode', type=str, choices=['abs', 'scaled', 'auto'], default='abs',
                            help="Modo para diferencia en 'y': abs (default), scaled o auto")
    try:
        _ = next(a for a in parser._actions if '--pair-hard-thresholds' in getattr(a, 'option_strings', []))
    except StopIteration:
        parser.add_argument('--pair-hard-thresholds', action='store_true',
                            help='Excluir aristas que violen umbrales antes de resolver (hard-thresholds)')
    try:
        _ = next(a for a in parser._actions if '--summary-mode' in getattr(a, 'option_strings', []))
    except StopIteration:
        parser.add_argument('--summary-mode', type=str, choices=['pre', 'postpair', 'auto'], default='auto',
                            help="Resumen: pre (antes de pairing), postpair (después), auto (regla de oro)")

    args = parser.parse_args()
    xml_path = args.xml_file
    k_use = args.k_use
    # Normalizar el prefijo de salida para incluir el stem del XML.  Si el
    # nombre base no contiene el stem, se añade automáticamente.  Esta
    # normalización unifica la nomenclatura entre los bloques 3–6.
    out_path = normalize_prefix(args.out_prefix, xml_path)
    out_dir = out_path.parent
    # Crear la carpeta padre si no existe
    if str(out_dir) != '' and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_path.name
    # Inicializar rutas de multiplicidad para imprimirlas más tarde
    multiplicity_path: Path | None = None
    multiplicity_fig_path: Path | None = None

    # Preparar paleta de colores coherente con Bloque 3 para las figuras
    # Se carga desde bloque3 si es posible; en caso de fallo, se define una
    # paleta por defecto.  Los colores se convertirán a RGBA y se ajustará
    # la transparencia con alpha_clusters para las barras, y alpha_base para
    # la categoría de ruido.
    try:
        from bloque3 import BUILTIN_PALETTES as _B3_PALETTES  # type: ignore
    except Exception:
        _B3_PALETTES = {
            'paper': (
                '#ff00ff', '#00e5ff', '#00a650', '#ff8c00', '#0077ff', '#000000'
            )
        }
    palette_name = (args.palette or 'paper').strip().lower()
    palette_raw = _B3_PALETTES.get(palette_name, _B3_PALETTES.get('paper'))
    # Convertir a lista de RGBA mediante matplotlib.colors.to_rgba
    conv_cols: List[Tuple[float, float, float, float]] = []
    for c in palette_raw:
        try:
            rgba = mcolors.to_rgba(c)
        except Exception:
            # Si es una tupla RGBA, usar tal cual
            if isinstance(c, (tuple, list)) and len(c) >= 3:
                rgba = tuple(float(x) for x in c[:4]) + (() if len(c) == 4 else (1.0,))
            else:
                rgba = (0.0, 0.0, 0.0, 1.0)
        conv_cols.append(rgba)
    # Asegurar al menos 5 colores repitiendo
    if len(conv_cols) < 5:
        conv_cols = (conv_cols * 5)[:5]
    # Construir diccionario de colores para tipos; aplicar alfa de clusters
    colors_palette: Dict[str, Tuple[float, float, float, float]] = {}
    type_order_for_colors = ['cavidad', 'superficial', 'corona', 'flotante', 'otras']
    for idx, tipo in enumerate(type_order_for_colors):
        col = list(mcolors.to_rgba(conv_cols[idx % len(conv_cols)]))
        # Ajustar la transparencia de la barra
        col[3] = max(0.0, min(1.0, args.alpha_clusters))
        colors_palette[tipo] = tuple(col)
    # Color para ruido basado en alpha_base
    base_alpha_val = max(0.0, min(1.0, args.alpha_base))
    colors_palette['ruido'] = (0.7, 0.7, 0.7, min(1.0, base_alpha_val * 0.85))
    # Cargar datos crudos desde el XML.  Utilizamos get() para aceptar
    # diferentes etiquetas alternativas: raw_y o pixel; times o ms; quantity o count.
    data = parse_xml_points(xml_path)
    raw_y_list = data.get('raw_y') if data.get('raw_y') is not None else data.get('pixel')
    times_list = data.get('times') if data.get('times') is not None else data.get('ms')
    qty_list = data.get('quantity') if data.get('quantity') is not None else data.get('count')
    sample_name = data.get('sample_name')
    # Convertir a arrays numpy y limpieza básica
    raw_y = np.array(raw_y_list, dtype=float)
    times = np.array(times_list, dtype=float)
    qty = np.array(qty_list, dtype=float)
    # Filtrar NaN e infinitos
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
    # Fase inicial a partir del tiempo y phase_shift
    phase_raw = phase_from_times(times, args.phase_shift)
    # Determinar sensor: si se pasa 'auto', utilizar identify_sensor_from_data
    sensor_norm_arg = (args.sensor or '').strip().lower()
    if sensor_norm_arg == 'auto':
        try:
            sensor_detected = identify_sensor_from_data(data)
        except Exception:
            sensor_detected = args.sensor
    else:
        sensor_detected = args.sensor
    # Calcular desplazamiento de fase según argumento
    align_spec = (args.phase_align or 'auto').lower()
    if align_spec == 'auto':
        shift_deg, shift_metrics = estimate_phase_shift_auto(phase_raw, y_norm, qty)
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
    # Fase corregida
    phase_corr = apply_phase_shift(phase_raw, shift_deg)
    # Determinar k_use si no proporcionado
    if k_use is None:
        k_use = 5
    # Mostrar en consola el número k de clusters utilizado para K‑Means
    try:
        print(f"k utilizado en Bloque 5: {k_use}")
    except Exception:
        pass
    # Elegir fase para el reclustering (si se activa) o original
    phase_for_clustering = phase_corr if args.recluster_after_align else phase_raw
    # Construir histograma para K‑Means con tamaño adaptativo (máx 240x120)
    H, xedges, yedges = prpd_hist2d(phase_for_clustering, y_norm, qty, bins_phase=240, bins_y=120)
    labels_k_grid, _sil = kmeans_over_bins(H, xedges, yedges, k_use)
    ny, nx = H.T.shape
    # Asignar cluster K a cada evento
    x_bins = np.digitize(phase_for_clustering, bins=xedges) - 1
    y_bins_evt = np.digitize(y_norm, bins=yedges) - 1
    # Inicializar etiquetas de eventos para K‑Means; usar la fase de reclustering como referencia
    labels_k_evt = np.full_like(phase_for_clustering, fill_value=-1, dtype=int)
    valid_mask = (x_bins >= 0) & (x_bins < nx) & (y_bins_evt >= 0) & (y_bins_evt < ny)
    labels_k_evt[valid_mask] = labels_k_grid[y_bins_evt[valid_mask], x_bins[valid_mask]]
    # HDBSCAN natural sobre bins y cálculo de estabilidad por punto
    # Construir puntos normalizados y pesos
    Xc, Yc = centers_from_edges(xedges, yedges)
    mask_bins = H.T > 0
    P = np.c_[Xc[mask_bins], Yc[mask_bins]]
    weights_bins = H.T[mask_bins]
    scaler = MinMaxScaler(); Pn = scaler.fit_transform(P)
    # Inicializar contenedores
    labels_h_bins = None
    bin_stability = None
    if hdbscan is not None:
        # Replicar puntos según peso
        w_int = np.round(weights_bins).astype(int)
        w_int[w_int < 1] = 1
        idx_map = np.repeat(np.arange(len(Pn)), w_int)
        Pn_rep = np.repeat(Pn, w_int, axis=0)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(50, int(0.005 * len(Pn))), cluster_selection_method='eom')
        labels_rep = clusterer.fit_predict(Pn_rep)
        # Probabilidades de estabilidad
        try:
            probs_rep = clusterer.probabilities_
        except Exception:
            probs_rep = np.ones_like(labels_rep, dtype=float)
        # Voto mayoritario y promedio de estabilidad por bin
        labels_h_bins = np.full(len(Pn), fill_value=-1, dtype=int)
        bin_stability = np.zeros(len(Pn), dtype=float)
        for i in range(len(Pn)):
            maski = idx_map == i
            labs, cnts = np.unique(labels_rep[maski], return_counts=True)
            labels_h_bins[i] = labs[np.argmax(cnts)]
            # Media de prob de estabilidad
            bin_stability[i] = float(np.mean(probs_rep[maski])) if np.any(maski) else 0.7
        noise_frac = float(np.mean(labels_rep < 0))
    else:
        # Fallback DBSCAN (sin estabilidad)
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=0.06, min_samples=5).fit(Pn, sample_weight=weights_bins)
        labels_h_bins = db.labels_
        bin_stability = np.full(len(labels_h_bins), 0.7, dtype=float)
        noise_frac = float(np.sum(weights_bins[labels_h_bins < 0])) / float(np.sum(weights_bins))
    # Reconstruir etiquetas y estabilidad HDBSCAN por evento
    labels_h_evt = np.full_like(phase_for_clustering, fill_value=-1, dtype=int)
    stability_evt = np.full_like(phase_for_clustering, fill_value=0.7, dtype=float)
    idx_bins = np.argwhere(mask_bins)
    bin_to_lab = {}
    bin_to_stab = {}
    for idx, (row, col) in enumerate(idx_bins):
        bin_to_lab[(row, col)] = labels_h_bins[idx]
        bin_to_stab[(row, col)] = bin_stability[idx]
    for i, (r, c) in enumerate(zip(y_bins_evt, x_bins)):
        if (r, c) in bin_to_lab:
            labels_h_evt[i] = bin_to_lab[(r, c)]
            stability_evt[i] = bin_to_stab[(r, c)]
    # Calcular características, pasando etiquetas HDBSCAN y estabilidad
    feats_k = compute_prpd_features(
        phase_corr, y_norm, qty, labels_k_evt,
        labels_hdb=labels_h_evt,
        stability_vals=stability_evt,
    )
    feats_h = compute_prpd_features(
        phase_corr, y_norm, qty, labels_h_evt,
        labels_hdb=labels_h_evt,
        stability_vals=stability_evt,
    )
    # Asignar probabilidades
    # Determinar parámetros para la categoría "Otras"
    allow_otras = str(args.allow_otras).strip().lower() not in ('false', '0', 'no')
    otras_min_score = float(args.otras_min_score)
    otras_cap = float(args.otras_cap)
    probs_k = map_scores_to_probs(
        feats_k,
        sensor_detected,
        args.tipo_tx,
        allow_otras=allow_otras,
        otras_min_score=otras_min_score,
        otras_cap=otras_cap,
    )
    probs_h = map_scores_to_probs(
        feats_h,
        sensor_detected,
        args.tipo_tx,
        allow_otras=allow_otras,
        otras_min_score=otras_min_score,
        otras_cap=otras_cap,
    )
    # Fusionar globalmente
    p_global, prob_n, ranking = ensemble_global(probs_k, probs_h, feats_k, feats_h)

    # Crear tarjetas de porcentajes por tipo (cards_path)
    # Mapa de colores por tipo (usar mismos colores que en DEFAULT_PALETTE y gris para ruido)
    cards_colors = {
        'cavidad': DEFAULT_PALETTE[0],
        'superficial': DEFAULT_PALETTE[1],
        'corona': DEFAULT_PALETTE[2],
        'flotante': DEFAULT_PALETTE[3],
        'otras': DEFAULT_PALETTE[4],
        'ruido': (0.7, 0.7, 0.7),
    }
    fig_cards, axs_cards = plt.subplots(2, 3, figsize=(9, 4))
    fig_cards.patch.set_facecolor('white')
    # Asegurar que axs_cards es 2x3
    idx = 0
    for t in TYPE_LIST:
        row = idx // 3
        col = idx % 3
        axc = axs_cards[row, col]
        axc.set_facecolor(cards_colors[t])
        # Texto grande con porcentaje
        pct = p_global.get(t, 0.0) * 100.0
        axc.text(
            0.5,
            0.5,
            f"{t.capitalize()}\n{pct:.0f}%",
            ha='center',
            va='center',
            fontsize=14,
            color='black' if sum(cards_colors[t][:3]) > 1.5 else 'white',
        )
        axc.set_xticks([]); axc.set_yticks([])
        axc.set_frame_on(True)
        idx += 1
    fig_cards.suptitle('Distribución de probabilidad por tipo', fontsize=12)
    fig_cards.tight_layout(rect=[0, 0.05, 1, 0.95])
    cards_path = out_dir / f"{out_base}_cards.png"
    fig_cards.savefig(cards_path, dpi=300)
    plt.close(fig_cards)
    # Preparar estructura para CSV de probabilidades
    # Combinar clusters de K y H para guardar
    csv_rows: List[List[Any]] = []
    for cl, p in probs_k.items():
        pref = f"K{cl}"
        feat = feats_k.get(cl, {})
        row = [pref] + [p.get(t, 0.0) for t in TYPE_LIST] + [
            # métricas básicas
            feat.get('stability_hdb', 0.7),
            feat.get('weight_qty_rel', 0.0),
            feat.get('size_frac', 0.0) * 100.0,
            # características añadidas (en el orden solicitado)
            feat.get('dens_60_120', 0.0),
            feat.get('dens_240_300', 0.0),
            feat.get('balance_semiciclos', 0.0),
            feat.get('lobe_width', 0.0),
            feat.get('simetria180', 0.0),
            # mejor corrimiento y flag de reclustering
            shift_deg,
            args.recluster_after_align,
            # métricas de alineación como información adicional
            shift_metrics.get('rho_sym_best', 0.0),
            shift_metrics.get('corona_score_best', 0.0),
        ]
        csv_rows.append(row)
    for cl, p in probs_h.items():
        pref = f"H{cl}"
        feat = feats_h.get(cl, {})
        row = [pref] + [p.get(t, 0.0) for t in TYPE_LIST] + [
            # métricas básicas
            feat.get('stability_hdb', 0.7),
            feat.get('weight_qty_rel', 0.0),
            feat.get('size_frac', 0.0) * 100.0,
            # características añadidas
            feat.get('dens_60_120', 0.0),
            feat.get('dens_240_300', 0.0),
            feat.get('balance_semiciclos', 0.0),
            feat.get('lobe_width', 0.0),
            feat.get('simetria180', 0.0),
            # mejor corrimiento y flag de reclustering
            shift_deg,
            args.recluster_after_align,
            # métricas de alineación
            shift_metrics.get('rho_sym_best', 0.0),
            shift_metrics.get('corona_score_best', 0.0),
        ]
        csv_rows.append(row)
    # Guardar CSV de probabilidades
    prob_csv_path = out_dir / f"{out_base}_probabilities.csv"
    with open(prob_csv_path, 'w') as f:
        # Header: primero las columnas de probabilidad y métricas básicas,
        # seguido de las características adicionales solicitadas. El orden
        # de las nuevas columnas sigue la especificación del enunciado
        # (dens_60_120, dens_240_300, balance_semiciclos, ancho_lobulo,
        # simetria180, best_shift_deg, reclustered).  Se dejan las métricas
        # de alineación (rho_sym_best, corona_score_best) al final como
        # información extra.
        headers = (
            ['cluster'] + TYPE_LIST +
            [
                'estabilidad',
                'densidad',
                'tamano_pct',
                'dens_60_120',
                'dens_240_300',
                'balance_semiciclos',
                'ancho_lobulo',
                'simetria180',
                'best_shift_deg',
                'reclustered',
                # métricas de alineación utilizadas internamente
                'rho_sym_best',
                'corona_score_best',
            ]
        )
        f.write(','.join(headers) + '\n')
        for row in csv_rows:
            f.write(','.join(str(round(x, 5)) if isinstance(x, float) else str(x) for x in row) + '\n')
    # Calcular evidencia global de "Otras" para la política
    # Se basa en la suma ponderada de scores brutos por cluster
    raw_sum = 0.0
    weight_sum = 0.0
    def compute_raw_other(feat: Dict[str, Any]) -> float:
        dens60 = feat.get('dens_60_120', 0.0)
        dens240 = feat.get('dens_240_300', 0.0)
        peaks_90_270 = feat.get('peaks_90_270', 0.0)
        phase_iqr = feat.get('phase_iqr', 0.0)
        phase_width = feat.get('phase_width', 0.0)
        penalty = (dens60 + dens240) + peaks_90_270 + (phase_iqr / 180.0) + (phase_width / 180.0)
        return max(0.0, 1.0 - penalty)
    # Considerar clusters de K y H para la evidencia global
    for cl, feat in feats_k.items():
        if int(cl) < 0:
            continue
        w = float(feat.get('weight_qty_rel', 0.0)) * float(feat.get('stability_hdb', 0.7))
        raw_sum += w * compute_raw_other(feat)
        weight_sum += w
    for cl, feat in feats_h.items():
        if int(cl) < 0:
            continue
        w = float(feat.get('weight_qty_rel', 0.0)) * float(feat.get('stability_hdb', 0.7))
        raw_sum += w * compute_raw_other(feat)
        weight_sum += w
    global_raw_other = raw_sum / weight_sum if weight_sum > 0 else 0.0
    global_oth_policy = 'suppressed' if ((not allow_otras) or (global_raw_other < otras_min_score)) else 'kept'
    # Se pospone la escritura del archivo de resumen hasta después del análisis de
    # subclusters (si corresponde), para incluir métricas de multiplicidad.  El
    # contenido de p_global, prob_n y la política de 'Otras' se guarda en
    # variables locales y se utilizará al final.
    summary_csv_path = out_dir / f"{out_base}_summary.csv"
    # Figura barras apiladas por cluster
    # Ordenar clusters por nombre
    cluster_names = [row[0] for row in csv_rows]
    cluster_probs = np.array([row[1:1 + len(TYPE_LIST)] for row in csv_rows], dtype=float)
    # Determinar modo de anotación: si se utiliza el flag --no-annot, se desactiva por completo
    annot_mode = 'none' if getattr(args, 'no_annot', False) else args.stack_labels
    # Calcular tamaño relativo de cada cluster para priorizar etiquetado cuando corresponde
    cluster_sizes: List[float] = []
    for name in cluster_names:
        # Extraer el identificador numérico y seleccionar la fuente de características
        if name.startswith('K'):
            cl_id = int(name[1:])
            size = feats_k.get(cl_id, {}).get('size_frac', 0.0)
        else:
            cl_id = int(name[1:])
            size = feats_h.get(cl_id, {}).get('size_frac', 0.0)
        cluster_sizes.append(float(size))
    # Indices ordenados por tamaño descendente
    sorted_indices = sorted(range(len(cluster_sizes)), key=lambda i: cluster_sizes[i], reverse=True)
    top_n = int(args.stack_top_n) if hasattr(args, 'stack_top_n') else 30
    top_indices = set(sorted_indices[:top_n])
    # Ajustar tamaño de figura según número de clusters
    n_clust = len(cluster_names)
    fig_width = 8.0
    if n_clust > 30:
        # Escalar ancho linealmente con el número de clusters, hasta un límite razonable
        fig_width = min(20.0, 6.0 + 0.15 * n_clust)
    fig_height = 4.0
    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    fig1.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    bottom = np.zeros(len(cluster_probs))
    # Usar la paleta preparada en colors_palette para los tipos
    colors = colors_palette.copy()
    # Dibujar barras por tipo
    for i, t in enumerate(TYPE_LIST):
        vals = cluster_probs[:, i]
        bars = ax1.bar(cluster_names, vals, bottom=bottom, color=colors[t], label=t.capitalize())
        # Anotar porcentaje dentro de cada segmento según modo
        for j, val in enumerate(vals):
            # Saltar anotaciones de cero o negativas
            if val <= 0.0:
                continue
            # Evitar saturación: no anotar porcentajes menores al 3%
            if val < 0.03:
                continue
            # Determinar si se debe anotar esta barra
            do_annot = False
            if annot_mode == 'all':
                do_annot = True
            elif annot_mode == 'top' and j in top_indices:
                do_annot = True
            elif annot_mode == 'none':
                do_annot = False
            if not do_annot:
                continue
            y_pos = bottom[j] + val / 2.0
            ax1.text(
                j,
                y_pos,
                f"{val * 100:.0f}%",
                ha='center',
                va='center',
                fontsize=6,
                color='black',
                bbox=dict(
                    facecolor=(0.85, 0.85, 0.85, 0.85),  # fondo gris claro con 85% opacidad
                    edgecolor='none',
                    boxstyle='round,pad=0.1'
                )
            )
        bottom += vals
    # Etiquetas de ejes
    ax1.set_xticks(range(len(cluster_names)))
    ax1.set_xticklabels(cluster_names, rotation=45, ha='right', fontsize=6)
    ax1.set_ylabel('Probabilidad')
    ax1.set_title('Distribución de probabilidad por cluster\nphase align: +{}° ({}) ; recluster={}'.format(int(round(shift_deg)), align_mode, 'sí' if args.recluster_after_align else 'no'))
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=7)
    # Añadir badge bajo cada cluster con su tipo y probabilidad máxima
    for j, name in enumerate(cluster_names):
        probs_vec = cluster_probs[j]
        top_idx = int(np.argmax(probs_vec))
        top_tipo = TYPE_LIST[top_idx]
        top_prob = probs_vec[top_idx]
        x_rel = (j + 0.5) / n_clust if n_clust > 0 else 0.5
        ax1.text(
            x_rel,
            -0.08,
            f"Top: {top_tipo.capitalize()} {top_prob * 100:.0f}%",
            transform=ax1.transAxes,
            ha='center',
            va='top',
            fontsize=6,
        )
    fig1.tight_layout(rect=[0, 0.1, 0.85, 1])
    stack_path = out_dir / f"{out_base}_stack_by_cluster.png"
    fig1.savefig(stack_path, dpi=300)
    plt.close(fig1)
    # Figura global mix
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    types = TYPE_LIST
    vals = [p_global[t] for t in types]
    ax2.bar(types, vals, color=[colors[t] for t in types])
    ax2.set_ylabel('Probabilidad global')
    ax2.set_title('Proporción global por tipo\nphase align: +{}° ({}); recluster={}'.format(int(round(shift_deg)), align_mode, 'sí' if args.recluster_after_align else 'no'))
    # Establecer primero las posiciones de ticks para evitar la advertencia
    ax2.set_xticks(range(len(types)))
    ax2.set_xticklabels([t.capitalize() for t in types], rotation=45, ha='right')
    # Añadir porcentaje numérico encima de cada barra
    for i, val in enumerate(vals):
        ax2.text(
            i,
            val + 0.02,
            f"{val*100:.0f}%",
            ha='center',
            va='bottom',
            fontsize=8,
        )
    fig2.tight_layout()
    global_mix_path = out_dir / f"{out_base}_global_mix.png"
    fig2.savefig(global_mix_path, dpi=300)
    plt.close(fig2)
    # Preparar ranking y resumen textual
    cluster_conf: Dict[str, Tuple[str, float]] = {}
    # Mapear features con prefijos
    feats_pref: Dict[str, Dict[str, Any]] = {}
    for cl, p in probs_k.items():
        name = f"K{cl}"
        top_type = max(TYPE_LIST, key=lambda t: p[t])
        conf = p[top_type] * feats_k.get(cl, {}).get('density', 0.0)
        cluster_conf[name] = (top_type, conf)
        # Añadir features prefijados
        fdict = feats_k.get(cl, {}).copy()
        feats_pref[name] = fdict
    for cl, p in probs_h.items():
        name = f"H{cl}"
        top_type = max(TYPE_LIST, key=lambda t: p[t])
        conf = p[top_type] * feats_h.get(cl, {}).get('density', 0.0)
        cluster_conf[name] = (top_type, conf)
        fdict = feats_h.get(cl, {}).copy()
        feats_pref[name] = fdict
    # Generar resumen textual, pero no imprimirlo en consola.  El resumen
    # se genera para completar la funcionalidad interna pero se omite la
    # impresión para cumplir la regla de UX que prohíbe texto extenso.
    resumen_text, lines = summarize_dx(p_global, cluster_conf, feats_pref)

    # Generar tabla de clusters principales (b5_top_clusters_table.csv)
    # Recorrer clusters K y H para construir filas con tipo predominante y probabilidades
    top_table_rows = []
    for cl, p in probs_k.items():
        name = f"K{cl}"
        top_type = max(TYPE_LIST, key=lambda t: p[t])
        top_prob = p[top_type]
        feat = feats_k.get(cl, {})
        size_pct = feat.get('size_frac', 0.0) * 100.0
        stability = feat.get('stability_hdb', 0.7)
        frac_noise = feat.get('frac_noise_hdb', 0.0)
        top_table_rows.append([
            name, top_type, f"{top_prob:.5f}", f"{size_pct:.2f}", f"{stability:.3f}", f"{frac_noise:.3f}"
        ])
    for cl, p in probs_h.items():
        name = f"H{cl}"
        top_type = max(TYPE_LIST, key=lambda t: p[t])
        top_prob = p[top_type]
        feat = feats_h.get(cl, {})
        size_pct = feat.get('size_frac', 0.0) * 100.0
        stability = feat.get('stability_hdb', 0.7)
        frac_noise = feat.get('frac_noise_hdb', 0.0)
        top_table_rows.append([
            name, top_type, f"{top_prob:.5f}", f"{size_pct:.2f}", f"{stability:.3f}", f"{frac_noise:.3f}"
        ])
    top_clusters_path = out_dir / f"{out_base}_top_clusters_table.csv"
    with open(top_clusters_path, 'w') as f:
        f.write('cluster_id,top_tipo,top_prob,size_pct,stability_hdb,frac_ruido_hdb\n')
        for row in top_table_rows:
            f.write(','.join(map(str, row)) + '\n')

    # Procesar subclusters si se solicita
    # Inicializar variables para multiplicidad global
    p_cavidad_ge2 = 0.0
    p_superficial_ge2 = 0.0
    p_corona_ge2 = 0.0
    p_flotante_ge2 = 0.0
    if args.subclusters:
        multiplicity_rows = []
        total_cav = 0
        total_corona = 0
        total_superf = 0
        subcluster_global_info = []  # para figura
        subcluster_counter = 1
        # Mapeo de eventos a identificadores de subcluster global.  Se
        # inicializa con cadenas vacías y se llenará al asignar
        # subclusters en cada cluster.
        evt_subcluster_id: list[str] = [''] * len(phase_corr)
        # Precompute event bin indices for cluster assignment to accelerate mapping
        for cl in sorted(np.unique(labels_k_evt)):
            if cl < 0:
                continue
            # Filtrar eventos del cluster
            mask_cl = labels_k_evt == cl
            idx_cl = np.nonzero(mask_cl)[0]
            if idx_cl.size == 0:
                continue
            phases_cl = phase_corr[idx_cl]
            y_cl = y_norm[idx_cl]
            qty_cl = qty[idx_cl]
            stability_cl = stability_evt[idx_cl]
            # histograma sobre cluster
            H_cl, xedges_cl, yedges_cl = prpd_hist2d(phases_cl, y_cl, qty_cl)
            # mínimo tamaño de cluster en términos de eventos (0.5% del número de eventos)
            n_ev_cl = len(idx_cl)
            min_cluster_size = max(2, int(0.005 * n_ev_cl))
            # ejecutar hdbscan_on_bins
            grid_sub, info_sub, used_hdb_sub = hdbscan_on_bins(H_cl, xedges_cl, yedges_cl, min_cluster_size=min_cluster_size)
            # asignar etiquetas por evento dentro del cluster
            labels_sub_evt = np.full(idx_cl.shape[0], fill_value=-1, dtype=int)
            # mapa de (bin) -> label
            # Calcular centros para cluster
            nx_sub = len(xedges_cl) - 1
            ny_sub = len(yedges_cl) - 1
            # Build mapping for bins with labels
            # For events, compute bin indices relative to cluster edges
            x_bins_evt_cl = np.digitize(phases_cl, bins=xedges_cl) - 1
            y_bins_evt_cl = np.digitize(y_cl, bins=yedges_cl) - 1
            for i_evt, (r_bin, c_bin) in enumerate(zip(y_bins_evt_cl, x_bins_evt_cl)):
                if 0 <= r_bin < grid_sub.shape[0] and 0 <= c_bin < grid_sub.shape[1]:
                    labels_sub_evt[i_evt] = int(grid_sub[r_bin, c_bin])
            # Calcular pesos por subcluster
            unique_sub = [lab for lab in np.unique(labels_sub_evt) if lab >= 0]
            sub_info_list = []
            total_qty_cluster = float(np.sum(qty_cl)) if np.sum(qty_cl) > 0 else 1.0
            for lab in unique_sub:
                mask_sub = labels_sub_evt == lab
                weight = float(np.sum(qty_cl[mask_sub]))
                if total_qty_cluster > 0:
                    frac = weight / total_qty_cluster
                else:
                    frac = 0.0
                # Filtrar subclusters con peso relativo menor al umbral especificado
                if frac < args.sub_min_pct:
                    continue
                # Estabilidad media del subcluster (usar valores de estabilidad natural)
                stab_mean = float(np.mean(stability_cl[mask_sub])) if np.any(mask_sub) else 0.7
                # Fase media ponderada
                phase_vals = phases_cl[mask_sub]
                weights_sub = qty_cl[mask_sub]
                if np.sum(weights_sub) > 0:
                    # Circular media ponderada
                    angles_rad = np.deg2rad(phase_vals)
                    C_sub = np.sum(weights_sub * np.cos(angles_rad)) / np.sum(weights_sub)
                    S_sub = np.sum(weights_sub * np.sin(angles_rad)) / np.sum(weights_sub)
                    phase_mean = (np.rad2deg(np.arctan2(S_sub, C_sub)) + 360.0) % 360.0
                else:
                    phase_mean = float(np.mean(phase_vals)) if len(phase_vals) > 0 else 0.0
                # Amplitud media ponderada
                y_mean = float(np.average(y_cl[mask_sub], weights=weights_sub)) if np.sum(weights_sub) > 0 else float(np.mean(y_cl[mask_sub]))
                sub_info_list.append({
                    'label': lab,
                    'weight': weight,
                    'frac': frac,
                    'stability': stab_mean,
                    'phase_mean': phase_mean,
                    'y_mean': y_mean,
                })
            # Ordenar subclusters por peso descendente
            sub_info_list.sort(key=lambda x: -x['weight'])
            # Métricas de multiplicidad
            n_sub = len(sub_info_list)
            if n_sub == 0:
                avg_stability = 0.0
            else:
                avg_stability = float(np.mean([si['stability'] for si in sub_info_list]))
            # Conteos de tipos de fenómenos
            n_cav_lobes = 0
            n_cor_groups = 0
            n_sup_bands = 0
            # Recorrer subclusters para contarlos
            for si in sub_info_list:
                phi = si['phase_mean']
                # cavidad: en 60–120 o 240–300
                if (60.0 <= phi <= 120.0) or (240.0 <= phi <= 300.0):
                    n_cav_lobes += 1
                # corona: cerca de 90 o 270 ±20
                if (70.0 <= phi <= 110.0) or (250.0 <= phi <= 290.0):
                    n_cor_groups += 1
                # superficial: predominio en un semiciclo
                # Se calcula tomando la proporción de eventos en semiciclo respecto al total
                mask_sub_local = labels_sub_evt == si['label']
                if np.sum(mask_sub_local) > 0:
                    phases_sub = phases_cl[mask_sub_local]
                    w_sub = qty_cl[mask_sub_local]
                    wl = float(np.sum(w_sub[(phases_sub >= 0.0) & (phases_sub < 180.0)]))
                    wr = float(np.sum(w_sub[(phases_sub >= 180.0) & (phases_sub < 360.0)]))
                    total_w = wl + wr
                    if total_w > 0:
                        ratio_dom = max(wl, wr) / total_w
                        if ratio_dom >= 0.66:
                            n_sup_bands += 1
                # Registrar para figura global y asignar ID a eventos
                sc_id = f"C{subcluster_counter}"
                # Asignar identificador de subcluster a eventos originales
                for evt_idx_local, assign_val in enumerate(mask_sub_local):
                    if assign_val:
                        global_idx = idx_cl[evt_idx_local]
                        if 0 <= global_idx < len(evt_subcluster_id):
                            evt_subcluster_id[global_idx] = sc_id
                # Añadir información global de subcluster
                subcluster_global_info.append({
                    'id': sc_id,
                    'phase': si['phase_mean'],
                    'y': si['y_mean'],
                    'weight': si['weight'],
                    'frac': si.get('frac', 0.0),
                    'stability': si.get('stability', 0.0),
                    # Etiqueta del cluster KMeans para permitir emparejar dentro del mismo K si se solicita
                    'k_label': cl,
                    # Tipo dominante del cluster padre (para colorear y clasificar)
                    'type': max(TYPE_LIST, key=lambda tt: probs_k[cl][tt] if cl in probs_k else 0.0),
                })
                subcluster_counter += 1
            # Determinar top tipo para cluster
            p_dict = probs_k.get(cl, {})
            top_tipo_cl = max(TYPE_LIST, key=lambda t: p_dict.get(t, 0.0)) if p_dict else ''
            multiplicity_rows.append([
                f"K{cl}",
                top_tipo_cl,
                n_sub,
                f"{avg_stability:.3f}",
                n_cav_lobes,
                n_cor_groups,
                n_sup_bands,
            ])
            total_cav += n_cav_lobes
            total_corona += n_cor_groups
            total_superf += n_sup_bands
        # Añadir fila global
        multiplicity_rows.append([
            'GLOBAL',
            '',
            '',
            '',
            total_cav,
            total_corona,
            total_superf,
        ])
        # Guardar CSV de multiplicidad
        multiplicity_path = out_dir / f"{out_base}_multiplicity.csv"
        with open(multiplicity_path, 'w') as f:
            f.write('cluster_id,top_tipo,n_subclusters,estabilidad_media,n_cavity_lobes,n_corona_groups,n_superficial_bands\n')
            for row in multiplicity_rows:
                f.write(','.join(map(str, row)) + '\n')
        # Generar figura de burbujas si hay subclusters
        if len(subcluster_global_info) > 0:
            fig_mul, ax_mul = plt.subplots(figsize=(6, 4))
            fig_mul.patch.set_facecolor('white')
            ax_mul.set_facecolor('white')
            # Mapear tipos a colores
            type_colors = {
                'cavidad': DEFAULT_PALETTE[0],
                'superficial': DEFAULT_PALETTE[1],
                'corona': DEFAULT_PALETTE[2],
                'flotante': DEFAULT_PALETTE[3],
                'otras': DEFAULT_PALETTE[4],
                'ruido': (0.7, 0.7, 0.7, 0.5),
            }
            for info in subcluster_global_info:
                ax_mul.scatter(
                    info['phase'],
                    info['y'],
                    s=50 * (info['weight'] / max(1.0, max(si['weight'] for si in subcluster_global_info))),
                    color=type_colors.get(info['type'], (0.5, 0.5, 0.5)),
                    alpha=0.7,
                )
                ax_mul.text(
                    info['phase'],
                    info['y'],
                    info['id'],
                    fontsize=7,
                    ha='center',
                    va='center',
                    color='black'
                )
            ax_mul.set_xlabel('Fase (°)')
            ax_mul.set_ylabel('Amplitud normalizada')
            ax_mul.set_xlim(0, 360)
            ax_mul.set_ylim(0, 100)
            ax_mul.set_title('Subclusters detectados\nphase align: +{}° ({}); recluster={}'.format(int(round(shift_deg)), align_mode, 'sí' if args.recluster_after_align else 'no'))
            ax_mul.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
            fig_mul.tight_layout()
            multiplicity_fig_path = out_dir / f"{out_base}_multiplicity.png"
            fig_mul.savefig(multiplicity_fig_path, dpi=300)
            plt.close(fig_mul)
        # Calcular probabilidad de multiplicidad (≥2 fuentes) para cada tipo basándonos en subclusters.
        # Cada subcluster válido contribuye un peso efectivo w = frac * max(stability, 0.3).
        # Considerar sólo subclusters cuyo w >= 0.05.
        type_candidates: Dict[str, list[float]] = {t: [] for t in ['cavidad', 'superficial', 'corona', 'flotante']}
        for info in subcluster_global_info:
            t = info.get('type', '')
            if t not in type_candidates:
                continue
            frac_val = float(info.get('frac', 0.0))
            stab_val = float(info.get('stability', 0.0))
            w_eff = frac_val * max(stab_val, 0.3)
            if w_eff >= 0.05:
                type_candidates[t].append(w_eff)
        def compute_p_ge2(lst: list[float]) -> float:
            total = 0.0
            n = len(lst)
            for i in range(n):
                for j in range(i + 1, n):
                    total += lst[i] * lst[j]
            # Clamp to [0,1]
            if total < 0.0:
                total = 0.0
            if total > 1.0:
                total = 1.0
            return total
        p_cavidad_ge2 = compute_p_ge2(type_candidates['cavidad'])
        p_superficial_ge2 = compute_p_ge2(type_candidates['superficial'])
        p_corona_ge2 = compute_p_ge2(type_candidates['corona'])
        p_flotante_ge2 = compute_p_ge2(type_candidates['flotante'])
        # Imprimir en consola un resumen de multiplicidad basado en subclusters
        try:
            print(
                f"Multiplicidad (subclusters): Cavidad p(≥2)={p_cavidad_ge2:.3f}, "
                f"Superficial p(≥2)={p_superficial_ge2:.3f}, Corona p(≥2)={p_corona_ge2:.3f}, "
                f"Flotante p(≥2)={p_flotante_ge2:.3f}"
            )
        except Exception:
            pass
        # Generar CSV con información de subclusters para reporte final.  Esto
        # incluye únicamente subclusters válidos (filtrados por frac >=
        # sub_min_pct) y ordenados por peso descendente.  Se registran el
        # identificador, tipo dominante, fracción relativa, estabilidad,
        # peso absoluto y promedios de fase y amplitud.
        try:
            subclusters_path = out_dir / f"{out_base}_subclusters.csv"
            with open(subclusters_path, 'w') as f:
                f.write('id,type,frac,stability,weight,phase,y\n')
                # Ordenar por peso descendente
                sorted_subs = sorted(subcluster_global_info, key=lambda x: -float(x.get('weight', 0.0)))
                for info in sorted_subs:
                    f.write(
                        f"{info.get('id','')},{info.get('type','')},{info.get('frac',0.0):.5f},"
                        f"{info.get('stability',0.0):.5f},{info.get('weight',0.0):.5f},"
                        f"{info.get('phase',0.0):.1f},{info.get('y',0.0):.1f}\n"
                    )
        except Exception:
            subclusters_path = None
        # Realizar emparejamiento de subclusters si pairing_utils está disponible y se activó subclusters
        paired_sources_info = []
        paired_mix: Dict[str, float] = {}
        p_ge2_paired: Dict[str, float] = {}
        paired_sources_path = None  # Ruta al CSV de fuentes emparejadas
        p_mult_csv_path = None  # Ruta al CSV de multiplicidad por tipo
        top_sources_path = None  # Ruta al CSV de ranking de fuentes
        paired_fig2d_path = None
        paired_fig3d_path = None
        paired_map_path = None
        if pair_subclusters is not None and len(subcluster_global_info) > 0:
            try:
                # Ejecutar algoritmo de emparejamiento sobre la lista de subclusters.
                # La función retorna una lista de pares y un mapa de subcluster→pair_id.
                pairs_ret = pair_subclusters(
                    subcluster_global_info,
                    pair_max_phase_deg=args.pair_max_phase_deg,
                    pair_max_y_ks=args.pair_max_y_ks,
                    pair_min_weight_ratio=args.pair_min_weight_ratio,
                    pair_miss_penalty=args.pair_miss_penalty,
                    y_mode=getattr(args, 'pair_y_mode', 'auto'),
                    hard_thresholds=getattr(args, 'pair_hard_thresholds', True),
                    enforce_same_k=getattr(args, 'pair_enforce_same_k', False),
                )
                # Soporta ambas firmas: si se devuelve sólo una lista, envolver en tupla.
                if isinstance(pairs_ret, tuple) and len(pairs_ret) == 2:
                    pairs, pair_map = pairs_ret
                else:
                    pairs = pairs_ret
                    pair_map = {}
                # Construir lista de fuentes emparejadas con campos agregados.
                paired_sources: List[Dict[str, Any]] = []
                for idx_pair, pair in enumerate(pairs, start=1):
                    # Recoger IDs de subclusters asociados
                    clusters_list: List[str] = []
                    id_pos = str(pair.get('id_pos', '')).strip()
                    id_neg = str(pair.get('id_neg', '')).strip()
                    if id_pos:
                        clusters_list.append(id_pos)
                    if id_neg:
                        clusters_list.append(id_neg)
                    # Recuperar fases y amplitudes originales de los subclusters
                    phase_centers: List[float] = []
                    y_stats: List[float] = []
                    for cid in clusters_list:
                        for info in subcluster_global_info:
                            if str(info.get('id')) == cid:
                                phase_centers.append(float(info.get('phase', 0.0)))
                                y_stats.append(float(info.get('y', 0.0)))
                                break
                    qty_weight = float(pair.get('weight_sum', 0.0))
                    # Convertir coste a puntuación (1 - coste).  Garantizar rango [0,1].
                    score_cost = float(pair.get('score', 0.0))
                    pair_score = max(0.0, min(1.0, 1.0 - score_cost))
                    paired_sources.append({
                        'source_id': f"S{idx_pair}",
                        'type_guess': pair.get('type', ''),
                        'clusters': clusters_list,
                        'phase_centers': phase_centers,
                        'y_stats': y_stats,
                        'qty_weight': qty_weight,
                        'pair_score': pair_score,
                    })
                # Guardar paired_sources.csv
                paired_sources_path = out_dir / f"{out_base}_paired_sources.csv"
                with open(paired_sources_path, 'w') as f:
                    f.write('source_id,type_guess,clusters,phase_centers,y_stats,qty_weight,pair_score\n')
                    for src in paired_sources:
                        clusters_str = ';'.join(str(c) for c in src.get('clusters', []))
                        phases_str = ';'.join(f"{float(p):.1f}" for p in src.get('phase_centers', []))
                        ys_str = ';'.join(f"{float(y):.1f}" for y in src.get('y_stats', []))
                        f.write(
                            f"{src.get('source_id','')},{src.get('type_guess','')},{clusters_str},{phases_str},{ys_str},"
                            f"{float(src.get('qty_weight',0.0)):.5f},{float(src.get('pair_score',0.0)):.4f}\n"
                        )
                # Guardar mapeo subcluster→pair_id en JSON
                try:
                    import json
                    paired_map_path = out_dir / f"{out_base}_paired_map.json"
                    # Si pair_map existe, utilizarlo; de lo contrario, construir a partir de paired_sources
                    map_dict: Dict[str, int] = {}
                    if 'pair_map' in locals() and isinstance(pair_map, dict):
                        map_dict = {str(k): int(v) for k, v in pair_map.items()}
                    else:
                        # Construir a partir de clusters en paired_sources
                        for idx, src in enumerate(paired_sources, start=1):
                            for cid in src.get('clusters', []):
                                map_dict[str(cid)] = idx
                    with open(paired_map_path, 'w', encoding='utf-8') as fpm:
                        json.dump(map_dict, fpm, ensure_ascii=False, indent=2)
                except Exception:
                    paired_map_path = None
                # Calcular mezcla global y multiplicidad a partir de fuentes emparejadas
                # Sumar pesos por tipo y calcular fracciones
                total_w = 0.0
                weight_by_type: Dict[str, float] = {}
                for src in paired_sources:
                    t = src.get('type_guess', '')
                    w = float(src.get('qty_weight', 0.0))
                    total_w += w
                    weight_by_type[t] = weight_by_type.get(t, 0.0) + w
                if total_w > 0:
                    for t in ['cavidad', 'superficial', 'corona', 'flotante', 'otras', 'ruido']:
                        paired_mix[t] = weight_by_type.get(t, 0.0) / total_w
                # Calcular probabilidad de ≥2 fuentes por tipo
                for t in ['cavidad', 'superficial', 'corona', 'flotante']:
                    # lista de fracciones de peso por tipo
                    fracs = []
                    for src in paired_sources:
                        if src.get('type_guess', '') == t:
                            w = float(src.get('qty_weight', 0.0))
                            if total_w > 0:
                                fracs.append(w / total_w)
                    # calcular probabilidad de dos o más
                    s = 0.0
                    n = len(fracs)
                    for i in range(n):
                        for j in range(i + 1, n):
                            s += fracs[i] * fracs[j]
                    p_ge2_paired[t] = min(max(s, 0.0), 1.0)
                # Guardar CSV de multiplicidad por tipo (post-fusión)
                p_mult_csv_path = out_dir / f"{out_base}_p_multiplicity.csv"
                with open(p_mult_csv_path, 'w') as f:
                    f.write('tipo,p_mix,p_ge2\n')
                    for t in ['cavidad', 'superficial', 'corona', 'flotante']:
                        f.write(
                            f"{t},{paired_mix.get(t,0.0):.5f},{p_ge2_paired.get(t,0.0):.5f}\n"
                        )
                # Generar ranking de fuentes por peso
                sorted_sources = sorted(paired_sources, key=lambda x: -float(x.get('qty_weight', 0.0)))
                top_sources_path = out_dir / f"{out_base}_top_sources_table.csv"
                with open(top_sources_path, 'w') as f:
                    f.write('source_id,type_guess,weight,clusters,phase_centers,y_stats,pair_score\n')
                    for src in sorted_sources:
                        clusters_str = ';'.join(str(c) for c in src.get('clusters', []))
                        phases_str = ';'.join(f"{float(p):.1f}" for p in src.get('phase_centers', []))
                        ys_str = ';'.join(f"{float(y):.1f}" for y in src.get('y_stats', []))
                        f.write(
                            f"{src.get('source_id','')},{src.get('type_guess','')},{float(src.get('qty_weight',0.0)):.5f},{clusters_str},{phases_str},{ys_str},{float(src.get('pair_score',0.0)):.4f}\n"
                        )
                # Generar figura 2D y 3D de eventos coloreados por fuente emparejada
                # Solo se generan si existe al menos un par (pair_id > 0)
                if paired_sources:
                    try:
                        # Construir mapa de subcluster id a índice de fuente
                        sc_to_pair_idx: Dict[str, int] = {}
                        for idx_pair, src in enumerate(paired_sources, start=1):
                            for sc_id in src.get('clusters', []):
                                sc_to_pair_idx[str(sc_id)] = idx_pair
                        # Generar lista de índices de par para cada evento (0 si sin pareja)
                        pair_ids_evt: List[int] = []
                        for sid in evt_subcluster_id:
                            if sid and sid in sc_to_pair_idx:
                                pair_ids_evt.append(sc_to_pair_idx[sid])
                            else:
                                pair_ids_evt.append(0)
                        # Elegir paleta discreta (tab20) según número de pares
                        max_pairs = max(pair_ids_evt) if pair_ids_evt else 0
                        if max_pairs > 0:
                            cmap = matplotlib.colormaps.get('tab20', max_pairs + 1)
                        else:
                            cmap = matplotlib.colormaps.get('tab20', 1)
                        # Construir listas de colores para Matplotlib (RGBA) y Plotly (rgba string)
                        colors_evt: List[Tuple[float, float, float, float]] = []
                        colors_hex: List[str] = []
                        for pid in pair_ids_evt:
                            if pid > 0:
                                rgba = cmap(pid)
                                colors_evt.append(rgba)
                                colors_hex.append(
                                    'rgba({},{},{},{})'.format(
                                        int(rgba[0] * 255),
                                        int(rgba[1] * 255),
                                        int(rgba[2] * 255),
                                        round(rgba[3], 3),
                                    )
                                )
                            else:
                                # Color para eventos no emparejados
                                colors_evt.append((0.7, 0.7, 0.7, 0.4))
                                colors_hex.append('rgba(200,200,200,0.5)')
                        # Figura 2D: nube de puntos coloreada por par
                        fig_pair2d, ax_pair2d = plt.subplots(figsize=(6, 4))
                        fig_pair2d.patch.set_facecolor('white')
                        ax_pair2d.set_facecolor('white')
                        ax_pair2d.scatter(
                            phase_corr,
                            y_norm,
                            s=np.maximum(1.0, qty) * 2.0,
                            c=colors_evt,
                            edgecolors='none',
                        )
                        ax_pair2d.set_xlabel('Fase (°)')
                        ax_pair2d.set_ylabel('Amplitud normalizada')
                        ax_pair2d.set_xlim(0, 360)
                        ax_pair2d.set_ylim(0, 100)
                        ax_pair2d.set_title(
                            'PRPD con fuentes emparejadas (2D)\nphase align: +{}° ({}) ; recluster={}'.format(
                                int(round(shift_deg)),
                                align_mode,
                                'sí' if args.recluster_after_align else 'no',
                            )
                        )
                        fig_pair2d.tight_layout()
                        paired_fig2d_path = out_dir / f"{out_base}_paired_2d.png"
                        fig_pair2d.savefig(paired_fig2d_path, dpi=300)
                        plt.close(fig_pair2d)
                        # Figura 3D interactiva: puntos por par y líneas opcionales
                        import plotly.graph_objects as go  # type: ignore
                        import plotly.io as pio  # type: ignore
                        fig3d = go.Figure()
                        fig3d.add_trace(
                            go.Scatter3d(
                                x=phase_corr,
                                y=y_norm,
                                z=np.arange(len(phase_corr)),
                                mode='markers',
                                marker=dict(size=3, color=colors_hex),
                                hovertemplate='Fase=%{x:.1f}°<br>Amp=%{y:.1f}<br>Index=%{z}<extra></extra>',
                                name='Eventos',
                                legendgroup='eventos',
                            )
                        )
                        show_lines_flag = bool(args.pairs_show_lines)
                        # Agrupar índices de eventos por par
                        pair_to_indices: Dict[int, List[int]] = {}
                        for idx_evt, pid in enumerate(pair_ids_evt):
                            pair_to_indices.setdefault(pid, []).append(idx_evt)
                        # Añadir líneas de unión para cada par, si se desea
                        for idx_pair, src in enumerate(paired_sources, start=1):
                            rgba = cmap(idx_pair)
                            col_hex = 'rgba({},{},{},{})'.format(
                                int(rgba[0] * 255),
                                int(rgba[1] * 255),
                                int(rgba[2] * 255),
                                rgba[3],
                            )
                            phases_c = src.get('phase_centers', [])
                            ys_c = src.get('y_stats', [])
                            if not phases_c or not ys_c:
                                continue
                            if len(phases_c) == 1:
                                phases_c = phases_c * 2
                                ys_c = ys_c * 2
                            idxs = pair_to_indices.get(idx_pair, [])
                            z_val = float(np.mean(idxs)) if idxs else 0.0
                            fig3d.add_trace(
                                go.Scatter3d(
                                    x=[phases_c[0], phases_c[1]],
                                    y=[ys_c[0], ys_c[1]],
                                    z=[z_val, z_val],
                                    mode='lines',
                                    line=dict(color=col_hex, width=3),
                                    name=f'Pair {idx_pair} line',
                                    legendgroup=f'pair_{idx_pair}',
                                    showlegend=False,
                                    visible=show_lines_flag,
                                )
                            )
                        fig3d.update_layout(
                            scene=dict(
                                xaxis_title='Fase (°)',
                                yaxis_title='Amplitud normalizada',
                                zaxis_title='Índice',
                            ),
                            title='PRPD con fuentes emparejadas (3D)',
                            margin=dict(l=0, r=0, b=0, t=40),
                        )
                        paired_fig3d_path = out_dir / f"{out_base}_paired_3d.html"
                        pio.write_html(
                            fig3d,
                            file=str(paired_fig3d_path),
                            auto_open=False,
                            include_plotlyjs='cdn',
                        )
                    except Exception:
                        # Si algo falla en la generación de figuras, continuar sin ellas
                        paired_fig2d_path = None
                        paired_fig3d_path = None
                else:
                    # No hay pares, no generar figuras de emparejamiento
                    paired_fig2d_path = None
                    paired_fig3d_path = None
            except Exception:
                # Si ocurre un error en el emparejamiento, dejar las variables como None
                paired_sources_path = None
                p_mult_csv_path = None
                top_sources_path = None
                paired_fig2d_path = None
                paired_fig3d_path = None

    # Al finalizar, imprimir las rutas de los archivos generados para cumplir
    # con la regla de UX (sin textos adicionales).  Se imprime cada ruta
    # en una línea independiente.
    # Escribir el archivo de resumen global b5_summary.csv.  Se incluyen
    # p_global, probabilidades de n fuentes y las nuevas métricas de
    # multiplicidad p_tipo_ge2.
    try:
        with open(summary_csv_path, 'w') as f:
            # p_global por tipo
            f.write('tipo,p_global\n')
            for t in TYPE_LIST:
                f.write(f"{t},{p_global.get(t, 0.0):.5f}\n")
            # Distribución de número de fuentes por tipo
            f.write('\ntipo,n,prob\n')
            for t, dist in prob_n.items():
                for n, p in dist.items():
                    f.write(f"{t},{n},{p:.5f}\n")
            # Métricas de multiplicidad p_ge2 por tipo (cavidad, superficial, corona, flotante)
            f.write('\np_cavidad_ge2,{:.5f}\n'.format(p_cavidad_ge2))
            f.write('p_superficial_ge2,{:.5f}\n'.format(p_superficial_ge2))
            f.write('p_corona_ge2,{:.5f}\n'.format(p_corona_ge2))
            f.write('p_flotante_ge2,{:.5f}\n'.format(p_flotante_ge2))
            # Política de Otras y parámetros
            f.write('\notras_policy,{}\n'.format(global_oth_policy))
            f.write('otras_raw_score,{:.5f}\n'.format(global_raw_other))
            f.write('otras_min_score,{:.5f}\n'.format(otras_min_score))
            f.write('otras_cap,{:.5f}\n'.format(otras_cap))
    except Exception:
        pass
    # Imprimir rutas de salida (probabilidades, resumen, gráficos y tablas)
    print(str(prob_csv_path))
    print(str(summary_csv_path))
    print(str(stack_path))
    print(str(global_mix_path))
    print(str(cards_path))
    print(str(top_clusters_path))
    if args.subclusters and multiplicity_path is not None:
        print(str(multiplicity_path))
        if multiplicity_fig_path is not None:
            print(str(multiplicity_fig_path))
        # Imprimir también el archivo de subclusters si se generó
        try:
            if 'subclusters_path' in locals() and subclusters_path is not None:
                print(str(subclusters_path))
        except Exception:
            pass
        # Imprimir archivos de emparejamiento y multiplicidad fusionada si existen
        try:
            if 'paired_sources_path' in locals() and paired_sources_path is not None:
                print(str(paired_sources_path))
            if 'paired_map_path' in locals() and paired_map_path is not None:
                print(str(paired_map_path))
            if 'p_mult_csv_path' in locals() and p_mult_csv_path is not None:
                print(str(p_mult_csv_path))
            if 'top_sources_path' in locals() and top_sources_path is not None:
                print(str(top_sources_path))
            if 'paired_fig2d_path' in locals() and paired_fig2d_path is not None:
                print(str(paired_fig2d_path))
            if 'paired_fig3d_path' in locals() and paired_fig3d_path is not None:
                print(str(paired_fig3d_path))
        except Exception:
            pass


if __name__ == '__main__':
    main()
