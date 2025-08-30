#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bloque 3 — Análisis con HDBSCAN sobre el mapa PRPD.

Este módulo contiene dos rutinas principales para la exploración de un
registro de descargas parciales almacenado en un XML de Megger.  La
primera parte (B3.A) realiza un clustering natural mediante HDBSCAN
ponderando cada bin del histograma PRPD por el número de impulsos
registrados.  Se genera una visualización coloreada con una paleta
discreta fijada y se calculan métricas básicas (#clusters, fracción
de ruido, DBCV aproximado).  La segunda parte (B3.B) alinea el
resultado de HDBSCAN con la partición obtenida por K‑Means (k
determinado en Bloque 2) y produce dos figuras: en 2D se muestra el
PRPD con los colores de K‑Means y contornos de HDBSCAN; en 3D se
grafica cada evento según su fase, amplitud y tiempo efectivo
estimado, coloreado por la etiqueta de K‑Means.  También se calculan
métricas de contingencia entre ambas particiones (puridad, ARI y NMI).

Uso (ejemplos):

    # Natural HDBSCAN
    python bloque3.py --mode natural time_20000101_022514.xml \
        --phase-shift 0 --out-prefix resultados/bloque3_natural

    # HDBSCAN alineado a k
    python bloque3.py --mode aligned time_20000101_022514.xml \
        --phase-shift 0 --k-target 5 --out-prefix resultados/bloque3_k

Dependencias: requiere que `prpd_mega.py` esté en el mismo directorio
o accesible en el PYTHONPATH, así como la biblioteca `hdbscan`.

"""

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker
from matplotlib import colors as mcolors
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

# Silenciar advertencias de sklearn deprecations para una experiencia limpia
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Importar utilidades de prpd_mega.  Estos incluyen parsing, normalización,
# histogramas, clustering K‑Means y reducción de HDBSCAN a un número de
# clusters deseado, así como funciones auxiliares de trazado.
try:
    from prpd_mega import (
        parse_xml_points,
        normalize_y,
        phase_from_times,
        prpd_hist2d,
        centers_from_edges,
        kmeans_over_bins,
        hdbscan_reduce_to_k,
        contours_from_grid,
    )
except ImportError as e:
    raise ImportError(
        "No se pudo importar prpd_mega. Asegúrate de que prpd_mega.py está en el mismo directorio."
    ) from e

try:
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None  # se verificará en tiempo de ejecución

# Importar Plotly para gráficos 3D interactivos.  Se utilizará para generar
# figuras en HTML y, cuando sea posible, archivos PNG mediante Kaleido.
try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.io as pio  # type: ignore
except Exception:
    go = None
    pio = None

# Paleta fija para la visualización 3D interactiva.  Coincide con la
# especificada en los bloques anteriores: magenta, cian, verde, naranja,
# azul y negro.  El ruido se asignará en gris claro.
# Paletas de colores predeterminadas para la visualización 3D interactiva.  La
# paleta "paper" corresponde a una gama bien diferenciada utilizada en los
# informes originales, mientras que "tab20" y "pastel" permiten estilos
# alternativos.  Puede cargarse una paleta adicional desde un archivo JSON
# mediante el argumento --palette-json.  El ruido se asignará siempre en
# gris claro.
PLOTLY_PALETTE = ("#ff00ff", "#00e5ff", "#00a650", "#ff8c00", "#0077ff", "#000000")

# Mapas con las paletas disponibles y sus listas de colores.  Estos se
# complementan con la paleta cargada desde JSON en tiempo de ejecución.
# Construir un diccionario de paletas predeterminadas.  Además de las
# opciones originales (paper, tab20, pastel), añadimos viridis, warm y cool.
# Usamos ``matplotlib.colormaps.get_cmap`` en lugar de plt.cm.get_cmap para
# evitar advertencias de deprecación de Matplotlib.  Cada paleta se
# almacena como una tupla de colores en formato RGBA o hex según
# corresponda.
from matplotlib import colormaps as _mpl_cmaps  # type: ignore

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

BUILTIN_PALETTES: dict[str, tuple[str, ...]] = {
    'paper': PLOTLY_PALETTE,
    # Para tab20 se obtienen 20 colores y se almacenan como tuplas RGBA
    'tab20': tuple(_mpl_cmaps.get_cmap('tab20')(i) for i in range(20)),
    # Pastel fija basada en el ejemplo original
    'pastel': (
        '#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6',
        '#ffffb3', '#fabd91', '#e5d8bd', '#fddaec', '#f2f2f2'
    ),
    # Paleta viridis (continua, pero se puede muestrear al vuelo)
    'viridis': tuple(_mpl_cmaps.get_cmap('viridis')(i / 19.0) for i in range(20)),
    # Paleta warm (usamos colormap 'autumn' como aproximación a tonos cálidos)
    'warm': tuple(_mpl_cmaps.get_cmap('autumn')(i / 19.0) for i in range(20)),
    # Paleta cool (colormap 'cool' de Matplotlib)
    'cool': tuple(_mpl_cmaps.get_cmap('cool')(i / 19.0) for i in range(20)),
}




def _weighted_hdbscan_labels(
    Pn: np.ndarray,
    weights: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, float, float, Any]:
    """
    Ejecuta HDBSCAN sobre un conjunto de puntos Pn ponderados por `weights`.

    Los pesos se implementan replicando cada punto según el valor entero
    de su peso (casteado a int).  Esto garantiza que los puntos con un
    número mayor de impulsos tengan mayor influencia en la topología
    densidad.

    Parameters
    ----------
    Pn : array, shape (n_bins, 2)
        Puntos normalizados de los centros de bins ocupados.
    weights : array, shape (n_bins,)
        Peso entero de cada bin (recuento de impulsos).  Se convertirá a
        `int` mediante `np.round` y se garantizará que los bins con
        peso > 0 tengan al menos una réplica.
    min_cluster_size : int, optional
        Parámetro `min_cluster_size` de HDBSCAN.  Por defecto 10.
    min_samples : int or None, optional
        Parámetro `min_samples` de HDBSCAN.  Si es None, se deja que
        HDBSCAN utilice `min_cluster_size` por defecto.
    random_state : int, optional
        Semilla para reproducibilidad.  Se utiliza internamente para
        inicializar HDBSCAN.

    Returns
    -------
    labels_grid : array, shape (n_bins,)
        Etiqueta asignada a cada bin (−1 ruido).  El orden corresponde
        al de Pn y `weights`.
    noise_frac : float
        Proporción de réplicas etiquetadas como ruido.
    dbcv : float or None
        Aproximación del índice DBCV (`relative_validity_`).  Si la
        implementación de HDBSCAN no dispone de dicho atributo, se
        devuelve None.
    clusterer : object
        Instancia del clusterer HDBSCAN entrenada.
    """
    # Asegurar que los pesos sean enteros y como mínimo 1 para cada bin
    w_int = np.asarray(np.round(weights), dtype=int)
    w_int[w_int < 1] = 1
    # Si hdbscan no está instalado, usar DBSCAN como alternativa
    if hdbscan is None:
        from sklearn.cluster import DBSCAN
        # Ejecutar DBSCAN ponderado a nivel de bin; se utilizan los pesos
        # como sample_weight.  Ajustamos eps e min_samples de forma
        # heurística: eps=0.06, min_samples=max(5, min_cluster_size//2).
        eps = 0.06
        ms = max(5, int(min_cluster_size // 2))
        clusterer = DBSCAN(eps=eps, min_samples=ms).fit(Pn, sample_weight=w_int)
        labels_bins = clusterer.labels_
        # Fracción de ruido ponderada por peso
        total_w = float(np.sum(w_int))
        noise_w = float(np.sum(w_int[labels_bins < 0]))
        noise_frac = noise_w / total_w if total_w > 0 else 1.0
        dbcv = None
        return labels_bins, noise_frac, dbcv, clusterer

    # Si hdbscan está disponible, replicar puntos según peso y ejecutar
    # HDBSCAN para aproximar un clustering ponderado.
    # Replicación de puntos y construcción de índice inverso
    idx_map = np.repeat(np.arange(len(Pn)), w_int)
    Pn_rep = np.repeat(Pn, w_int, axis=0)

    # Ejecutar HDBSCAN sobre las réplicas
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        prediction_data=False,
    )
    labels_rep = clusterer.fit_predict(Pn_rep)
    # Calcular fracción de ruido a nivel de réplicas
    noise_frac = float(np.mean(labels_rep < 0)) if labels_rep.size > 0 else 1.0
    # Asignar etiqueta por bin mediante voto mayoritario
    labels_bins = np.full(len(Pn), fill_value=-1, dtype=int)
    # Agrupar índices de réplicas por bin
    for bin_idx in range(len(Pn)):
        mask_idx = (idx_map == bin_idx)
        if not np.any(mask_idx):
            continue
        lab_vals, counts = np.unique(labels_rep[mask_idx], return_counts=True)
        # Ordenar por recuento descendente
        order = np.argsort(-counts)
        best_lab = int(lab_vals[order[0]])
        labels_bins[bin_idx] = best_lab
    # DBCV aproximado
    dbcv = None
    if hasattr(clusterer, "relative_validity_"):
        try:
            dbcv = float(clusterer.relative_validity_)
        except Exception:
            dbcv = None
    return labels_bins, noise_frac, dbcv, clusterer


def run_hdbscan_natural(
    xml_path: str,
    phase_shift: float = 0.0,
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    random_state: int = 42,
    palette: Optional[Tuple[str, ...]] = None,
    alpha_base: float = 0.25,
    alpha_clusters: float = 0.85,
    sub_min_pct: float = 0.02,
) -> Tuple[plt.Figure, str]:
    """
    Ejecuta HDBSCAN natural ponderando cada bin y genera figura y resumen.

    Parameters
    ----------
    xml_path : str
        Ruta al archivo XML de medición.
    phase_shift : float, optional
        Corrimiento de fase a aplicar antes del análisis. Por defecto 0°.
    min_cluster_size : int, optional
        Parámetro `min_cluster_size` para HDBSCAN. Por defecto 10.
    min_samples : int or None, optional
        Parámetro `min_samples` de HDBSCAN. Por defecto None (igual a
        `min_cluster_size`).
    random_state : int, optional
        Semilla para reproducibilidad.  Se usa en K‑Means si fuera
        necesario; HDBSCAN no tiene random_state explícito.
    palette : tuple of str, optional
        Lista de colores para los clusters (en orden).  Si no se
        especifica se usa la paleta fija ["magenta", "cyan", "green",
        "orange", "blue", "black"].

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura con el PRPD natural coloreado por clusters de HDBSCAN.
    summary : str
        Resumen textual de las métricas (#clusters, %ruido, DBCV).
    """
    # Paleta predeterminada para clusters de HDBSCAN natural
    if palette is None:
        palette = ("magenta", "cyan", "green", "orange", "blue", "black")

    # Cargar datos del XML
    data = parse_xml_points(xml_path)
    raw_y, times, qty = data["raw_y"], data["times"], data["quantity"]
    # Normalizar eje vertical
    y_norm, _ = normalize_y(raw_y, data.get("sample_name"))
    # Convertir tiempo a fase con el corrimiento solicitado
    phase = phase_from_times(times, phase_shift)
    # Construir histograma PRPD
    H, xedges, yedges = prpd_hist2d(phase, y_norm, qty)
    ny, nx = H.T.shape
    # Obtener centros de bins ocupados y sus pesos (cantidad de impulsos)
    Xc, Yc = centers_from_edges(xedges, yedges)
    mask = H.T > 0
    if not np.any(mask):
        raise RuntimeError("No hay bins ocupados en el histograma; no se puede aplicar HDBSCAN.")
    # Matriz de puntos originales (fase, amplitud)
    P_orig = np.c_[Xc[mask], Yc[mask]]
    weights_orig = H.T[mask]
    # Para evitar cortes en 0/360, duplicar puntos trasladando fase ±360°.
    # Se generarán réplicas de cada punto con fase-360 y fase+360.
    # Construir matriz ampliada y pesos correspondientes.
    P_aug = np.concatenate([
        P_orig,
        np.column_stack((P_orig[:, 0] - 360.0, P_orig[:, 1])),
        np.column_stack((P_orig[:, 0] + 360.0, P_orig[:, 1]))
    ], axis=0)
    weights_aug = np.concatenate([weights_orig, weights_orig, weights_orig], axis=0)
    # Escalar a [0,1]
    from sklearn.preprocessing import MinMaxScaler  # import local
    scaler = MinMaxScaler()
    Pn_aug = scaler.fit_transform(P_aug)
    # Determinar min_cluster_size adaptativo si no se proporcionó explícitamente.
    # Para obtener clusters robustos, se calcula como max(30, round(0.005 * n_points)),
    # donde n_points es el número de bins ocupados (sin réplicas).  P_orig contiene
    # los puntos originales; usar su longitud como referencia.
    n_points = P_orig.shape[0]
    if min_cluster_size is None:
        min_cluster_size_local = max(30, int(round(0.005 * n_points)))
    else:
        min_cluster_size_local = min_cluster_size
    # Ejecutar HDBSCAN ponderado sobre el conjunto aumentado
    labels_aug, noise_frac_aug, dbcv, clusterer = _weighted_hdbscan_labels(
        Pn_aug, weights_aug, min_cluster_size=min_cluster_size_local, min_samples=min_samples, random_state=random_state
    )
    # Recortar etiquetas para conservar sólo las correspondientes a los puntos originales
    n_orig = len(P_orig)
    labels_bins = labels_aug[:n_orig]
    # Calcular fracción de ruido ponderada utilizando sólo puntos originales
    total_weight_orig = float(np.sum(weights_orig)) if np.sum(weights_orig) > 0 else 1.0
    noise_weight_orig = float(np.sum(weights_orig[np.array(labels_bins) < 0]))
    noise_frac = noise_weight_orig / total_weight_orig if total_weight_orig > 0 else 0.0
    # Filtrado de micro‑clusters: descartar clusters con una contribución inferior a
    # sub_min_pct del total de bins ocupados.  Los bins de micro‑clusters se
    # reclasifican como ruido (-1).
    if sub_min_pct is not None and sub_min_pct > 0.0:
        total_bins = len(labels_bins)
        if total_bins > 0:
            # Calcular tamaño de cada etiqueta válida
            unique_labs, counts_labs = np.unique(labels_bins[labels_bins >= 0], return_counts=True)
            # Etiquetas a descartar
            to_noise = set()
            for lab, count in zip(unique_labs, counts_labs):
                if (count / total_bins) < sub_min_pct:
                    to_noise.add(int(lab))
            if to_noise:
                for idx_lab, lab in enumerate(labels_bins):
                    if lab in to_noise:
                        labels_bins[idx_lab] = -1
    # Reconstruir grilla completa con etiquetas (ny,nx)
    grid = np.full((ny, nx), fill_value=-1, dtype=int)
    # Mapear los índices planos de mask a coordenadas de grilla
    bin_indices = np.argwhere(mask)
    # bin_indices nos da [row, col] en la grilla H.T (ny,nx).  labels_bins
    # está indexado en el mismo orden que bin_indices.
    for idx, (row, col) in enumerate(bin_indices):
        grid[row, col] = labels_bins[idx]
    # Métricas
    labs_valid = np.unique(labels_bins[labels_bins >= 0])
    n_clusters = int(len(labs_valid))
    perc_noise = noise_frac * 100.0
    # Figura
    fig, ax = plt.subplots(figsize=(9, 5))
    # Base en escala de grises (densidad PRPD); aplicar transparencia configurable
    im_base = ax.imshow(
        H.T + 1e-9,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        norm=LogNorm(vmin=1e-9, vmax=max(1.0, H.max())),
        cmap="gray_r",
        alpha=max(0.0, min(1.0, alpha_base)),
    )
    # Capa de clusters
    # Preparar imagen RGBA para clusters
    cluster_img = np.zeros((ny, nx, 4), dtype=float)
    # Mapear cada cluster a un color de la paleta
    for lab in labs_valid:
        color = palette[int(lab) % len(palette)]
        # Convertir a RGBA con alpha
        rgba = list(mcolors.to_rgba(color))
        rgba[3] = max(0.0, min(1.0, alpha_clusters))  # transparencia configurable
        cluster_img[grid == lab] = rgba
    # Ruido: gris claro
    noise_color = list(mcolors.to_rgba("lightgray"))
    # Ajustar alfa del ruido proporcional a la base (un poco menor)
    noise_color[3] = max(0.0, min(1.0, alpha_base * 0.8))
    cluster_img[grid < 0] = noise_color
    ax.imshow(
        cluster_img,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        interpolation="nearest",
    )
    # Contornos blancos de clusters
    # Dibujar contornos de clusters en blanco.  Dado que contours_from_grid
    # utiliza por defecto líneas negras, se sobreescribe el color.
    CS = ax.contour(
        np.linspace(xedges[0], xedges[-1], H.shape[0]),
        np.linspace(yedges[0], yedges[-1], H.shape[1]),
        grid,
        levels=np.unique(grid[grid >= 0]),
        linewidths=0.6,
        alpha=0.9,
        colors='white'
    )
    # Ajustes de figura
    ax.set_title("HDBSCAN natural (ponderado) — {}".format(Path(xml_path).name))
    ax.set_xlabel("Fase (°)")
    ax.set_ylabel("Muestra normalizada (0–100)")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    # Barra de colores sólo para densidad
    cbar = fig.colorbar(im_base, ax=ax)
    cbar.set_label("Recuento (log)")
    # Texto de métricas debajo de la figura
    summary = (
        f"Clusters: {n_clusters} (sin ruido). "
        f"Ruido: {perc_noise:.1f}%. "
        + (f"DBCV≈{dbcv:.2f}." if dbcv is not None else "DBCV no disponible.")
    )
    fig.text(0.02, -0.15, summary, ha="left", va="top", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig, summary


def _compute_contingency_metrics(
    labels_kmeans: np.ndarray,
    labels_hdb: np.ndarray,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Calcula métricas de contingencia entre dos particiones sobre una
    colección de elementos.

    Se asume que `labels_kmeans` y `labels_hdb` son vectores de la
    misma longitud que contienen las etiquetas asignadas por K‑Means y
    HDBSCAN respectivamente.  Las etiquetas con valor -1 se consideran
    ruido y se excluyen de los cálculos.

    Returns
    -------
    purity : float
        Medida de pureza (0–1) considerando HDBSCAN como predicción y
        K‑Means como referencia.
    ari : float
        Índice de Rand ajustado.
    nmi : float
        Información mutua normalizada.
    conf_mat : array
        Matriz de contingencia (confusión) entre las etiquetas sin
        ruido.
    """
    # Filtrar ruido
    mask = (labels_kmeans >= 0) & (labels_hdb >= 0)
    if not np.any(mask):
        return 0.0, 0.0, 0.0, np.zeros((1, 1), dtype=int)
    y_true = labels_kmeans[mask]
    y_pred = labels_hdb[mask]
    # Confusion matrix con filas = predicciones (HDBSCAN) y columnas = referencia (KMeans)
    # Ajustar índices para que sean consecutivos
    # Mapear etiquetas únicas a índices
    uniq_pred = np.unique(y_pred)
    uniq_true = np.unique(y_true)
    # Construir contadores
    conf_mat = np.zeros((len(uniq_pred), len(uniq_true)), dtype=int)
    for i, pval in enumerate(uniq_pred):
        for j, tval in enumerate(uniq_true):
            conf_mat[i, j] = int(np.sum((y_pred == pval) & (y_true == tval)))
    # Purity: para cada cluster predicho se toma su intersección máxima con un
    # cluster de referencia y se suma, normalizando por total
    max_per_cluster = conf_mat.max(axis=1)
    purity = float(max_per_cluster.sum() / y_true.size)
    # ARI y NMI
    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    return purity, ari, nmi, conf_mat


def _estimate_t_eff(times_ms: np.ndarray, k: int = 15) -> np.ndarray:
    """
    Estima el tiempo efectivo t_eff para cada evento basándose en la
    distancia a sus vecinos temporales más cercanos.

    Para cada muestra se computa la distancia promedio (en milisegundos)
    a sus k vecinos más cercanos en el eje temporal y se convierte a
    microsegundos.  Se utiliza NearestNeighbors de scikit‑learn para
    acelerar el cálculo.

    Parameters
    ----------
    times_ms : array, shape (n_samples,)
        Tiempos de los eventos en milisegundos.
    k : int, optional
        Número de vecinos a considerar (default=15).

    Returns
    -------
    t_eff_us : array, shape (n_samples,)
        Tiempo efectivo estimado en microsegundos.
    """
    if times_ms.ndim != 1:
        times_ms = times_ms.ravel()
    # Reshape para usar con NearestNeighbors
    X = times_ms.reshape(-1, 1)
    # k+1 porque el vecino más cercano es el propio punto
    n_neighbors = min(k + 1, X.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    # distancias en el eje de los tiempos (ms); excluir la distancia a sí mismo (cero)
    if distances.shape[1] > 1:
        t_eff_ms = distances[:, 1:].mean(axis=1)
    else:
        # Caso trivial: solo un punto
        t_eff_ms = np.zeros_like(times_ms)
    # Convertir a microsegundos
    return t_eff_ms * 1000.0


def run_hdbscan_3d_plotly(
    phase_deg: np.ndarray,
    y_norm: np.ndarray,
    qty: np.ndarray,
    labels_evt: np.ndarray,
    *,
    palette: Tuple[str, ...],
    palette_options: dict[str, tuple[str, ...]] | None = None,
    out_prefix: Path,
    title: str,
    note: str | None = None,
    show_noise_toggle: bool = True,
) -> Tuple[str, str]:
    """
    Genera un gráfico 3D interactivo de los eventos PRPD utilizando Plotly
    con soporte para cambiar paletas de colores y ocultar/mostrar ruido.

    Se admiten múltiples paletas de colores definidas en ``palette_options``.
    El usuario podrá seleccionar la paleta mediante un menú desplegable en el
    propio HTML.  Adicionalmente, se proporciona un control para ocultar
    las trazas que representan ruido (label < 0).

    Parameters
    ----------
    phase_deg, y_norm, qty, labels_evt : arrays
        Datos de fase, amplitud, peso y etiquetas por evento.
    palette : tuple of str
        Paleta por defecto para colorear los clusters.
    palette_options : dict, optional
        Diccionario con todas las paletas disponibles.  Cada clave es el
        nombre que aparecerá en el menú y el valor es una lista o tupla
        de colores.  Si es None, se usará solamente la paleta indicada.
    out_prefix : Path
        Prefijo para los nombres de los archivos de salida (sin extensión).
    title : str
        Título del gráfico.
    note : str, optional
        Nota adicional para el título.
    show_noise_toggle : bool, optional
        Incluir un menú para ocultar/mostrar la traza de ruido.  Por defecto
        True.

    Returns
    -------
    (png_path, html_path) : tuple of str
        Rutas a los archivos PNG y HTML generados.  Si Plotly no está
        disponible, se devolverán cadenas vacías.
    """
    # Verificar disponibilidad de Plotly
    if go is None or pio is None:
        return "", ""
    # Aplanar datos
    phase_arr = np.asarray(phase_deg).ravel()
    y_arr = np.asarray(y_norm).ravel()
    qty_arr = np.asarray(qty).ravel()
    labels_arr = np.asarray(labels_evt).ravel()
    qty_clipped = np.clip(qty_arr, 0.0, None)
    z_vals = np.log10(qty_clipped + 1.0)
    # Traza separada por etiqueta
    fig = go.Figure()
    # Determinar orden único de etiquetas para consistencia en menús
    unique_labels = np.unique(labels_arr)
    # Construir traza para cada etiqueta
    sizes = None
    if np.max(qty_clipped) > 0:
        sizes = 4.0 + 6.0 * (qty_clipped / np.max(qty_clipped))
    else:
        sizes = np.full_like(qty_clipped, fill_value=4.0)
    for lab in unique_labels:
        mask = labels_arr == lab
        if not np.any(mask):
            continue
        name = f"Cluster {lab}" if lab >= 0 else "Ruido"
        default_col = 'lightgray' if lab < 0 else palette[int(lab) % len(palette)]
        fig.add_trace(
            go.Scatter3d(
                x=phase_arr[mask],
                y=y_arr[mask],
                z=z_vals[mask],
                mode='markers',
                name=name,
                marker=dict(
                    size=sizes[mask],
                    color=default_col,
                    opacity=0.7 if lab >= 0 else 0.3,
                ),
                hovertemplate=(
                    'Fase: %{x:.1f}°<br>'
                    'Amplitud: %{y:.1f}<br>'
                    'Log10(qty+1): %{z:.3f}<br>'
                    'Cluster: '+name
                ),
            )
        )
    # Preparar título
    full_title = title + (f" {note}" if note else "")
    # Definir layout básico
    fig.update_layout(
        title=full_title,
        template='plotly_white',
        margin=dict(l=40, r=40, b=40, t=60),
        scene=dict(
            xaxis=dict(title='Fase (°)', range=[0, 360], gridcolor='lightgray'),
            yaxis=dict(title='Amplitud normalizada', range=[0, 100], gridcolor='lightgray'),
            zaxis=dict(title='log10(Qty+1)', gridcolor='lightgray'),
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
        ),
        showlegend=True,
    )
    # Preparar menús interactivos si se proporcionan opciones
    updatemenus = []
    # Paleta: si existen varias opciones, crear un menú desplegable
    if palette_options:
        buttons_pal = []
        # Precomputar los índices de trazas que corresponden a ruido
        noise_indices = [i for i, lab in enumerate(unique_labels) if lab < 0]
        for pal_name, pal_values in palette_options.items():
            # Asegurar que la paleta tenga suficientes colores; si no, repetir
            pal_list = list(pal_values)
            # Preparar vector de colores para todas las trazas
            colors_for_traces = []
            for lab in unique_labels:
                if lab < 0:
                    colors_for_traces.append('lightgray')
                else:
                    colors_for_traces.append(pal_list[int(lab) % len(pal_list)])
            buttons_pal.append(
                dict(
                    args=[{'marker.color': [colors_for_traces[i] for i in range(len(unique_labels))]}],
                    label=pal_name,
                    method='update',
                )
            )
        updatemenus.append(
            dict(
                buttons=buttons_pal,
                direction='down',
                showactive=True,
                x=0.0,
                xanchor='left',
                y=1.15,
                yanchor='top',
            )
        )
    # Menú para ocultar/mostrar ruido
    if show_noise_toggle and len(unique_labels) > 1:
        noise_indices = [i for i, lab in enumerate(unique_labels) if lab < 0]
        # Si no hay ruido, no añadir
        if noise_indices:
            # Lista de visibilidad para cada traza
            show_all = [True] * len(unique_labels)
            hide_noise = [False if i in noise_indices else True for i in range(len(unique_labels))]
            updatemenus.append(
                dict(
                    buttons=[
                        dict(
                            args=[{'visible': hide_noise}],
                            label='Ocultar ruido',
                            method='update',
                        ),
                        dict(
                            args=[{'visible': show_all}],
                            label='Mostrar ruido',
                            method='update',
                        ),
                    ],
                    direction='down',
                    showactive=True,
                    x=0.25,
                    xanchor='left',
                    y=1.15,
                    yanchor='top',
                )
            )
    if updatemenus:
        fig.update_layout(updatemenus=updatemenus)
    # Guardar archivos
    html_path = Path(str(out_prefix) + '_plotly3d.html')
    try:
        fig.write_html(html_path, include_plotlyjs='cdn')
    except Exception:
        # Si falla, escribir un HTML básico
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(fig.to_html(include_plotlyjs='cdn'))
    png_path = Path(str(out_prefix) + '_3d.png')
    # Guardar PNG usando kaleido si está disponible
    try:
        fig.write_image(png_path, format='png', scale=2)
    except Exception:
        # Intento de fallback: usar Matplotlib 3D
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig_png = plt.figure(figsize=(9, 6))
            ax_png = fig_png.add_subplot(111, projection='3d')
            for idx, lab in enumerate(unique_labels):
                mask = labels_arr == lab
                if not np.any(mask):
                    continue
                col = 'lightgray' if lab < 0 else palette[int(lab) % len(palette)]
                alpha = 0.7 if lab >= 0 else 0.3
                ax_png.scatter(
                    phase_arr[mask],
                    y_arr[mask],
                    z_vals[mask],
                    c=[col],
                    s=sizes[mask],
                    alpha=alpha,
                    marker='o',
                    label=f'Cluster {lab}' if lab >= 0 else 'Ruido'
                )
            ax_png.set_xlabel('Fase (°)')
            ax_png.set_ylabel('Amplitud normalizada')
            ax_png.set_zlabel('log10(Qty+1)')
            ax_png.set_xlim(0, 360)
            ax_png.set_ylim(0, 100)
            zmin, zmax = z_vals.min(), z_vals.max()
            ax_png.set_zlim(zmin, zmax)
            ax_png.view_init(elev=20., azim=30)
            ax_png.set_title(full_title)
            ax_png.legend(loc='upper right', fontsize=7)
            fig_png.tight_layout()
            fig_png.savefig(png_path, dpi=300)
            plt.close(fig_png)
        except Exception:
            png_path = Path('')
    return str(png_path), str(html_path)


def run_hdbscan_aligned(
    xml_path: str,
    k_target: int,
    phase_shift: float = 0.0,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    random_state: int = 42,
    palette: Optional[Tuple[str, ...]] = None,
    knn_k: int = 15,
    *,
    alpha_base: float = 0.25,
    alpha_clusters: float = 0.85,
    events3d: bool = False,
) -> Tuple[plt.Figure, plt.Figure | None, str]:
    """
    Ejecuta HDBSCAN alineado a un número k de clusters y genera figuras en 2D y 3D.

    Parameters
    ----------
    xml_path : str
        Ruta al archivo XML de medición.
    k_target : int
        Número de clusters objetivo (k) obtenido en Bloque 2.
    phase_shift : float, optional
        Corrimiento de fase a aplicar antes del análisis. Por defecto 0°.
    min_cluster_size : int, optional
        Parámetro `min_cluster_size` para HDBSCAN natural. Por defecto 10.
    min_samples : int or None, optional
        Parámetro `min_samples` de HDBSCAN. Por defecto None.
    random_state : int, optional
        Semilla para reproducibilidad en K‑Means.
    palette : tuple of str, optional
        Paleta de colores para los clusters K‑Means/HDBSCAN. Si no se
        especifica se usa una paleta `tab20` de Matplotlib.
    knn_k : int, optional
        Número de vecinos para estimar t_eff en el scatter 3D. Por
        defecto 15.

    Returns
    -------
    fig2d : matplotlib.figure.Figure
        Figura 2D con el PRPD coloreado por K‑Means y contornos de HDBSCAN.
    fig3d : matplotlib.figure.Figure
        Figura 3D con los eventos coloreados por cluster de K‑Means.
    summary : str
        Texto resumen con las métricas de contingencia.
    """
    # Paleta por defecto: utilizar colormap discreto tab20 si no se especifica
    if palette is None:
        # Generar paleta de tamaño k_target a partir de tab20 utilizando API moderna
        from matplotlib import colormaps as _cm
        cmap = _cm.get_cmap('tab20').resampled(max(1, k_target))
        palette = tuple(cmap(i) for i in range(max(1, k_target)))

    # Cargar datos
    data = parse_xml_points(xml_path)
    raw_y, times, qty = data["raw_y"], data["times"], data["quantity"]
    # Normalización
    y_norm, _ = normalize_y(raw_y, data.get("sample_name"))
    # Fase con shift
    phase = phase_from_times(times, phase_shift)
    # Histograma PRPD
    H, xedges, yedges = prpd_hist2d(phase, y_norm, qty)
    ny, nx = H.T.shape
    # Clustering K‑Means sobre bins para k_target
    labels_k_grid, sil = kmeans_over_bins(H, xedges, yedges, k_target)
    # HDBSCAN natural ponderado
    # Datos de bins ocupados
    Xc, Yc = centers_from_edges(xedges, yedges)
    mask = H.T > 0
    P = np.c_[Xc[mask], Yc[mask]]
    weights = H.T[mask]
    from sklearn.preprocessing import MinMaxScaler  # local import
    scaler = MinMaxScaler(); Pn = scaler.fit_transform(P)
    labels_bins_hdb, noise_frac, dbcv, clusterer = _weighted_hdbscan_labels(
        Pn, weights, min_cluster_size=min_cluster_size, min_samples=min_samples, random_state=random_state
    )
    # Reconstruir grilla hdbscan
    labels_h_grid = np.full((ny, nx), fill_value=-1, dtype=int)
    bin_indices = np.argwhere(mask)
    for idx, (row, col) in enumerate(bin_indices):
        labels_h_grid[row, col] = labels_bins_hdb[idx]
    # Contingencia entre kmeans y hdbscan natural a nivel de bins
    # Convertir grillas a vectores
    flat_k = labels_k_grid.flatten()
    flat_h = labels_h_grid.flatten()
    purity, ari, nmi, conf_mat = _compute_contingency_metrics(flat_k, flat_h)
    # ----- Generar figura 2D -----
    fig2d, ax2d = plt.subplots(figsize=(9, 5))
    # Base en escala de grises con transparencia configurable
    im_dens = ax2d.imshow(
        H.T + 1e-9,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        norm=LogNorm(vmin=1e-9, vmax=max(1.0, H.max())),
        cmap="gray_r",
        alpha=max(0.0, min(1.0, alpha_base)),
    )
    # Capa de clusters K‑Means con colores exactos
    cluster_img2 = np.zeros((ny, nx, 4), dtype=float)
    for lab in range(k_target):
        color = palette[lab % len(palette)]
        # Asegurar color en formato RGBA
        if hasattr(color, '__iter__') and len(color) == 4:
            rgba = list(color)
        else:
            rgba = list(mcolors.to_rgba(color))
        rgba[3] = max(0.0, min(1.0, alpha_clusters))
        cluster_img2[labels_k_grid == lab] = rgba
    # Ruido (bins sin datos) transparente
    cluster_img2[labels_k_grid < 0] = (0, 0, 0, 0)
    ax2d.imshow(
        cluster_img2,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        interpolation="nearest",
    )
    # Contornos de HDBSCAN natural en blanco
    # Dibujar contornos blancos de HDBSCAN natural
    CS2 = ax2d.contour(
        np.linspace(xedges[0], xedges[-1], H.shape[0]),
        np.linspace(yedges[0], yedges[-1], H.shape[1]),
        labels_h_grid,
        levels=np.unique(labels_h_grid[labels_h_grid >= 0]),
        linewidths=0.6,
        alpha=0.9,
        colors='white'
    )
    # Ajustes de la figura 2D
    ax2d.set_title(
        f"PRPD K‑Means (k={k_target}) con contornos HDBSCAN — {Path(xml_path).name}"
    )
    ax2d.set_xlabel("Fase (°)")
    ax2d.set_ylabel("Muestra normalizada (0–100)")
    ax2d.set_xlim(0, 360)
    ax2d.set_ylim(0, 100)
    ax2d.xaxis.set_major_locator(ticker.MultipleLocator(30))
    cbar2 = fig2d.colorbar(im_dens, ax=ax2d)
    cbar2.set_label("Recuento (log)")
    # Texto de métricas al pie
    summary2d = (
        f"Purity={purity:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}. "
        f"Clusters HDBSCAN (sin ruido)={int(len(np.unique(labels_bins_hdb[labels_bins_hdb>=0])))}"
    )
    fig2d.text(0.02, -0.15, summary2d, ha="left", va="top", fontsize=8)
    fig2d.tight_layout(rect=[0, 0.05, 1, 0.95])
    # ----- Generar figura 3D -----
    # Mapear cada evento a su cluster de K‑Means usando grilla labels_k_grid
    # Determinar bin index para cada evento
    # edges: xedges (len nx+1), yedges (len ny+1)
    # Buscar índices binarios para cada fase/y_norm
    x_bins = np.digitize(phase, bins=xedges) - 1
    y_bins = np.digitize(y_norm, bins=yedges) - 1
    # Asegurar límites válidos
    valid_mask = (x_bins >= 0) & (x_bins < nx) & (y_bins >= 0) & (y_bins < ny)
    # Inicializar labels por evento
    labels_events = np.full_like(phase, fill_value=-1, dtype=int)
    labels_events[valid_mask] = labels_k_grid[y_bins[valid_mask], x_bins[valid_mask]]
    # Estimar t_eff (usando k vecinos)
    t_eff_us = _estimate_t_eff(times, k=knn_k)
    # Crear figura 3D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    fig3d = None
    if events3d:
        fig3d = plt.figure(figsize=(9, 6))
        ax3d = fig3d.add_subplot(111, projection='3d')
    # Para ejes log se requiere evitar ceros
    amp = y_norm.copy()
    amp[amp < 1e-3] = 1e-3
    t_eff_nonzero = t_eff_us.copy()
    t_eff_nonzero[t_eff_nonzero < 1e-6] = 1e-6
    if events3d and fig3d is not None:
        # Dibujar cada cluster por separado
        for lab in range(k_target):
            mask_evt = labels_events == lab
            if not np.any(mask_evt):
                continue
            color = palette[lab % len(palette)]
            if hasattr(color, '__iter__') and len(color) == 4:
                rgba = list(color)
            else:
                rgba = list(mcolors.to_rgba(color))
            ax3d.scatter(
                phase[mask_evt], amp[mask_evt], t_eff_nonzero[mask_evt],
                c=[rgba], s=3, alpha=0.6, marker='o', label=f'Cluster {lab}'
            )
        # Dibujar ruido en gris
        mask_noise = labels_events < 0
        if np.any(mask_noise):
            ax3d.scatter(
                phase[mask_noise], amp[mask_noise], t_eff_nonzero[mask_noise],
                c=[(0.7, 0.7, 0.7, 0.5)], s=2, marker='.', label='Ruido'
            )
        # Configurar ejes
        ax3d.set_xlabel('Fase (°)')
        ax3d.set_ylabel('Amplitud (rel.)')
        ax3d.set_zlabel('t_eff (µs)')
        ax3d.set_xlim(0, 360)
        ax3d.set_ylim(0, 100)
        # Escala log para t_eff si la variación es significativa
        z_min = t_eff_nonzero.min(); z_max = t_eff_nonzero.max()
        if z_max > z_min * 10:
            ax3d.set_zscale('log')
            ax3d.set_zlim(max(z_min, 1e-6), z_max * 1.1)
        else:
            ax3d.set_zlim(z_min, z_max)
        ax3d.view_init(elev=20., azim=30)
        ax3d.set_title(
            f"Eventos PRPD en 3D (K‑Means k={k_target}) — {Path(xml_path).name}"
        )
        ax3d.legend(loc='upper right', fontsize=7)
        fig3d.tight_layout()
    # Resumen final
    summary3 = (
        f"Contingencia KMeans↔HDBSCAN: purity={purity:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}, "
        f"clusters HDBSCAN={int(len(np.unique(labels_bins_hdb[labels_bins_hdb>=0])))}"
    )
    return fig2d, fig3d, summary3


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bloque 3: clustering HDBSCAN natural y alineado a k. "
            "Elige un modo de operación con --mode {natural, aligned}."
        )
    )
    parser.add_argument("xml_file", help="Ruta al archivo XML a procesar.")
    parser.add_argument("--mode", choices=["natural", "aligned"], required=True,
                        help="Modo de operación: natural (B3.A) o aligned (B3.B).")
    # Permitir especificar corrimiento de fase mediante --phase-align.  La opción
    # --phase-shift se mantiene por compatibilidad, pero si se especifica
    # --phase-align, tiene prioridad.  --phase-align acepta 'auto' o un número
    # (en grados).  'auto' se interpreta como 0 en este bloque (no se
    # aplica corrimiento).
    parser.add_argument("--phase-shift", type=float, default=None,
                        help="(Obsoleto) Corrimiento de fase en grados.")
    parser.add_argument("--phase-align", type=str, default=None,
                        help="Alineación de fase: 'auto' (0°) o número de grados. Tiene prioridad sobre phase-shift.")
    parser.add_argument("--k-target", type=int, default=None,
                        help="Número k para la alineación (modo aligned).")
    parser.add_argument("--k-use", type=int, default=None,
                        help="Número de clusters K‑Means a utilizar de forma directa, anulando el cálculo automático.")
    parser.add_argument("--out-prefix", type=str, default="bloque3",
                        help="Prefijo de salida para guardar las figuras.")
    parser.add_argument("--min-cluster-size", type=int, default=10,
                        help="min_cluster_size para HDBSCAN.")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="min_samples para HDBSCAN (None = min_cluster_size).")
    parser.add_argument("--knn-k", type=int, default=15,
                        help="Número de vecinos para estimar t_eff en 3D.")
    # Generar gráfica 3D interactiva con Plotly.  El pipeline requiere
    # siempre la versión interactiva en HTML, por lo que se invoca
    # independientemente de este argumento.  Se mantiene el flag
    # únicamente por compatibilidad; si se desea desactivar la
    # generación interactiva deberá modificarse el código.
    parser.add_argument("--plotly-3d", dest="plotly_3d", action='store_true',
                        help="(Obsoleto) Generar gráfica 3D interactiva con Plotly.  La gráfica se generará siempre.")
    parser.add_argument("--palette", type=str, default='paper',
                        help="Paleta de colores base a utilizar (paper, tab20, pastel, viridis, warm, cool).")
    parser.add_argument("--palette-json", type=str, default=None,
                        help="Ruta a un archivo JSON con una lista de colores personalizada.")
    parser.add_argument("--alpha-base", type=float, default=0.25,
                        help="Transparencia de la capa base (densidad). Valor entre 0 y 1 (default 0.25).")
    parser.add_argument("--alpha-clusters", type=float, default=0.85,
                        help="Transparencia para la capa de clusters en 2D (default 0.85).")
    parser.add_argument("--events3d", action='store_true',
                        help="Incluir figura 3D de eventos individuales por KMeans (solo modo aligned).")
    parser.add_argument("--sub-min-pct", type=float, default=0.02,
                        help="Umbral mínimo de contribución para descartar micro‑clusters en HDBSCAN natural.")
    args = parser.parse_args()

    xml_path = args.xml_file
    # Prefijo de salida: se usa tal cual, creando su carpeta padre si no existe
    # Normalizar el prefijo de salida para incluir el stem del XML
    out_pref = normalize_prefix(args.out_prefix, xml_path)
    # Crear carpeta de salida si no existe
    out_pref.parent.mkdir(parents=True, exist_ok=True)
    # Preparar paletas disponibles
    palette_name = (args.palette or 'paper').strip().lower()
    palette_options = {}
    # Copiar paletas predeterminadas
    for key, val in BUILTIN_PALETTES.items():
        # Convertir colores en formato matplotlib RGBA a hex si fuera necesario
        # Matplotlib colormap devuelve RGBA; convertir a hex
        conv = []
        for c in val:
            if isinstance(c, tuple) or isinstance(c, list):
                # RGBA: convertir a hex
                try:
                    import matplotlib.colors as _mcols
                    conv.append(_mcols.to_hex(c))
                except Exception:
                    conv.append(str(c))
            else:
                conv.append(str(c))
        palette_options[key] = tuple(conv)
    # Si se especifica un JSON con paletas personalizadas
    if args.palette_json:
        try:
            import json
            with open(args.palette_json, 'r', encoding='utf-8') as f:
                custom_pal = json.load(f)
            # El JSON puede ser un dict {nombre: [colores]} o una lista
            if isinstance(custom_pal, dict):
                for k, v in custom_pal.items():
                    if isinstance(v, (list, tuple)):
                        palette_options[str(k)] = tuple(str(col) for col in v)
            elif isinstance(custom_pal, list):
                palette_options['custom'] = tuple(str(col) for col in custom_pal)
        except Exception:
            pass
    # Seleccionar paleta por defecto para esta ejecución
    palette_default = palette_options.get(palette_name, palette_options.get('paper', PLOTLY_PALETTE))

    # Determinar corrimiento de fase a utilizar.  Si se especifica --phase-align
    # usamos dicha opción; de lo contrario, usamos --phase-shift (o 0 si None).
    if args.phase_align is not None:
        spec = (args.phase_align or '').strip().lower()
        if spec == 'auto' or spec == 'none':
            phase_shift_param = 0.0
        else:
            try:
                phase_shift_param = float(spec)
                # Indicar de forma explícita que se aplicó un desfase manual
                try:
                    print(f"PHASE_ALIGN_APPLIED: {phase_shift_param}")
                except Exception:
                    pass
            except Exception:
                phase_shift_param = 0.0
    elif args.phase_shift is not None:
        phase_shift_param = float(args.phase_shift)
    else:
        phase_shift_param = 0.0

    if args.mode == "natural":
        # Ejecutar HDBSCAN natural y guardar la figura 2D
        fig, summary = run_hdbscan_natural(
            xml_path,
            phase_shift=phase_shift_param,
            # Pasar None para activar el cálculo adaptativo del tamaño de clúster si
            # min_cluster_size no se especifica en la línea de comandos
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            palette=palette_default,
            alpha_base=args.alpha_base,
            alpha_clusters=args.alpha_clusters,
            sub_min_pct=args.sub_min_pct,
        )
        # Nombre base para 2D natural: <out_prefix>.png
        out_png = out_pref.parent / f"{out_pref.name}.png"
        fig.savefig(out_png, dpi=300)
        print(str(out_png))
        # Imprimir el resumen de forma segura evitando caracteres no ASCII (p.ej. ↔)
        try:
            safe_summary = str(summary).replace('↔', '<->')
            print(safe_summary)
        except Exception:
            try:
                print(summary.encode('cp1252', 'ignore').decode('cp1252', 'ignore'))
            except Exception:
                print(str(summary))
        # Generar figura 3D interactiva SIEMPRE para natural
        # Se utilizan las mismas fases corregidas aplicadas en HDBSCAN natural.
        try:
            # Cargar datos para asignar etiquetas de HDBSCAN natural a eventos
            data = parse_xml_points(xml_path)
            raw_y, times, qty = data["raw_y"], data["times"], data["quantity"]
            y_norm, _ = normalize_y(raw_y, data.get("sample_name"))
            # Aplicar el mismo corrimiento de fase empleado en la ejecución principal
            phase = phase_from_times(times, phase_shift_param)
            # Reconstruir histograma y xedges/yedges
            H, xedges, yedges = prpd_hist2d(phase, y_norm, qty)
            ny, nx = H.T.shape
            # Obtener centros y pesos
            Xc, Yc = centers_from_edges(xedges, yedges)
            mask_bins = H.T > 0
            P = np.c_[Xc[mask_bins], Yc[mask_bins]]
            weights = H.T[mask_bins]
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(); Pn = scaler.fit_transform(P)
            # Determinar min_cluster_size adaptativo para HDBSCAN natural
            mcs = args.min_cluster_size if args.min_cluster_size is not None else max(30, int(round(0.005 * len(Pn))))
            labels_bins_hdb, _, _, _ = _weighted_hdbscan_labels(
                Pn, weights, min_cluster_size=mcs, min_samples=args.min_samples, random_state=42
            )
            # Reconstruir grilla
            grid_labels = np.full((ny, nx), fill_value=-1, dtype=int)
            bin_indices = np.argwhere(mask_bins)
            for idx, (row, col) in enumerate(bin_indices):
                grid_labels[row, col] = labels_bins_hdb[idx]
            # Asignar etiquetas de cluster a eventos
            x_bins_evt = np.digitize(phase, bins=xedges) - 1
            y_bins_evt = np.digitize(y_norm, bins=yedges) - 1
            labels_evt = np.full_like(phase, fill_value=-1, dtype=int)
            mask_valid_evt = (x_bins_evt >= 0) & (x_bins_evt < nx) & (y_bins_evt >= 0) & (y_bins_evt < ny)
            labels_evt[mask_valid_evt] = grid_labels[y_bins_evt[mask_valid_evt], x_bins_evt[mask_valid_evt]]
            note = "(fallback DBSCAN)" if hdbscan is None else None
            # Definir prefijo específico para natural
            out3d_prefix = out_pref.parent / f"{out_pref.name}_natural"
            png_path, html_path = run_hdbscan_3d_plotly(
                phase,
                y_norm,
                qty,
                labels_evt,
                palette=palette_default,
                palette_options=palette_options,
                out_prefix=out3d_prefix,
                title=f"PRPD 3D HDBSCAN natural — {Path(xml_path).name}",
                note=note,
            )
            if png_path:
                print(str(png_path))
            if html_path:
                print(str(html_path))
        except Exception:
            # En caso de error, continuar sin interrumpir el flujo
            pass
    elif args.mode == "aligned":
        # En modo alineado se requiere k_target
        # Determinar k a usar: si se especifica --k-use, tiene prioridad sobre --k-target
        k_target_value: int | None = args.k_use if args.k_use is not None else args.k_target
        if k_target_value is None:
            raise ValueError("Debe especificar --k-target o --k-use en modo 'aligned'.")
        # Ejecutar clustering alineado y obtener figura 2D y resumen
        fig2d, fig3d, summary = run_hdbscan_aligned(
            xml_path,
            k_target=k_target_value,
            phase_shift=phase_shift_param,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            palette=palette_default,
            knn_k=args.knn_k,
            alpha_base=args.alpha_base,
            alpha_clusters=args.alpha_clusters,
            events3d=args.events3d,
        )
        # Guardar sólo la figura 2D con el nombre <out_prefix>_2d.png
        out_png2d = out_pref.parent / f"{out_pref.name}_2d.png"
        fig2d.savefig(out_png2d, dpi=300)
        print(str(out_png2d))
        # Imprimir el resumen de forma segura evitando caracteres no ASCII (p.ej. ↔)
        try:
            safe_summary = str(summary).replace('↔', '<->')
            print(safe_summary)
        except Exception:
            try:
                print(summary.encode('cp1252', 'ignore').decode('cp1252', 'ignore'))
            except Exception:
                print(str(summary))
        # Loggear en consola el número k utilizado
        print(f"k utilizado: {k_target_value}")
        # Generar gráfica 3D interactiva SIEMPRE para el clustering alineado
        try:
            # Cargar datos
            data = parse_xml_points(xml_path)
            raw_y, times, qty = data["raw_y"], data["times"], data["quantity"]
            y_norm, _ = normalize_y(raw_y, data.get("sample_name"))
            # Utilizar el mismo corrimiento de fase que en la ejecución principal
            phase = phase_from_times(times, phase_shift_param)
            # Histograma y clustering K‑Means sobre bins
            H, xedges, yedges = prpd_hist2d(phase, y_norm, qty)
            ny, nx = H.T.shape
            labels_k_grid, _ = kmeans_over_bins(H, xedges, yedges, k_target_value)
            # Asignar etiquetas por evento
            x_bins_evt = np.digitize(phase, bins=xedges) - 1
            y_bins_evt = np.digitize(y_norm, bins=yedges) - 1
            labels_evt = np.full_like(phase, fill_value=-1, dtype=int)
            mask_valid_evt = (x_bins_evt >= 0) & (x_bins_evt < nx) & (y_bins_evt >= 0) & (y_bins_evt < ny)
            labels_evt[mask_valid_evt] = labels_k_grid[y_bins_evt[mask_valid_evt], x_bins_evt[mask_valid_evt]]
            note = "(fallback DBSCAN)" if hdbscan is None else None
            # Definir prefijo específico para aligned
            out3d_prefix = out_pref.parent / f"{out_pref.name}_aligned"
            png_path, html_path = run_hdbscan_3d_plotly(
                phase,
                y_norm,
                qty,
                labels_evt,
                palette=palette_default,
                palette_options=palette_options,
                out_prefix=out3d_prefix,
                title=f"PRPD 3D KMeans (k={k_target_value}) — {Path(xml_path).name}",
                note=note,
            )
            if png_path:
                print(str(png_path))
            if html_path:
                print(str(html_path))
        except Exception:
            pass


if __name__ == "__main__":
    main()