#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bloque 2 — Selección de k óptimo para clustering de PRPD.

Este módulo implementa la lógica para evaluar distintos valores de k en un
clustering K‑Means aplicado a los bins ocupados del histograma PRPD y
determinar un valor óptimo basándose en la combinación de la métrica
Silhouette y el criterio del codo (Elbow) detectado mediante el algoritmo
Kneedle.  Además, se permite un override manual del codo/k propuesto y se
generan figuras ilustrativas de las curvas Silhouette/Inercia y de los
resultados de clustering sobre el PRPD crudo para los casos automático y
manual.

Uso (ejemplo):

    python bloque2.py time_20000101_022514.xml \
        --phase-shift 0 --k-min 2 --k-max 8 \
        --k-manual 5 --random-state 42 \
        --out-prefix resultados/bloque2

Esto generará tres archivos PNG:
  1. resultados/bloque2_curvas.png – curvas Silhouette e Inercia con
     indicación del codo y del k óptimo.
  2. resultados/bloque2_prpd_auto.png – PRPD coloreado usando el k óptimo
     seleccionado automáticamente (combinación Silhouette+Elbow).
  3. resultados/bloque2_prpd_manual.png (solo si se especifica --k-manual
     distinto del k óptimo automático) – PRPD coloreado usando el k óptimo
     recalculado con el codo manual.

Las figuras se guardan como archivos PNG.  También se imprime en consola un
resumen de los resultados y de los parámetros empleados.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

# Importar funciones desde prpd_mega.  Se asume que prpd_mega.py está
# disponible en el mismo directorio o en el PYTHONPATH.
try:
    from prpd_mega import (
        parse_xml_points,
        identify_sensor_from_data,
        normalize_y,
        phase_from_times,
        prpd_hist2d,
        centers_from_edges,
        kneedle_k_from_elbow,
        prpd_clusters_on_raw,
        kmeans_over_bins,
    )
except ImportError as e:
    raise ImportError(
        "No se pudo importar prpd_mega. Asegúrate de que prpd_mega.py está en el mismo directorio."
    ) from e


def choose_k_opt(
    ks: List[int],
    silhouettes: List[float],
    k_elbow: int,
    eps_tie: float = 0.02,
    eps_marginal: float = 0.03,
) -> Tuple[int, str]:
    """Aplica reglas heurísticas para elegir k óptimo dado un codo.

    Parameters
    ----------
    ks : list of int
        Lista de valores de k evaluados, ordenada ascendentemente.
    silhouettes : list of float
        Lista de valores de la métrica Silhouette correspondientes a cada k en `ks`.
    k_elbow : int
        Valor del codo (Elbow) detectado o definido manualmente.  Debe pertenecer a `ks`.
    eps_tie : float, optional
        Tolerancia para considerar un empate/meseta de Silhouette (por defecto 0.02).
    eps_marginal : float, optional
        Tolerancia para considerar una mejora marginal cuando el máximo de
        Silhouette está a la derecha del codo (por defecto 0.03).

    Returns
    -------
    k_opt : int
        Valor óptimo de k según las reglas.
    razon : str
        Cadena explicando la regla aplicada (informativa).

    Reglas implementadas
    --------------------
    1) Elige el k con Silhouette máximo.
    2) Si hay empate/meseta (diferencia <= eps_tie respecto al máximo), toma
       el menor k que esté en o después del codo.
    3) Si el Silhouette máximo está a la derecha del codo pero la mejora es
       marginal (diferencia <= eps_marginal entre el máximo y el mejor
       Silhouette en el rango [k_elbow, k_sil_max]), utiliza el k del codo.
    """
    if len(ks) == 0:
        raise ValueError("La lista ks no puede estar vacía.")
    ks_arr = np.asarray(ks, dtype=int)
    sil_arr = np.asarray(silhouettes, dtype=float)
    # Índice del máximo de Silhouette (primera ocurrencia)
    idx_silmax = int(np.nanargmax(sil_arr))
    k_silmax = int(ks_arr[idx_silmax])
    sil_max = float(sil_arr[idx_silmax])

    # Meseta: valores de k con Silhouette dentro de eps_tie del máximo
    plateau_mask = (sil_arr >= sil_max - eps_tie)
    plateau_ks = ks_arr[plateau_mask]
    # Candidatos en/tras el codo
    cand_after_elbow = plateau_ks[plateau_ks >= k_elbow]
    if cand_after_elbow.size > 0:
        k_opt = int(cand_after_elbow.min())
        razon = (
            f"Meseta de Silhouette; se elige el menor k ≥ codo ({k_elbow}) en la meseta."
        )
        return k_opt, razon

    # Sin meseta adecuada.  Si el máximo está a la derecha del codo,
    # evaluar si la mejora es marginal
    if k_silmax > k_elbow:
        # Mejor Silhouette en el rango [k_elbow, k_silmax]
        mask_range = (ks_arr >= k_elbow) & (ks_arr <= k_silmax)
        best_right = float(np.max(sil_arr[mask_range]))
        if sil_max - best_right <= eps_marginal:
            # Mejora marginal → elegir el codo
            k_opt = int(k_elbow)
            razon = (
                f"Silhouette máxima a la derecha del codo, pero mejora marginal (Δ≤{eps_marginal:.2f}); "
                f"se toma k_elbow={k_elbow}."
            )
            return k_opt, razon
    # En cualquier otro caso, usar k_silmax
    k_opt = k_silmax
    razon = "Silhouette máximo se encuentra en k_silmax."
    return k_opt, razon


def run_k_selection(
    xml_path: str,
    phase_shift: float = 0.0,
    k_min: int = 2,
    k_max: int = 12,
    k_manual: Optional[int] = None,
    random_state: int = 42,
    eps_tie: float = 0.02,
    eps_marginal: float = 0.03,
) -> Dict[str, Optional[plt.Figure]]:
    """Evalúa distintos k y genera figuras de selección y PRPD coloreados.

    Parameters
    ----------
    xml_path : str
        Ruta al archivo XML de medición.
    phase_shift : float, optional
        Corrimiento de fase en grados aplicado antes del clustering.  Por
        defecto 0.0.
    k_min : int, optional
        Valor mínimo de k a evaluar (incluido).  Por defecto 2.
    k_max : int, optional
        Valor máximo de k a evaluar (incluido).  Por defecto 12.
    k_manual : int or None, optional
        Valor manual de codo/k a utilizar para recalcular k óptimo.  Si es
        None, solo se utiliza el valor automático.
    random_state : int, optional
        Semilla para la inicialización de K‑Means.  Por defecto 42.
    eps_tie : float, optional
        Tolerancia para mesetas de Silhouette.  Por defecto 0.02.
    eps_marginal : float, optional
        Tolerancia para mejoras marginales de Silhouette.  Por defecto 0.03.

    Returns
    -------
    figs : dict
        Diccionario con las figuras generadas.  Las claves pueden ser:
          - 'curvas' → figura de Silhouette/Inercia y selección de k.
          - 'prpd_auto' → figura del PRPD coloreado usando k_opt_auto.
          - 'prpd_manual' → figura del PRPD coloreado usando k_opt_manual
            (solo si k_manual está definido y distinto del k_opt_auto).

    El lado del terminal también mostrará un resumen del proceso.
    """
    # Parsear el XML y preparar los datos
    data = parse_xml_points(xml_path)
    raw_y = data["raw_y"]
    times = data["times"]
    qty = data["quantity"]
    sample_name = data.get("sample_name")

    # Normalizar la muestra/pixel y calcular fase con corrimiento
    y_norm, _ = normalize_y(raw_y, sample_name)
    phase = phase_from_times(times, phase_shift)

    # Generar histograma PRPD
    H, xedges, yedges = prpd_hist2d(phase, y_norm, qty)

    # Construir matriz de puntos (centros de bins) para los bins ocupados
    Xc, Yc = centers_from_edges(xedges, yedges)
    mask = H.T > 0  # notación: H.T es (ny, nx)
    # Si no hay bins ocupados, no se puede continuar
    if not np.any(mask):
        raise RuntimeError("No hay bins ocupados en el histograma. No se puede ejecutar clustering.")
    P = np.c_[Xc[mask], Yc[mask]]
    scaler = MinMaxScaler()
    Pn = scaler.fit_transform(P)

    # Vectores para almacenar métricas
    ks = list(range(k_min, k_max + 1))
    inertias: List[float] = []
    silhouettes: List[float] = []

    # Preentrenamiento para cada k
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Pn)
        inertias.append(float(km.inertia_))
        labels = km.labels_
        # Calcular Silhouette solo si hay más de un cluster
        if len(np.unique(labels)) > 1:
            sil = float(silhouette_score(Pn, labels))
        else:
            sil = -1.0
        silhouettes.append(sil)

    # Detectar el codo (Elbow) automáticamente mediante Kneedle
    k_elbow_auto, idx_elbow_auto, _ = kneedle_k_from_elbow(ks, inertias)

    # Calcular k óptimo automático según las reglas
    k_opt_auto, razon_auto = choose_k_opt(ks, silhouettes, k_elbow_auto, eps_tie, eps_marginal)

    # Obtener Silhouette del k_opt_auto para mostrarlo
    sil_auto_value = float(silhouettes[ks.index(k_opt_auto)]) if k_opt_auto in ks else float('nan')

    # Manejo de override manual
    k_opt_manual: Optional[int] = None
    sil_manual_value: Optional[float] = None
    razon_manual: Optional[str] = None
    if k_manual is not None:
        # Validar que k_manual esté en el rango evaluado
        if k_manual < k_min or k_manual > k_max:
            raise ValueError(f"k_manual={k_manual} está fuera del rango evaluado [{k_min},{k_max}].")
        # Recalcular k óptimo usando k_manual como codo
        k_opt_manual, razon_manual = choose_k_opt(ks, silhouettes, k_manual, eps_tie, eps_marginal)
        sil_manual_value = float(silhouettes[ks.index(k_opt_manual)])

    # ---------------------------
    # Construcción de figuras
    # ---------------------------
    figs: Dict[str, Optional[plt.Figure]] = {
        "curvas": None,
        "prpd_auto": None,
        "prpd_manual": None,
    }

    # === Figura de Silhouette + Elbow ===
    fig_cur, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    # Trazar Silhouette
    l1, = ax1.plot(ks, silhouettes, marker="o", color="C0", label="Silhouette")
    ax1.set_ylabel("Silhouette promedio", color="C0")
    ax1.tick_params(axis='y', labelcolor="C0")
    # Trazar Inercia
    l2, = ax2.plot(ks, inertias, marker="s", linestyle="--", color="C3", label="Inercia (Elbow)")
    ax2.set_ylabel("Inercia (Within-Cluster SS)", color="C3")
    ax2.tick_params(axis='y', labelcolor="C3")
    ax1.set_xlabel("Número de clusters (k)")
    ax1.set_title("Selección de k: Silhouette vs Elbow")
    # Subtítulo
    fig_cur.suptitle("Codo detectado automáticamente con Kneedle (Elbow)", y=0.98, fontsize=10)
    # Vertical para codo automático
    v_codo_auto = ax1.axvline(k_elbow_auto, color="gray", linestyle=":", linewidth=1.5)
    # Vertical para k_opt_auto
    v_opt_auto = ax1.axvline(k_opt_auto, color="green", linestyle="-", linewidth=2.0)
    # Etiqueta para k_opt_auto
    ax1.text(k_opt_auto, ax1.get_ylim()[1]*0.95,
             f"k_opt={k_opt_auto}\\nSil={sil_auto_value:.2f}\\nCodo≈{k_elbow_auto}",
             rotation=90, va="top", ha="left", color="green", fontsize=8)
    # Si hay override manual distinto del óptimo automático, marcarlo
    v_codo_manual = None
    v_opt_manual_line = None
    if k_manual is not None and k_opt_manual is not None:
        # Línea para codo manual (solo si distinto de codo auto)
        v_codo_manual = ax1.axvline(k_manual, color="purple", linestyle="--", linewidth=1.2)
        # Línea para k_opt_manual
        v_opt_manual_line = ax1.axvline(k_opt_manual, color="orange", linestyle="-", linewidth=1.8)
        # Etiqueta para k_opt_manual
        ax1.text(k_opt_manual, ax1.get_ylim()[1]*0.75,
                 f"k_opt(manual)={k_opt_manual}\\nSil={sil_manual_value:.2f}\\nCodo≈{k_manual}",
                 rotation=90, va="top", ha="left", color="orange", fontsize=8)
    # Leyenda
    handles = [l1, l2, v_opt_auto]
    labels = ["Silhouette", "Inercia (Elbow)", "k_opt auto"]
    if v_codo_auto is not None:
        handles.append(v_codo_auto)
        labels.append("Codo auto")
    if k_manual is not None and k_opt_manual is not None:
        handles.append(v_codo_manual)
        labels.append("Codo manual")
        handles.append(v_opt_manual_line)
        labels.append("k_opt manual")
    ax1.legend(handles, labels, loc="best", fontsize=8)
    # Mini explicaciones debajo de la figura
    expl_text = (
        "Silhouette (–1..1): mayor = mejor separación/compactación.\\n"
        "Elbow/Inercia: punto de rendimientos decrecientes (parsimonia).\\n"
        "Reglas combinan calidad (Silhouette) + parsimonia (Elbow)."
    )
    # Usamos fig.text para colocar la explicación en la parte inferior
    fig_cur.text(0.02, -0.18, expl_text, ha="left", va="top", fontsize=8, wrap=True)
    fig_cur.tight_layout(rect=[0, 0.05, 1, 0.95])
    figs["curvas"] = fig_cur

    # === Figura PRPD coloreado – automático ===
    # Calcular clusters para k_opt_auto usando kmeans_over_bins
    labels_grid_auto, sil_auto_grid = kmeans_over_bins(H, xedges, yedges, k_opt_auto)
    title_auto = (
        f"PRPD coloreado — k_opt auto={k_opt_auto}, Sil={sil_auto_value:.2f} — {Path(xml_path).name}"
    )
    fig_auto = None
    try:
        fig_auto, _ax = plt.subplots(figsize=(9, 5))
        # Dibujar densidad (base)
        im = _ax.imshow(
            H.T + 1e-9,
            origin='lower',
            aspect='auto',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            norm=LogNorm(vmin=1e-9, vmax=max(1.0, H.max())),
            cmap='inferno',
        )
        _ax.set_title(title_auto)
        _ax.set_xlabel("Fase (°)")
        _ax.set_ylabel("Muestra normalizada (0–100)")
        cbar = fig_auto.colorbar(im, ax=_ax)
        cbar.set_label("Recuento (log)")
        _ax.set_xlim(0, 360)
        _ax.set_ylim(0, 100)
        _ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
        # Crear overlay de clusters coloreado
        ny, nx = labels_grid_auto.shape
        # Colormap discreto para clusters
        cmap_clusters = plt.cm.get_cmap('tab20', np.max(labels_grid_auto) + 1 if np.max(labels_grid_auto)>=0 else 1)
        # Construir imagen RGBA para clusters con transparencia
        cluster_img = np.zeros((ny, nx, 4), dtype=float)
        # Asignar color a cada etiqueta de cluster (etiquetas negativas se dejan transparentes)
        unique_labs = np.unique(labels_grid_auto)
        for lab in unique_labs:
            if lab < 0:
                continue
            # Obtener color del colormap
            color = cmap_clusters(lab % cmap_clusters.N)
            # Definir alpha para clusters (0.4)
            color = (color[0], color[1], color[2], 0.4)
            cluster_img[labels_grid_auto == lab] = color
        # Extender cluster_img sobre las coordenadas
        _ax.imshow(
            cluster_img,
            origin='lower',
            aspect='auto',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            interpolation='nearest'
        )
        # Contornos en blanco para delinear clusters
        contours = _ax.contour(
            np.linspace(xedges[0], xedges[-1], H.shape[0]),
            np.linspace(yedges[0], yedges[-1], H.shape[1]),
            labels_grid_auto,
            levels=np.unique(labels_grid_auto[labels_grid_auto >= 0]),
            colors='white', linewidths=0.6, alpha=0.9
        )
        try:
            _ax.clabel(contours, inline=True, fontsize=7, fmt="%d")
        except Exception:
            pass
        fig_auto.tight_layout()
        figs["prpd_auto"] = fig_auto
    except Exception as ex:
        print(f"[WARN] No se pudo generar PRPD auto: {ex}")
        figs["prpd_auto"] = None

    # === Figura PRPD coloreado – manual (solo si corresponde) ===
    if k_manual is not None and k_opt_manual is not None and k_opt_manual != k_opt_auto:
        labels_grid_manual, sil_manual_grid = kmeans_over_bins(H, xedges, yedges, k_opt_manual)
        title_manual = (
            f"PRPD coloreado — k_opt manual={k_opt_manual}, Sil={sil_manual_value:.2f} — {Path(xml_path).name}"
        )
        try:
            fig_manual, _axm = plt.subplots(figsize=(9, 5))
            # Dibujar densidad base
            im_m = _axm.imshow(
                H.T + 1e-9,
                origin='lower',
                aspect='auto',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                norm=LogNorm(vmin=1e-9, vmax=max(1.0, H.max())),
                cmap='inferno',
            )
            _axm.set_title(title_manual)
            _axm.set_xlabel("Fase (°)")
            _axm.set_ylabel("Muestra normalizada (0–100)")
            cbar_m = fig_manual.colorbar(im_m, ax=_axm)
            cbar_m.set_label("Recuento (log)")
            _axm.set_xlim(0, 360)
            _axm.set_ylim(0, 100)
            _axm.xaxis.set_major_locator(ticker.MultipleLocator(30))
            # Overlay de clusters
            ny2, nx2 = labels_grid_manual.shape
            cmap_clusters_m = plt.cm.get_cmap('tab20', np.max(labels_grid_manual) + 1 if np.max(labels_grid_manual)>=0 else 1)
            cluster_img_m = np.zeros((ny2, nx2, 4), dtype=float)
            unique_labs_m = np.unique(labels_grid_manual)
            for lab in unique_labs_m:
                if lab < 0:
                    continue
                color = cmap_clusters_m(lab % cmap_clusters_m.N)
                color = (color[0], color[1], color[2], 0.4)
                cluster_img_m[labels_grid_manual == lab] = color
            _axm.imshow(
                cluster_img_m,
                origin='lower',
                aspect='auto',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                interpolation='nearest'
            )
            # Contornos de clusters
            contours_m = _axm.contour(
                np.linspace(xedges[0], xedges[-1], H.shape[0]),
                np.linspace(yedges[0], yedges[-1], H.shape[1]),
                labels_grid_manual,
                levels=np.unique(labels_grid_manual[labels_grid_manual >= 0]),
                colors='white', linewidths=0.6, alpha=0.9
            )
            try:
                _axm.clabel(contours_m, inline=True, fontsize=7, fmt="%d")
            except Exception:
                pass
            fig_manual.tight_layout()
            figs["prpd_manual"] = fig_manual
        except Exception as ex:
            print(f"[WARN] No se pudo generar PRPD manual: {ex}")
            figs["prpd_manual"] = None

    # Resumen en consola
    print("=== Resumen Bloque 2 ===")
    print(f"Archivo: {Path(xml_path).name}")
    print(f"Fase shift aplicado: {phase_shift}°")
    print(f"Rango k evaluado: [{k_min}, {k_max}]")
    print(f"Codo automático (k_elbow): {k_elbow_auto}")
    print(f"k óptimo automático: {k_opt_auto} (Silhouette={sil_auto_value:.2f}) -> {razon_auto}")
    if k_manual is not None:
        print(f"Codo/k manual: {k_manual}")
        print(f"k óptimo manual: {k_opt_manual} (Silhouette={sil_manual_value:.2f}) -> {razon_manual}")
        if k_opt_manual == k_opt_auto:
            print("Nota: El valor manual coincide con el valor automático.")
    return figs


def main() -> None:
    """Punto de entrada para ejecución desde consola."""
    parser = argparse.ArgumentParser(
        description=(
            "Bloque 2: evalúa un rango de k para clustering K‑Means sobre un "
            "PRPD y genera figuras de Silhouette/Inercia y de PRPD coloreado."
        )
    )
    parser.add_argument("xml_file", help="Ruta al archivo XML de medición.")
    parser.add_argument(
        "--phase-shift", type=float, default=0.0,
        help="Corrimiento de fase (grados). Ej.: 0, 120, 240."
    )
    parser.add_argument(
        "--k-min", type=int, default=2, help="Valor mínimo de k a evaluar (incluido)."
    )
    parser.add_argument(
        "--k-max", type=int, default=12, help="Valor máximo de k a evaluar (incluido)."
    )
    parser.add_argument(
        "--k-manual", type=int, default=None,
        help="Valor manual de codo/k para override. Si se omite, se usa solo el automático."
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Semilla para K‑Means (por defecto 42)."
    )
    parser.add_argument(
        "--out-prefix", type=str, default="bloque2_result",
        help="Prefijo para los archivos PNG de salida. Se crearán _curvas.png, _prpd_auto.png y opcionalmente _prpd_manual.png."
    )
    args = parser.parse_args()

    figs = run_k_selection(
        xml_path=args.xml_file,
        phase_shift=args.phase_shift,
        k_min=args.k_min,
        k_max=args.k_max,
        k_manual=args.k_manual,
        random_state=args.random_state,
    )

    # Guardar figuras
    out_pref = Path(args.out_prefix)
    out_pref.parent.mkdir(parents=True, exist_ok=True)
    if figs.get("curvas") is not None:
        curves_path = out_pref.parent / f"{out_pref.name}_curvas.png"
        figs["curvas"].savefig(curves_path, dpi=300)
        print(f"Figura de curvas guardada en {curves_path}")
    if figs.get("prpd_auto") is not None:
        auto_path = out_pref.parent / f"{out_pref.name}_prpd_auto.png"
        figs["prpd_auto"].savefig(auto_path, dpi=300)
        print(f"Figura PRPD auto guardada en {auto_path}")
    if figs.get("prpd_manual") is not None:
        manual_path = out_pref.parent / f"{out_pref.name}_prpd_manual.png"
        figs["prpd_manual"].savefig(manual_path, dpi=300)
        print(f"Figura PRPD manual guardada en {manual_path}")


if __name__ == "__main__":
    main()
