#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PRPD MEGA — análisis de descargas parciales desde XML Megger UHF PDD (y similares).

Uso:
  python prpd_mega.py --asset "transformador seco" file1.xml [file2.xml ...]
Opciones útiles:
  --phase-shift 240          # aplica corrimiento de fase (°)
  --auto-suggest-shift       # sólo sugiere shift (no aplica)
  --kmin 2 --kmax 8          # rango de k a evaluar
"""

import os, sys, math, argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

# hdbscan opcional
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False


# ===========================
# Utilidades de parsing y PRPD
# ===========================

def parse_xml_points(xml_path):
    """
    Lee XML con listas 'sample'|'pixel', 'times', 'quantity' separadas por espacios.
    Devuelve dict con:
      raw_sample (o pixel), times (ms), quantity
      sample_name: 'sample' o 'pixel'
    """
    root = ET.parse(xml_path).getroot()
    text_map = {}

    # buscar nodos típicos por nombre (case-insensitive)
    for tag in root.iter():
        tname = (tag.tag or "").lower()
        if tname.endswith("sample") or tname.endswith("pixel"):
            text_map["sample_or_pixel"] = (tname, (tag.text or "").strip())
        elif tname.endswith("times"):
            text_map["times"] = (tname, (tag.text or "").strip())
        elif tname.endswith("quantity"):
            text_map["quantity"] = (tname, (tag.text or "").strip())

    if "sample_or_pixel" not in text_map or "times" not in text_map or "quantity" not in text_map:
        raise ValueError(f"{xml_path}: no encuentro listas sample/pixel, times y quantity.")

    name_sp, s_str = text_map["sample_or_pixel"]
    name_tm, t_str = text_map["times"]
    name_qt, q_str = text_map["quantity"]

    def to_float_vec(s):
        s = s.replace("\n", " ").replace("\t", " ")
        parts = [p for p in s.split(" ") if p != ""]
        return np.asarray(list(map(float, parts)), dtype=float)

    raw_y = to_float_vec(s_str)     # sample o pixel (invertido en megger)
    times = to_float_vec(t_str)     # ms (máx ~16.6ms)
    qty   = to_float_vec(q_str)

    n = min(len(raw_y), len(times), len(qty))
    return {
        "raw_y": raw_y[:n],
        "times": times[:n],
        "quantity": qty[:n],
        "sample_name": name_sp
    }


def identify_sensor_from_data(raw_y, sample_name):
    """
    Heurística muy simple: usar nombre + rango/amplitud para etiquetar sensor.
    No es definitivo, pero ayuda a títulos.
    """
    name = (sample_name or "").lower()
    rng = np.nanmax(raw_y) - np.nanmin(raw_y)
    avg = float(np.nanmean(raw_y))

    if "uhf" in name:
        return "UHF"
    if "tev" in name:
        return "TEV"
    if "hfct" in name:
        return "HFCT"

    # heurística por rango
    if rng <= 400 and avg <= 400:
        return "UHF"
    return "TEV/HFCT"


def normalize_y(raw_y, sample_name):
    """
    Normaliza 'sample' o invierte 'pixel' para que el eje vaya 0..100 (abajo->arriba).
    """
    n = raw_y.astype(float)
    if "pixel" in (sample_name or "").lower():
        n = 100.0 - (n * 100.0 / max(1.0, np.nanmax(n)))
        label = "Pixel invertido (0–100)"
    else:
        n = (n - np.nanmin(n)) / (np.nanmax(n) - np.nanmin(n) + 1e-12) * 100.0
        label = "Sample normalizado (0–100)"
    return n, label


def phase_from_times(times_ms, shift_deg=0.0):
    """
    Convierte ms a grados (0..360) asumiendo periodo ~16.6667ms (60Hz). Aplica shift si se indica.
    """
    phase = (times_ms / 16.6667) * 360.0
    if shift_deg:
        phase = (phase + shift_deg) % 360.0
    return phase


def suggest_phase_shift(phase, y_norm):
    """
    Sugeridor muy simple: proyecta densidad sobre fase y busca máximos en cuadrantes.
    Devuelve 0, 120 o 240 aprox según correlación (heurístico).
    """
    # hist de fase
    hist, edges = np.histogram(phase % 360.0, bins=36, range=(0,360), weights=None)
    # correlación con tres plantillas (desfase 0/120/240)
    tpl = np.array([1,0,0,  1,0,0,  1,0,0]*4)[:36]  # patrón 3 picos
    c0 = np.correlate(hist, np.roll(tpl, 0))[0]
    c1 = np.correlate(hist, np.roll(tpl, 12))[0]
    c2 = np.correlate(hist, np.roll(tpl, 24))[0]
    ops = [0,120,240]
    best = int(ops[int(np.argmax([c0,c1,c2]))])
    return best


def prpd_hist2d(phase, y_norm, quantity, bins_phase=360, bins_y=100):
    """
    Histograma 2D (fase vs y_norm) ponderado por quantity.
    Devuelve H (nx,ny, ya transpuesto para imagen), xedges, yedges.
    """
    H, xedges, yedges = np.histogram2d(
        phase, y_norm,
        bins=[bins_phase, bins_y],
        range=[[0,360],[0,100]],
        weights=quantity
    )
    # para pintar en imshow usamos H.T (ny,nx)
    return H, xedges, yedges


def centers_from_edges(xedges, yedges):
    Xc = 0.5*(xedges[:-1]+xedges[1:])
    Yc = 0.5*(yedges[:-1]+yedges[1:])
    Xc2, Yc2 = np.meshgrid(Xc, Yc, indexing='xy')  # cuidado: xy => X filas horizontal
    return Xc2, Yc2


# ===========================
# Plot helpers
# ===========================

def prpd_density_plot(phase, y_norm, quantity, title):
    H, xedges, yedges = prpd_hist2d(phase, y_norm, quantity)
    fig, ax = plt.subplots(figsize=(9,5))
    im = ax.imshow(
        H.T + 1e-9,
        origin='lower', aspect='auto',
        extent=[xedges[0],xedges[-1], yedges[0],yedges[-1]],
        norm=LogNorm(vmin=1e-9, vmax=max(1.0, H.max()))
    )
    ax.set_title(title)
    ax.set_xlabel("Fase (°)")
    ax.set_ylabel("Sample normalizado (0–100)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Recuento (log)")
    ax.set_xlim(0,360); ax.set_ylim(0,100)
    # ejes bonitos
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    plt.tight_layout()
    return H, xedges, yedges


def contours_from_grid(grid, xedges, yedges, ax, alpha=0.9, lw=1.0):
    """
    Dibuja contornos finos de los clusters (grid ny,nx con labels -1 ruido).
    """
    from matplotlib import cm
    ny, nx = grid.shape
    xs = np.linspace(xedges[0], xedges[-1], nx)
    ys = np.linspace(yedges[0], yedges[-1], ny)
    L = grid.copy().astype(float)
    L[L<0] = np.nan  # no contornear ruido

    CS = ax.contour(xs, ys, L, levels=np.unique(L[np.isfinite(L)]),
                    linewidths=lw, alpha=alpha, colors='k')
    # opcional: etiquetas
    try:
        ax.clabel(CS, inline=True, fontsize=7, fmt="%d")
    except Exception:
        pass


def prpd_clusters_on_raw(phase, y_norm, quantity, labels_grid, xedges, yedges, title):
    """
    Render: crudo (densidad) + contornos finos del grid de clusters.
    """
    H, _, _ = prpd_hist2d(phase, y_norm, quantity)
    fig, ax = plt.subplots(figsize=(9,5))
    im = ax.imshow(
        H.T + 1e-9, origin='lower', aspect='auto',
        extent=[xedges[0],xedges[-1], yedges[0],yedges[-1]],
        norm=LogNorm(vmin=1e-9, vmax=max(1.0, H.max()))
    )
    ax.set_title(title)
    ax.set_xlabel("Fase (°)")
    ax.set_ylabel("Sample normalizado (0–100)")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Recuento (log)")
    ax.set_xlim(0,360); ax.set_ylim(0,100)
    contours_from_grid(labels_grid, xedges, yedges, ax, alpha=0.9, lw=1.0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    plt.tight_layout()


# ===========================
# Selección de k (Kneedle + Silhouette)
# ===========================

def kneedle_k_from_elbow(ks, inertias):
    x = np.array(ks, dtype=float); y = np.array(inertias, dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
    y_line = y_n[0] + (y_n[-1] - y_n[0]) * x_n
    diff = y_line - y_n  # curvatura (codo)
    idx = int(np.argmax(diff))
    return int(x[idx]), idx, diff


def select_k_with_rules(ks, inertias, silhouettes, eps_tie=0.02, eps_marginal=0.03):
    ks = np.array(ks, dtype=int)
    inertias = np.array(inertias, dtype=float)
    silhouettes = np.array(silhouettes, dtype=float)

    k_codo, _, _ = kneedle_k_from_elbow(ks, inertias)

    idx_max = int(np.nanargmax(silhouettes))
    k_sil = int(ks[idx_max])
    sil_max = float(silhouettes[idx_max])

    # meseta alrededor del máximo
    meseta_mask = (silhouettes >= sil_max - eps_tie)
    candidatos = ks[meseta_mask]

    razon = []
    if k_sil > k_codo:
        rango = (ks >= k_codo) & (ks <= k_sil)
        best_right = float(np.max(silhouettes[rango]))
        if sil_max - best_right <= eps_marginal:
            razon.append(f"Mejora marginal a la derecha del codo (Δ≤{eps_marginal:.02f}); elijo k_codo={k_codo}")
            return k_codo, k_codo, k_sil, "; ".join(razon)

    cand_post_codo = candidatos[candidatos >= k_codo]
    if cand_post_codo.size > 0:
        razon.append(f"Empate/meseta (≤{eps_tie:.02f}) cerca del máximo; elijo menor k post-codo={int(cand_post_codo.min())}")
        return int(cand_post_codo.min()), k_codo, k_sil, "; ".join(razon)

    razon.append(f"Silhouette máximo en k={k_sil}")
    return k_sil, k_codo, k_sil, "; ".join(razon)


def combined_elbow_silhouette_plot(ks, inertias, silhouettes, k_auto, k_manual=None, rules_text=""):
    fig, ax1 = plt.subplots(figsize=(9,5))
    ax2 = ax1.twinx()
    l1, = ax1.plot(ks, silhouettes, marker="o", color="C0", label="Silhouette")
    l2, = ax2.plot(ks, inertias, marker="o", linestyle="--", color="C3", label="Inercia (Elbow)")
    ax1.set_ylabel("Silhouette promedio", color="C0")
    ax2.set_ylabel("Inercia (Within-Cluster SS)", color="C3")
    ax1.set_xlabel("Número de clusters (k)")
    ax1.set_title("Selección de k: Kneedle (auto) + Silhouette")

    v_auto = ax1.axvline(k_auto, color="C2", linestyle=":", linewidth=1.8)
    ax1.text(k_auto, ax1.get_ylim()[1]*0.92, f"k_auto={k_auto}", rotation=90, va="top", color="C2")
    v_man = None
    if k_manual is not None and (k_manual != k_auto):
        v_man = ax1.axvline(k_manual, color="purple", linestyle="-.", linewidth=1.5)
        ax1.text(k_manual, ax1.get_ylim()[1]*0.65, f"k_manual={k_manual}", rotation=90, va="top", color="purple")

    handles = [l1, l2, v_auto]
    labels  = ["Silhouette", "Inercia (Elbow)", "k_auto (Kneedle)"]
    if v_man is not None:
        handles.append(v_man); labels.append("k_manual")
    ax1.legend(handles, labels, loc="best")

    if rules_text:
        fig.text(0.02, -0.15, rules_text, ha="left", va="top", fontsize=8, wrap=True)

    fig.tight_layout(); plt.show()


# ===========================
# K-Means sobre bins
# ===========================

def kmeans_over_bins(H, xedges, yedges, k):
    """
    Clustering K-Means sobre bins ocupados del histograma.
    Devuelve grilla ny,nx con labels (-1 ruido).
    """
    Xc, Yc = centers_from_edges(xedges, yedges)
    mask = H.T > 0
    P = np.c_[Xc[mask], Yc[mask]]
    scaler = MinMaxScaler(); Pn = scaler.fit_transform(P)

    grid = np.full_like(H.T, fill_value=-1, dtype=int)
    if Pn.shape[0] == 0:
        return grid, None

    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Pn)
    labels = km.labels_
    grid[mask] = labels

    sil = -1.0
    if len(np.unique(labels))>1:
        sil = silhouette_score(Pn, labels)
    return grid, sil


# ===========================
# HDBSCAN / DBSCAN
# ===========================

def hdbscan_on_bins(H, xedges, yedges, min_cluster_size=10, min_samples=None):
    """
    Ejecuta HDBSCAN sobre los bins ocupados (o DBSCAN si HDBSCAN no está disponible).
    Devuelve: grid(int), info(dict), used_hdb(bool)
    """
    Xc, Yc = centers_from_edges(xedges, yedges)
    mask = H.T > 0
    P = np.c_[Xc[mask], Yc[mask]]
    scaler = MinMaxScaler(); Pn = scaler.fit_transform(P)

    grid = np.full_like(H.T, fill_value=-1, dtype=int)
    info = {"method": None, "n_clusters": 0, "noise_frac": 0.0,
            "persistence_avg": None, "cluster_sizes": {}, "msg": ""}

    if Pn.shape[0] == 0:
        info["msg"] = "Sin bins ocupados."
        return grid, info, HDBSCAN_AVAILABLE

    if HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    cluster_selection_method='eom')
        labels = clusterer.fit_predict(Pn)
        probs = getattr(clusterer, "probabilities_", None)
        grid[mask] = labels
        used_hdb = True

        labs_valid = np.unique(labels[labels >= 0])
        n_clusters = int(len(labs_valid))
        noise_frac = float(np.mean(labels < 0))

        # “persistencia” aprox como media de probs por cluster
        pers_map = {}
        if n_clusters > 0:
            for lab in labs_valid:
                idxs = (labels == lab)
                p_mean = float(np.mean(probs[idxs])) if (probs is not None and np.any(idxs)) else None
                pers_map[int(lab)] = {"p_mean": p_mean}
            persistence_avg = float(np.nanmean([v["p_mean"] for v in pers_map.values()
                                                if v["p_mean"] is not None])) if pers_map else None
        else:
            persistence_avg = None

        sizes = {int(l): int(np.sum(labels == l)) for l in labs_valid}
        info.update({
            "method": "HDBSCAN",
            "n_clusters": n_clusters,
            "noise_frac": noise_frac,
            "persistence_avg": persistence_avg,
            "cluster_sizes": sizes,
            "msg": f"HDBSCAN: {n_clusters} clusters, ruido {noise_frac*100:.1f}%, "
                   f"p≈{(persistence_avg*100 if persistence_avg is not None else 0):.1f}%"
        })
        return grid, info, used_hdb
    else:
        # Fallback DBSCAN
        eps = 0.06; ms = 5
        db = DBSCAN(eps=eps, min_samples=ms).fit(Pn)
        labels = db.labels_
        grid[mask] = labels
        used_hdb = False

        labs_valid = np.unique(labels[labels >= 0])
        n_clusters = int(len(labs_valid))
        noise_frac = float(np.mean(labels < 0))
        sizes = {int(l): int(np.sum(labels == l)) for l in labs_valid}

        info.update({
            "method": "DBSCAN",
            "n_clusters": n_clusters,
            "noise_frac": noise_frac,
            "persistence_avg": None,
            "cluster_sizes": sizes,
            "msg": f"DBSCAN: {n_clusters} clusters, ruido {noise_frac*100:.1f}% "
                   f"(eps={eps}, min_samples={ms})"
        })
        return grid, info, used_hdb


def hdbscan_reduce_to_k(labels_grid, xedges, yedges, target_k):
    """
    Reduce/fusiona los clusters de labels_grid hasta target_k (si hay más).
    Devuelve: (new_grid, info_fusion)
    """
    labs = np.unique(labels_grid[labels_grid >= 0])
    n_before = int(len(labs))
    info = {"n_before": n_before, "n_after": n_before, "action": "identico", "mapping": {}}

    if n_before == 0:
        return labels_grid, info
    if n_before <= target_k:
        info["action"] = "ya_menos_que_k"
        return labels_grid, info

    Xc, Yc = centers_from_edges(xedges, yedges)
    cents, lab_list = [], []
    for lab in labs:
        m = (labels_grid == lab)
        xs = Xc[m]; ys = Yc[m]
        cents.append([np.mean(xs), np.mean(ys)])
        lab_list.append(int(lab))
    cents = np.array(cents, dtype=float)

    scaler = MinMaxScaler(); Cn = scaler.fit_transform(cents)
    km = KMeans(n_clusters=target_k, n_init=10, random_state=42).fit(Cn)
    assign = km.labels_

    mapping = { lab_list[i]: int(assign[i]) for i in range(len(lab_list)) }
    new_grid = np.full_like(labels_grid, fill_value=-1, dtype=int)
    for lab in lab_list:
        new_grid[labels_grid == lab] = mapping[lab]

    info["n_after"] = int(target_k)
    info["action"] = f"fusion_{n_before}_a_{target_k}"
    info["mapping"] = mapping
    return new_grid, info


def grid_cluster_count(labels_grid):
    """Cuenta clusters válidos (labels>=0) en una grilla."""
    return int(len(np.unique(labels_grid[labels_grid >= 0])))


# ===========================
# Métricas heurísticas y probas
# ===========================

def features_from_grid(labels_grid, xedges, yedges):
    """
    Extrae rasgos por cluster: ancho en fase, altura media en y, densidad relativa, etc.
    """
    Xc, Yc = centers_from_edges(xedges, yedges)
    labs = np.unique(labels_grid[labels_grid>=0])
    feats = {}
    for lab in labs:
        m = (labels_grid==lab)
        xs = Xc[m]; ys = Yc[m]
        if xs.size==0: continue
        ph_min, ph_max = np.min(xs), np.max(xs)
        phase_width = float(ph_max - ph_min)
        y_mean = float(np.mean(ys))
        y_std  = float(np.std(ys))
        area   = float(np.sum(m))  # #bins
        feats[int(lab)] = dict(phase_width=phase_width, y_mean=y_mean, y_std=y_std, area=area)
    return feats


def typify_cluster(feat):
    """
    Asigna etiqueta heurística básica por rasgos PRPD (super simplificada).
    """
    pw = feat.get("phase_width", 0.0)
    ym = feat.get("y_mean", 50.0)
    # heurísticas simples
    if pw < 40 and 25 <= ym <= 75:
        return "cavidad"
    if pw < 40 and ym < 25:
        return "corona"
    if pw < 40 and ym > 75:
        return "flotante"
    if 40 <= pw <= 120:
        return "superficial"
    return "otras"


def final_probabilities(phase, y_norm, quantity, scenarios, xedges, yedges):
    """
    Consolida escenarios (kmeans/hdbscan) y estima % por tipo + prob(n fuentes).
    """
    type_counts = {"cavidad":0, "superficial":0, "corona":0, "flotante":0, "otras":0, "ruido":0}
    prob_n = {"cavidad":{}, "corona":{}, "superficial":{}, "flotante":{}}
    type_by_lab_any = {}

    for name, grid in scenarios.items():
        feats = features_from_grid(grid, xedges, yedges)
        labs  = sorted(list(feats.keys()))
        n_src = len(labs)

        # contar tipos
        local_types = []
        for lab in labs:
            t = typify_cluster(feats[lab])
            local_types.append(t)
            type_counts[t] += 1
            type_by_lab_any[f"{name}:{lab}"] = t

        # prob(n fuentes) por tipo (conteo simple)
        for t in ["cavidad","corona","superficial","flotante"]:
            prob_n.setdefault(t, {})
            prob_n[t][n_src] = prob_n[t].get(n_src, 0) + local_types.count(t)

    # normalizar a %
    total = float(sum(type_counts.values()) + 1e-12)
    perc = {k: v/total for k,v in type_counts.items()}

    # normalizar prob_n por tipo
    for t in prob_n:
        s = float(sum(prob_n[t].values()) + 1e-12)
        if s==0: continue
        for n in prob_n[t]:
            prob_n[t][n] = prob_n[t][n] / s

    return perc, prob_n, type_by_lab_any


# ===========================
# MAIN
# ===========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, default="desconocido", help="tipo de activo (p.ej. transformador seco)")
    parser.add_argument("--phase-shift", type=float, default=0.0, help="corrimiento de fase (°)")
    parser.add_argument("--auto-suggest-shift", action="store_true", help="sugiere shift (no aplica)")
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    plt.ion()  # modo interactivo

    for f in args.files:
        data = parse_xml_points(f)
        sensor = identify_sensor_from_data(data["raw_y"], data["sample_name"])
        y_norm, y_label = normalize_y(data["raw_y"], data["sample_name"])

        # shift
        phase0 = phase_from_times(data["times"], 0.0)
        if args.auto_suggest_shift:
            sug = suggest_phase_shift(phase0, y_norm)
            print(f"[SUGERENCIA] {Path(f).name}: shift ≈ +{sug}° (no aplicado). Úsalo con --phase-shift {sug}")

        phase = phase_from_times(data["times"], args.phase_shift)
        base = {"path": f, "phase": phase, "y": y_norm, "quantity": data["quantity"]}

        # ---------- A) PRPD densidad (crudo)
        title_A = f"PRPD — {sensor} — {Path(f).name}"
        H, xedges, yedges = prpd_density_plot(base["phase"], base["y"], base["quantity"], title_A)

        # ---------- B) Selección de k
        Xc, Yc = centers_from_edges(xedges, yedges)
        mask = H.T > 0
        P = np.c_[Xc[mask], Yc[mask]]
        scaler = MinMaxScaler(); Pn = scaler.fit_transform(P)

        ks = list(range(args.kmin, args.kmax+1))
        inertias = []; silhouettes = []
        for k in ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Pn)
            inertias.append(km.inertia_)
            lab = km.labels_
            silhouettes.append(silhouette_score(Pn, lab) if len(np.unique(lab))>1 else -1.0)

        k_auto, k_codo, k_sil, razon = select_k_with_rules(ks, inertias, silhouettes)
        rules = ("Reglas:\n"
                 "a) Elige el k con Silhouette máximo.\n"
                 "b) Si hay empate/meseta, toma el menor k en o después del codo.\n"
                 "c) Si el Silhouette máximo está a la derecha del codo y la mejora es marginal, quédate con el k del codo.")
        combined_elbow_silhouette_plot(ks, inertias, silhouettes, k_auto, None, rules_text=rules)
        print(f"[AUTO] k_codo={k_codo}, k_silmax={k_sil}, k_auto={k_auto} :: {razon}")

        # override manual opcional
        try:
            user = input("¿k manual? (ENTER = no, ó número): ").strip()
        except Exception:
            user = ""
        k_manual = None
        if user:
            try:
                val = int(user)
                if args.kmin <= val <= args.kmax:
                    k_manual = val
                    print(f"[MANUAL] k_manual={k_manual}")
                    combined_elbow_silhouette_plot(ks, inertias, silhouettes, k_auto, k_manual, rules_text=rules)
                else:
                    print("[AVISO] k fuera de rango; sigo con auto.")
            except:
                print("[AVISO] entrada inválida; sigo con auto.")

        # ---------- C) K-Means auto
        labels_grid_auto, sil_auto = kmeans_over_bins(H, xedges, yedges, k_auto)
        title_C = f"PRPD coloreado por K-Means (k_auto={k_auto}, silhouette={sil_auto:.2f}) — {Path(f).name}"
        prpd_clusters_on_raw(base["phase"], base["y"], base["quantity"],
                             labels_grid_auto, xedges, yedges, title_C)

        # ---------- D) K-Means manual (si hay)
        labels_grid_man = None
        if k_manual is not None:
            labels_grid_man, sil_man = kmeans_over_bins(H, xedges, yedges, k_manual)
            title_D = f"K-Means (k_manual={k_manual}, silhouette={sil_man:.2f}) — {Path(f).name}"
            prpd_clusters_on_raw(base["phase"], base["y"], base["quantity"],
                                 labels_grid_man, xedges, yedges, title_D)

        # ---------- E) HDBSCAN natural
        grid_E, info_E, used_hdb = hdbscan_on_bins(H, xedges, yedges)
        method_E = info_E["method"]
        nE = info_E["n_clusters"]; noiseE = info_E["noise_frac"]; pE = info_E["persistence_avg"]
        title_E = f"{method_E} Natural — {Path(f).name} (clusters={nE}, ruido={noiseE*100:.1f}%"
        if pE is not None:
            title_E += f", p≈{pE*100:.0f}%"
        title_E += ")"
        prpd_clusters_on_raw(base["phase"], base["y"], base["quantity"],
                             grid_E, xedges, yedges, title_E)
        print(f"[INFO] E: {info_E['msg']}")

        # ---------- F) HDBSCAN → k (resumen/fusión)
        k_target = k_manual if (k_manual is not None) else k_auto
        grid_F, info_F = hdbscan_reduce_to_k(grid_E, xedges, yedges, k_target)
        nF = grid_cluster_count(grid_F)
        if info_F["action"] == "identico" or nF == info_E["n_clusters"]:
            print(f"[INFO] F: Resumen idéntico al natural (n={nF} ya coincide con k_target={k_target}).")
        elif info_F["action"].startswith("fusion"):
            print(f"[INFO] F: {info_F['action']} — mapping: {info_F['mapping']}")
        elif info_F["action"] == "ya_menos_que_k":
            print(f"[INFO] F: Natural n={info_F['n_before']} <= k_target={k_target}. No se fuerza creación de clusters.")
        title_F = f"{method_E} → k={k_target} (Resumen) — {Path(f).name} (n_final={nF})"
        prpd_clusters_on_raw(base["phase"], base["y"], base["quantity"],
                             grid_F, xedges, yedges, title_F)

        # ---------- G) Probabilidades y resumen ----------
        scenarios = {"kmeans_auto": labels_grid_auto,
                     "hdbscan_E": grid_E,
                     "hdbscan_to_k": grid_F}
        if labels_grid_man is not None:
            scenarios["kmeans_manual"] = labels_grid_man

        perc, prob_n, type_by_lab_any = final_probabilities(
            base["phase"], base["y"], base["quantity"], scenarios, xedges, yedges
        )

        print("\n=== RESULTADO FINAL (resumen) ===")
        print("Porcentajes por tipo (aprox, heurístico):")
        for t in ["cavidad","superficial","corona","flotante","otras","ruido"]:
            print(f"  {t:12s}: {perc.get(t,0.0)*100:5.1f}%")

        print("\nProbabilidad de n fuentes (escenarios bootstrap):")
        for t in ["cavidad","corona","superficial","flotante"]:
            pmap = prob_n.get(t,{})
            if not pmap:
                print(f"  {t:12s}: (sin datos)")
                continue
            s = ", ".join([f'{n}->{p*100:.0f}%' for n,p in pmap.items()])
            print(f"  {t:12s}: {s}")

        print("\nNotas: Los % se basan en rasgos PRPD "
              "(simetría, ventanas de fase, ancho, amplitud, densidad) "
              "y consistencia entre escenarios/clusters.")
        if not HDBSCAN_AVAILABLE:
            print("[AVISO] HDBSCAN no instalado. Para vistas E/F robustas: pip install hdbscan")

        input("\nPulsa ENTER para cerrar las figuras...")  # mantener ventanas abiertas

    plt.ioff()


if __name__ == "__main__":
    if len(sys.argv)==1:
        print('Uso: python prpd_mega.py --asset "metal-clad" file1.xml [file2.xml ...]')
    else:
        main()
