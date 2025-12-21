# report.py
# Exporta un PDF multipagina con resumen, figuras y KPIs (estilo informe)
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib import gridspec
from matplotlib.lines import Line2D

from PRPDapp.conclusion_rules import build_conclusion_block
from PRPDapp.clouds import pixel_cluster_clouds, combine_clouds, select_dominant_clouds
from PRPDapp.logic_hist import compute_semicycle_histograms_from_aligned
from PRPDapp.pipeline import ANN_DISPLAY, ANN_CLASSES


A4 = (8.27, 11.69)


def _add_header(fig: Figure, title: str, subtitle: str, page_num: int, total_pages: int) -> None:
    fig.text(0.04, 0.965, title, fontsize=16, fontweight="bold", ha="left", va="top")
    fig.text(0.04, 0.94, subtitle, fontsize=9, color="#555555", ha="left", va="top")
    fig.text(0.96, 0.965, f"Pag {page_num}/{total_pages}", fontsize=8, color="#666666", ha="right", va="top")
    line = Line2D([0.04, 0.96], [0.925, 0.925], transform=fig.transFigure, color="#d0d7e2", linewidth=1.0)
    fig.add_artist(line)


def _draw_table(ax, rows: list[tuple[str, str]], *, y_top: float = 0.9, y_step: float = 0.065) -> None:
    ax.axis("off")
    y = y_top
    for key, value in rows:
        ax.text(0.02, y, f"{key}:", fontsize=10, fontweight="bold", ha="left", va="top")
        ax.text(0.38, y, str(value), fontsize=10, ha="left", va="top")
        y -= y_step


def _draw_ann_bar(ax, ann_block: dict) -> None:
    ax.clear()
    ax.set_facecolor("#f7f9fc")
    display = ann_block.get("display") if isinstance(ann_block, dict) else None
    labels = []
    values = []
    colors = []
    if isinstance(display, dict):
        labels = list(display.get("labels") or [])
        values = [float(v) for v in (display.get("values") or [])]
        colors = list(display.get("colors") or [])
    if not labels:
        probs = ann_block.get("probs") if isinstance(ann_block, dict) else {}
        for cls in ANN_CLASSES:
            meta = ANN_DISPLAY.get(cls, {})
            labels.append(meta.get("label", cls))
            colors.append(meta.get("color", "#999999"))
            values.append(float(probs.get(cls, 0.0) or 0.0))
    x = np.arange(len(labels), dtype=float)
    bars = ax.bar(x, values, color=colors, edgecolor="#37474f", linewidth=1.0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probabilidad")
    ax.set_title("ANN Predicted PD Source", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" / ", "\n") for l in labels], fontsize=9)
    max_val = max(values) if values else 0.0
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02, f"{values[i]*100:.1f}%", ha="center", va="bottom", fontsize=9)
        if values and values[i] == max_val:
            bar.set_linewidth(2.0)
            bar.set_edgecolor("#0d1117")


def _draw_prpd_scatter(ax, phase, amp, title: str) -> None:
    ax.scatter(phase, amp, s=4, alpha=0.6, color="#1f77b4")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Fase (deg)")
    ax.set_ylabel("Amplitud")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)


def _draw_prpd_quintiles(ax, phase: np.ndarray, amp: np.ndarray, quint: np.ndarray, title: str) -> None:
    ax.clear()
    ax.set_facecolor("#ffffff")
    if phase.size and amp.size:
        if quint is not None and quint.size == phase.size:
            colors = {
                1: "#0066CC",
                2: "#009900",
                3: "#FFCC00",
                4: "#FF8000",
                5: "#CC0000",
            }
            for q in range(1, 6):
                m = quint == q
                if np.any(m):
                    ax.scatter(phase[m], amp[m], s=4, alpha=0.7, color=colors[q], label=f"Q{q}")
            try:
                ax.legend(loc="upper right", fontsize=7, frameon=True)
            except Exception:
                pass
        else:
            ax.scatter(phase, amp, s=4, alpha=0.6, color="#1f77b4")
    else:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Fase (deg)")
    ax.set_ylabel("Amplitud")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)


def _cluster_profiles() -> dict[str, dict]:
    return {
        "S1 Weak": {"eps": 0.055, "min_samples": 8, "force_multi": False},
        "S2 Strong": {"eps": 0.030, "min_samples": 14, "force_multi": True},
    }


def _build_cluster_variants(
    ph: np.ndarray,
    amp: np.ndarray,
    filter_level: str | None,
) -> tuple[dict[str, list[dict]], list[dict], list[dict], list[dict]]:
    variants: dict[str, list[dict]] = {}
    if not (ph.size and amp.size):
        return variants, [], [], []
    total_points = float(ph.size)
    for name, params in _cluster_profiles().items():
        clouds = pixel_cluster_clouds(ph, amp, **params)
        for idx, c in enumerate(clouds):
            c = dict(c)
            c["legend"] = f"{name.split()[0]}-{idx + 1}"
            c["frac"] = c.get("frac", c.get("count", 0) / max(1.0, total_points))
            clouds[idx] = c
        variants[name] = clouds
    combined_all = [dict(c) for arr in variants.values() for c in arr]
    if not combined_all:
        return variants, [], [], []
    filt = (filter_level or "").lower()
    if "strong" in filt or "s2" in filt:
        base_s3 = variants.get("S2 Strong", []) or combined_all
    elif "weak" in filt or "s1" in filt:
        base_s3 = variants.get("S1 Weak", []) or combined_all
    else:
        base_s3 = combined_all
    clouds_s4 = combine_clouds(base_s3)
    for idx, c in enumerate(clouds_s4):
        c["legend"] = f"C{idx + 1}"
    base_for_s5 = clouds_s4 if clouds_s4 else base_s3
    clouds_s5 = select_dominant_clouds(base_for_s5, min_frac=0.10)
    for idx, c in enumerate(clouds_s5):
        c["legend"] = f"C{idx + 1}"
    return variants, base_s3, clouds_s4, clouds_s5


def _draw_clusters(ax, ph: np.ndarray, amp: np.ndarray, clouds: list[dict], title: str, *, color_points: bool = True, include_k: bool = True, max_labels: int = 10) -> None:
    ax.clear()
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
        "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
        "#dbdb8d", "#9edae5",
    ]
    n_clusters = len(clouds) if clouds else 0
    if include_k and n_clusters:
        title = f"{title} - k={n_clusters}"
    if ph.size and amp.size:
        if color_points and n_clusters > 0:
            centers = np.array([[c.get("phase_mean", 0.0), c.get("y_mean", 0.0)] for c in clouds], dtype=float)
            if centers.size:
                dp = np.abs(((ph[:, None] - centers[:, 0] + 180.0) % 360.0) - 180.0)
                dy = np.abs(amp[:, None] - centers[:, 1])
                lbl = np.argmin(0.6 * dp + 0.4 * dy, axis=1)
                for j in range(n_clusters):
                    m = lbl == j
                    if np.any(m):
                        color = palette[j % len(palette)]
                        label = clouds[j].get("legend") or (f"C{j + 1}" if j < max_labels else None)
                        ax.scatter(ph[m], amp[m], s=5, alpha=0.65, color=color, label=label)
        else:
            ax.scatter(ph, amp, s=3, alpha=0.4, color="#bfbfbf")
    for j, c in enumerate(clouds or []):
        px = float(c.get("phase_mean", 0.0))
        py = float(c.get("y_mean", 0.0))
        color = palette[j % len(palette)]
        label = c.get("legend") or (f"C{j + 1}" if j < max_labels else None)
        ax.scatter([px], [py], s=60, color=color, edgecolors="black", label=label)
    try:
        handles, labels = ax.get_legend_handles_labels()
        handles_labels = [(h, l) for h, l in zip(handles, labels) if l]
        if len(handles_labels) > max_labels:
            handles_labels = handles_labels[:max_labels]
            extra = Line2D([], [], marker="o", color="white", markerfacecolor="#bfbfbf", markersize=6, linestyle="None")
            handles_labels.append((extra, "..."))
        if handles_labels:
            ax.legend([h for h, _ in handles_labels], [l for _, l in handles_labels], loc="upper right", fontsize=7)
    except Exception:
        pass
    ax.set_title(title)
    ax.set_xlim(0, 360)
    ax.set_xlabel("Fase (deg)")
    ax.set_ylabel("Amplitud")


def _draw_gap_chart(ax, gap_stats: dict | None) -> None:
    ax.clear()
    ax.set_facecolor("#f5f6fb")
    if not gap_stats:
        ax.text(0.5, 0.5, "Gap-time no disponible", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    times = np.asarray(gap_stats.get("time_s", []), dtype=float)
    amps = np.asarray(gap_stats.get("series", []), dtype=float)
    if not times.size or not amps.size or times.size != amps.size:
        ax.text(0.5, 0.5, "Gap-time sin serie asociada", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    base = float(gap_stats.get("base", np.median(amps)))
    tol = float(gap_stats.get("tolerance", 5.0))
    mask = np.asarray(gap_stats.get("mask", []), dtype=int)
    quiet_hi = base + tol
    quiet_lo = base - tol
    ax.fill_between(times, quiet_lo, quiet_hi, color="#dcedc8", alpha=0.5)
    ax.plot(times, amps, color="#0d47a1", linewidth=1.2, label="Magnitud <max>")
    if mask.size == amps.size:
        shots = mask.astype(bool)
        ax.scatter(times[shots], amps[shots], color="#ff7043", s=12, alpha=0.7, label="Descarga detectada")
    ax.plot(times, np.full_like(times, quiet_hi), "--", color="#43a047", linewidth=1.0)
    ax.axhline(quiet_lo, color="#43a047", linestyle="--", linewidth=1.0)
    ax.set_title("Gap-time (serie)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Magnitud (dBm)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)


def _draw_histograms(ax_amp, ax_phase, ph: np.ndarray, amp: np.ndarray) -> None:
    hist = compute_semicycle_histograms_from_aligned({"phase_deg": ph, "amplitude": amp}, N=16)
    xi = np.arange(1, 17)
    Ha_pos = hist.get("H_amp_pos", np.zeros(16))
    Ha_neg = hist.get("H_amp_neg", np.zeros(16))
    Hp_pos = hist.get("H_ph_pos", np.zeros(16))
    Hp_neg = hist.get("H_ph_neg", np.zeros(16))
    ax_amp.plot(xi, Ha_pos, "-o", label="H_amp+")
    ax_amp.plot(xi, Ha_neg, "-o", label="H_amp-")
    ax_amp.set_title("Histograma de Amplitud (N=16)")
    ax_amp.set_xlabel("Indice")
    ax_amp.set_ylabel("H_amp (norm)")
    ax_amp.grid(True, alpha=0.2)
    ax_amp.legend(fontsize=8)

    ax_phase.plot(xi, Hp_pos, "-o", label="H_ph+")
    ax_phase.plot(xi, Hp_neg, "-o", label="H_ph-")
    ax_phase.set_title("Histograma de Fase (N=16)")
    ax_phase.set_xlabel("Indice")
    ax_phase.set_ylabel("H_ph (norm)")
    ax_phase.grid(True, alpha=0.2)
    ax_phase.legend(fontsize=8)


def _draw_angpd(ax, ang: dict, *, quantity: bool = False) -> None:
    x = np.asarray(ang.get("phi_centers", []), dtype=float)
    if quantity:
        y_primary = np.asarray(ang.get("n_angpd_qty", []), dtype=float)
        y_secondary = np.asarray(ang.get("angpd_qty", []), dtype=float) * 100.0
        title = "ANGPD / N-ANGPD (quantity)"
        sec_color = "#2ca02c"
    else:
        y_primary = np.asarray(ang.get("n_angpd", []), dtype=float)
        y_secondary = np.asarray(ang.get("angpd", []), dtype=float) * 100.0
        title = "ANGPD / N-ANGPD"
        sec_color = "#1f77b4"
    if not x.size:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    ax.fill_between(x, 0, y_primary, color="#f4c2c2", alpha=0.12)
    ax.plot(x, y_primary, color="#d62728", linewidth=2.0, label="N-ANGPD (max=1)")
    ax.set_xlim(0, 360)
    ax.set_xlabel("Fase (deg)")
    ax.set_ylabel("N-ANGPD")
    ax.set_title(title)
    twin = ax.twinx()
    twin.plot(x, y_secondary, color=sec_color, label="ANGPD (sum=1) x100")
    twin.set_ylabel("ANGPD x100")
    try:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = twin.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    except Exception:
        pass


def _draw_angpd_proj(ax_phase, ax_amp, ang_proj: dict | None) -> None:
    if not ang_proj:
        for ax in (ax_phase, ax_amp):
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        return
    phase_pos = np.asarray(ang_proj.get("phase_pos", []), dtype=float)
    phase_neg = np.asarray(ang_proj.get("phase_neg", []), dtype=float)
    amp_pos = np.asarray(ang_proj.get("amp_pos", []), dtype=float)
    amp_neg = np.asarray(ang_proj.get("amp_neg", []), dtype=float)
    n_points = int(ang_proj.get("n_points") or 0)
    if n_points <= 0:
        n_points = max(phase_pos.size, amp_pos.size, 0)
    phase_x = np.linspace(0.0, 360.0, n_points) if n_points > 0 else np.zeros(0)
    amp_min = float(ang_proj.get("amp_min", 0.0))
    amp_max = float(ang_proj.get("amp_max", 1.0))
    amp_x = np.linspace(amp_min, amp_max, n_points) if n_points > 0 else np.zeros(0)

    ax_phase.plot(phase_x, phase_pos, color="#1f77b4", linewidth=2.0, label="P+")
    ax_phase.plot(phase_x, phase_neg, color="#d62728", linewidth=2.0, label="P-")
    ax_phase.fill_between(phase_x, 0, phase_pos, color="#1f77b4", alpha=0.12)
    ax_phase.fill_between(phase_x, 0, phase_neg, color="#d62728", alpha=0.12)
    ax_phase.set_xlim(0, 360)
    ax_phase.set_title("Proyeccion fase (ANGPD 2.0)")
    ax_phase.set_xlabel("Fase (deg)")
    ax_phase.set_ylabel("Densidad (norm.)")
    ax_phase.grid(True, alpha=0.2)
    ax_phase.legend(loc="upper right", fontsize=8)

    ax_amp.plot(amp_x, amp_pos, color="#2ca02c", linewidth=2.0, label="P+")
    ax_amp.plot(amp_x, amp_neg, color="#ff7f0e", linewidth=2.0, label="P-")
    ax_amp.fill_between(amp_x, 0, amp_pos, color="#2ca02c", alpha=0.12)
    ax_amp.fill_between(amp_x, 0, amp_neg, color="#ff7f0e", alpha=0.12)
    ax_amp.set_title("Proyeccion amplitud (ANGPD 2.0)")
    ax_amp.set_xlabel("Amplitud")
    ax_amp.set_ylabel("Densidad (norm.)")
    ax_amp.grid(True, alpha=0.2)
    ax_amp.legend(loc="upper right", fontsize=8)


def _draw_amp_hist_norm(ax, metrics_adv: dict | None, hist_kpi: dict | None) -> None:
    ax.clear()
    ax.set_facecolor("#f7f9fc")
    metrics_adv = metrics_adv or {}
    hist = metrics_adv.get("hist", {}) if isinstance(metrics_adv, dict) else {}
    amp_pos = np.asarray(hist.get("amp_hist_pos", []), dtype=float)
    amp_neg = np.asarray(hist.get("amp_hist_neg", []), dtype=float)
    edges = np.asarray(hist.get("amp_edges_pos", []), dtype=float)
    centers = np.arange(max(amp_pos.size, amp_neg.size), dtype=float)
    if edges.size >= 2:
        centers = (edges[:-1] + edges[1:]) / 2.0
    if amp_pos.size:
        norm = float(np.max(amp_pos) or 1.0)
        ax.plot(centers, amp_pos / norm, "-o", color="#1f77b4", linewidth=1.6, markersize=3.5, label="Amp+")
        ax.fill_between(centers, 0, amp_pos / norm, color="#1f77b4", alpha=0.08)
    if amp_neg.size:
        norm = float(np.max(amp_neg) or 1.0)
        ax.plot(centers, amp_neg / norm, "-o", color="#d62728", linewidth=1.6, markersize=3.5, label="Amp-")
        ax.fill_between(centers, 0, amp_neg / norm, color="#d62728", alpha=0.08)
    ax.set_title("Amp hist (norm.)", fontweight="bold")
    ax.set_xlabel("Bins amp")
    ax.set_ylabel("Norm.")
    ax.grid(True, alpha=0.2, linestyle="--")
    try:
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white", edgecolor="#d9d9d9")
    except Exception:
        pass

    rows = []
    hist_kpi = hist_kpi or {}
    if isinstance(hist_kpi, dict) and hist_kpi:
        rows = [
            ("Bins amp (+)", hist_kpi.get("hist_amp_active_bins_pos")),
            ("Bins amp (-)", hist_kpi.get("hist_amp_active_bins_neg")),
            ("Bins fase (+)", hist_kpi.get("hist_ph_active_bins_pos")),
            ("Bins fase (-)", hist_kpi.get("hist_ph_active_bins_neg")),
            ("Corr amp +/-", hist_kpi.get("hist_amp_pos_neg_corr")),
            ("Corr fase +/-", hist_kpi.get("hist_ph_pos_neg_corr")),
        ]
    if rows:
        txt = "\n".join([f"{k}: {v:.3f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in rows])
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="#d7dce5", boxstyle="round,pad=0.3"))


def _draw_fa_profile(ax, fa_profile: dict | None) -> None:
    ax.clear()
    if not fa_profile:
        ax.text(0.5, 0.5, "FA profile no disponible", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    phase = np.asarray(fa_profile.get("phase_bins_deg", []), dtype=float)
    max_amp = np.asarray(fa_profile.get("max_amp_smooth", []), dtype=float)
    if not phase.size or not max_amp.size:
        ax.text(0.5, 0.5, "FA profile vacio", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    ax.plot(phase, max_amp, color="#f28b30", linewidth=2.2, label="Envolvente")
    ax.set_xlim(0, 360)
    ax.set_xlabel("Fase (deg)")
    ax.set_ylabel("Amplitud")
    ax.set_title("FA profile")
    ax.grid(True, alpha=0.2)


def export_pdf_report(result: dict, out_root: Path) -> Path:
    out_root = Path(out_root)
    (out_root / "reports").mkdir(parents=True, exist_ok=True)
    run_id = result.get("run_id", "run")
    pdf_path = out_root / "reports" / f"{run_id}_report.pdf"

    raw = result.get("raw", {}) if isinstance(result, dict) else {}
    aligned = result.get("aligned", {}) if isinstance(result, dict) else {}
    ann_block = result.get("ann", {}) if isinstance(result, dict) else {}
    rule = result.get("rule_pd", {}) if isinstance(result, dict) else {}
    kpis = result.get("kpis", {}) if isinstance(result, dict) else {}
    hist_kpi = (result.get("kpi", {}) or {}).get("hist", {}) if isinstance(result, dict) else {}
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    metrics_adv = result.get("metrics_advanced", {}) if isinstance(result, dict) else {}
    gap_raw = result.get("gap_stats_raw") or result.get("gap_stats") or {}
    gap_total = result.get("gap_stats_total") or result.get("gap_stats") or {}
    ang = result.get("angpd", {}) if isinstance(result, dict) else {}
    ang_proj = result.get("ang_proj", {}) if isinstance(result, dict) else {}
    filter_level = result.get("filter_level") if isinstance(result, dict) else None
    fa_profile = result.get("fa_profile") if isinstance(result, dict) else None

    ph_raw = np.asarray(raw.get("phase_deg", []), dtype=float)
    amp_raw = np.asarray(raw.get("amplitude", []), dtype=float)
    ph_al = np.asarray(aligned.get("phase_deg", []), dtype=float)
    amp_al = np.asarray(aligned.get("amplitude", []), dtype=float)
    quint = np.asarray(aligned.get("qty_quintiles", []), dtype=int)
    _, clouds_s3, clouds_s4, clouds_s5 = _build_cluster_variants(ph_al, amp_al, filter_level)

    conclusion_block = result.get("conclusion_block") if isinstance(result, dict) else {}
    if not conclusion_block:
        try:
            conclusion_block = build_conclusion_block(result, rule)
        except Exception:
            conclusion_block = {}

    subtitle = f"Archivo: {Path(result.get('source_path','')).name} | Fecha: {datetime.now():%Y-%m-%d %H:%M}"

    pages = 6
    with PdfPages(pdf_path) as pdf:
        # Pagina 1: resumen y conclusiones
        fig = Figure(figsize=A4, dpi=120)
        _add_header(fig, "PRPD - Informe tecnico", subtitle, 1, pages)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.02, 0.88, "Resumen ejecutivo", fontsize=13, fontweight="bold")
        rows_left = [
            ("Modo dominante", conclusion_block.get("dominant_discharge", rule.get("class_label", "N/D"))),
            ("Etapa", conclusion_block.get("rule_pd_stage", rule.get("stage", "N/D"))),
            ("Riesgo", conclusion_block.get("risk_level", rule.get("risk_level", "N/D"))),
            ("Severidad", conclusion_block.get("severity_level", rule.get("severity_level", "N/D"))),
            ("Ubicacion", conclusion_block.get("location_hint", rule.get("location_hint", "N/D"))),
            ("Vida remanente", conclusion_block.get("lifetime_score_band", "N/D")),
        ]
        rows_right = [
            ("Gap P50 (ms)", gap_total.get("p50_ms", "N/D")),
            ("Gap P5 (ms)", gap_total.get("p5_ms", "N/D")),
            ("FA width (deg)", kpis.get("fa_phase_width_deg", "N/D")),
            ("FA symmetry", kpis.get("fa_symmetry_index", "N/D")),
            ("ANGPD ratio", kpis.get("n_angpd_angpd_ratio", metrics.get("n_ang_ratio", "N/D"))),
            ("P95 amplitud", metrics.get("p95_mean", "N/D")),
        ]
        ax_left = fig.add_axes([0.04, 0.46, 0.45, 0.36])
        ax_right = fig.add_axes([0.52, 0.46, 0.44, 0.36])
        _draw_table(ax_left, rows_left, y_top=0.95, y_step=0.09)
        _draw_table(ax_right, rows_right, y_top=0.95, y_step=0.09)

        actions = conclusion_block.get("actions") or rule.get("actions") or []
        if isinstance(actions, str):
            actions = [a.strip() for a in actions.split(".") if a.strip()]
        ax.text(0.02, 0.38, "Acciones recomendadas", fontsize=12, fontweight="bold")
        y = 0.34
        for idx, act in enumerate(actions[:5], 1):
            ax.text(0.04, y, f"{idx:02d}. {act}", fontsize=10)
            y -= 0.045
        pdf.savefig(fig); fig.clear()

        # Pagina 2: PRPD + ANN
        fig = Figure(figsize=A4, dpi=120)
        _add_header(fig, "PRPD - Visualizacion", subtitle, 2, pages)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.28, wspace=0.25)
        ax_raw = fig.add_subplot(gs[0, 0])
        ax_al = fig.add_subplot(gs[0, 1])
        ax_ann = fig.add_subplot(gs[1, 0])
        ax_kpi = fig.add_subplot(gs[1, 1])
        _draw_prpd_scatter(ax_raw, ph_raw, amp_raw, "PRPD crudo")
        _draw_prpd_scatter(ax_al, ph_al, amp_al, "Alineado / filtrado")
        _draw_ann_bar(ax_ann, ann_block)
        ax_kpi.axis("off")
        kpi_rows = [
            ("Total pulsos", metrics.get("total_count", "N/D")),
            ("Phase width", metrics.get("phase_width", "N/D")),
            ("Phase center", metrics.get("phase_center", "N/D")),
            ("P95 amp", metrics.get("p95_mean", "N/D")),
            ("Pulses ratio", metrics.get("pulses_ratio", "N/D")),
            ("Skew pos/neg", f"{metrics_adv.get('skewness', {}).get('pos_skew', 'N/D')} / {metrics_adv.get('skewness', {}).get('neg_skew', 'N/D')}")
        ]
        _draw_table(ax_kpi, kpi_rows, y_top=0.9, y_step=0.12)
        pdf.savefig(fig); fig.clear()

        # Pagina 3: Nubes (S3/S4/S5)
        fig = Figure(figsize=A4, dpi=120)
        _add_header(fig, "PRPD - Nubes y clustering", subtitle, 3, pages)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.30, wspace=0.26)
        ax_al = fig.add_subplot(gs[0, 0])
        ax_s3 = fig.add_subplot(gs[0, 1])
        ax_s4 = fig.add_subplot(gs[1, 0])
        ax_s5 = fig.add_subplot(gs[1, 1])
        _draw_prpd_quintiles(ax_al, ph_al, amp_al, quint, "Alineado / filtrado")
        _draw_clusters(ax_s3, ph_al, amp_al, clouds_s3, "Nubes crudas (S3)")
        _draw_clusters(ax_s4, ph_al, amp_al, clouds_s4, "Nubes combinadas (S4)")
        _draw_clusters(ax_s5, ph_al, amp_al, clouds_s5, "Nubes dominantes (S5)")
        pdf.savefig(fig); fig.clear()

        # Pagina 4: Gap-time + Histogramas
        fig = Figure(figsize=A4, dpi=120)
        _add_header(fig, "PRPD - Gap-time e Histogramas", subtitle, 4, pages)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.32, wspace=0.26)
        ax_gap = fig.add_subplot(gs[0, :])
        ax_amp = fig.add_subplot(gs[1, 0])
        ax_phase = fig.add_subplot(gs[1, 1])
        _draw_gap_chart(ax_gap, gap_raw)
        _draw_histograms(ax_amp, ax_phase, ph_al, amp_al)
        pdf.savefig(fig); fig.clear()

        # Pagina 5: KPI avanzados (ANGPD 2.0 + hist)
        fig = Figure(figsize=A4, dpi=120)
        _add_header(fig, "PRPD - KPI avanzados", subtitle, 5, pages)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.32, wspace=0.26)
        ax_hist = fig.add_subplot(gs[0, 0])
        ax_adv = fig.add_subplot(gs[0, 1])
        ax_phase = fig.add_subplot(gs[1, 0])
        ax_amp = fig.add_subplot(gs[1, 1])
        _draw_amp_hist_norm(ax_hist, metrics_adv, hist_kpi)
        adv_rows = [
            ("phase_corr", metrics_adv.get("phase_corr", "N/D")),
            ("pulses_ratio", metrics_adv.get("pulses_ratio", "N/D")),
            ("heuristic_top", metrics_adv.get("heuristic_top", "N/D")),
            ("skewness", metrics_adv.get("skewness", "N/D")),
            ("kurtosis", metrics_adv.get("kurtosis", "N/D")),
            ("p95_amp", metrics_adv.get("phase_medians_p95", {}).get("p95_amp", "N/D")),
        ]
        _draw_table(ax_adv, adv_rows, y_top=0.92, y_step=0.12)
        _draw_angpd_proj(ax_phase, ax_amp, ang_proj)
        pdf.savefig(fig); fig.clear()

        # Pagina 6: ANGPD + FA profile
        fig = Figure(figsize=A4, dpi=120)
        _add_header(fig, "PRPD - ANGPD y FA profile", subtitle, 6, pages)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.30, wspace=0.26)
        ax_ang = fig.add_subplot(gs[0, 0])
        ax_ang_q = fig.add_subplot(gs[0, 1])
        ax_fa = fig.add_subplot(gs[1, 0])
        ax_adv = fig.add_subplot(gs[1, 1])
        _draw_angpd(ax_ang, ang, quantity=False)
        _draw_angpd(ax_ang_q, ang, quantity=True)
        _draw_fa_profile(ax_fa, fa_profile)
        ax_adv.axis("off")
        ang_rows = [
            ("phase_width", kpis.get("phase_width_deg", "N/D")),
            ("phase_center", kpis.get("phase_center_deg", "N/D")),
            ("fa_width", kpis.get("fa_phase_width_deg", "N/D")),
            ("fa_sym", kpis.get("fa_symmetry_index", "N/D")),
            ("n_ang_ratio", kpis.get("n_angpd_angpd_ratio", "N/D")),
            ("p95_amp", metrics.get("p95_mean", "N/D")),
        ]
        _draw_table(ax_adv, ang_rows, y_top=0.9, y_step=0.12)
        pdf.savefig(fig); fig.clear()

        meta = pdf.infodict()
        meta["Title"] = f"PRPD Report - {run_id}"
        meta["Author"] = "PRPD-GUI"
        meta["Subject"] = "PRPD analysis"
        meta["Keywords"] = "PRPD, ANN, KPI, gap-time"

    return pdf_path
