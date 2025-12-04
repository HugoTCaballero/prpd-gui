# report.py
# Exporta un PDF multipágina con resumen, figuras y métricas rápidas (Windows friendly)
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from PRPDapp.config_pd import CLASS_NAMES, CLASS_INFO

def _page_title(fig: Figure, title: str, subtitle: str = ""):
    ax = fig.add_subplot(111)
    ax.axis("off")
    y = 0.75
    ax.text(0.02, y, title, fontsize=20, weight="bold", va="top")
    if subtitle:
        ax.text(0.02, y-0.08, subtitle, fontsize=11, va="top", color="#444444")

def _page_table(fig: Figure, title: str, items: list[tuple[str, str]]):
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.02, 0.95, title, fontsize=14, weight="bold", va="top")
    y = 0.88
    for k, v in items:
        ax.text(0.04, y, f"{k}:", fontsize=10, weight="bold", va="top")
        ax.text(0.26, y, str(v), fontsize=10, va="top")
        y -= 0.06

def _scatter(ax, phase, amp, title):
    ax.scatter(phase, amp, s=4, alpha=0.7)
    ax.set_xlim(0, 360); ax.set_xlabel("Fase (°)")
    ax.set_ylabel("Amplitud")
    ax.set_title(title)

def export_pdf_report(result: dict, out_root: Path) -> Path:
    out_root = Path(out_root)
    (out_root / "reports").mkdir(parents=True, exist_ok=True)
    run_id = result.get("run_id", "run")
    pdf_path = out_root / "reports" / f"{run_id}_report.pdf"

    with PdfPages(pdf_path) as pdf:
        # portada
        fig = Figure(figsize=(8.27, 11.69), dpi=120)  # A4
        subtitle = f"Archivo: {Path(result.get('source_path','')).name}    |    Fecha: {datetime.now():%Y-%m-%d %H:%M}"
        _page_title(fig, "PRPD – Informe rápido", subtitle)
        pdf.savefig(fig); fig.clear()

        # resumen
        items = [
            ("Clase predicha", result["predicted"]),
            ("Severidad (0–100)", f"{result['severity_score']:.1f}"),
            ("Fase (offset)", f"{result['phase_offset']}°"),
            ("Vector de fase R", f"{result['phase_vector_R']:.3f}"),
            ("Ruido detectado", "Sí" if result["has_noise"] else "No"),
            ("Clusters conservados", f"{result['n_clusters']}"),
            ("Puntos crudos", f"{len(result['raw']['phase_deg'])}"),
            ("Puntos filtrados", f"{len(result['aligned']['phase_deg'])}"),
        ]
        fig = Figure(figsize=(8.27, 11.69), dpi=120)
        _page_table(fig, "Resumen del procesamiento", items)
        pdf.savefig(fig); fig.clear()

        # figuras
        fig = Figure(figsize=(8.27, 11.69), dpi=120); gs = fig.add_gridspec(2,2, hspace=0.28, wspace=0.20)
        ax1 = fig.add_subplot(gs[0,0])
        _scatter(ax1, result["raw"]["phase_deg"], result["raw"]["amplitude"], "PRPD crudo")

        ax2 = fig.add_subplot(gs[0,1])
        _scatter(ax2, result["aligned"]["phase_deg"], result["aligned"]["amplitude"],
                 f"Alineado/filtrado (offset={result['phase_offset']}°)")

        ax3 = fig.add_subplot(gs[1,0])
        classes = list(CLASS_NAMES)
        probs = [result["probs"].get(k,0.0) for k in classes]
        colors = [CLASS_INFO.get(k,{}).get("color", "#888888") for k in classes]
        labels = [CLASS_INFO.get(k,{}).get("name", k) for k in classes]
        ax3.bar(labels, probs, color=colors)
        ax3.set_ylim(0,1); ax3.set_title("Probabilidades por clase")
        ax3.set_ylabel("Probabilidad")

        ax4 = fig.add_subplot(gs[1,1])
        ax4.axis("off")
        f = result["features"]
        lines = [
            "Características:",
            f"• p95 |amplitud|   : {f['p95_amp']:.3f}",
            f"• densidad guardada: {f['dens']:.3f}",
            f"• concentración fase R: {f['R_phase']:.3f}",
            f"• balance de polaridad: {f['polarity_balance']:.3f}",
            "",
            "Meta:",
            "Este reporte es un MVP (ANN heurística + severidad compuesta).",
        ]
        ax4.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=10)
        pdf.savefig(fig); fig.clear()

        # metadatos JSON embebidos
        d = pdf.infodict()
        d['Title'] = f"PRPD Report – {run_id}"
        d['Author'] = "PRPD-GUI MVP"
        d['Subject'] = "Clasificación y filtros PRPD"
        d['Keywords'] = "PRPD, clustering, fase, ANN, severidad"
    return pdf_path
