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
from PRPDapp.conclusion_rules import build_conclusion_block

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

        # resumen (usa rule_pd si está disponible)
        rule = result.get("rule_pd", {}) if isinstance(result, dict) else {}
        try:
            conclusion_block = result.get("conclusion_block", {}) if isinstance(result, dict) else {}
        except Exception:
            conclusion_block = {}
        if not conclusion_block:
            try:
                conclusion_block = build_conclusion_block(result if isinstance(result, dict) else {}, rule)
            except Exception:
                conclusion_block = {}
        items = [
            ("Modo dominante", rule.get("class_label") or result.get("predicted", "N/D")),
            ("Etapa", rule.get("stage", "N/D")),
            ("Riesgo", rule.get("risk_level", "N/D")),
            ("Ubicación probable", rule.get("location_hint", "N/D")),
            ("Ruleset", rule.get("ruleset_version", "N/D")),
            ("Severidad (0–100)", f"{result.get('severity_score', 0):.1f}"),
            ("Fase (offset)", f"{result.get('phase_offset', 0)}°"),
            ("Vector de fase R", f"{result.get('phase_vector_R', 0):.3f}"),
            ("Ruido detectado", "Sí" if result.get("has_noise") else "No"),
            ("Clusters conservados", f"{result.get('n_clusters', 0)}"),
            ("Puntos crudos", f"{len(result.get('raw', {}).get('phase_deg', []))}"),
            ("Puntos filtrados", f"{len(result.get('aligned', {}).get('phase_deg', []))}"),
        ]
        fig = Figure(figsize=(8.27, 11.69), dpi=120)
        _page_table(fig, "Resumen del procesamiento", items)
        pdf.savefig(fig); fig.clear()

        # KPIs consolidados
        kpis = result.get("kpis", {}) if isinstance(result, dict) else {}
        kpi_items = [
            ("FA: anchura fase (°)", kpis.get("fa_phase_width_deg", "N/D")),
            ("FA: simetría", kpis.get("fa_symmetry_index", "N/D")),
            ("FA: conc. amplitud", kpis.get("fa_concentration_index", "N/D")),
            ("FA: P95 amplitud", kpis.get("fa_p95_amplitude", "N/D")),
            ("N-ANGPD/ANGPD", kpis.get("n_angpd_angpd_ratio", "N/D")),
            ("Gap-time P50 (ms)", kpis.get("gap_p50_ms", "N/D")),
            ("Gap-time P5 (ms)", kpis.get("gap_p5_ms", "N/D")),
            ("ANGPD2: picos fase", kpis.get("ang_phase_peaks", "N/D")),
            ("ANGPD2: conc. amplitud", kpis.get("ang_amp_concentration", "N/D")),
        ]
        fig = Figure(figsize=(8.27, 11.69), dpi=120)
        _page_table(fig, "KPIs consolidados", kpi_items)
        pdf.savefig(fig); fig.clear()

        # Conclusiones estandarizadas
        if conclusion_block:
            fig = Figure(figsize=(8.27, 11.69), dpi=120)
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.02, 0.94, "Conclusiones (aceite sumergido)", fontsize=16, weight="bold", va="top")
            y = 0.86
            conc_items = [
                ("Modo dominante", conclusion_block.get("dominant_discharge", "N/D")),
                ("Riesgo", conclusion_block.get("risk_level", "N/D")),
                ("Etapa (regla)", conclusion_block.get("rule_pd_stage", "N/D")),
                ("Vida (banda)", conclusion_block.get("lifetime_score_band", "N/D")),
                ("Vida (texto)", conclusion_block.get("lifetime_score_text", "N/D")),
                ("Ubicación", conclusion_block.get("location_hint", "N/D")),
                ("FA", conclusion_block.get("fa_value", "N/D")),
                ("Evolución", conclusion_block.get("evolution_stage", "N/D")),
            ]
            for k, v in conc_items:
                ax.text(0.04, y, f"{k}:", fontsize=11, weight="bold", va="top")
                ax.text(0.42, y, str(v), fontsize=11, va="top")
                y -= 0.06
            actions = conclusion_block.get("actions")
            action_list = []
            if isinstance(actions, str):
                action_list = [a.strip() for a in actions.split(".") if a.strip()]
            elif isinstance(actions, list):
                action_list = [str(a).strip() for a in actions if a]
            if action_list:
                ax.text(0.02, y - 0.02, "Acciones sugeridas:", fontsize=12, weight="bold", va="top")
                y -= 0.10
                for act in action_list[:5]:
                    ax.text(0.04, y, f"• {act}", fontsize=10, va="top")
                    y -= 0.055
            pdf.savefig(fig); fig.clear()

        # figuras
        fig = Figure(figsize=(8.27, 11.69), dpi=120); gs = fig.add_gridspec(2,2, hspace=0.28, wspace=0.20)
        ax1 = fig.add_subplot(gs[0,0])
        _scatter(ax1, result["raw"]["phase_deg"], result["raw"]["amplitude"], "PRPD crudo")

        ax2 = fig.add_subplot(gs[0,1])
        _scatter(ax2, result["aligned"]["phase_deg"], result["aligned"]["amplitude"],
                 f"Alineado/filtrado (offset={result['phase_offset']}°)")

        ax3 = fig.add_subplot(gs[1,0])
        rule_probs = rule.get("class_probs", {}) if isinstance(rule, dict) else {}
        classes = list(rule_probs.keys()) or list(CLASS_NAMES)
        probs = [rule_probs.get(k, 0.0) for k in classes] if rule_probs else [result.get("probs", {}).get(k,0.0) for k in classes]
        colors = [CLASS_INFO.get(k,{}).get("color", "#888888") for k in classes]
        labels = [CLASS_INFO.get(k,{}).get("name", k) for k in classes]
        ax3.bar(labels, probs, color=colors)
        ax3.set_ylim(0,1); ax3.set_title("Probabilidades por clase")
        ax3.set_ylabel("Probabilidad")

        ax4 = fig.add_subplot(gs[1,1])
        ax4.axis("off")
        rule_lines = [
            "Clasificador por reglas",
            f"Modo dominante: {rule.get('class_label', 'N/D')}",
            f"Etapa: {rule.get('stage', 'N/D')}",
            f"Riesgo: {rule.get('risk_level', 'N/D')}",
            f"Ubicación: {rule.get('location_hint', 'N/D')}",
            f"Ruleset: {rule.get('ruleset_version', 'N/D')}",
        ]
        ax4.text(0.02, 0.98, "\n".join(rule_lines), va="top", fontsize=10)
        pdf.savefig(fig); fig.clear()

        # metadatos JSON embebidos
        d = pdf.infodict()
        d['Title'] = f"PRPD Report – {run_id}"
        d['Author'] = "PRPD-GUI"
        d['Subject'] = "Clasificación y filtros PRPD"
        d['Keywords'] = "PRPD, clustering, fase, ANN, severidad"
    return pdf_path
