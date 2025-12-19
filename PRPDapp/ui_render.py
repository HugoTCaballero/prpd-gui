from matplotlib.patches import FancyBboxPatch
import numpy as np

from PRPDapp import ui_draw


def render_nubes_grid(wnd, ph_al, amp_al, quint_idx, clouds_s3, clouds_s4, clouds_s5):
    """Panel 2x2: Alineado/filtrado + S3 + S4 + S5."""
    wnd.ax_raw.clear()
    use_hist2d = wnd.chk_hist2d.isChecked() and ph_al.size and amp_al.size
    if use_hist2d:
        wnd.ax_raw.set_facecolor("#0f172a")
        h = wnd.ax_raw.hist2d(
            ph_al,
            amp_al,
            bins=[72, 50],
            range=[[0, 360], [0, 100]],
            cmap="viridis",
            cmin=1,
        )
        wnd._last_hist2d_raw = h
        wnd.ax_raw.set_ylim(0, 100)
    else:
        wnd.ax_raw.set_facecolor("#ffffff")
        if quint_idx is not None and ph_al.size and amp_al.size:
            quint_colors = {
                1: "#0066CC",  # Azul
                2: "#009900",  # Verde
                3: "#FFCC00",  # Amarillo
                4: "#FF8000",  # Naranja
                5: "#CC0000",  # Rojo
            }
            used = False
            for q_idx in range(1, 6):
                mask = quint_idx == q_idx
                if not mask.any():
                    continue
                used = True
                wnd.ax_raw.scatter(
                    ph_al[mask],
                    amp_al[mask],
                    s=6,
                    color=quint_colors.get(q_idx, "#999999"),
                    alpha=0.85,
                    label=f"Q{q_idx}",
                )
            if used:
                try:
                    wnd.ax_raw.legend(loc="upper right", fontsize=8)
                except Exception:
                    pass
        else:
            wnd.ax_raw.scatter(ph_al, amp_al, s=4, alpha=0.7, color="#4a90e2")
    wnd.ax_raw.set_title("Alineado / filtrado")
    wnd.ax_raw.set_xlim(0, 360)
    wnd._apply_auto_ylim(wnd.ax_raw, amp_al)
    wnd.ax_raw.set_xlabel("Fase (°)")
    wnd.ax_raw.set_ylabel("Amplitud")

    # S3, S4, S5
    wnd.ax_filtered.clear()
    wnd.ax_probs.clear()
    wnd.ax_text.clear()
    wnd._plot_clusters_on_ax(
        wnd.ax_filtered,
        ph_al,
        amp_al,
        clouds_s3,
        title="Nubes crudas (S3)",
        color_points=True,
        include_k=True,
        max_labels=10,
    )
    wnd._plot_clusters_on_ax(
        wnd.ax_probs,
        ph_al,
        amp_al,
        clouds_s4,
        title="Nubes combinadas (S4)",
        color_points=True,
        include_k=True,
        max_labels=10,
    )
    wnd._plot_clusters_on_ax(
        wnd.ax_text,
        ph_al,
        amp_al,
        clouds_s5,
        title="Nubes dominantes (S5)",
        color_points=True,
        include_k=True,
        max_labels=10,
    )
    for ax in (wnd.ax_filtered, wnd.ax_probs, wnd.ax_text):
        ax.set_xlim(0, 360)
        wnd._apply_auto_ylim(ax, amp_al)
        ax.set_xlabel("Fase (°)")
        ax.set_ylabel("Amplitud")
    wnd._apply_auto_ylim(wnd.ax_probs, amp_al)


def render_conclusions(wnd, result: dict, payload: dict | None = None) -> None:
    """Render de conclusiones compacto (dos columnas)."""
    if payload is None:
        text, payload = wnd._get_conclusion_insight(result)
        wnd.last_conclusion_text = text
        wnd.last_conclusion_payload = payload
    else:
        text = wnd.last_conclusion_text

    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    if summary is None:
        summary = {}
    rule_pd = result.get("rule_pd", {}) if isinstance(result, dict) else {}
    if rule_pd is None:
        rule_pd = {}
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    metrics_adv = payload.get("metrics_advanced", {}) if isinstance(payload, dict) else {}
    if not metrics_adv:
        metrics_adv = result.get("metrics_advanced", {})
    kpis = result.get("kpis", {}) if isinstance(result, dict) else {}
    fa_kpis = result.get("fa_kpis", {}) if isinstance(result, dict) else {}
    manual = wnd.manual_override if getattr(wnd, "manual_override", {}).get("enabled") else None
    gap_stats = payload.get("gap") if isinstance(payload, dict) else None
    conclusion_block = payload.get("conclusion_block", {}) if isinstance(payload, dict) else {}
    if not conclusion_block and isinstance(result, dict):
        conclusion_block = result.get("conclusion_block", {})

    wnd.ax_gap_wide.set_visible(False)
    for ax_top in (wnd.ax_raw, wnd.ax_filtered):
        ax_top.clear()
        ax_top.set_facecolor("#fafafa")
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.axis("off")

    wnd._clear_conclusion_artists()
    ax = wnd._ensure_conclusion_axis()
    ax.set_visible(True)
    ax.clear()
    try:
        ax.set_position([0.0, 0.0, 1.0, 1.0])
    except Exception:
        pass
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("#ffffff")

    card = FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.02", linewidth=1.2, facecolor="#fffdf5", edgecolor="#c8c6c3")
    wnd._register_conclusion_artist(ax.add_patch(card))

    palette = {
        "critico": "#B00000",
        "crítico": "#B00000",
        "critica": "#B00000",
        "alta": "#B00000",
        "alto": "#B00000",
        "media": "#1565c0",
        "grave": "#FF8C00",
        "moderado": "#1565c0",
        "leve": "#1565c0",
        "bajo": "#00B050",
        "baja": "#00B050",
        "aceptable": "#00B050",
        "incipiente": "#00B050",
        "descargas parciales no detectadas": "#00B050",
        "sin descargas": "#00B050",
        "alerta": "#FF8C00",
        "info": "#0d47a1",
        "mitigado": "#00B050",
    }
    wnd._draw_conclusion_header(ax, "Resultados de los principales KPI", None)

    # Cards más altos: la vista Conclusiones suele tener poco alto útil por los controles/banners.
    cards_y = 0.08
    cards_h = 0.82
    left_card = FancyBboxPatch((0.02, cards_y), 0.47, cards_h, boxstyle="round,pad=0.02", linewidth=1.0, facecolor="#ffffff", edgecolor="#d9d4c7")
    right_card = FancyBboxPatch((0.51, cards_y), 0.47, cards_h, boxstyle="round,pad=0.02", linewidth=1.0, facecolor="#ffffff", edgecolor="#d9d4c7")
    wnd._register_conclusion_artist(ax.add_patch(left_card))
    wnd._register_conclusion_artist(ax.add_patch(right_card))

    left_ax = ax.inset_axes([0.03, cards_y + 0.03, 0.44, cards_h - 0.06])
    right_ax = ax.inset_axes([0.52, cards_y + 0.03, 0.44, cards_h - 0.06])
    for sub in (left_ax, right_ax):
        sub.set_axis_off()
        sub.set_xlim(0, 1)
        sub.set_ylim(0, 1)
        sub.set_facecolor("none")
    wnd._conclusion_subaxes.extend([left_ax, right_ax])

    def _fmt_value(value, decimals=1, suffix=""):
        if value is None:
            return "N/D"
        if isinstance(value, str):
            val = value.strip().lower()
            if val in ("", "nd", "n/d", "n/a", "-", "nan"):
                return "N/D"
        try:
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                return "N/D"
            if isinstance(value, (int, np.integer)):
                return f"{int(value):,}{suffix}"
            return f"{float(value):.{decimals}f}{suffix}"
        except Exception:
            return "N/D"

    def _fmt_angle(value):
        if value is None or value == "N/D":
            return "N/D"
        try:
            return f"{float(value):.1f}°"
        except Exception:
            return f"{value}°"

    def _status_or_default(status):
        if isinstance(status, tuple) and len(status) == 2:
            return status
        if isinstance(status, str):
            return (status, palette.get(status.lower(), "#00B050"))
        return ("", "#ffffff")

    def _gap_badge_tuple(value, classification, *, default="Aceptable"):
        if classification:
            label = classification.get("level_name") or classification.get("label") or "Gap-time"
            color = classification.get("color", "#00B050")
            return (label, color)
        label, color = wnd._status_from_gap(value)
        if label == "Sin gap time":
            return (default, "#00B050")
        return (label, color)

    gap_info = (gap_stats or {}).get("classification") if isinstance(gap_stats, dict) else None
    class_p5 = (gap_stats or {}).get("classification_p5") if isinstance(gap_stats, dict) else None
    gap_p50 = kpis.get("gap_p50_ms", metrics.get("gap_p50"))
    gap_p5 = kpis.get("gap_p5_ms", metrics.get("gap_p5"))

    # Métricas avanzadas (skew/kurt/correlación/medianas, etc.)
    m_adv = metrics_adv or {}
    skew_pos = m_adv.get("skewness", {}).get("pos_skew")
    skew_neg = m_adv.get("skewness", {}).get("neg_skew")
    kurt_pos = m_adv.get("kurtosis", {}).get("pos_kurt")
    kurt_neg = m_adv.get("kurtosis", {}).get("neg_kurt")
    corr_phase = m_adv.get("phase_corr")
    med_pos = m_adv.get("phase_medians_p95", {}).get("median_pos_phase")
    med_neg = m_adv.get("phase_medians_p95", {}).get("median_neg_phase")
    p95_adv = m_adv.get("phase_medians_p95", {}).get("p95_amp")
    pulses_ratio = m_adv.get("pulses_ratio")

    # Lista compacta: Conclusiones debe mostrar "principales KPI" sin saturar la tarjeta.
    triplets = [
        ("Total pulsos utiles",
         _fmt_value(metrics.get("total_count"), decimals=0),
         _fmt_value(metrics.get("count_pos"), decimals=0),
         _fmt_value(metrics.get("count_neg"), decimals=0),
         wnd._status_from_total(metrics.get("total_count")),
         False,
         0.26),
        ("Anchura fase",
         _fmt_angle(metrics.get("phase_width")),
         _fmt_angle(metrics.get("phase_width_pos")),
         _fmt_angle(metrics.get("phase_width_neg")),
         wnd._status_from_width(metrics.get("phase_width")),
         False,
         0.26),
        ("Centro",
         _fmt_angle(metrics.get("phase_center")),
         _fmt_angle(metrics.get("phase_center_pos")),
         _fmt_angle(metrics.get("phase_center_neg")),
         None,
         False,
         0.26),
        ("Numero de picos de fase",
         _fmt_value(metrics.get("n_peaks"), decimals=0),
         _fmt_value(metrics.get("n_peaks_pos"), decimals=0),
         _fmt_value(metrics.get("n_peaks_neg"), decimals=0),
         None,
         False,
         0.26),
        ("P95 amplitud",
         _fmt_value(metrics.get("p95_mean")),
         _fmt_value(metrics.get("amp_p95_pos")),
         _fmt_value(metrics.get("amp_p95_neg")),
         wnd._status_from_amp(metrics.get("p95_mean")),
         False,
         0.26),
        ("Correlacion fases",
         _fmt_value(corr_phase, decimals=2),
         "-",
         "-",
         None,
         False,
         0.26),
        ("Pulsos N+/N-",
         _fmt_value(pulses_ratio, decimals=3),
         "-",
         "-",
         None,
          False,
          0.26),
    ]

    def _draw_card_section(ax_target, title: str, *, y: float = 0.97, x: float = 0.02, size: int = 12) -> float:
        title_art = ax_target.text(
            x,
            y,
            title.upper(),
            fontsize=size,
            fontweight="bold",
            ha="left",
            va="center",
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#e8eefc", edgecolor="none"),
        )
        line = ax_target.plot([x, 0.98], [y - 0.032, y - 0.032], color="#e1e5eb", linewidth=1.2)[0]
        wnd._register_conclusion_artist(title_art)
        wnd._register_conclusion_artist(line)
        return y - 0.085

    def _draw_triplet_row(ax_target, y_val, label, total_val, pos_val, neg_val, badge, show_badge, badge_x):
        label_fmt = (label[:1].upper() + label[1:]) if label else "N/D"
        wnd._register_conclusion_artist(ax_target.text(0.02, y_val, label_fmt, fontsize=10.5, fontweight="bold", ha="left", va="center", color="#0f172a"))
        wnd._register_conclusion_artist(ax_target.text(0.38, y_val, total_val, fontsize=10.5, ha="left", va="center", color="#111827"))
        wnd._register_conclusion_artist(ax_target.text(0.60, y_val, pos_val, fontsize=10.2, ha="left", va="center", color="#0d47a1"))
        wnd._register_conclusion_artist(ax_target.text(0.80, y_val, neg_val, fontsize=10.2, ha="left", va="center", color="#b23c17"))
        if badge and show_badge:
            text_badge, color_badge = badge
            wnd._register_conclusion_artist(wnd._draw_status_tag(ax_target, text_badge, badge_x, y_val, color=color_badge, size=10))

    # -----------------------
    # Left card: tabla KPI
    # -----------------------
    y_left = _draw_card_section(left_ax, "Indicadores clave", y=0.97)
    header_y = y_left + 0.028
    wnd._register_conclusion_artist(left_ax.text(0.38, header_y, "TOTAL", fontsize=9, fontweight="bold", color="#1f2933"))
    wnd._register_conclusion_artist(left_ax.text(0.60, header_y, "SEMICICLO +", fontsize=9, fontweight="bold", color="#0d47a1"))
    wnd._register_conclusion_artist(left_ax.text(0.80, header_y, "SEMICICLO -", fontsize=9, fontweight="bold", color="#b23c17"))
    wnd._register_conclusion_artist(left_ax.plot([0.02, 0.95], [header_y - 0.02, header_y - 0.02], color="#e5e7eb", linewidth=1.0)[0])

    # Relación (usa ANN si está disponible para inferir el tipo)
    badge_rel = None
    pd_type_effective = summary.get("pd_type", "") if isinstance(summary.get("pd_type"), str) else ""
    ann_probs_badge = wnd._last_ann_probs or {}
    if ann_probs_badge:
        try:
            ann_norm = {str(k).lower(): float(v) for k, v in ann_probs_badge.items() if v is not None}
            dom_key = max(ann_norm, key=ann_norm.get)
            if dom_key.startswith("cavidad"):
                pd_type_effective = "Cavidad interna"
            elif dom_key.startswith("super"):
                pd_type_effective = "Superficial / Tracking"
            elif dom_key.startswith("corona"):
                pd_type_effective = "Corona"
        except Exception:
            pass
    pd_type_lower = pd_type_effective.lower()
    rel_val_raw = kpis.get("n_angpd_angpd_ratio") if kpis else None
    if rel_val_raw is None:
        rel_val_raw = metrics.get("n_ang_ratio")
    try:
        rel_val = float(rel_val_raw)
    except Exception:
        rel_val = None
    if "superficial" in pd_type_lower or "tracking" in pd_type_lower:
        if rel_val is not None:
            if rel_val > 3.0:
                badge_rel = ("Crítico", "#B00000")
            elif rel_val > 1.0:
                badge_rel = ("Grave", "#FF8C00")
            else:
                badge_rel = ("Estable", "#1565c0")
    rel_txt = f"{rel_val:.2f}" if rel_val is not None else "N/D"

    # KPIs FA profile (resumen)
    def _fmt_fa(val, decimals=2):
        try:
            if val is None:
                return "N/D"
            if isinstance(val, (float, np.floating)) and np.isnan(val):
                return "N/D"
            if isinstance(val, (int, np.integer)):
                return f"{int(val):,}"
            return f"{float(val):.{decimals}f}"
        except Exception:
            return "N/D"

    fa_rows = [
        ("Simetría semicírculos", _fmt_fa(kpis.get("fa_symmetry_index") or fa_kpis.get("symmetry_index"))),
        ("Centro de fase (°)", _fmt_fa(kpis.get("fa_phase_center_deg") or fa_kpis.get("phase_center_deg"))),
        ("Anchura fase (°)", _fmt_fa(kpis.get("fa_phase_width_deg") or fa_kpis.get("phase_width_deg"))),
        ("Índice concentración FA", _fmt_fa(kpis.get("fa_concentration_index") or fa_kpis.get("ang_amp_concentration_index"))),
        ("P95 amplitud (FA)", _fmt_fa(kpis.get("fa_p95_amplitude") or fa_kpis.get("p95_amplitude"))),
    ]

    # Espaciado adaptativo: evita que la tabla "se salga" del recuadro en pantallas bajas
    y_cursor = y_left - 0.018
    y_min = 0.055
    gap_units = 0.45
    fa_gap_units = 0.55
    total_units = len(triplets) + gap_units + 2 + gap_units + 1 + fa_gap_units + len(fa_rows)
    unit = (y_cursor - y_min) / max(total_units, 1.0)

    for idx, entry in enumerate(triplets):
        label, total_val, pos_val, neg_val, status = entry[:5]
        show_badge = entry[5] if len(entry) > 5 else False
        badge_x = entry[6] if len(entry) > 6 else 0.26
        _draw_triplet_row(left_ax, y_cursor, label, total_val, pos_val, neg_val, _status_or_default(status), show_badge, badge_x)
        if idx < len(triplets) - 1:
            sep_y = y_cursor - unit * 0.48
            wnd._register_conclusion_artist(left_ax.plot([0.02, 0.95], [sep_y, sep_y], color="#f3f4f6", linewidth=0.8)[0])
        y_cursor -= unit

    y_cursor -= unit * gap_units
    _draw_triplet_row(
        left_ax,
        y_cursor,
        "Gap-Time P50",
        _fmt_value(gap_p50, decimals=2, suffix=" ms"),
        "-",
        "-",
        _status_or_default(_gap_badge_tuple(gap_p50, gap_info)),
        True,
        0.58,
    )
    y_cursor -= unit
    _draw_triplet_row(
        left_ax,
        y_cursor,
        "Gap-Time P5",
        _fmt_value(gap_p5, decimals=2, suffix=" ms"),
        "-",
        "-",
        _status_or_default(_gap_badge_tuple(gap_p5, class_p5, default="Sin dato")),
        True,
        0.58,
    )
    y_cursor -= unit

    y_cursor -= unit * gap_units
    _draw_triplet_row(
        left_ax,
        y_cursor,
        "Relación N-ANGPD/ANGPD",
        rel_txt,
        "-",
        "-",
        badge_rel if badge_rel else ("", "#ffffff"),
        badge_rel is not None,
        0.58,
    )
    y_cursor -= unit

    y_cursor -= unit * fa_gap_units
    for label, val in fa_rows:
        _draw_triplet_row(left_ax, y_cursor, label, val, "-", "-", ("", "#ffffff"), False, 0.26)
        y_cursor -= unit

    # -----------------------
    # Right card: resumen / criticidad
    # -----------------------
    def _coerce_str(value) -> str:
        if value is None:
            return "N/D"
        if isinstance(value, str):
            txt = value.strip()
            return txt if txt else "N/D"
        return str(value)

    def _titlecase_first(value: str) -> str:
        txt = _coerce_str(value)
        if txt == "N/D":
            return txt
        return txt[:1].upper() + txt[1:]

    def _short_line(value: str, *, max_chars: int) -> str:
        txt = " ".join(_coerce_str(value).split())
        if txt == "N/D":
            return txt
        if len(txt) <= max_chars:
            return txt
        trimmed = txt[: max(0, max_chars - 1)].rstrip(" .,;:")
        return f"{trimmed}…"

    def _wrap_badge(value: str, *, max_chars: int, max_lines: int = 2) -> str:
        import textwrap

        txt = " ".join(_coerce_str(value).split())
        if txt == "N/D":
            return txt
        lines = textwrap.wrap(txt, width=max_chars, break_long_words=False, break_on_hyphens=False)
        if not lines:
            return "N/D"
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = _short_line(lines[-1], max_chars=max_chars)
        return "\n".join(lines)

    def _split_lines(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            items = []
            for item in value:
                s = _coerce_str(item)
                if s != "N/D":
                    items.append(s)
            return items
        if isinstance(value, str):
            raw = value.replace("\r", "\n")
            parts = []
            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts.extend([p.strip() for p in line.split(".") if p.strip()])
            return parts
        return []

    def _draw_right_row(ax_target, y_val, label, value, *, color, badge_x=0.44):
        badge_txt = _wrap_badge(value, max_chars=32, max_lines=2)
        wnd._register_conclusion_artist(ax_target.text(0.02, y_val, label.upper(), fontsize=9, fontweight="bold", ha="left", va="center", color="#111827"))
        wnd._register_conclusion_artist(wnd._draw_status_tag(ax_target, _titlecase_first(badge_txt), badge_x, y_val, color=color, text_color="#ffffff", size=10))
        n_lines = badge_txt.count("\n") + 1
        return y_val - (0.070 + 0.032 * (n_lines - 1))

    y_right = _draw_card_section(right_ax, "Seguimiento y criticidad", y=0.97)

    # Prioridad: bloque de conclusiones > rule_pd > summary
    dom_pd = conclusion_block.get("dominant_discharge") or rule_pd.get("class_label") or rule_pd.get("dominant_pd") or summary.get("pd_type") or "N/D"
    location = conclusion_block.get("location_hint") or rule_pd.get("location_hint", "N/D")
    stage = conclusion_block.get("rule_pd_stage") or rule_pd.get("stage", "N/D")
    sev_level = rule_pd.get("severity_level") or summary.get("risk") or "N/D"
    sev_idx = rule_pd.get("severity_index")
    risk_label = conclusion_block.get("risk_level") or rule_pd.get("risk_level", summary.get("risk", "N/D"))
    lifetime_score = conclusion_block.get("lifetime_score") or rule_pd.get("lifetime_score")
    lifetime_band = conclusion_block.get("lifetime_score_band") or rule_pd.get("lifetime_band")
    lifetime_text = conclusion_block.get("lifetime_score_text") or rule_pd.get("lifetime_text")

    # Overrides manuales (si están activos)
    if manual:
        dom_pd = manual.get("mode") or dom_pd
        location = manual.get("location") or location
        stage = manual.get("stage") or stage
        risk_label = manual.get("risk") or risk_label

    if isinstance(stage, str) and stage.lower().startswith("evoluc"):
        stage = "Avanzada"

    # Badge resumen (sin categoría, como pidió el usuario)
    risk_key = risk_label.lower() if isinstance(risk_label, str) else ""
    estado_map = {
        "bajo": ("Aceptable", "#00B050"),
        "baja": ("Baja", "#00B050"),
        "media": ("Media", "#1565c0"),
        "moderado": ("Moderado", "#1565c0"),
        "alta": ("Alta", "#FF8C00"),
        "alto": ("Grave", "#FF8C00"),
        "grave": ("Grave", "#FF8C00"),
        "critico": ("Crítico", "#B00000"),
        "critica": ("Crítico", "#B00000"),
        "incipiente": ("Incipiente", "#00B050"),
        "descargas parciales no detectadas": ("Sin descargas", "#00B050"),
        "sin descargas": ("Sin descargas", "#00B050"),
    }
    estado_general, risk_color = estado_map.get(
        risk_key,
        (risk_label if isinstance(risk_label, str) else "N/D", palette.get(risk_key, "#00B050")),
    )
    life_txt = f"{lifetime_score:.1f}" if isinstance(lifetime_score, (int, float)) else "N/D"
    vida_txt = lifetime_band or "N/D"
    if manual:
        estado_general = manual.get("header_risk") or estado_general
        life_txt = manual.get("header_score") or life_txt
        vida_txt = manual.get("header_life") or vida_txt
        risk_color = manual.get("header_color", risk_color)
    summary_badge = f"{estado_general}  |  LifeScore: {life_txt}  |  Vida remanente: {vida_txt}"
    summary_badge = _wrap_badge(summary_badge, max_chars=48, max_lines=2)

    y_cursor = y_right - 0.02
    wnd._register_conclusion_artist(wnd._draw_status_tag(right_ax, summary_badge, 0.02, y_cursor, color=risk_color, text_color="#ffffff", size=10))
    y_cursor -= 0.11 if "\n" not in summary_badge else 0.15

    # Filas con categoría + badge (alineadas)
    y_cursor = _draw_right_row(right_ax, y_cursor, "Modo dominante", dom_pd, color=(manual.get("mode_color", "#1e88e5") if manual else "#1e88e5"))
    y_cursor = _draw_right_row(right_ax, y_cursor, "Ubicación probable", location, color=(manual.get("location_color", "#1e88e5") if manual else "#1e88e5"))
    y_cursor = _draw_right_row(right_ax, y_cursor, "Etapa", stage, color=(manual.get("stage_color", "#1e88e5") if manual else "#1e88e5"))
    sev_text = f"{sev_level} (Índice {sev_idx:.1f}/10)" if sev_idx is not None else sev_level
    y_cursor = _draw_right_row(right_ax, y_cursor, "Severidad", sev_text, color="#1e88e5")

    # LifeTime (badge + texto corto)
    if lifetime_score is not None:
        lt_line = f"LifeTime score: {lifetime_score}/100"
        if lifetime_band:
            lt_line += f" ({lifetime_band})"
        lt_line = _wrap_badge(lt_line, max_chars=44, max_lines=2)
        wnd._register_conclusion_artist(wnd._draw_status_tag(right_ax, lt_line, 0.02, y_cursor, color="#0d47a1", text_color="#ffffff", size=10))
        y_cursor -= 0.08 if "\n" not in lt_line else 0.12
        if lifetime_text:
            wnd._register_conclusion_artist(right_ax.text(0.02, y_cursor, _short_line(lifetime_text, max_chars=72), fontsize=9, ha="left", va="top", color="#111827", wrap=True))
            y_cursor -= 0.06

    # Acciones recomendadas + resumen de reglas (compacto)
    actions_block = conclusion_block.get("actions")
    actions_list = _split_lines(manual.get("action_reco") if manual else None) or _split_lines(actions_block)
    if not actions_list:
        actions_list = _split_lines(summary.get("actions")) or _split_lines(rule_pd.get("actions"))
    gap_action = _split_lines(manual.get("action_gap") if manual else None) or _split_lines((gap_info or {}).get("action") if isinstance(gap_info, dict) else None)
    if gap_action:
        gap_first = _short_line(gap_action[0], max_chars=60)
        actions_list = actions_list + [f"Gap-time P50: {gap_first}"]

    actions_list = [_short_line(a, max_chars=64) for a in actions_list if a][:4]
    if actions_list and y_cursor > 0.22:
        y_cursor = _draw_card_section(right_ax, "Acciones recomendadas", y=y_cursor, size=11)
        for act in actions_list:
            wnd._register_conclusion_artist(right_ax.text(0.03, y_cursor, f"• {act}", fontsize=9, ha="left", va="top", color="#111827", wrap=True))
            y_cursor -= 0.055
        y_cursor -= 0.02

    explanation_lines = _split_lines(rule_pd.get("explanation"))
    explanation_lines = [_short_line(e, max_chars=68) for e in explanation_lines if e][:3]
    if explanation_lines and y_cursor > 0.14:
        y_cursor = _draw_card_section(right_ax, "Resumen de reglas", y=y_cursor, size=11)
        for line in explanation_lines:
            wnd._register_conclusion_artist(right_ax.text(0.03, y_cursor, f"- {line}", fontsize=8.8, ha="left", va="top", color="#111827", wrap=True))
            y_cursor -= 0.05



def render_ann_gap_view(wnd, result: dict, payload: dict | None = None) -> None:
    """Vista combinada ANN + Gap-time (cuatro paneles)."""
    if payload is None:
        _, payload = wnd._get_conclusion_insight(result)
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    gap_stats = payload.get("gap") if isinstance(payload, dict) else None
    if not gap_stats:
        gap_stats = result.get("gap_stats")
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    metrics_adv = payload.get("metrics_advanced", {}) if isinstance(payload, dict) else {}
    if not metrics_adv:
        metrics_adv = result.get("metrics_advanced", {})

    wnd.ax_gap_wide.set_visible(False)
    wnd.ax_raw.set_visible(True)
    wnd.ax_raw.set_axis_on()
    wnd._draw_ann_prediction_panel(result, summary, metrics, metrics_adv)

    wnd.ax_filtered.set_visible(True)
    wnd.ax_filtered.set_axis_on()
    wnd._draw_gap_chart(wnd.ax_filtered, gap_stats)

    wnd.ax_probs.clear()
    wnd.ax_probs.set_facecolor("#fffdf5")
    wnd.ax_probs.set_xticks([])
    wnd.ax_probs.set_yticks([])
    wnd.ax_probs.axis("off")
    wnd._draw_gap_summary_split(wnd.ax_probs, metrics, gap_stats, side="p50")

    wnd.ax_text.clear()
    wnd.ax_text.set_facecolor("#fffdf5")
    wnd.ax_text.set_xticks([])
    wnd.ax_text.set_yticks([])
    wnd.ax_text.axis("off")
    wnd._draw_gap_summary_split(wnd.ax_text, metrics, gap_stats, side="p5")


def render_gap_time_full(wnd, result: dict, payload: dict | None = None) -> None:
    """Vista de Gap-time con PRPD crudo y alineado abajo."""
    if payload is None:
        _, payload = wnd._get_conclusion_insight(result)
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    gap_stats = payload.get("gap") if isinstance(payload, dict) else None
    if not gap_stats:
        gap_stats = result.get("gap_stats")

    wnd.ax_raw.set_visible(False)
    wnd.ax_filtered.set_visible(False)
    wnd.ax_gap_wide.set_visible(True)
    wnd._draw_gap_chart(wnd.ax_gap_wide, gap_stats)

    wnd.ax_probs.set_visible(True)
    wnd.ax_probs.clear()
    wnd.ax_probs.set_facecolor("#fafafa")
    raw = result.get("raw", {}) or {}
    ph_raw = np.asarray(raw.get("phase_deg", []), dtype=float)
    amp_raw = np.asarray(raw.get("amplitude", []), dtype=float)
    if ph_raw.size and amp_raw.size:
        if wnd.chk_hist2d.isChecked():
            try:
                H0, xe0, ye0 = np.histogram2d(ph_raw, amp_raw, bins=[72, 50], range=[[0, 360], [0, 100]])
                wnd.ax_probs.imshow(H0.T + 1e-9, origin="lower", aspect="auto", extent=[xe0[0], xe0[-1], ye0[0], ye0[-1]])
            except Exception:
                pass
        wnd.ax_probs.scatter(ph_raw, amp_raw, s=3, alpha=0.4, color="#1f77b4")
    else:
        wnd.ax_probs.text(0.5, 0.5, "Sin PRPD crudo", ha="center", va="center")
    wnd.ax_probs.set_xlim(0, 360)
    wnd._apply_auto_ylim(wnd.ax_probs, amp_raw)
    wnd.ax_probs.set_xlabel("Fase (°)")
    wnd.ax_probs.set_ylabel("Amplitud")
    wnd.ax_probs.set_title("PRPD crudo")

    wnd.ax_text.set_visible(True)
    wnd.ax_text.clear()
    wnd.ax_text.set_facecolor("#fafafa")
    aligned = result.get("aligned", {}) or {}
    ph_al = np.asarray(aligned.get("phase_deg", []), dtype=float)
    amp_al = np.asarray(aligned.get("amplitude", []), dtype=float)
    quint_idx = wnd._resolve_quintiles(ph_al, result, aligned)
    quint_colors = {1: "#0066CC", 2: "#009900", 3: "#FFCC00", 4: "#FF8000", 5: "#CC0000"}
    if ph_al.size and amp_al.size:
        colored = False
        if wnd.chk_hist2d.isChecked():
            try:
                H2, xe2, ye2 = np.histogram2d(ph_al, amp_al, bins=[72, 50], range=[[0, 360], [0, 100]])
                wnd.ax_text.imshow(H2.T + 1e-9, origin="lower", aspect="auto", extent=[xe2[0], xe2[-1], ye2[0], ye2[-1]])
            except Exception:
                pass
        if quint_idx is not None and quint_idx.size == ph_al.size:
            for q_idx in range(1, 6):
                mask = quint_idx == q_idx
                if not np.any(mask):
                    continue
                colored = True
                wnd.ax_text.scatter(
                    ph_al[mask],
                    amp_al[mask],
                    s=5,
                    color=quint_colors.get(q_idx, "#999999"),
                    alpha=0.8,
                    label=f"Q{q_idx}",
                )
            if colored:
                try:
                    wnd.ax_text.legend(loc="lower right", fontsize=8, framealpha=0.7)
                except Exception:
                    pass
        if not colored:
            wnd.ax_text.scatter(ph_al, amp_al, s=3, alpha=0.6, color="#1565c0")
    else:
        wnd.ax_text.text(0.5, 0.5, "Sin PRPD alineado", ha="center", va="center")
    wnd.ax_text.set_xlim(0, 360)
    wnd._apply_auto_ylim(wnd.ax_text, amp_al)
    wnd.ax_text.set_xlabel("Fase (°)")
    wnd.ax_text.set_ylabel("Amplitud")
    wnd.ax_text.set_title(f"Alineado/filtrado (offset={result.get('phase_offset', 0)}°)")
