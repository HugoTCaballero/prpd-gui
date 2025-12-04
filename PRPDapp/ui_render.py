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
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    metrics_adv = payload.get("metrics_advanced", {}) if isinstance(payload, dict) else {}
    if not metrics_adv:
        metrics_adv = result.get("metrics_advanced", {})
    manual = wnd.manual_override if getattr(wnd, "manual_override", {}).get("enabled") else None
    gap_stats = payload.get("gap") if isinstance(payload, dict) else None

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
        "alta": "#B00000",
        "alto": "#B00000",
        "grave": "#FF8C00",
        "moderado": "#1565c0",
        "leve": "#1565c0",
        "bajo": "#00B050",
        "aceptable": "#00B050",
        "incipiente": "#00B050",
        "descargas parciales no detectadas": "#00B050",
        "sin descargas": "#00B050",
        "alerta": "#FF8C00",
        "info": "#0d47a1",
        "mitigado": "#00B050",
    }
    wnd._draw_conclusion_header(ax, "Resultados de los principales KPI", None)

    left_card = FancyBboxPatch((0.02, 0.15), 0.47, 0.74, boxstyle="round,pad=0.02", linewidth=1.0, facecolor="#ffffff", edgecolor="#d9d4c7")
    right_card = FancyBboxPatch((0.51, 0.15), 0.47, 0.74, boxstyle="round,pad=0.02", linewidth=1.0, facecolor="#ffffff", edgecolor="#d9d4c7")
    wnd._register_conclusion_artist(ax.add_patch(left_card))
    wnd._register_conclusion_artist(ax.add_patch(right_card))

    left_ax = ax.inset_axes([0.03, 0.18, 0.44, 0.66])
    right_ax = ax.inset_axes([0.52, 0.18, 0.44, 0.66])
    for sub in (left_ax, right_ax):
        sub.set_axis_off()
        sub.set_xlim(0, 1)
        sub.set_ylim(0, 1)
        sub.set_facecolor("none")
    wnd._conclusion_subaxes.extend([left_ax, right_ax])

    def _fmt_value(value, decimals=1, suffix=""):
        if value is None:
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
    gap_p50 = metrics.get("gap_p50")
    gap_p5 = metrics.get("gap_p5")

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
        ("Skewness (fase)",
         "-",
         _fmt_value(skew_pos, decimals=2),
         _fmt_value(skew_neg, decimals=2),
         None,
         False,
         0.26),
        ("Kurtosis (fase)",
         "-",
         _fmt_value(kurt_pos, decimals=2),
         _fmt_value(kurt_neg, decimals=2),
         None,
         False,
         0.26),
        ("Mediana fase",
         "-",
         _fmt_angle(med_pos),
         _fmt_angle(med_neg),
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
        ("P95 amplitud (qty)",
         _fmt_value(p95_adv),
         "-",
         "-",
         None,
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

    def _draw_triplet_row(ax_target, y_val, label, total_val, pos_val, neg_val, badge, show_badge, badge_x):
        label_fmt = (label[:1].upper() + label[1:]) if label else "N/D"
        wnd._register_conclusion_artist(ax_target.text(0.02, y_val, label_fmt, fontsize=12, fontweight="bold", ha="left", va="center", color="#0f172a"))
        wnd._register_conclusion_artist(ax_target.text(0.38, y_val, total_val, fontsize=12, ha="left", va="center", color="#111827"))
        wnd._register_conclusion_artist(ax_target.text(0.60, y_val, pos_val, fontsize=11.5, ha="left", va="center", color="#0d47a1"))
        wnd._register_conclusion_artist(ax_target.text(0.80, y_val, neg_val, fontsize=11.5, ha="left", va="center", color="#b23c17"))
        if badge and show_badge:
            text_badge, color_badge = badge
            wnd._register_conclusion_artist(wnd._draw_status_tag(ax_target, text_badge, badge_x, y_val, color=color_badge))

    y_left = wnd._draw_section_title(left_ax, "Indicadores clave", y=0.98)
    header_y = y_left + 0.05
    wnd._register_conclusion_artist(left_ax.text(0.38, header_y, "TOTAL", fontsize=10, fontweight="bold", color="#1f2933"))
    wnd._register_conclusion_artist(left_ax.text(0.60, header_y, "SEMICICLO +", fontsize=10, fontweight="bold", color="#0d47a1"))
    wnd._register_conclusion_artist(left_ax.text(0.80, header_y, "SEMICICLO -", fontsize=10, fontweight="bold", color="#b23c17"))
    wnd._register_conclusion_artist(left_ax.plot([0.02, 0.95], [header_y - 0.02, header_y - 0.02], color="#e0e0e0", linewidth=1.0)[0])
    y_left -= 0.025
    for entry in triplets:
        label, total_val, pos_val, neg_val, status = entry[:5]
        show_badge = entry[5] if len(entry) > 5 else False
        badge_x = entry[6] if len(entry) > 6 else 0.26
        _draw_triplet_row(left_ax, y_left, label, total_val, pos_val, neg_val, _status_or_default(status), show_badge, badge_x)
        wnd._register_conclusion_artist(left_ax.plot([0.02, 0.95], [y_left - 0.025, y_left - 0.025], color="#f0f0f0", linewidth=0.8)[0])
        y_left -= 0.07

    y_left -= 0.02
    _draw_triplet_row(left_ax, y_left, "Gap-Time P50", _fmt_value(gap_p50, decimals=2, suffix=" ms"), "-", "-", _status_or_default(_gap_badge_tuple(gap_p50, gap_info)), True, 0.58)
    y_left -= 0.075
    _draw_triplet_row(left_ax, y_left, "Gap-Time P5", _fmt_value(gap_p5, decimals=2, suffix=" ms"), "-", "-", _status_or_default(_gap_badge_tuple(gap_p5, class_p5, default="Sin dato")), True, 0.58)
    y_left -= 0.075

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
    if "superficial" in pd_type_lower or "tracking" in pd_type_lower:
        try:
            rel_val = float(metrics.get("n_ang_ratio"))
        except Exception:
            rel_val = None
        if rel_val is not None:
            if rel_val > 3.0:
                badge_rel = ("Crítico", "#B00000")
            elif rel_val > 1.0:
                badge_rel = ("Grave", "#FF8C00")
            else:
                badge_rel = ("Estable", "#1565c0")
    _draw_triplet_row(left_ax, y_left, "Relacion N-ANGPD/ANGPD", f"{metrics.get('n_ang_ratio', 'N/D')}", "-", "-", badge_rel if badge_rel else ("", "#ffffff"), badge_rel is not None, 0.58)
    y_left -= 0.075

    def _wrap_action_text(text_value: str) -> list[str]:
        parts = []
        for raw in (text_value or "").split("."):
            chunk = raw.strip()
            if not chunk:
                continue
            if len(chunk) > 60:
                mid = chunk[:60].rfind(" ")
                if mid > 20:
                    parts.append(chunk[:mid].strip())
                    parts.append(chunk[mid:].strip())
                else:
                    parts.append(chunk)
            else:
                parts.append(chunk)
        return parts or ["Sin acciones registradas"]

    def _render_action_badges(ax_target, y_start, label, text_value, color):
        ax_target.text(0.02, y_start, label, fontsize=11, fontweight="bold", ha="left", va="center")
        y_pos = y_start - 0.07
        for part in _wrap_action_text(text_value):
            wnd._draw_status_tag(ax_target, part, 0.32, y_pos, color=color, text_color="#ffffff")
            y_pos -= 0.07
        return y_pos + 0.01

    def _draw_right_row(ax_target, y_val, label, value, *, color="#0d47a1", text_color="#ffffff"):
        pretty = value if value and value != "N/D" else "N/D"
        if isinstance(pretty, str):
            pretty = pretty.strip()
            pretty = pretty[0].upper() + pretty[1:] if pretty else "N/D"
        wnd._register_conclusion_artist(ax_target.text(0.02, y_val, label.upper(), fontsize=10, fontweight="bold", ha="left", va="center"))
        wnd._register_conclusion_artist(wnd._draw_status_tag(ax_target, pretty, 0.42, y_val, color=color, text_color=text_color))
        return y_val - 0.08

    header_y = wnd._draw_section_title(right_ax, "Seguimiento y criticidad", y=0.96)
    risk_label = summary.get("risk", "N/D")
    risk_key = risk_label.lower() if isinstance(risk_label, str) else ""
    estado_map = {
        "bajo": ("Aceptable", "#00B050"),
        "moderado": ("Moderado", "#1565c0"),
        "alto": ("Grave", "#FF8C00"),
        "grave": ("Grave", "#FF8C00"),
        "critico": ("Crítico", "#B00000"),
        "crítico": ("Crítico", "#B00000"),
        "incipiente": ("Incipiente", "#00B050"),
        "descargas parciales no detectadas": ("Sin descargas", "#00B050"),
        "sin descargas": ("Sin descargas", "#00B050"),
    }
    estado_general, risk_color = estado_map.get(risk_key, (risk_label if isinstance(risk_label, str) else "N/D", palette.get(risk_key, "#00B050")))
    life_years = summary.get("life_years")
    life_score = summary.get("life_score")
    y_right = header_y - 0.03
    life_txt = f"{life_score:.1f}" if isinstance(life_score, (int, float)) else "N/A"
    vida_txt = "N/A"
    if summary.get("life_interval"):
        vida_txt = summary.get("life_interval")
    elif isinstance(life_years, (int, float)):
        vida_txt = f"{life_years:.1f} años"
    if manual:
        estado_general = manual.get("header_risk") or estado_general
        life_txt = manual.get("header_score") or life_txt
        vida_txt = manual.get("header_life") or vida_txt
        risk_color = manual.get("header_color", risk_color)
    summary_badge = f"{estado_general}   |   LifeScore: {life_txt}   |   Vida remanente: {vida_txt}"
    wnd._draw_status_tag(right_ax, summary_badge, 0.02, y_right, color=risk_color, text_color="#ffffff", size=12)
    y_right -= 0.14

    action_general = manual.get("action_reco") if manual else None
    if not action_general:
        action_general = summary.get("actions", "Sin acciones registradas.")
    action_color = manual.get("action_reco_color", "#0d47a1") if manual else "#0d47a1"
    y_right = _render_action_badges(right_ax, y_right, "ACCIÓN RECOMENDADA", action_general, action_color) - 0.04

    # Indicadores avanzados (skew/kurt/corr/medianas)
    y_right = wnd._draw_section_title(right_ax, "Indicadores avanzados", y=y_right - 0.02)
    adv_rows = [
        ("Skewness ±", f"{_fmt_value(skew_pos, decimals=2)} / {_fmt_value(skew_neg, decimals=2)}"),
        ("Kurtosis ±", f"{_fmt_value(kurt_pos, decimals=2)} / {_fmt_value(kurt_neg, decimals=2)}"),
        ("Correlación fases", _fmt_value(corr_phase, decimals=2)),
        ("Mediana fase ±", f"{_fmt_angle(med_pos)} / {_fmt_angle(med_neg)}"),
    ]
    for label, val in adv_rows:
        y_right = _draw_right_row(right_ax, y_right, label, val, color="#4b5563", text_color="#ffffff")

    if gap_info:
        gap_text = manual.get("action_gap") if manual else None
        gap_color = manual.get("action_gap_color", gap_info.get("color", "#00B050")) if manual else gap_info.get("color", "#00B050")
        if not gap_text:
            gap_text = gap_info.get("action", "")
        y_right = _render_action_badges(right_ax, y_right, "ACCIÓN GAP-TIME P50", gap_text, gap_color) - 0.08

    stage = (manual.get("stage") if manual else None) or summary.get("stage", "N/D")
    pd_type = (manual.get("mode") if manual else None) or summary.get("pd_type", "N/D")
    location = (manual.get("location") if manual else None) or summary.get("location", "N/D")
    risk_manual_text = manual.get("risk") if manual else None

    ann_probs = wnd._last_ann_probs or {}
    if ann_probs:
        try:
            ann_norm = {str(k).lower(): float(v) for k, v in ann_probs.items() if v is not None}
            dom_key = max(ann_norm, key=ann_norm.get)
            if dom_key in CLASS_INFO:
                pd_type = CLASS_INFO[dom_key].get("name", pd_type)
        except Exception:
            pass
    if isinstance(stage, str) and stage.lower().startswith("evoluc"):
        stage = "Avanzada"

    y_right = _draw_right_row(right_ax, y_right, "ETAPA PROBABLE", stage, color=manual.get("stage_color", "#1e88e5") if manual else "#1e88e5")
    y_right = _draw_right_row(right_ax, y_right, "MODO DOMINANTE", pd_type, color=manual.get("mode_color", "#1e88e5") if manual else "#1e88e5")
    y_right = _draw_right_row(right_ax, y_right, "UBICACIÓN PROBABLE", location, color=manual.get("location_color", "#1e88e5") if manual else "#1e88e5")
    _draw_right_row(right_ax, y_right, "RIESGO", risk_manual_text or risk_label, color=manual.get("risk_color", risk_color) if manual else risk_color)


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
