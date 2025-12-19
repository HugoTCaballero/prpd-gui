from matplotlib.patches import FancyBboxPatch


def draw_status_tag(ax, text: str, x: float, y: float, *, color: str, text_color: str = "#ffffff", size: int = 11):
    """Small colored tag similar to a badge."""
    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=size,
        color=text_color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=color, edgecolor="none"),
    )


def draw_section_title(ax, text: str, *, y: float = 0.95, x: float = 0.02, register=None) -> float:
    """Stylized section header; returns next y coordinate."""
    title = ax.text(
        x,
        y,
        text.upper(),
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="center",
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#e8eefc", edgecolor="none"),
    )
    line = ax.plot([x, 0.98], [y - 0.04, y - 0.04], color="#e1e5eb", linewidth=1.4)[0]
    if register:
        register(title)
        register(line)
    return y - 0.12


def _default_action_lines(classification):
    label = (classification or {}).get("label", "").lower()
    if "3 ms < tiempo" in label:
        return ["Pruebas especializadas", "↑ Planear sustitución", "Monitoreo cada 3 meses"]
    if "gap-time < 3" in label or "< 3" in label:
        return ["Insatisfactorio.", "Planear retiro inmediato o a corto plazo", "Monitoreo semanal o continuo"]
    if "gap-time > 7" in label:
        return ["Continuar operación normal.", "Monitoreo cada 6 meses"]
    if "sin gap" in label:
        return ["Continuar operación normal.", "Monitoreo cada 12 meses"]
    action = (classification or {}).get("action", "")
    if not action:
        return ["Sin información"]
    return [p.strip() for p in action.split(".") if p.strip()]


def _default_interval_text(classification):
    if not classification:
        return "Sin dato"
    label = (classification.get("label") or "").lower()
    if "3 ms < tiempo" in label:
        return "3 ms < Gap-time < 7 ms"
    if "gap-time < 3" in label or "< 3" in label:
        return "< 3 ms"
    if "gap-time > 7" in label:
        return "> 7 ms"
    if "sin gap" in label:
        return "Sin gap time"
    return classification.get("label", "Gap-time")


def draw_gap_summary_panel(ax, metrics: dict, gap_stats: dict | None, *, draw_tag=draw_status_tag, draw_title=draw_section_title, register=None) -> None:
    """Single gap summary panel (used in conclusiones)."""
    ax.clear()
    ax.set_facecolor("#fffdf5")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    card = FancyBboxPatch(
        (0.02, 0.08),
        0.96,
        0.84,
        boxstyle="round,pad=0.02",
        linewidth=1.0,
        facecolor="#ffffff",
        edgecolor="#d9d4c7",
    )
    ax.add_patch(card)

    header_y = draw_title(ax, "Resumen gap-time", y=0.90, register=register)
    y = header_y - 0.05
    rows = [
        ("ACCION GAP-TIME P50", gap_stats.get("classification") if gap_stats else None, _default_action_lines),
        ("TIEMPO ENTRE PULSOS P50", gap_stats.get("classification") if gap_stats else None, lambda c: [_default_interval_text(c)]),
    ]

    for label, classification, builder in rows:
        color = (classification or {}).get("color", "#607d8b")
        ax.text(0.06, y, label, fontsize=12, fontweight="bold", ha="left", va="center", color="#0f172a")
        y -= 0.055
        for line in builder(classification):
            parts = [line]
            if len(line) > 45:
                mid = line[:45].rfind(" ")
                if mid > 20:
                    parts = [line[:mid].strip(), line[mid:].strip()]
            for part in parts:
                draw_tag(ax, part, 0.34, y, color=color, text_color="#ffffff")
                y -= 0.065
        ax.plot([0.06, 0.94], [y + 0.02, y + 0.02], color="#e0e0e0", linewidth=1.0)
        y -= 0.025


def draw_gap_summary_split(ax, metrics: dict, gap_stats: dict | None, side: str, *, draw_tag=draw_status_tag, draw_title=draw_section_title, register=None) -> None:
    """Split gap summary for P50 or P5."""
    ax.clear()
    ax.set_facecolor("#fffdf5")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    card = FancyBboxPatch(
        (0.015, 0.05),
        0.97,
        0.90,
        boxstyle="round,pad=0.02",
        linewidth=1.0,
        facecolor="#ffffff",
        edgecolor="#d7dce5",
    )
    ax.add_patch(card)

    if side == "p5":
        clas_main = gap_stats.get("classification_p5") if isinstance(gap_stats, dict) else None
        title_action = "ACCION GAP-TIME P5"
        title_interval = "TIEMPO ENTRE PULSOS P5"
    else:
        clas_main = gap_stats.get("classification") if isinstance(gap_stats, dict) else None
        title_action = "ACCION GAP-TIME P50"
        title_interval = "TIEMPO ENTRE PULSOS P50"

    import textwrap

    def _wrap_label(label: str) -> list[str]:
        if not label:
            return [""]
        parts = textwrap.wrap(label, width=20, break_long_words=False, break_on_hyphens=False)
        if len(parts) <= 2:
            return parts
        return [parts[0], " ".join(parts[1:])]

    def _wrap_badge(line: str) -> list[str]:
        if not line:
            return [""]
        parts = textwrap.wrap(line, width=34, break_long_words=False, break_on_hyphens=False)
        if len(parts) <= 2:
            return parts
        return [parts[0], " ".join(parts[1:])]

    header_y = draw_title(ax, "Resumen gap-time", y=0.93, register=register)
    y = header_y - 0.05
    label_x = 0.06
    badge_x = 0.46
    rows = [
        (title_action, clas_main, _default_action_lines),
        (title_interval, clas_main, lambda c: [_default_interval_text(c)]),
    ]
    for label, classification, builder in rows:
        color = (classification or {}).get("color", "#607d8b")
        label_lines = _wrap_label(label)
        for line in label_lines:
            ax.text(label_x, y, line, fontsize=10.5, fontweight="bold", ha="left", va="center", color="#0f172a")
            y -= 0.045
        y -= 0.01
        for line in builder(classification):
            for part in _wrap_badge(line):
                draw_tag(ax, part, badge_x, y, color=color, text_color="#ffffff", size=10)
                y -= 0.062
        ax.plot([label_x, 0.94], [y + 0.02, y + 0.02], color="#e3e6ec", linewidth=1.0)
        y -= 0.03
