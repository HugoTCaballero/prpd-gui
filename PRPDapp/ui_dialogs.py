from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QColorDialog, QCheckBox, QFormLayout, QInputDialog
)


def open_manual_phase_dialog(window, current_offset: int | None) -> int | None:
    """Diálogo para capturar fase manual; devuelve offset (int) o None si se cancela."""
    val, ok = QInputDialog.getDouble(
        window,
        "Fase manual",
        "Offset de fase (°):",
        float(current_offset or 0),
        -360.0,
        720.0,
        1,
    )
    if not ok:
        return None
    return int(round(val)) % 360


def open_manual_override_dialog(window, current: dict) -> dict | None:
    """Formulario para sobrescribir conclusiones manualmente. Devuelve dict o None si cancela."""
    dlg = QDialog(window)
    dlg.setWindowTitle("Conclusiones manuales")
    layout = QVBoxLayout(dlg)

    def color_button(default: str):
        btn = QPushButton("")
        btn.setFixedWidth(28)
        btn.setStyleSheet(f"background-color: {default}")
        btn.color = default

        def pick():
            col = QColorDialog.getColor()
            if col.isValid():
                btn.color = col.name()
                btn.setStyleSheet(f"background-color: {btn.color}")
        btn.clicked.connect(pick)
        return btn

    form = QFormLayout()
    header_risk = QLineEdit(current.get("header_risk", ""))
    header_score = QLineEdit(current.get("header_score", ""))
    header_life = QLineEdit(current.get("header_life", ""))
    btn_header_color = color_button(current.get("header_color", "#00B050"))
    header_box = QHBoxLayout(); header_box.addWidget(header_risk); header_box.addWidget(header_score); header_box.addWidget(header_life); header_box.addWidget(btn_header_color)
    form.addRow("Riesgo / LifeScore / Vida:", header_box)

    action_reco = QPlainTextEdit(current.get("action_reco", ""))
    btn_action_reco = color_button(current.get("action_reco_color", "#1f4e78"))
    box_reco = QHBoxLayout(); box_reco.addWidget(action_reco); box_reco.addWidget(btn_action_reco)
    form.addRow("Acción recomendada:", box_reco)

    action_gap = QPlainTextEdit(current.get("action_gap", ""))
    btn_action_gap = color_button(current.get("action_gap_color", "#ff8c00"))
    box_gap = QHBoxLayout(); box_gap.addWidget(action_gap); box_gap.addWidget(btn_action_gap)
    form.addRow("Acción Gap-time P50:", box_gap)

    stage = QLineEdit(current.get("stage", ""))
    btn_stage = color_button(current.get("stage_color", "#1565c0"))
    mode = QLineEdit(current.get("mode", ""))
    btn_mode = color_button(current.get("mode_color", "#1565c0"))
    location = QLineEdit(current.get("location", ""))
    btn_location = color_button(current.get("location_color", "#1565c0"))
    risk = QLineEdit(current.get("risk", ""))
    btn_risk = color_button(current.get("risk_color", "#1565c0"))
    row_stage = QHBoxLayout(); row_stage.addWidget(stage); row_stage.addWidget(btn_stage)
    row_mode = QHBoxLayout(); row_mode.addWidget(mode); row_mode.addWidget(btn_mode)
    row_loc = QHBoxLayout(); row_loc.addWidget(location); row_loc.addWidget(btn_location)
    row_risk = QHBoxLayout(); row_risk.addWidget(risk); row_risk.addWidget(btn_risk)
    form.addRow("Etapa probable:", row_stage)
    form.addRow("Modo dominante:", row_mode)
    form.addRow("Ubicación probable:", row_loc)
    form.addRow("Riesgo:", row_risk)

    layout.addLayout(form)
    self_chk = QCheckBox("Usar estos valores en Conclusiones")
    self_chk.setChecked(current.get("enabled", False))
    layout.addWidget(self_chk)

    btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    btn_clear = btns.addButton("Limpiar", QDialogButtonBox.ActionRole)
    layout.addWidget(btns)
    btns.accepted.connect(dlg.accept)
    btns.rejected.connect(dlg.reject)

    def do_clear():
        header_risk.clear(); header_score.clear(); header_life.clear()
        action_reco.clear(); action_gap.clear()
        stage.clear(); mode.clear(); location.clear(); risk.clear()
    btn_clear.clicked.connect(do_clear)

    if dlg.exec() == QDialog.Accepted:
        return {
            "enabled": self_chk.isChecked(),
            "header_risk": header_risk.text().strip(),
            "header_score": header_score.text().strip(),
            "header_life": header_life.text().strip(),
            "header_color": btn_header_color.color,
            "action_reco": action_reco.toPlainText().strip(),
            "action_reco_color": btn_action_reco.color,
            "action_gap": action_gap.toPlainText().strip(),
            "action_gap_color": btn_action_gap.color,
            "stage": stage.text().strip(),
            "stage_color": btn_stage.color,
            "mode": mode.text().strip(),
            "mode_color": btn_mode.color,
            "location": location.text().strip(),
            "location_color": btn_location.color,
            "risk": risk.text().strip(),
            "risk_color": btn_risk.color,
        }
    return None

