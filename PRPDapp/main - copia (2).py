#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

 

import sys, os, json, traceback, re, subprocess
from pathlib import Path
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QCheckBox,
    QPlainTextEdit, QLineEdit, QDialog, QDialogButtonBox, QTextBrowser,
    QSizePolicy, QFrame, QMenu
)
from PySide6.QtGui import QPixmap, QFont, QAction
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.patches import FancyBboxPatch

# Imports locales (soportar ejecución como script o módulo)
import pathlib as _pl
_THIS = _pl.Path(__file__).resolve(); _ROOT = _THIS.parents[1]; _PKG = _THIS.parent
for _p in (str(_ROOT), str(_PKG)):
    (_p in sys.path) or sys.path.insert(0, _p)

import prpd_core as core
from report import export_pdf_report
from datetime import datetime
from utils_io import ensure_out_dirs, time_tag
from prpd_ann import PRPDANN
from PRPDapp.clouds import pixel_cluster_clouds, combine_clouds, select_dominant_clouds



# ANN loader opcional (models/ann_loader)
try:
    from models.ann_loader import (
        load_ann_model as _load_ann_model,
        predict_proba as _ann_predict_proba,
    )
except Exception:
    _load_ann_model = None
    _ann_predict_proba = None


APP_TITLE = "PRPD GUI — Unificada (exports v2)"


class PRPDWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 740)

        # Estado
        self.current_path: Path | None = None
        self.last_result: dict | None = None
        self.last_run_profile: dict | None = None
        self.out_root: Path | None = None
        self.auto_phase = True
        self.pixel_deciles_enabled: set[int] = set(range(1, 11))
        self.qty_deciles_enabled: set[int] = set(range(1, 11))
        self.last_qty_deciles = None
        self.last_conclusion_text: str = ""
        self.last_conclusion_payload: dict = {}
        self._ann_history_written: bool = False
        self._ann_history_recent: list[dict] = []

        # ANN (fallback PRPDANN si no hay loader)
        self.ann = PRPDANN(class_names=["cavidad","superficial","corona","flotante","ruido"])
        self.ann_model = None
        self.ann_classes: list[str] = []
        # Auto-cargar models/ann.pkl si existe
        try:
            if _load_ann_model is not None:
                self.ann_model, self.ann_classes = _load_ann_model(None)
        except Exception:
            pass

        # UI
        central = QWidget(self); self.setCentralWidget(central)
        v = QVBoxLayout(central)

        # Barra superior
        top = QHBoxLayout()
        self.btn_open = QPushButton("Abrir PRPD…")
        self.btn_run  = QPushButton("Procesar")
        self.btn_pdf  = QPushButton("Exportar PDF")
        self.btn_load_ann = QPushButton("Cargar ANN")
        self.btn_batch = QPushButton("Procesar carpeta")
        self.btn_help = QPushButton("Ayuda/README")
        self.btn_reset_base = QPushButton("Reset baseline")
        self.btn_pdf.setEnabled(True)
        # Seguimiento y criticidad (gap-time opcional)
        self.chk_gap = QCheckBox("Gap-time XML")
        self.btn_gap_pick = QPushButton("...")
        self.btn_compare = QPushButton("Comparar vs base")

        self.cmb_phase = QComboBox(); self.cmb_phase.addItems(["Auto (0/120/240)", "0°", "120°", "240°"]); self.cmb_phase.setCurrentIndex(0)
        self.cmb_filter = QComboBox(); self.cmb_filter.addItems(["S1 Weak","S2 Strong"]); self.cmb_filter.setCurrentIndex(0)
        self.cmb_masks = QComboBox()
        self.cmb_masks.addItems(["Ninguna","Corona +","Corona -","Superficial","Void","Manual"])
        self.cmb_masks.setCurrentIndex(0)
        self.cmb_masks.currentTextChanged.connect(self._on_mask_mode_changed)
        self.mask_manual_label = QLabel("Intervalos (°):")
        self.mask_manual_label.setVisible(False)
        self.mask_interval_1 = QLineEdit()
        self.mask_interval_1.setPlaceholderText("Int1 (ej: 45-135)")
        self.mask_interval_1.setMaximumWidth(120)
        self.mask_interval_1.setVisible(False)
        self.mask_interval_2 = QLineEdit()
        self.mask_interval_2.setPlaceholderText("Int2 (ej: 225-315)")
        self.mask_interval_2.setMaximumWidth(120)
        self.mask_interval_2.setVisible(False)
        self._mask_manual_widgets = [self.mask_manual_label, self.mask_interval_1, self.mask_interval_2]
        self._on_mask_mode_changed(self.cmb_masks.currentText())
        self.btn_pixel = QPushButton("Pixel: D1-D10")
        self.btn_pixel.setToolTip("Selecciona los deciles (D1-D10) que deseas conservar.")
        self.btn_pixel.clicked.connect(self._open_pixel_dialog)
        self._update_pixel_button_text()
        self.btn_qty = QPushButton("Qty: DQ1-DQ10")
        self.btn_qty.setToolTip("Selecciona los deciles (DQ1-DQ10) de quantity que deseas conservar.")
        self.btn_qty.clicked.connect(self._open_qty_dialog)
        self._update_qty_button_text()
        self.chk_hist2d = QCheckBox("Densidad (hist2D)"); self.chk_hist2d.setChecked(True)

        top.addWidget(self.btn_open)
        top.addWidget(self.btn_run)
        top.addWidget(QLabel("Fase:")); top.addWidget(self.cmb_phase)
        top.addWidget(QLabel("Filtro:")); top.addWidget(self.cmb_filter)
        top.addWidget(QLabel("Máscara:")); top.addWidget(self.cmb_masks)
        for w in self._mask_manual_widgets:
            top.addWidget(w)
        top.addWidget(QLabel("Pixel:")); top.addWidget(self.btn_pixel)
        top.addWidget(QLabel("Qty:")); top.addWidget(self.btn_qty)
        top.addStretch(1)
        top.addWidget(self.chk_hist2d)
        top.addWidget(self.btn_load_ann)
        top.addWidget(self.btn_batch)
        top.addWidget(self.btn_pdf)
        top.addWidget(self.chk_gap)
        top.addWidget(self.btn_gap_pick)
        top.addWidget(self.btn_compare)
        top.addWidget(self.btn_reset_base)
        top.addWidget(self.btn_help)
        v.addLayout(top)

        # Figuras
        fig = Figure(figsize=(10,6), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvas(fig)
        self._gs_main = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        self.ax_raw      = fig.add_subplot(self._gs_main[0, 0])
        self.ax_filtered = fig.add_subplot(self._gs_main[0, 1])
        self.ax_probs    = fig.add_subplot(self._gs_main[1, 0])
        self.ax_text     = fig.add_subplot(self._gs_main[1, 1])
        self.ax_gap_wide = fig.add_subplot(self._gs_main[0, :])
        self.ax_gap_wide.set_visible(False)
        self.ax_gap_wide.set_facecolor("#f5f6fb")
        for a in [self.ax_raw, self.ax_filtered, self.ax_probs, self.ax_text]:
            a.set_facecolor("#fafafa")
        v.addWidget(self.canvas)
        self.ax_raw_twin = None
        self.ax_probs_twin = None
        self.ax_conclusion_box = fig.add_subplot(self._gs_main[1, :])
        self.ax_conclusion_box.set_visible(False)
        self._conclusion_artists: list = []
        self._conclusion_subaxes: list = []

        # Estado panel inferior izquierdo
        # Cambiar la opción "ANGPD" por "Histogramas" para reflejar que esta vista
        # muestra histogramas de amplitud/fase y curvas ANGPD/N‑ANGPD.
        self.cmb_plot = QComboBox();
        self.cmb_plot.addItems(["Conclusiones", "ANN / Gap-time", "Gap-time", "Histogramas", "Combinada", "Nubes (S3)", "Nubes (S4)", "Nubes (S5)"])
        self.cmb_plot.setCurrentIndex(0)
        sub = QHBoxLayout(); sub.addWidget(QLabel("Vista:")); sub.addWidget(self.cmb_plot); sub.addStretch(1)
        # Botón de exportación total
        self.btn_export_all = QPushButton("Exportar todo los resultados")
        sub.addWidget(self.btn_export_all)
        v.addLayout(sub)

        # Área resumen/batch se usa como banner inferior dinámico.
        batch_bar = QHBoxLayout()
        self.banner_max_height = 150
        self.signature_label = QLabel()
        self.signature_label.setAlignment(Qt.AlignCenter)
        # Permitir que el banner crezca/encojja con la ventana pero sin exceder banner_max_height.
        self.signature_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.signature_label.setMinimumHeight(int(self.banner_max_height * 0.5))
        self.signature_label.setMaximumHeight(self.banner_max_height)
        self.signature_label.setScaledContents(False)
        self._signature_pixmap: QPixmap | None = None
        self._refresh_signature_banner()
        batch_bar.addWidget(self.signature_label)
        v.addLayout(batch_bar)

        # Eventos
        self.btn_open.clicked.connect(self.open_file_dialog)
        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_pdf.clicked.connect(self.export_pdf_clicked)
        self.btn_export_all.clicked.connect(self.on_export_all_clicked)
        self.btn_load_ann.clicked.connect(self.on_btnLoadANN_clicked)
        self.btn_batch.clicked.connect(self.on_btnBatch_clicked)
        self.btn_help.clicked.connect(self.on_open_readme)
        self.btn_reset_base.clicked.connect(self.on_reset_baseline)
        self.btn_gap_pick.clicked.connect(self.on_pick_gap_xml)
        self.btn_compare.clicked.connect(self.on_compare_vs_base)

        # When the user changes the phase selection in the combo, update the auto_phase flag
        # and refresh the display if there is a loaded result. This ensures phase gating
        # follows the combo box rather than always remaining in auto mode.
        self.cmb_phase.currentIndexChanged.connect(self._on_phase_changed)


    def _on_phase_changed(self, idx: int) -> None:
        """Actualizar auto_phase en función del índice de la fase seleccionada.
        Fase "Auto" (índice 0) habilita auto_phase; cualquier otra desactiva auto_phase.
        Redibuja el resultado si hay uno cargado.
        """
        # Auto (0/120/240) => auto_phase True; valores 0°, 120°, 240° => auto_phase False
        self.auto_phase = (idx == 0)
        # Si hay un resultado procesado, volver a renderizar con la nueva fase seleccionada
        try:
            if self.last_result:
                self.render_result(self.last_result)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Helper methods for phase offsets and filter labels
    #
    # Para garantizar coherencia en el uso de filtros y fases a lo largo de la
    # aplicación, estas funciones devuelven los valores correspondientes a
    # partir de los widgets de la interfaz. Usar estas funciones en lugar de
    # acceder directamente a cmb_phase/cmb_filter en otras partes del código.
    def _get_force_offsets(self) -> list[int] | None:
        """Devuelve la lista de offsets de fase seleccionados, o None si es modo auto.

        La fase se selecciona en el combobox cmb_phase. El índice 0 corresponde a
        modo automático (0/120/240), mientras que índices 1–3 corresponden a
        offsets fijos 0°, 120° y 240° respectivamente.
        """
        try:
            idx = self.cmb_phase.currentIndex()
        except Exception:
            return None
        # Índice 0 => auto_phase (usar todos los offsets cand)
        if idx == 0:
            return None
        # Convertir índice a valor de fase
        try:
            phase_val = [0, 0, 120, 240][idx]
            return [int(phase_val)]
        except Exception:
            return None

    def _get_filter_label(self) -> str:
        """Devuelve el texto del filtro seleccionado (S1 Weak, S2 Strong, etc.)."""
        try:
            return self.cmb_filter.currentText().strip()
        except Exception:
            return "S1 Weak"

    def _on_mask_mode_changed(self, _: str) -> None:
        """Activa la edición de intervalos solo cuando se elige el modo Manual."""
        try:
            current = self.cmb_masks.currentText().strip().lower()
        except Exception:
            current = ""
        manual = current.startswith("manual")
        for w in getattr(self, "_mask_manual_widgets", []):
            try:
                w.setVisible(manual)
                w.setEnabled(manual)
            except Exception:
                continue

    def _get_phase_mask_ranges(self) -> list[tuple[float, float]]:
        """Devuelve la lista de intervalos de fase que se deben conservar."""
        try:
            mode = self.cmb_masks.currentText().strip().lower()
        except Exception:
            mode = "ninguna"
        clean = mode.replace(" ", "")
        presets: dict[str, list[tuple[float, float]]] = {
            "ninguna": [],
            "corona+": [(45.0, 135.0)],
            "corona-": [(225.0, 315.0)],
            "superficial": [(45.0, 135.0), (225.0, 315.0)],
            "void": [(0.0, 45.0), (135.0, 225.0), (315.0, 360.0)],
        }
        if clean in presets:
            return [tuple(pair) for pair in presets[clean]]
        if clean.startswith("manual"):
            intervals: list[tuple[float, float]] = []
            for edit in (getattr(self, "mask_interval_1", None), getattr(self, "mask_interval_2", None)):
                rng = self._parse_manual_interval(edit.text() if edit is not None else "")
                if rng:
                    intervals.append(rng)
            return intervals
        return []

    def _parse_manual_interval(self, text: str) -> tuple[float, float] | None:
        """Parsea un texto tipo '45-135' devolviendo el par normalizado en 0..360."""
        if not text:
            return None
        norm = text.replace(",", " ").replace(";", " ")
        # Tratar guiones (clásico y variantes unicode) como separadores entre números.
        norm = re.sub(r"(?<=\d)\s*[-–—]\s*(?=\d)", " ", norm)
        nums = re.findall(r"-?\d+(?:\.\d+)?", norm)
        if len(nums) < 2:
            return None
        try:
            start = float(nums[0]) % 360.0
            end = float(nums[1]) % 360.0
        except Exception:
            return None
        if abs(start - end) < 1e-6:
            return None
        return (start, end)

    def _open_pixel_dialog(self) -> None:
        """Permite seleccionar qué deciles de pixel se conservan."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Filtro por deciles de pixel")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Activa los deciles (D1 = menor magnitud, D10 = mayor) que deseas conservar."))
        rows = [QHBoxLayout(), QHBoxLayout()]
        checks: list[tuple[int, QCheckBox]] = []
        for idx, dec in enumerate(range(1, 11)):
            cb = QCheckBox(f"D{dec}")
            cb.setChecked(dec in self.pixel_deciles_enabled)
            rows[0 if idx < 5 else 1].addWidget(cb)
            checks.append((dec, cb))
        for row in rows:
            layout.addLayout(row)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_all = btn_box.addButton("Todos", QDialogButtonBox.ActionRole)
        btn_none = btn_box.addButton("Vaciar", QDialogButtonBox.ActionRole)

        def _set_all(state: bool) -> None:
            for _, cb in checks:
                cb.setChecked(state)

        btn_all.clicked.connect(lambda: _set_all(True))
        btn_none.clicked.connect(lambda: _set_all(False))
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)
        if dlg.exec() == QDialog.Accepted:
            selected = {dec for dec, cb in checks if cb.isChecked()}
            self.pixel_deciles_enabled = selected
            self._update_pixel_button_text()

    def _update_pixel_button_text(self) -> None:
        """Actualiza el texto del botón Pixel con un resumen de deciles."""
        if not hasattr(self, "btn_pixel"):
            return
        sels = sorted(self.pixel_deciles_enabled)
        if not sels:
            text = "Pixel: (ninguno)"
        elif len(sels) == 10:
            text = "Pixel: D1-D10"
        else:
            parts: list[str] = []
            start = prev = sels[0]
            for v in sels[1:]:
                if v == prev + 1:
                    prev = v
                    continue
                parts.append(self._format_decile_range(start, prev))
                start = prev = v
            parts.append(self._format_decile_range(start, prev))
            text = f"Pixel: {','.join(parts)}"
        self.btn_pixel.setText(text)

    @staticmethod
    def _format_decile_range(a: int, b: int) -> str:
        if a == b:
            return f"D{a}"
        return f"D{a}-D{b}"

    def _get_pixel_deciles_selection(self) -> list[int]:
        """Devuelve la lista ordenada de deciles a conservar (1..10)."""
        return sorted(self.pixel_deciles_enabled)

    def _clear_twin_axis(self, attr: str) -> None:
        twin = getattr(self, attr, None)
        if twin is not None:
            try:
                twin.remove()
            except Exception:
                pass
            setattr(self, attr, None)

    def _prepare_twin_axis(self, base_ax, attr: str):
        self._clear_twin_axis(attr)
        twin = base_ax.twinx()
        setattr(self, attr, twin)
        return twin

    def _ensure_conclusion_axis(self):
        """Devuelve el eje que ocupa todo el rectángulo inferior."""
        if self.ax_conclusion_box is not None:
            self.ax_conclusion_box.set_facecolor("#fffdf5")
            return self.ax_conclusion_box
        # Fallback de emergencia: usar un eje que ocupe toda la fila inferior
        fig = self.canvas.figure
        gs = fig.add_gridspec(2, 1)
        self.ax_conclusion_box = fig.add_subplot(gs[1, 0])
        self.ax_conclusion_box.set_facecolor("#fffdf5")
        return self.ax_conclusion_box

    def _set_conclusion_mode(self, enable: bool) -> None:
        """Oculta/muestra los ejes inferiores y el rectángulo de conclusiones."""
        if enable:
            self.ax_probs.set_visible(False)
            self.ax_text.set_visible(False)
            ax = self._ensure_conclusion_axis()
            ax.set_visible(True)
        else:
            self.ax_probs.set_visible(True)
            self.ax_text.set_visible(True)
            if self.ax_conclusion_box is not None:
                self.ax_conclusion_box.set_visible(False)
                self._clear_conclusion_artists()

    @staticmethod
    def _draw_status_tag(ax, text: str, x: float, y: float, *, color: str, text_color: str = "#ffffff", size: int = 11) -> None:
        """Etiqueta coloreada al estilo dashboard."""
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
            bbox=dict(boxstyle="round,pad=0.35", facecolor=color, edgecolor="none"),
        )

    def _draw_conclusion_header(self, ax, title: str, subtitle: str | None = None) -> None:
        """Encabezado principal de la vista Conclusiones."""
        title_art = ax.text(0.03, 0.96, title.upper(), fontsize=18, fontweight="bold", ha="left", va="center", color="#0f172a")
        self._register_conclusion_artist(title_art)
        if subtitle:
            sub = ax.text(0.03, 0.91, subtitle, fontsize=11, fontweight="bold", ha="left", va="center", color="#5f6c7b")
            self._register_conclusion_artist(sub)
        line = ax.plot([0.03, 0.97], [0.89, 0.89], color="#d0d7de", linewidth=1.6)
        self._register_conclusion_artist(line[0])

    def _append_ann_history(self, result: dict) -> None:
        if self._ann_history_written:
            return
        stem = self.current_path.stem if self.current_path else None
        if not stem:
            return
        probs = result.get("probs") or {}
        if not probs:
            return
        norm = {str(k).lower(): float(v) for k, v in probs.items() if v is not None}
        if not norm:
            return
        record = {
            "timestamp": time_tag(),
            "file": str(self.current_path),
            "filter": self._get_filter_label(),
            "probs": norm,
        }
        history_dir = Path("seguimiento") / "ann_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        hist_file = history_dir / f"{stem}_ann_history.json"
        try:
            data = json.loads(hist_file.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
        data.append(record)
        data = data[-120:]
        hist_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._ann_history_recent = data
        self._ann_history_written = True

    def _load_ann_history(self, stem: str | None) -> list[dict]:
        if not stem:
            self._ann_history_recent = []
            return []
        hist_file = Path("seguimiento") / "ann_history" / f"{stem}_ann_history.json"
        try:
            data = json.loads(hist_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._ann_history_recent = data[-120:]
            else:
                self._ann_history_recent = []
        except Exception:
            self._ann_history_recent = []
        return self._ann_history_recent

    def _draw_section_title(self, ax, text: str, *, y: float = 0.95, x: float = 0.02) -> float:
        """Dibuja un título estilizado dentro de un subpanel y devuelve la siguiente coordenada Y."""
        title = ax.text(x, y, text.upper(), fontsize=13, fontweight="bold", ha="left", va="center", color="#1d3557")
        self._register_conclusion_artist(title)
        line = ax.plot([x, 0.98], [y - 0.04, y - 0.04], color="#e1e5eb", linewidth=1.4)
        self._register_conclusion_artist(line[0])
        return y - 0.12

    def _register_conclusion_artist(self, artist):
        if artist is not None:
            self._conclusion_artists.append(artist)
        return artist

    def _clear_conclusion_artists(self):
        for artist in getattr(self, "_conclusion_artists", []):
            try:
                artist.remove()
            except Exception:
                pass
        self._conclusion_artists = []
        for sub_ax in getattr(self, "_conclusion_subaxes", []):
            try:
                sub_ax.remove()
            except Exception:
                pass
        self._conclusion_subaxes = []

    def _open_qty_dialog(self) -> None:
        """Selector de deciles para quantity."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Filtro por deciles de quantity")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Activa los deciles DQ1..DQ10 (Por repetición) que deseas conservar."))
        rows = [QHBoxLayout(), QHBoxLayout()]
        checks: list[tuple[int, QCheckBox]] = []
        for idx, dec in enumerate(range(1, 11)):
            cb = QCheckBox(f"DQ{dec}")
            cb.setChecked(dec in self.qty_deciles_enabled)
            rows[0 if idx < 5 else 1].addWidget(cb)
            checks.append((dec, cb))
        for row in rows:
            layout.addLayout(row)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_all = btn_box.addButton("Todos", QDialogButtonBox.ActionRole)
        btn_none = btn_box.addButton("Vaciar", QDialogButtonBox.ActionRole)

        def _set_all(state: bool) -> None:
            for _, cb in checks:
                cb.setChecked(state)

        btn_all.clicked.connect(lambda: _set_all(True))
        btn_none.clicked.connect(lambda: _set_all(False))
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)
        if dlg.exec() == QDialog.Accepted:
            selected = {dec for dec, cb in checks if cb.isChecked()}
            self.qty_deciles_enabled = selected
            self._update_qty_button_text()

    def _update_qty_button_text(self) -> None:
        if not hasattr(self, "btn_qty"):
            return
        sels = sorted(self.qty_deciles_enabled)
        if not sels:
            text = "Qty: (ninguno)"
        elif len(sels) == 10:
            text = "Qty: DQ1-DQ10"
        else:
            parts: list[str] = []
            start = prev = sels[0]
            for v in sels[1:]:
                if v == prev + 1:
                    prev = v
                    continue
                parts.append(self._format_decile_range(start, prev).replace("D", "DQ"))
                start = prev = v
            parts.append(self._format_decile_range(start, prev).replace("D", "DQ"))
            text = f"Qty: {','.join(parts)}"
        self.btn_qty.setText(text)

    def _get_qty_deciles_selection(self) -> list[int]:
        return sorted(self.qty_deciles_enabled)

    def _collect_current_profile(self) -> dict:
        """Recopila los parámetros actuales para comparaciones/export."""
        try:
            phase = self.cmb_phase.currentText().strip()
        except Exception:
            phase = "Auto (0/120/240)"
        filt = self._get_filter_label()
        try:
            mask = self.cmb_masks.currentText().strip()
        except Exception:
            mask = "Ninguna"
        manual_vals = (
            self.mask_interval_1.text().strip() if hasattr(self, "mask_interval_1") else "",
            self.mask_interval_2.text().strip() if hasattr(self, "mask_interval_2") else "",
        )
        return {
            "phase": phase,
            "filter": filt,
            "mask": mask,
            "mask_manual": manual_vals,
            "pixel": tuple(sorted(self.pixel_deciles_enabled)),
            "qty": tuple(sorted(self.qty_deciles_enabled)),
        }

    @staticmethod
    def _format_decile_token(values: tuple[int, ...]) -> str:
        if not values:
            return "0"
        vals = sorted(int(v) for v in values)
        if not vals:
            return "0"
        contiguous = vals == list(range(vals[0], vals[-1] + 1))
        if contiguous:
            return f"{vals[0]}{vals[-1]}"
        return "".join(str(v) for v in vals)

    def _build_profile_tag(self, profile: dict | None = None) -> str:
        profile = profile or self.last_run_profile or self._collect_current_profile()
        phase = profile.get("phase", "").lower()
        filt = profile.get("filter", "").lower()
        mask = profile.get("mask", "").lower()
        pixel_vals = profile.get("pixel", tuple())
        qty_vals = profile.get("qty", tuple())
        phase_tag = "F?"
        if "auto" in phase:
            phase_tag = "FA"
        elif "120" in phase:
            phase_tag = "F120"
        elif "240" in phase:
            phase_tag = "F240"
        elif "0" in phase:
            phase_tag = "F0"
        filter_tag = "F?"
        if "s1" in filt or "weak" in filt:
            filter_tag = "FS1"
        elif "s2" in filt or "strong" in filt:
            filter_tag = "FS2"
        mask_tag = "MN"
        if "manual" in mask:
            mask_tag = "MM"
        elif "corona" in mask and "+" in mask:
            mask_tag = "MCP"
        elif "corona" in mask and "-" in mask:
            mask_tag = "MCN"
        elif "super" in mask:
            mask_tag = "MS"
        elif "void" in mask:
            mask_tag = "MV"
        pixel_tag = f"PD{self._format_decile_token(pixel_vals)}"
        qty_tag = f"QDQ{self._format_decile_token(qty_vals)}"
        return f"{phase_tag}{filter_tag}{mask_tag}{pixel_tag}{qty_tag}"

    def _create_session_dir(self, base: Path, tag: str) -> Path:
        base = Path(base)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = base / f"{tag}_{ts}"
        counter = 1
        final = candidate
        while final.exists():
            counter += 1
            final = base / f"{tag}_{ts}_{counter}"
        final.mkdir(parents=True, exist_ok=True)
        return final

    @staticmethod
    def _resolve_export_dir(outdir: Path, subfolder: str | None) -> Path:
        target = Path(outdir)
        if subfolder:
            target = target / subfolder
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _open_folder(self, path: Path | str) -> None:
        try:
            target = Path(path)
            if sys.platform.startswith("win"):
                os.startfile(str(target))
            elif sys.platform.startswith("darwin"):
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception:
            pass

    def _apply_qty_decile_filter(self, result: dict, keep_deciles: list[int]) -> None:
        aligned = result.get("aligned", {})
        qty_dec = np.asarray(aligned.get("qty_deciles", []), dtype=int)
        phase = np.asarray(aligned.get("phase_deg", []), dtype=float)
        if qty_dec.size == 0 or qty_dec.size != phase.size:
            return
        keep_deciles = keep_deciles or list(range(1, 11))
        keep_set = {int(k) for k in keep_deciles if 1 <= int(k) <= 10}
        keep_mask = np.isin(qty_dec, list(keep_set)) if keep_set else np.zeros(qty_dec.shape, dtype=bool)
        for key in ("phase_deg", "amplitude", "quantity", "qty_quintiles", "pixel", "labels_aligned"):
            arr = aligned.get(key, None)
            if arr is None:
                continue
            aligned[key] = np.asarray(arr)[keep_mask]
        aligned["qty_deciles"] = qty_dec[keep_mask]

    @staticmethod
    def _equal_frequency_bucket(values: np.ndarray, groups: int = 5) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        idx_all = np.zeros(arr.shape, dtype=int)
        if groups <= 0:
            return idx_all
        finite = np.isfinite(arr)
        if not finite.any():
            return idx_all
        vals = arr[finite]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(vals.size)
        grp = (ranks * groups) // max(vals.size, 1)
        grp = np.clip(grp, 0, groups - 1)
        idx = (grp + 1).astype(int)
        idx_all[finite] = idx
        return idx_all

    def _cluster_profiles(self) -> dict[str, dict]:
        return {
            "S1 Weak": {"eps": 0.055, "min_samples": 8, "force_multi": False},
            "S2 Strong": {"eps": 0.030, "min_samples": 14, "force_multi": True},
        }

    def _build_cluster_variants(self, ph: np.ndarray, amp: np.ndarray) -> tuple[dict[str, list[dict]], list[dict], list[dict], list[dict]]:
        variants: dict[str, list[dict]] = {}
        if not (ph.size and amp.size):
            return variants, [], [], []
        total_points = float(ph.size)
        for name, params in self._cluster_profiles().items():
            clouds = pixel_cluster_clouds(ph, amp, **params)
            for idx, c in enumerate(clouds):
                c = dict(c)
                c["source"] = name
                c["legend"] = f"{name.split()[0]}-{idx + 1}"
                c["frac"] = c.get("frac", c.get("count", 0) / max(1.0, total_points))
                clouds[idx] = c
            variants[name] = clouds
        combined = [dict(c) for arr in variants.values() for c in arr]
        if not combined:
            return variants, [], [], []
        clouds_s4 = combine_clouds(combined)
        base_for_s5 = clouds_s4 if clouds_s4 else combined
        clouds_s5 = select_dominant_clouds(base_for_s5)
        return variants, combined, clouds_s4, clouds_s5

    # -------------------------------------------------------------------------
    # Métodos para trazar nubes/clusters en un eje (ax) de matplotlib.
    #
    # Se utiliza una función estática para poder reutilizar la misma lógica tanto
    # en la interfaz gráfica como en los exportes de PNG. Esta función dibuja
    # puntos y centros de nubes, coloreando los centros con una paleta
    # reutilizable. Los clusters con índice superior a max_labels no aparecen en
    # la leyenda (sólo se muestran los primeros max_labels). Si el número total
    # de clusters excede max_labels, se añade un elemento "…" al final de la
    # leyenda para indicar que hay más.
    @staticmethod
    def _plot_clusters_on_ax(ax, ph: np.ndarray, amp: np.ndarray, clouds: list[dict], *,
                             title: str, color_points: bool = False, include_k: bool = False,
                             max_labels: int = 10) -> None:
        """Dibuja las nubes/clusters en el eje dado.

        Parámetros:
        - ax: eje de matplotlib donde dibujar.
        - ph, amp: arrays de fase y amplitud (mismos tamaños).
        - clouds: lista de dicts con claves 'phase_mean' y 'y_mean'.
        - title: título del gráfico. Si include_k es True, se añade "— k=n".
        - color_points: si True, colorea cada punto según su cluster; si False,
          dibuja todos los puntos en gris claro y sólo colorea los centros.
        - max_labels: número máximo de etiquetas de clusters que aparecerán en la leyenda.
        """
        # Definir paleta de colores ampliada (hasta 20 colores distintos) para evitar
        # repetición excesiva. Estos colores se inspiran en la paleta 'tab20'.
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
            "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
            "#dbdb8d", "#9edae5",
        ]
        n_clusters = len(clouds) if clouds else 0
        if include_k and n_clusters:
            try:
                title = f"{title} — k={n_clusters}"
            except Exception:
                pass
        # Dibujar puntos base
        try:
            if ph.size and amp.size:
                if color_points and n_clusters > 0:
                    # Asignar cada punto a un cluster usando la misma heurística que en la GUI
                    import numpy as _np
                    centers = _np.array([[c.get("phase_mean", 0.0), c.get("y_mean", 0.0)] for c in clouds], dtype=float)
                    lbl = _np.zeros(ph.shape[0], dtype=int)
                    for i in range(ph.shape[0]):
                        # Distancia circular en fase (considerando simetría semiciclo)
                        dp = _np.minimum.reduce([
                            _np.abs(ph[i] - centers[:, 0]),
                            _np.abs(ph[i] - centers[:, 0] - 180),
                            _np.abs(ph[i] - centers[:, 0] + 180),
                        ])
                        dy = _np.abs(amp[i] - centers[:, 1])
                        j = int(_np.argmin(0.6 * dp + 0.4 * dy))
                        lbl[i] = j
                    # Colorear puntos según su cluster
                    for j in range(n_clusters):
                        m = (lbl == j)
                        if np.any(m):
                            color = palette[j % len(palette)]
                            label = c.get("legend") or (f"C{j + 1}" if j < max_labels else None)
                            ax.scatter(ph[m], amp[m], s=6, alpha=0.65, color=color, label=label)
                else:
                    # Todos los puntos en gris claro
                    ax.scatter(ph, amp, s=3, alpha=0.4, color="#bfbfbf")
        except Exception:
            pass
        # Dibujar centros de clusters con colores
        try:
            for j, c in enumerate(clouds):
                px = float(c.get("phase_mean", 0.0))
                py = float(c.get("y_mean", 0.0))
                color = palette[j % len(palette)]
                label = c.get("legend") or (f"C{j + 1}" if j < max_labels else None)
                ax.scatter([px], [py], s=70, color=color, edgecolors="black", label=label)
        except Exception:
            pass
        # Configurar leyenda: sólo mostrar hasta max_labels y añadir '…' si hay más
        try:
            # Recoger handles y labels únicos (excluyendo None)
            handles, labels = ax.get_legend_handles_labels()
            # Filtrar etiquetas vacías
            handles_labels = [(h, l) for h, l in zip(handles, labels) if l]
            # Limitar el número de labels
            if len(handles_labels) > max_labels:
                handles_labels = handles_labels[:max_labels]
                # Añadir un marcador simple para indicar que hay más clusters
                from matplotlib.lines import Line2D
                extra = Line2D([], [], marker='o', color='white', markerfacecolor='#bfbfbf', markersize=6, linestyle='None')
                handles_labels.append((extra, '…'))
            if handles_labels:
                ax.legend([h for h, _ in handles_labels], [l for _, l in handles_labels], loc='upper right', fontsize=8)
        except Exception:
            pass
        # Configurar títulos y ejes
        ax.set_title(title)
        ax.set_xlim(0, 360)
        ax.set_xlabel("Fase (°)")
        ax.set_ylabel("Amplitud")
        ax.set_ylim(0, 100)


    def _get_output_dir(self, force_prompt: bool = False) -> Path:
        """Ensure that an output directory exists and return it.
        Prompts the user to select a directory the first time (or when force_prompt=True);
        caches it in self.out_root."""
        try:
            # If out_root not yet defined or empty, prompt the user once
            needs_prompt = force_prompt or (not hasattr(self, "out_root") or self.out_root is None)
            if needs_prompt:
                directory = QFileDialog.getExistingDirectory(self, "Selecciona carpeta de salida", str(Path.cwd()))
                if directory:
                    self.out_root = Path(directory)
                else:
                    self.out_root = Path("out")
            return ensure_out_dirs(Path(self.out_root))
        except Exception:
            # fallback to local "out" directory
            try:
                return ensure_out_dirs(Path("out"))
            except Exception:
                return Path("out")

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        try:
            self._update_signature_pixmap()
        except Exception:
            pass

    def _refresh_signature_banner(self) -> None:
        """Carga/actualiza el banner inferior si existe firma_banner.png."""
        sig_path = Path("firma_banner.png")
        try:
            if sig_path.exists():
                pix = QPixmap(str(sig_path))
                if pix.isNull():
                    raise ValueError("Imagen inválida")
                self._signature_pixmap = pix
                self.signature_label.setText("")
                self._update_signature_pixmap()
            else:
                self._signature_pixmap = None
                self.signature_label.setText("\nFirma/Autoría no encontrada.\nColoque 'firma_banner.png' en la carpeta del programa.")
        except Exception:
            self._signature_pixmap = None
            self.signature_label.setText("\nFirma/Autoría no encontrada.\nColoque 'firma_banner.png' en la carpeta del programa.")
        self.signature_label.setAlignment(Qt.AlignCenter)

    def _update_signature_pixmap(self) -> None:
        """Escala el banner sin exceder la altura máxima manteniento relación de aspecto."""
        pix = getattr(self, "_signature_pixmap", None)
        if pix is None or not hasattr(self, "signature_label"):
            return
        # Usar el ancho real del QLabel (o del central widget) para auto-ajustar el banner.
        label_w = max(self.signature_label.width(), 0)
        if label_w <= 2:
            central = self.centralWidget()
            if central is not None:
                label_w = max(central.width(), label_w)
        if label_w <= 2:
            label_w = pix.width()
        current_h = self.signature_label.height()
        if current_h <= 0:
            current_h = int(self.banner_max_height * 0.7)
        target_h = min(self.banner_max_height, max(current_h, int(self.banner_max_height * 0.6)))
        scaled = pix.scaled(int(label_w), target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.signature_label.setPixmap(scaled)
        # Ajustar altura mínima para evitar que el layout la colapse al redimensionar.
        self.signature_label.setMinimumHeight(min(self.banner_max_height, scaled.height()))

    def run_pipeline(self) -> None:
        if not self.current_path:
            QMessageBox.warning(self, "Falta archivo", "Primero carga un CSV/XML de PRPD.")
            return

        try:
            # Obtener directorio de salida usando el helper
            outdir = self._get_output_dir()

            # Calcular offset de fase seleccionado (o None si modo auto)
            force_offsets = self._get_force_offsets()

            # Filtro actual
            filt_label = self._get_filter_label()

            # Máscara/intervalos seleccionados y deciles de pixel
            mask_ranges = self._get_phase_mask_ranges()
            pixel_deciles = self._get_pixel_deciles_selection()
            qty_deciles = self._get_qty_deciles_selection()

            # Procesar PRPD utilizando el offset de fase, filtro y máscara/pixel
            result = core.process_prpd(
                path=self.current_path,
                out_root=outdir,
                force_phase_offsets=force_offsets,
                fast_mode=False,
                filter_level=filt_label,
                phase_mask=mask_ranges,
                pixel_deciles_keep=pixel_deciles,
                qty_deciles_keep=qty_deciles,
            )
            if self.chk_gap.isChecked():
                gx = getattr(self, "_gap_xml_path", None)
                if gx:
                    result["gap_stats"] = self._compute_gap(gx) or {}
            self.last_result = result
            self.last_run_profile = self._collect_current_profile()
            self._apply_qty_decile_filter(result, qty_deciles)
            self._ann_history_written = False

            # Pintar en GUI
            self.render_result(result)
            self.btn_pdf.setEnabled(True)

            # Export básico del filtro actual (ANGPD con qty en CSV)
            out_reports = (Path(outdir) / "reports")
            out_reports.mkdir(parents=True, exist_ok=True)
            stem = self.current_path.stem if self.current_path else "session"

            ang = result.get("angpd", {})
            x  = np.asarray(ang.get("phi_centers", []), dtype=float)
            y1 = np.asarray(ang.get("angpd", []), dtype=float)
            y2 = np.asarray(ang.get("n_angpd", []), dtype=float)
            y3 = np.asarray(ang.get("angpd_qty", []), dtype=float)
            y4 = np.asarray(ang.get("n_angpd_qty", []), dtype=float)

            if x.size:
                with open(out_reports / f"{stem}_angpd.csv", "w", encoding="utf-8") as f:
                    f.write("phi_deg,angpd,n_angpd,angpd_qty,n_angpd_qty\n")
                    for i in range(x.size):
                        a1 = float(y1[i]) if i < y1.size else 0.0
                        a2 = float(y2[i]) if i < y2.size else 0.0
                        a3 = float(y3[i]) if i < y3.size else 0.0
                        a4 = float(y4[i]) if i < y4.size else 0.0
                        f.write(f"{float(x[i]):.3f},{a1:.6f},{a2:.6f},{a3:.6f},{a4:.6f}\n")

        except FileNotFoundError as e:
            # Archivo o ruta no encontrada
            QMessageBox.critical(self, "Error en pipeline", f"Archivo o ruta no encontrado: {e}")
        except ValueError as e:
            # Errores de valor durante el procesamiento
            QMessageBox.critical(self, "Error en pipeline", f"Error de valor: {e}")
        except Exception as e:
            # Otras excepciones imprevistas
            traceback.print_exc()
            QMessageBox.critical(self, "Error en pipeline", str(e))

    # Ayuda
    def on_open_readme(self) -> None:
        readme = _ROOT / "README.md"
        try:
            if os.name == "nt":
                os.startfile(str(readme))  # type: ignore[attr-defined]
            else:
                import webbrowser
                webbrowser.open(str(readme))
        except Exception:
            QMessageBox.information(self, "README", f"Abre: {readme}")


        # UI handlers
    def open_file_dialog(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "Selecciona archivo PRPD (CSV/XML)", "", "PRPD (*.csv *.xml);;CSV (*.csv);;XML (*.xml);;Todos (*.*)")
        if not fn:
            return
        self.current_path = Path(fn)
        self.last_result = None
        self.last_run_profile = None
        try:
            self._get_output_dir(force_prompt=True)
        except Exception:
            pass
        try:
            data = core.load_prpd(self.current_path)
            self.plot_raw(data)
        except FileNotFoundError:
            QMessageBox.critical(self, "Error al cargar", "Archivo no encontrado.")
        except ValueError as e:
            QMessageBox.critical(self, "Error al cargar", f"Error en el formato de archivo: {e}")
        except Exception as e:
            # Otras excepciones imprevistas
            QMessageBox.critical(self, "Error al cargar", str(e))

    def export_pdf_clicked(self) -> None:
        if not self.last_result:
            QMessageBox.warning(self, "Sin resultados", "Ejecuta primero el procesamiento.")
            return
        try:
            # Exportar PDF en la carpeta de salida seleccionada
            outdir = self._get_output_dir()
            pdf_path = export_pdf_report(self.last_result, outdir)
            QMessageBox.information(self, "PDF exportado", f"Guardado en:\n{pdf_path}")
        except PermissionError as e:
            QMessageBox.critical(self, "Error exportando PDF", f"Permiso denegado: {e}")
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Error exportando PDF", f"Archivo no encontrado: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error exportando PDF", str(e))

    def on_reset_baseline(self) -> None:
        try:
            if not self.last_result:
                QMessageBox.information(self, "Baseline", "No hay resultados actuales.")
                return
            outdir = self._get_output_dir()
            out_reports = (Path(outdir) / 'reports'); out_reports.mkdir(parents=True, exist_ok=True)
            stem = self.current_path.stem if self.current_path else 'session'
            bd = self.last_result.get('severity_breakdown', {})
            # Extender baseline con KPIs PRPD y metadatos, manteniendo claves originales para compatibilidad
            ph = np.asarray(self.last_result.get('aligned',{}).get('phase_deg', []), dtype=float)
            raw_amp = np.asarray(self.last_result.get('raw',{}).get('amplitude', []), dtype=float)
            def _circ_mean_deg(arr):
                try:
                    if arr is None or arr.size == 0: return None
                    sx = float(np.cos(np.deg2rad(arr)).sum()); sy = float(np.sin(np.deg2rad(arr)).sum())
                    return float((np.rad2deg(np.arctan2(sy, sx)) + 360.0) % 360.0)
                except Exception:
                    return None
            def _circ_width2_deg(arr):
                try:
                    if arr is None or arr.size == 0: return None
                    C = float(np.cos(np.deg2rad(arr)).mean()); S = float(np.sin(np.deg2rad(arr)).mean())
                    R = (C*C + S*S) ** 0.5
                    if R <= 0: return None
                    std_rad = float((-2.0 * np.log(max(R, 1e-12))) ** 0.5)
                    return float(np.rad2deg(std_rad) * 2.0)
                except Exception:
                    return None
            cur = {
                # Claves históricas
                'p95_amp': float(bd.get('p95_amp', 0.0)),
                'dens': float(bd.get('dens', 0.0)),
                'R_phase': float(bd.get('R_phase', 0.0)),
                'std_circ_deg': float(bd.get('std_circ_deg', 0.0)),
                'severity': float(self.last_result.get('severity_score', 0.0)),
                # Metadatos y KPIs extendidos (no rompen lecturas existentes)
                '__meta__': {
                    'source_file': str(self.current_path) if self.current_path else None,
                    'source_stem': stem,
                    'created_utc': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                    'filter_level': self.cmb_filter.currentText().strip() if hasattr(self, 'cmb_filter') else None,
                    'phase_offset': int(self.last_result.get('phase_offset', 0)),
                },
                'kpi_ext': {
                    'total_count': int(ph.size) if ph is not None else None,
                    'tev_db': float(np.percentile(raw_amp, 95)) if raw_amp.size else None,
                    'ang_width_deg': _circ_width2_deg(ph),
                    'phase_center_deg': _circ_mean_deg(ph),
                    'p50_ms': None,
                    'p5_ms': None,
                }
            }
            (out_reports / f"{stem}_baseline.json").write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding='utf-8')
            QMessageBox.information(self, "Baseline", "Baseline actualizado con valores actuales.")
            self.render_result(self.last_result)
        except Exception as e:
            QMessageBox.warning(self, "Baseline", f"No se pudo escribir baseline: {e}")

    # ---- Seguimiento y criticidad ----
    def on_pick_gap_xml(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "Selecciona XML gap-time (opcional)", "", "XML (*.xml);;Todos (*.*)")
        if fn:
            setattr(self, '_gap_xml_path', fn)
            QMessageBox.information(self, "Gap-time", f"Usando XML para gap-time:\n{fn}")

    @staticmethod
    def _circ_mean_deg(ph: np.ndarray) -> float | None:
        try:
            if ph is None or ph.size == 0:
                return None
            sx = float(np.cos(np.deg2rad(ph)).sum())
            sy = float(np.sin(np.deg2rad(ph)).sum())
            ang = float((np.rad2deg(np.arctan2(sy, sx)) + 360.0) % 360.0)
            return ang
        except Exception:
            return None

    @staticmethod
    def _circ_std2_deg(ph: np.ndarray) -> float | None:
        try:
            if ph is None or ph.size == 0:
                return None
            C = float(np.cos(np.deg2rad(ph)).mean())
            S = float(np.sin(np.deg2rad(ph)).mean())
            R = (C*C + S*S) ** 0.5
            if R <= 0:
                return None
            std_rad = float((-2.0 * np.log(max(R, 1e-12))) ** 0.5)
            return float(np.rad2deg(std_rad) * 2.0)
        except Exception:
            return None

    @staticmethod
    def _compute_gap(xml_path: str) -> dict:
        """Calcula P5/P50 de gap-time a partir de XML con <times> o <max>."""
        try:
            import xml.etree.ElementTree as ET
            import re
            root = ET.parse(xml_path).getroot()

            def _extract_numeric_list(tag_name: str) -> list[float]:
                values: list[float] = []
                pattern = re.compile(r"[-+]?\d+(?:\.\d+)?")
                for node in root.iter():
                    tag = (node.tag or "").strip().lower()
                    if tag != tag_name:
                        continue
                    text = (node.text or "").strip()
                    if not text:
                        continue
                    for m in pattern.finditer(text):
                        try:
                            values.append(float(m.group(0)))
                        except Exception:
                            continue
                return values

            def _build_from_diffs(samples: np.ndarray, unit_ms: float, has_activity: bool) -> dict:
                if samples.size < 2:
                    return {}
                diffs = np.diff(np.sort(samples.astype(float)))
                diffs = diffs[diffs > 0]
                if diffs.size == 0:
                    return {}
                gaps_ms = diffs * unit_ms
                p50 = float(np.percentile(gaps_ms, 50))
                p5 = float(np.percentile(gaps_ms, 5))
                class_p50 = PRPDWindow._gap_condition(p50, has_activity=has_activity)
                class_p5 = PRPDWindow._gap_condition(p5, has_activity=has_activity)
                return {
                    "p50_ms": p50,
                    "p5_ms": p5,
                    "gaps_ms": gaps_ms.tolist(),
                    "classification": class_p50,
                    "classification_p5": class_p5,
                }

            # Caso 1: XML con lista explícita de tiempos de pulsos
            raw_times = _extract_numeric_list("times")
            if raw_times:
                times = np.asarray(raw_times, dtype=float)
                out = _build_from_diffs(times, 1000.0, has_activity=True)  # se asume segundos -> ms
                if out:
                    out["source"] = "times"
                    return out

            # Caso 2: XML tipo niveles con 500 muestras <max>
            max_values = _extract_numeric_list("max")
            if not max_values:
                return {}
            amp = np.asarray(max_values, dtype=float)
            if amp.size < 5:
                return {}
            base = float(np.median(amp))
            tolerance = 5.0
            mask = np.abs(amp - base) > tolerance

            def _safe_float(text: str | None) -> float | None:
                try:
                    if text is None:
                        return None
                    return float(text.strip())
                except Exception:
                    return None

            dt_ms = _safe_float(root.findtext(".//gating_time"))
            if dt_ms is None or dt_ms <= 0.0:
                timescale = _safe_float(root.findtext(".//timescale"))
                nsamples = _safe_float(root.findtext(".//nsamples"))
                if timescale and nsamples and nsamples > 0:
                    dt_ms = (timescale / nsamples) * 1000.0
            if dt_ms is None or dt_ms <= 0.0:
                # Escenario descrito: 500 muestras cubren 0.5 s
                dt_ms = 0.5 / max(1, amp.size) * 1000.0

            time_ms = np.arange(amp.size, dtype=float) * dt_ms
            gaps: list[float] = []
            current = 0.0
            for flag in mask:
                if flag:
                    if current > 0.0:
                        gaps.append(current)
                        current = 0.0
                else:
                    current += dt_ms
            if current > 0.0:
                gaps.append(current)
            gaps_arr = np.asarray(gaps, dtype=float)
            if gaps_arr.size == 0:
                gaps_arr = np.zeros(1, dtype=float)
            p50 = float(np.percentile(gaps_arr, 50))
            p5 = float(np.percentile(gaps_arr, 5))
            has_activity = bool(np.any(mask))
            classification = PRPDWindow._gap_condition(p50, has_activity=has_activity)
            classification_p5 = PRPDWindow._gap_condition(p5, has_activity=has_activity)
            result = {
                "source": "levels",
                "p50_ms": p50,
                "p5_ms": p5,
                "gaps_ms": gaps_arr.tolist(),
                "base": base,
                "tolerance": tolerance,
                "dt_ms": dt_ms,
                "time_s": (time_ms / 1000.0).tolist(),
                "series": amp.tolist(),
                "mask": mask.astype(int).tolist(),
                "classification": classification,
                "classification_p5": classification_p5,
            }
            return result
        except Exception:
            traceback.print_exc()
            return {}

    @staticmethod
    def _gap_condition(value_ms: float | None, has_activity: bool) -> dict:
        """Clasificación textual según la tabla de severidad."""
        table = [
            {
                "code": "sin_dp",
                "label": "Sin gap time",
                "level_name": "Aceptable",
                "color": "#00B050",
                "threshold": 9999.0,
                "action": "Continuar operación normal. Monitoreo cada 12 meses",
                "action_short": "Continuar operación normal. Monitoreo cada 12 meses",
            },
            {
                "code": "pd_leve",
                "label": "Gap-time > 7 ms",
                "level_name": "Descarga Parcial Leve",
                "color": "#1565c0",
                "threshold": 7.0,
                "action": "Monitoreo cada 6 meses",
                "action_short": "Monitoreo cada 6 meses",
            },
            {
                "code": "pd_critico",
                "label": "3 ms < Tiempo entre pulsos < 7 ms",
                "level_name": "Descarga Parcial Grave",
                "color": "#FF8C00",
                "threshold": 3.0,
                "action": "Requiere evaluación de pruebas especializadas. Si la tendencia es creciente planear sustitución. en un futuro cercano. Monitoreo cada 3 meses",
                "action_short": "Requiere evaluación de pruebas especializadas. Si la tendencia es creciente planear sustitución. en un futuro cercano. Monitoreo cada 3 meses",
            },
            {
                "code": "pd_critico",
                "label": "Gap-Time < 3 ms",
                "level_name": "Crítico",
                "color": "#B00000",
                "threshold": 0.0,
                "action": "Insatisfactorio. Planear retiro inmediato o a corto plazo. Monitreo semanal o contínuo",
                "action_short": "Insatisfactorio. Planear retiro inmediato o a corto plazo. Monitreo semanal o contínuo",
            },
        ]
        if value_ms is None or not has_activity:
            return table[0]
        if value_ms > 7.0:
            return table[1]
        if value_ms > 3.0:
            return table[2]
        return table[3]

    @staticmethod
    def _safe_append_summary(path: Path, kv: dict) -> Path | None:
        try:
            txt0 = path.read_text(encoding='utf-8') if path.exists() else ''
            lines = [l for l in txt0.splitlines()]
            if lines and lines[-1].strip() != '':
                lines.append('')
            for k,v in kv.items():
                lines.append(f"{k},{v}")
            tmp = path.with_suffix(path.suffix + '.tmp')
            tmp.write_text("\n".join(lines) + "\n", encoding='utf-8')
            try:
                tmp.replace(path); return path
            except PermissionError:
                safe = path.with_name(path.stem + '_safe' + path.suffix)
                tmp.replace(safe); return safe
        except Exception:
            return None

    def on_compare_vs_base(self) -> None:
        if not self.last_result:
            QMessageBox.information(self, "Comparar", "Procesa primero un PRPD.")
            return
        try:
            outdir = self._get_output_dir()
            out_reports = Path(outdir)/'reports'
            out_reports.mkdir(parents=True, exist_ok=True)
            stem = self.current_path.stem if self.current_path else 'session'

            r = self.last_result
            ph = np.asarray(r.get('aligned',{}).get('phase_deg', []), dtype=float)
            amp = np.asarray(r.get('aligned',{}).get('amplitude', []), dtype=float)
            raw_amp = np.asarray(r.get('raw',{}).get('amplitude', []), dtype=float)

            # KPIs actuales
            total_count = int(ph.size)
            tev_db = float(np.percentile(raw_amp, 95)) if raw_amp.size else None
            ang_width = self._circ_std2_deg(ph)
            phase_center = self._circ_mean_deg(ph)

            # Gap opcional
            gap = {}
            if self.chk_gap.isChecked():
                gx = getattr(self, '_gap_xml_path', None)
                if gx:
                    gap = self._compute_gap(gx) or {}

            # Baseline
            base_path = out_reports / f"{stem}_baseline.json"
            base = {}
            if base_path.exists():
                try:
                    base = json.loads(base_path.read_text(encoding='utf-8'))
                except Exception:
                    base = {}
            # Validar correspondencia por fuente (si tiene meta)
            try:
                meta = base.get('__meta__', {}) if isinstance(base, dict) else {}
                src_ok = (not meta) or (meta.get('source_stem') == stem)
                if not src_ok:
                    base = {}
            except Exception:
                pass

            def pct_delta(a, b):
                try:
                    if a is None or b is None:
                        return None
                    b = float(b)
                    if b == 0:
                        return float('inf') if (a or 0) > 0 else 0.0
                    return (float(a) - b) / b
                except Exception:
                    return None

            def circ_delta(a, b):
                try:
                    if a is None or b is None:
                        return None
                    return abs(((float(a) - float(b)) + 180.0) % 360.0 - 180.0)
                except Exception:
                    return None

            # Baseline extendido puede venir en kpi_ext
            bx = base.get('kpi_ext', {}) if isinstance(base, dict) else {}
            count_base = (bx.get('total_count') if bx else base.get('total_count')) if isinstance(base, dict) else None
            count_dpct = pct_delta(total_count, count_base)

            tev_base = (bx.get('tev_db') if bx else base.get('tev_db')) if isinstance(base, dict) else None
            tev_delta_db = None
            try:
                if tev_db is not None and tev_base is not None and float(tev_base) > 0:
                    tev_delta_db = 20.0 * np.log10(max(float(tev_db) / max(float(tev_base), 1e-9), 1e-9))
            except Exception:
                tev_delta_db = None

            ang_base = (bx.get('ang_width_deg') if bx else base.get('ang_width_deg')) if isinstance(base, dict) else None
            if ang_base is None:
                ang_base = base.get('std_circ_deg') if isinstance(base, dict) else None
            ang_dpct = pct_delta(ang_width, ang_base)

            phase_base = (bx.get('phase_center_deg') if bx else base.get('phase_center_deg')) if isinstance(base, dict) else None
            phase_shift_deg = circ_delta(phase_center, phase_base)

            # TEV asimetría (aprox con raw_amp por semiciclo)
            tev_asym_db = None
            try:
                if ph.size and raw_amp.size:
                    pos = raw_amp[(ph % 360.0) < 180.0]
                    neg = raw_amp[(ph % 360.0) >= 180.0]
                    if pos.size and neg.size:
                        p95_pos = float(np.percentile(pos, 95)); p95_neg = float(np.percentile(neg, 95))
                        tev_asym_db = abs(p95_pos - p95_neg)
            except Exception:
                tev_asym_db = None

            # Flags
            flag_count = 'green'
            if count_dpct is not None:
                if count_dpct >= 0.50: flag_count = 'red'
                elif count_dpct >= 0.30: flag_count = 'orange'

            flag_tev = 'green'
            if tev_delta_db is not None:
                if tev_delta_db > 6.0: flag_tev = 'red'
                elif tev_delta_db >= 3.0: flag_tev = 'orange'
                try:
                    if tev_asym_db is not None and tev_asym_db <= 3.0 and flag_tev != 'green':
                        flag_tev = 'red'
                except Exception:
                    pass

            flag_ancho = 'green'
            if ang_dpct is not None:
                if ang_dpct >= 0.40: flag_ancho = 'red'
                elif ang_dpct >= 0.20: flag_ancho = 'orange'

            flag_phase = 'green'
            if phase_shift_deg is not None:
                if phase_shift_deg >= 20.0: flag_phase = 'red'
                elif phase_shift_deg >= 10.0: flag_phase = 'orange'

            # Gap rules
            flag_gap = 'green'
            GAP_P50_RED = 7.0; GAP_P5_RED = 3.0
            p50_cur = gap.get('p50_ms'); p5_cur = gap.get('p5_ms')
            p50_base = (bx.get('p50_ms') if bx else base.get('p50_ms')) if isinstance(base, dict) else None
            p5_base = (bx.get('p5_ms') if bx else base.get('p5_ms')) if isinstance(base, dict) else None
            try:
                severe_now = ((p50_cur is not None and p50_cur < GAP_P50_RED) or (p5_cur is not None and p5_cur < GAP_P5_RED))
                drop_p50 = None
                if p50_cur is not None and p50_base is not None and float(p50_base) > 0:
                    drop_p50 = max(0.0, (float(p50_base) - float(p50_cur)) / float(p50_base))
                if severe_now: flag_gap = 'red'
                elif drop_p50 is not None and drop_p50 >= 0.30: flag_gap = 'red'
                elif drop_p50 is not None and drop_p50 >= 0.10: flag_gap = 'orange'
            except Exception:
                flag_gap = 'green'

            # Criticidad global
            flags = [flag_count, flag_tev, flag_ancho, flag_phase, flag_gap]
            criticidad = 'red' if 'red' in flags else ('orange' if 'orange' in flags else 'green')
            decision = 'Inspección prioritaria' if criticidad == 'red' else ('Monitorear y re-evaluar' if criticidad == 'orange' else 'OK')

            # Persistencia: summary.csv (si existe en out) y JSON de tracking en reports
            nd = 'N/D'
            kv = {
                'kpi_count_delta_pct': (f"{count_dpct:.4f}" if count_dpct is not None else nd),
                'tev_delta_db': (f"{tev_delta_db:.3f}" if tev_delta_db is not None else nd),
                'tev_asym_db': (f"{tev_asym_db:.3f}" if tev_asym_db is not None else nd),
                'ang_width_deg_act': (f"{ang_width:.3f}" if ang_width is not None else nd),
                'ang_width_deg_base': (f"{(ang_base if ang_base is not None else 0):.3f}" if ang_base is not None else nd),
                'ang_width_delta_pct': (f"{ang_dpct:.4f}" if ang_dpct is not None else nd),
                'phase_shift_deg': (f"{(phase_shift_deg if phase_shift_deg is not None else 0):.3f}" if phase_shift_deg is not None else nd),
                'p50_ms': (f"{p50_cur:.3f}" if p50_cur is not None else nd),
                'p5_ms': (f"{p5_cur:.3f}" if p5_cur is not None else nd),
                'kpi_flags.count': flag_count,
                'kpi_flags.tev': flag_tev,
                'kpi_flags.ancho': flag_ancho,
                'kpi_flags.phase': flag_phase,
                'kpi_flags.gap': flag_gap,
                'criticidad_global': criticidad,
                'decision_recomendada': decision,
            }
            # summary.csv si existe
            summary_csv = Path(outdir) / f"{stem}_summary.csv"
            wrote_csv = self._safe_append_summary(summary_csv, kv) if summary_csv.exists() else None
            # Tracking JSON
            tracking = {
                'actual': {
                    'total_count': total_count,
                    'tev_db': tev_db,
                    'ang_width_deg': ang_width,
                    'phase_center_deg': phase_center,
                    **(gap or {}),
                },
                'baseline': base,
                'flags': {
                    'count': flag_count, 'tev': flag_tev, 'ancho': flag_ancho, 'phase': flag_phase, 'gap': flag_gap,
                },
                'criticidad_global': criticidad,
                'decision_recomendada': decision,
            }
            (out_reports / f"{stem}_kpi_tracking.json").write_text(json.dumps(tracking, ensure_ascii=False, indent=2), encoding='utf-8')

            msg = (
                f"?Conteo: {kv['kpi_count_delta_pct']} | ?TEV(dB): {kv['tev_delta_db']} | Asim: {kv['tev_asym_db']}\n"
                f"Ancho act/base/?%: {kv['ang_width_deg_act']} / {kv['ang_width_deg_base']} / {kv['ang_width_delta_pct']} | Fase ?°: {kv['phase_shift_deg']}\n"
                f"Gap p50/p5 ms: {kv['p50_ms']} / {kv['p5_ms']}\n"
                f"Flags: count={flag_count}, tev={flag_tev}, ancho={flag_ancho}, phase={flag_phase}, gap={flag_gap}\n"
                f"Criticidad: {criticidad} | Decisión: {decision} | summary.csv: {'OK' if wrote_csv else 'N/D'}"
            )
            QMessageBox.information(self, "Comparar vs base", msg)
        except Exception as e:
            QMessageBox.warning(self, "Comparar vs base", f"Error: {e}")

    # Plots
    def plot_raw(self, data: dict) -> None:
        self.ax_raw.clear()
        try:
            ph = np.asarray(data.get("phase_deg", []), dtype=float)
            amp = np.asarray(data.get("amplitude", []), dtype=float)
            if self.chk_hist2d.isChecked() and ph.size and amp.size:
                H, xedges, yedges = np.histogram2d(ph, amp, bins=[72, 50], range=[[0,360],[0,100]])
                self.ax_raw.imshow(H.T + 1e-9, origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            else:
                self.ax_raw.scatter(ph, amp, s=4, alpha=0.6)
        except Exception:
            pass
        self.ax_raw.set_title("PRPD crudo")
        self.ax_raw.set_xlim(0, 360); self.ax_raw.set_xlabel("Fase (°)")
        self.ax_raw.set_ylabel("Amplitud")
        self.ax_raw.set_ylim(0, 100)
        self.canvas.draw_idle()

    def _draw_histograms_semiciclo(self, r: dict) -> None:
        """Dibuja Histogramas por semiciclos en dos paneles:
        - ax_filtered: H_amp+ y H_amp- (N=16)
        - ax_text:     H_ph+ y H_ph-   (N=16)
        """
        ax = self.ax_filtered
        ax.clear()
        try:
            ph = np.asarray(r.get("aligned", {}).get("phase_deg", []), dtype=float)
            amp = np.asarray(r.get("aligned", {}).get("amplitude", []), dtype=float)
            if not (ph.size and amp.size):
                ax.text(0.5,0.5,'Sin datos',ha='center',va='center');
                self.ax_text.clear(); self.ax_text.text(0.5,0.5,'Sin datos',ha='center',va='center');
                return
            N = 16
            phi = ph % 360.0
            pos = (phi < 180.0)
            neg = ~pos
            # H_amp en panel superior derecho
            a_pos, _ = np.histogram(amp[pos], bins=N, range=(0.0, 100.0))
            a_neg, _ = np.histogram(amp[neg], bins=N, range=(0.0, 100.0))
            Ha_pos = np.log10(1.0 + a_pos.astype(float))
            Ha_neg = np.log10(1.0 + a_neg.astype(float))
            m_amp = float(max(Ha_pos.max() if Ha_pos.size else 0.0, Ha_neg.max() if Ha_neg.size else 0.0, 1.0))
            Ha_pos = Ha_pos / m_amp; Ha_neg = Ha_neg / m_amp
            xi = np.arange(1, N+1)
            ax.plot(xi, Ha_pos, '-o', color='#1f77b4', label='H_amp+')
            ax.plot(xi, Ha_neg, '-o', color='#d62728', label='H_amp-')
            ax.set_xlabel('Indice de ventana (N=16)'); ax.set_ylabel('H_amp (norm)'); ax.set_title('Histograma de Amplitud (N=16)')
            ax.legend(loc='upper right', fontsize=8)
            # H_ph en panel inferior derecho
            bx = self.ax_text
            bx.clear()
            phi_pos = phi[pos]; phi_neg = (phi[neg] - 180.0)
            p_pos, _ = np.histogram(phi_pos, bins=N, range=(0.0, 180.0))
            p_neg, _ = np.histogram(phi_neg, bins=N, range=(0.0, 180.0))
            Hp_pos = np.log10(1.0 + p_pos.astype(float))
            Hp_neg = np.log10(1.0 + p_neg.astype(float))
            m_ph = float(max(Hp_pos.max() if Hp_pos.size else 0.0, Hp_neg.max() if Hp_neg.size else 0.0, 1.0))
            Hp_pos = Hp_pos / m_ph; Hp_neg = Hp_neg / m_ph
            bx.plot(xi, Hp_pos, '-o', color='#1f77b4', label='H_ph+')
            bx.plot(xi, Hp_neg, '-o', color='#d62728', label='H_ph-')
            bx.set_xlabel('Indice de ventana (N=16)'); bx.set_ylabel('H_ph (norm)'); bx.set_title('Histograma de Fase (N=16)')
            bx.legend(loc='upper right', fontsize=8)
        except Exception:
            ax.text(0.5,0.5,'Error histogramas',ha='center',va='center');
            try:
                self.ax_text.text(0.5,0.5,'Error histogramas',ha='center',va='center')
            except Exception:
                pass

    def _draw_combined_panels(self, r: dict, ph_al: np.ndarray, amp_al: np.ndarray) -> None:
        self._draw_combined_overlay(
            ax=self.ax_raw,
            twin_attr="ax_raw_twin",
            ph=ph_al,
            amp=amp_al,
            ang=r.get("angpd", {}),
            quantity=False,
        )
        self._draw_combined_overlay(
            ax=self.ax_probs,
            twin_attr="ax_probs_twin",
            ph=ph_al,
            amp=amp_al,
            ang=r.get("angpd", {}),
            quantity=True,
        )

    def _render_conclusions(self, result: dict, payload: dict | None = None) -> None:
        if payload is None:
            text, payload = self._get_conclusion_insight(result)
            self.last_conclusion_text = text
            self.last_conclusion_payload = payload
        else:
            text = self.last_conclusion_text
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        gap_stats = payload.get("gap") if isinstance(payload, dict) else None
        self.ax_gap_wide.set_visible(False)
        for ax_top in (self.ax_raw, self.ax_filtered):
            ax_top.clear()
            ax_top.set_facecolor("#fafafa")
            ax_top.set_xticks([])
            ax_top.set_yticks([])
            ax_top.axis("off")

        self._clear_conclusion_artists()
        ax = self._ensure_conclusion_axis()
        ax.set_visible(True)
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor("#ffffff")
        card = FancyBboxPatch(
            (0.01, 0.02),
            0.98,
            0.96,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            facecolor="#fffdf5",
            edgecolor="#c8c6c3",
        )
        self._register_conclusion_artist(ax.add_patch(card))
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
            "alerta": "#FF8C00",
            "info": "#0d47a1",
            "mitigado": "#00B050",
        }
        self._draw_conclusion_header(ax, "Resultados de los principales KPI", None)

        left_card = FancyBboxPatch(
            (0.02, 0.05),
            0.47,
            0.82,
            boxstyle="round,pad=0.02",
            linewidth=1.0,
            facecolor="#ffffff",
            edgecolor="#d9d4c7",
        )
        right_card = FancyBboxPatch(
            (0.51, 0.05),
            0.47,
            0.82,
            boxstyle="round,pad=0.02",
            linewidth=1.0,
            facecolor="#ffffff",
            edgecolor="#d9d4c7",
        )
        self._register_conclusion_artist(ax.add_patch(left_card))
        self._register_conclusion_artist(ax.add_patch(right_card))

        left_ax = ax.inset_axes([0.03, 0.08, 0.44, 0.76])
        right_ax = ax.inset_axes([0.52, 0.08, 0.44, 0.76])
        for sub in (left_ax, right_ax):
            sub.set_axis_off()
            sub.set_xlim(0, 1)
            sub.set_ylim(0, 1)
            sub.set_facecolor("none")
        self._conclusion_subaxes.extend([left_ax, right_ax])

        def _fmt_value(value, decimals=1, suffix=""):
            if value is None:
                return "N/D"
            try:
                if isinstance(value, (int, np.integer)):
                    return f"{int(value):,}{suffix}"
                return f"{float(value):.{decimals}f}{suffix}"
            except Exception:
                return f"{value}"

        def _fmt_angle(value):
            if value is None or value == "N/D":
                return "N/D"
            try:
                return f"{float(value):.1f}°"
            except Exception:
                return f"{value}°"

        def _ratio_status(value):
            try:
                val = float(value)
            except Exception:
                return ("Aceptable", "#00B050")
            if val > 0.6:
                return ("Crítico", "#B00000")
            if val > 0.35:
                return ("Grave", "#FF8C00")
            return ("Aceptable", "#00B050")

        def _status_or_default(status):
            if isinstance(status, tuple):
                return status
            if isinstance(status, str):
                return (status, "#00B050")
            return ("Aceptable", "#00B050")

        def _gap_badge_tuple(value, classification, *, default="Aceptable"):
            if classification:
                label = classification.get("level_name") or classification.get("label") or "Gap-time"
                color = classification.get("color", "#00B050")
                return (label, color)
            label, color = self._status_from_gap(value)
            if label == "Sin gap time":
                return (default, "#00B050")
            return (label, color)

        delta_phase = result.get("phase_offset")
        gap_info = (gap_stats or {}).get("classification") if isinstance(gap_stats, dict) else None
        class_p5 = (gap_stats or {}).get("classification_p5") if isinstance(gap_stats, dict) else None
        gap_p50 = metrics.get("gap_p50")
        gap_p5 = metrics.get("gap_p5")
        rows = [
            ("Total pulsos útiles", _fmt_value(metrics.get("total_count"), decimals=0), self._status_from_total(metrics.get("total_count"))),
            ("Anchura fase", _fmt_angle(metrics.get("phase_width")), self._status_from_width(metrics.get("phase_width"))),
            ("Centro", _fmt_angle(metrics.get("phase_center")), None),
            ("Número de picos de fase", _fmt_value(metrics.get("n_peaks"), decimals=0), None),
            ("p95 medio", _fmt_value(metrics.get("p95_mean")), self._status_from_amp(metrics.get("p95_mean"))),
            ("p95 amplitud pos/neg", f"{_fmt_value(metrics.get('amp_p95_pos'))} / {_fmt_value(metrics.get('amp_p95_neg'))}", self._status_from_amp(metrics.get("amp_p95_pos"))),
            ("Δ fase", _fmt_angle(delta_phase), None),
            ("Gap-Time P50", _fmt_value(gap_p50, decimals=2, suffix=" ms"), _gap_badge_tuple(gap_p50, gap_info)),
            ("Gap-Time P5", _fmt_value(gap_p5, decimals=2, suffix=" ms"), _gap_badge_tuple(gap_p5, class_p5, default="Sin dato")),
            ("Relación N-ANGPD/ANGPD", f"{metrics.get('n_ang_ratio', 'N/D')}", _ratio_status(metrics.get("n_ang_ratio"))),
        ]

        def _draw_kpi_row(ax_target, y_val, label, value, badge):
            label_fmt = (label[:1].upper() + label[1:]) if label else "N/D"
            self._register_conclusion_artist(ax_target.text(0.02, y_val, label_fmt, fontsize=11, fontweight="bold", ha="left", va="center"))
            self._register_conclusion_artist(ax_target.text(0.42, y_val, value, fontsize=11, ha="left", va="center"))
            if badge:
                text_badge, color_badge = badge
                self._register_conclusion_artist(self._draw_status_tag(ax_target, text_badge, 0.78, y_val, color=color_badge))

        y = self._draw_section_title(left_ax, "Indicadores clave", y=0.93)
        for label, value, status in rows:
            _draw_kpi_row(left_ax, y, label, value, _status_or_default(status))
            y -= 0.075

        def _render_action_badges(ax_target, y_start, label, text_value, color):
            ax_target.text(0.02, y_start, label, fontsize=11, fontweight="bold", ha="left", va="center")
            y_pos = y_start - 0.07
            parts = [p.strip() for p in text_value.split(".") if p.strip()]
            if not parts:
                parts = ["Sin acciones registradas"]
            for part in parts:
                self._draw_status_tag(ax_target, part, 0.32, y_pos, color=color, text_color="#ffffff")
                y_pos -= 0.07
            return y_pos + 0.01

        def _draw_right_row(ax_target, y_val, label, value, *, color="#0d47a1", text_color="#ffffff"):
            pretty = value if value and value != "N/D" else "N/D"
            if isinstance(pretty, str):
                pretty = pretty.strip()
                if pretty:
                    pretty = pretty[0].upper() + pretty[1:]
                else:
                    pretty = "N/D"
            self._register_conclusion_artist(ax_target.text(0.02, y_val, label.upper(), fontsize=10, fontweight="bold", ha="left", va="center"))
            self._register_conclusion_artist(self._draw_status_tag(ax_target, pretty, 0.42, y_val, color=color, text_color=text_color))
            return y_val - 0.08

        # Contenido tarjeta derecha
        header_y = self._draw_section_title(right_ax, "Seguimiento y criticidad", y=0.93)

        risk_label = summary.get("risk", "N/D")
        risk_key = risk_label.lower() if isinstance(risk_label, str) else ""
        estado_map = {
            "bajo": ("Aceptable", "#00B050"),
            "moderado": ("Moderado", "#1565c0"),
            "alto": ("Grave", "#FF8C00"),
            "grave": ("Grave", "#FF8C00"),
            "critico": ("Crítico", "#B00000"),
            "crítico": ("Crítico", "#B00000"),
        }
        estado_general, risk_color = estado_map.get(risk_key, (risk_label if isinstance(risk_label, str) else "N/D", palette.get(risk_key, "#00B050")))
        life_years = summary.get("life_years")
        life_score = summary.get("life_score")

        y = header_y - 0.03
        life_txt = f"{life_score:.1f}" if isinstance(life_score, (int, float)) else "N/D"
        vida_txt = "N/D"
        if isinstance(life_years, (int, float)):
            vida_txt = f"≈ {life_years:.1f} años"
        summary_badge = f"{estado_general}   |   LifeScore: {life_txt}   |   Vida remanente: {vida_txt}"
        self._draw_status_tag(right_ax, summary_badge, 0.02, y, color=risk_color, text_color="#ffffff", size=12)
        y -= 0.14
        action_general = summary.get("actions", "Sin acciones registradas.")
        y = _render_action_badges(right_ax, y, "ACCIÓN RECOMENDADA", action_general, "#0d47a1") - 0.04

        if gap_info:
            y = _render_action_badges(right_ax, y, "ACCIÓN GAP-TIME P50", gap_info.get("action", ""), gap_info.get("color", "#00B050")) - 0.08

        stage = summary.get("stage", "N/D")
        pd_type = summary.get("pd_type", "N/D")
        location = self._normalize_location_label(summary.get("location", "N/D"))
        y = _draw_right_row(right_ax, y, "ETAPA PROBABLE", stage, color="#1e88e5")
        y = _draw_right_row(right_ax, y, "MODO DOMINANTE", pd_type, color="#1e88e5")
        y = _draw_right_row(right_ax, y, "UBICACIÓN PROBABLE", location, color="#1e88e5")
        y = _draw_right_row(right_ax, y, "RIESGO", estado_general, color=risk_color)

    def _render_ann_gap_view(self, result: dict, payload: dict | None = None) -> None:
        if payload is None:
            _, payload = self._get_conclusion_insight(result)
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        gap_stats = payload.get("gap") if isinstance(payload, dict) else None
        if not gap_stats:
            gap_stats = result.get("gap_stats")
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}

        # Mostrar ANN en el eje superior izquierdo
        self.ax_gap_wide.set_visible(False)
        self.ax_raw.set_visible(True)
        self.ax_raw.set_axis_on()
        self._draw_ann_prediction_panel(result, summary)

        # Gap-time en el panel superior derecho
        self.ax_filtered.set_visible(True)
        self.ax_filtered.set_axis_on()
        self._draw_gap_chart(self.ax_filtered, gap_stats)

        # Liberar panel inferior izquierdo y mostrar resumen GAP en el derecho
        self.ax_probs.clear()
        self.ax_probs.set_facecolor("#f5f6fb")
        self.ax_probs.set_xticks([])
        self.ax_probs.set_yticks([])
        self.ax_probs.axis("off")

        self._draw_gap_summary_panel(self.ax_text, metrics, gap_stats)

    def _render_gap_time_full(self, result: dict, payload: dict | None = None) -> None:
        if payload is None:
            _, payload = self._get_conclusion_insight(result)
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        gap_stats = payload.get("gap") if isinstance(payload, dict) else None
        if not gap_stats:
            gap_stats = result.get("gap_stats")

        # Ocultar ejes individuales y mostrar el eje ancho
        self.ax_raw.set_visible(False)
        self.ax_filtered.set_visible(False)
        self.ax_gap_wide.set_visible(True)
        self._draw_gap_chart(self.ax_gap_wide, gap_stats)

        # Panel inferior izquierdo sin contenido
        self.ax_probs.clear()
        self.ax_probs.set_facecolor("#f5f6fb")
        self.ax_probs.set_xticks([])
        self.ax_probs.set_yticks([])
        self.ax_probs.axis("off")

        # Panel inferior derecho también se limpia para esta vista
        self.ax_text.clear()
        self.ax_text.set_facecolor("#f5f6fb")
        self.ax_text.set_xticks([])
        self.ax_text.set_yticks([])
        self.ax_text.axis("off")

    def _draw_gap_summary_panel(self, ax, metrics: dict, gap_stats: dict | None) -> None:
        ax.clear()
        ax.set_facecolor("#fffdf5")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        card = FancyBboxPatch(
            (0.02, 0.05),
            0.96,
            0.90,
            boxstyle="round,pad=0.02",
            linewidth=1.0,
            facecolor="#ffffff",
            edgecolor="#d9d4c7",
        )
        ax.add_patch(card)

        def _badge(text, x, y, color, text_color="#ffffff"):
            return self._draw_status_tag(ax, text, x, y, color=color, text_color=text_color)

        def _action_lines(classification):
            label = (classification or {}).get("label", "").lower()
            if "3 ms < tiempo" in label:
                return [
                    ("Pruebas especializadas", False, "#ffffff"),
                    ("↑ Planear sustitución", True, "#ffffff"),
                    ("Monitoreo cada 3 meses", True, "#ffffff"),
                ]
            if "gap-time < 3" in label or "< 3" in label:
                return [
                    ("Insatisfactorio.", False, "#ffffff"),
                    ("Planear retiro inmediato o a corto plazo", True, "#ffffff"),
                    ("Monitoreo semanal o contínuo", True, "#ffffff"),
                ]
            if "gap-time > 7" in label:
                return [
                    ("Monitoreo cada 6 meses", False, "#ffffff"),
                ]
            if "sin gap" in label:
                return [("Continuar operación normal. Monitoreo cada 12 meses", False, "#ffffff")]
            action = (classification or {}).get("action", "")
            if not action:
                return [("Sin información", False, "#ffffff")]
            parts = [p.strip() for p in action.split(".") if p.strip()]
            return [(p, False, "#ffffff") for p in parts]

        def _interval_text(classification):
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

        def _render_action(y, title, classification, default_color):
            color = classification.get("color", default_color) if classification else default_color
            lines = _action_lines(classification)
            ax.text(0.04, y, title.upper(), fontsize=12, fontweight="bold", ha="left", va="center")
            y -= 0.075
            for text_line, _, txt_color in lines:
                self._draw_status_tag(ax, text_line, 0.42, y, color=color, text_color=txt_color)
                y -= 0.075
            return y - 0.025

        def _render_interval(y, label, classification):
            color = classification.get("color", "#00B050") if classification else "#90a4ae"
            interval = _interval_text(classification)
            ax.text(0.04, y, label.upper(), fontsize=12, fontweight="bold", ha="left", va="center")
            y -= 0.075
            self._draw_status_tag(ax, interval, 0.42, y, color=color)
            return y - 0.025

        header_y = self._draw_section_title(ax, "Resumen gap-time", y=0.93)
        y = header_y - 0.03
        gap_info = (gap_stats or {}).get("classification") if isinstance(gap_stats, dict) else None
        class_p5 = (gap_stats or {}).get("classification_p5") if isinstance(gap_stats, dict) else None

        y = _render_action(y, "ACCIÓN GAP-TIME P50", gap_info or {}, "#00B050")
        y = _render_interval(y, "TIEMPO ENTRE PULSOS P50", gap_info)
        y = _render_action(y, "ACCIÓN GAP-TIME P5", class_p5 or {}, "#B00000")
        _render_interval(max(y, 0.20), "TIEMPO ENTRE PULSOS P5", class_p5)

    def _normalize_location_label(self, text: str) -> str:
        if not isinstance(text, str):
            return "N/D"
        cleaned = text.strip()
        if not cleaned:
            return "N/D"
        lowered = cleaned.lower()
        if "ranura" in lowered or "ranuras" in lowered or "generador" in lowered:
            return "Transformador seco (resina epóxica) / transformador sumergido en aceite (mineral o vegetal)"
        return cleaned

    def _restore_standard_axes(self) -> None:
        self.ax_gap_wide.set_visible(False)
        for ax in (self.ax_raw, self.ax_filtered, self.ax_probs, self.ax_text):
            ax.set_visible(True)
            ax.set_axis_on()

    def _draw_ann_prediction_panel(self, result: dict, summary: dict) -> None:
        ax = self.ax_raw
        ax.clear()
        ax.set_facecolor("#f3f6fb")
        ann = result.get("probs", {}) or {}
        ann_norm = {str(k).lower(): float(v) for k, v in ann.items() if v is not None}
        labels = ["Corona +", "Superficial", "Corona -", "Cavidad"]
        color_map = ["#FFCC00", "#0066CC", "#CC0000", "#8E44AD"]
        keys = ["corona+", "superficial", "corona-", "cavidad"]
        values = [ann_norm.get(k, 0.0) for k in keys]
        if not any(values):
            inferred = (summary.get("pd_type") or "").lower()
            if "superficial" in inferred or "tracking" in inferred:
                values[1] = 1.0
            elif "corona" in inferred and "-" in inferred:
                values[2] = 1.0
            elif "corona" in inferred:
                values[0] = 1.0
            elif "cavidad" in inferred:
                values[3] = 1.0
            else:
                values[3] = 0.5
        values = [max(0.0, float(v)) for v in values]
        total = sum(values)
        if total <= 0:
            total = 1.0
        values = [min(1.0, max(0.0, v / total)) for v in values]
        bars = ax.bar(labels, values, color=color_map, edgecolor="#37474f", linewidth=1.0)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Probabilidad")
        ax.set_title("ANN Predicted PD Source", fontweight="bold")
        max_val = max(values) if values else 0.0
        tol = 0.02
        top_idx = [i for i, v in enumerate(values) if max_val - v <= tol]
        for i, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.02,
                f"{values[i]*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#0f172a",
            )
            if i in top_idx:
                bar.set_linewidth(2.3)
                bar.set_edgecolor("#0d1117")
        if top_idx:
            dom_labels = " / ".join(labels[i] for i in top_idx)
            text = f"Predicción dominante: {dom_labels} ({max_val*100:.1f}%)"
        else:
            text = "Predicción dominante: N/D"
        ax.text(
            0.5,
            -0.25,
            text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="#1f2933",
        )

    def _draw_gap_chart(self, ax, gap_stats: dict | None) -> None:
        ax.clear()
        ax.set_facecolor("#f5f6fb")
        if not gap_stats:
            ax.text(0.5, 0.5, "Carga un XML Gap-time para visualizar.", ha="center", va="center", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        times = np.asarray(gap_stats.get("time_s", []), dtype=float)
        amps = np.asarray(gap_stats.get("series", []), dtype=float)
        if not times.size or not amps.size or times.size != amps.size:
            ax.text(0.5, 0.5, "Gap-time sin serie asociada.", ha="center", va="center", fontsize=11)
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
        line_zona, = ax.plot(times, np.full_like(times, quiet_hi), "--", color="#43a047", linewidth=1.0)
        ax.axhline(quiet_lo, color="#43a047", linestyle="--", linewidth=1.0)
        ax.set_xlim(float(min(times.min(), 0.0)), float(times.max()))
        ylim_min = float(np.min(amps))
        ylim_max = float(np.max(amps))
        pad = max(1.0, (ylim_max - ylim_min) * 0.15)
        ax.set_ylim(ylim_min - pad, ylim_max + pad)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Magnitud (dBm)")
        ax.set_title("Gap-time (serie de 0.5 s)")
        ax.grid(True, alpha=0.25, linestyle="--")
        p50 = gap_stats.get("p50_ms")
        p5 = gap_stats.get("p5_ms")
        classification = gap_stats.get("classification") or {}
        classification_p5 = gap_stats.get("classification_p5") or {}
        top_y = ylim_max + pad * 0.6
        if p50 is not None:
            ax.text(
                times.min(),
                top_y,
                f"P50 = {p50:.1f} ms",
                fontsize=11,
                fontweight="bold",
                color="#ffffff",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=classification.get("color", "#4a4a4a"), edgecolor="none"),
            )
        if p5 is not None:
            ax.text(
                times.min() + (np.ptp(times) * 0.25),
                top_y,
                f"P5 = {p5:.1f} ms",
                fontsize=11,
                fontweight="bold",
                color="#ffffff",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=classification_p5.get("color", "#4a4a4a"), edgecolor="none"),
            )
        if classification:
            label = classification.get("level_name", "Gap")
            color = classification.get("color", "#546e7a")
            self._draw_status_tag(ax, label, 0.78, 0.94, color=color)
            action = classification.get("action") or classification.get("action_short")
            if action:
                ax.text(
                    0.5,
                    -0.20,
                    self._format_gap_action_text(action),
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="top",
                    color="#212121",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor=color),
                    clip_on=False,
                )
        handles, labels = ax.get_legend_handles_labels()
        legend_entries = []
        legend_labels = []
        for h, l in zip(handles, labels):
            if "Zona sin DP" in legend_labels:
                if l == "Zona sin DP":
                    continue
            legend_entries.append(h)
            legend_labels.append(l)
        if line_zona not in legend_entries:
            legend_entries.insert(0, line_zona)
            legend_labels.insert(0, "Zona sin DP")
        if legend_entries:
            ax.legend(legend_entries, legend_labels, loc="lower left", fontsize=8)

    @staticmethod
    def _format_gap_action_text(text: str | None) -> str:
        if not text:
            return ""
        txt = text.strip()
        if not txt:
            return ""
        inserted = False
        for token in (" Monitoreo", " Monitreo"):
            idx = txt.find(token)
            if idx > 0:
                txt = txt[:idx] + "\n" + txt[idx + 1 :]
                inserted = True
                break
        if not inserted and ". " in txt:
            parts = txt.split(". ", 1)
            if len(parts) == 2:
                txt = parts[0] + ".\n" + parts[1]
        return txt

    @staticmethod
    def _status_from_total(count: int | None) -> tuple[str, str]:
        if count is None:
            return ("N/D", "#757575")
        if count > 60000:
            return ("Crítico", "#B00000")
        if count > 25000:
            return ("Grave", "#FF8C00")
        return ("Aceptable", "#00B050")

    @staticmethod
    def _status_from_width(width: float | None) -> tuple[str, str]:
        if width is None:
            return ("N/D", "#757575")
        if width > 220:
            return ("Crítico", "#B00000")
        if width > 140:
            return ("Grave", "#FF8C00")
        return ("Aceptable", "#00B050")

    @staticmethod
    def _status_from_amp(p95_mean: float | None) -> tuple[str, str]:
        if p95_mean is None:
            return ("N/D", "#757575")
        if p95_mean > 85:
            return ("Crítico", "#B00000")
        if p95_mean > 70:
            return ("Grave", "#FF8C00")
        return ("Leve", "#1565c0")

    @staticmethod
    def _status_from_gap(p50: float | None) -> tuple[str, str]:
        if p50 is None:
            return ("Sin gap time", "#00B050")
        if p50 < 3.0:
            return ("Crítico", "#B00000")
        if p50 < 7.0:
            return ("Grave", "#FF8C00")
        return ("Leve", "#1565c0")

    def _draw_combined_overlay(self, ax, twin_attr: str, ph: np.ndarray, amp: np.ndarray, ang: dict, *, quantity: bool) -> None:
        ax.clear()
        if ph.size and amp.size:
            ax.scatter(ph, amp, s=4, alpha=0.25, color="#bdbdbd", label="Nubes S3")
        else:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            return
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Fase (°)")
        ax.set_ylabel("Amplitud")
        x = np.asarray(ang.get("phi_centers", []), dtype=float)
        twin = self._prepare_twin_axis(ax, twin_attr)
        if quantity:
            y_primary = np.asarray(ang.get("n_angpd_qty", []), dtype=float)
            y_secondary = np.asarray(ang.get("angpd_qty", []), dtype=float) * 100.0
            twin.set_ylabel("Densidad qty / Densidad(x100)")
            title = "Nubes + ANGPD qty"
            color_secondary = "#2ca02c"
        else:
            y_primary = np.asarray(ang.get("n_angpd", []), dtype=float)
            y_secondary = np.asarray(ang.get("angpd", []), dtype=float) * 100.0
            twin.set_ylabel("Densidad / Densidad(x100)")
            title = "Nubes + ANGPD"
            color_secondary = "#1f77b4"
        if x.size and (y_primary.size or y_secondary.size):
            if y_primary.size:
                twin.plot(x, y_primary, label="N-ANGPD" + (" qty" if quantity else ""), color="#d62728")
            if y_secondary.size:
                twin.plot(x, y_secondary, label=("ANGPD qty" if quantity else "ANGPD") + " x100", color=color_secondary)
            max_val = max(1.0, float(y_primary.max()) if y_primary.size else 0.0, float(y_secondary.max()) if y_secondary.size else 0.0)
            twin.set_ylim(0, max_val * 1.05)
            try:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = twin.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
            except Exception:
                pass
        ax.set_title(title)

    def _get_conclusion_insight(self, result: dict) -> tuple[str, dict]:
        try:
            gap_stats = None
            if self.chk_gap.isChecked():
                gap_stats = result.get("gap_stats")
                if not gap_stats:
                    gx = getattr(self, "_gap_xml_path", None)
                    if gx:
                        gap_stats = self._compute_gap(gx)
            metrics = compute_pd_metrics(result, gap_stats=gap_stats)
        except Exception:
            metrics = {}
        if not metrics.get("total_count"):
            return ("Sin datos alineados para analizar conclusiones.", {})
        summary = classify_pd(metrics)
        payload = {
            "metrics": metrics,
            "summary": summary,
            "phase_offset": result.get("phase_offset"),
            "filter": self._get_filter_label(),
            "gap": gap_stats or {},
        }
        def fmt(key, prec=1):
            val = metrics.get(key)
            if val is None:
                return "N/D"
            if isinstance(val, float):
                return f"{val:.{prec}f}"
            return str(val)
        lines = [
            f"Tipo PD estimado: {summary.get('pd_type','N/D')}",
            f"Ubicación probable: {summary.get('location','N/D')}",
            f"Riesgo: {summary.get('risk','N/D')}  |  Etapa: {summary.get('stage','N/D')}",
            f"Vida remanente estimada: {summary.get('life_years','N/D')} años (score {summary.get('life_score','N/D')})",
            "",
            f"Total de pulsos útiles: {metrics.get('total_count',0)}",
            f"p95 amplitud pos/neg: {fmt('amp_p95_pos')}/{fmt('amp_p95_neg')}  (ratio {fmt('amp_ratio')})",
            f"Anchura de fase: {fmt('phase_width')}°  |  Centro: {fmt('phase_center')}°",
            f"Número de picos de fase: {metrics.get('n_peaks','N/D')}",
            f"Relación N-ANGPD/ANGPD: {fmt('n_ang_ratio',2)}",
        ]
        gap_text = []
        if metrics.get("gap_p50") is not None:
            gap_text.append(f"Gap p50={metrics['gap_p50']:.2f} ms")
        if metrics.get("gap_p5") is not None:
            gap_text.append(f"p5={metrics['gap_p5']:.2f} ms")
        if gap_text:
            lines.append("Gap-time: " + ", ".join(gap_text))
        if gap_stats and isinstance(gap_stats, dict):
            classification = gap_stats.get("classification") or {}
            if classification:
                lines.append(f"Condición gap-time: {classification.get('level_name','N/D')} ({classification.get('label','')})")
                action = classification.get("action")
                if action:
                    lines.append("Recomendación gap: " + action)
        actions = summary.get("actions")
        if actions:
            lines.append("")
            lines.append("Acciones sugeridas:")
            lines.append(actions)
        return ("\n".join(lines), payload)

    def render_result(self, r: dict) -> None:
        view_mode = self.cmb_plot.currentText().strip().lower()
        is_conclusion = view_mode.startswith("conclusiones")
        is_ann_gap = view_mode.startswith("ann")
        is_gap_full = view_mode.startswith("gap-time")

        text, payload = self._get_conclusion_insight(r)
        self.last_conclusion_text = text
        self.last_conclusion_payload = payload
        stem_for_history = self.current_path.stem if self.current_path else None
        self._append_ann_history(r)
        self._load_ann_history(stem_for_history)

        self._set_conclusion_mode(is_conclusion)
        self._clear_twin_axis("ax_raw_twin")
        self._clear_twin_axis("ax_probs_twin")
        self.ax_gap_wide.set_visible(False)

        if is_conclusion:
            self._render_conclusions(r, payload)
            self.canvas.draw_idle()
            return
        if is_ann_gap:
            self._render_ann_gap_view(r, payload)
            self.canvas.draw_idle()
            return
        if is_gap_full:
            self._render_gap_time_full(r, payload)
            self.canvas.draw_idle()
            return

        # Vista estándar: mostrar PRPD crudo/histogramas/etc.
        self._set_conclusion_mode(False)
        self._restore_standard_axes()
        self.ax_raw.clear()
        try:
            ph0 = np.asarray(r.get("raw", {}).get("phase_deg", []), dtype=float)
            a0 = np.asarray(r.get("raw", {}).get("amplitude", []), dtype=float)
            if self.chk_hist2d.isChecked() and ph0.size and a0.size:
                H0, xe0, ye0 = np.histogram2d(ph0, a0, bins=[72,50], range=[[0,360],[0,100]])
                self.ax_raw.imshow(H0.T + 1e-9, origin='lower', aspect='auto', extent=[xe0[0], xe0[-1], ye0[0], ye0[-1]])
                # Overlay ruido (gris tenue)
                labels = np.asarray(r.get('labels', []))
                keep = np.asarray(r.get('keep_mask', []), dtype=bool)
                if labels.size and keep.size and ph0.size == labels.size and a0.size == labels.size:
                    raw_keep = keep & (labels >= 0)
                    noise_idx = ~raw_keep
                    self.ax_raw.scatter(ph0[noise_idx], a0[noise_idx], s=2, alpha=0.12, color='#888888')
            else:
                self.ax_raw.scatter(ph0, a0, s=3, alpha=0.4)
        except Exception:
            pass
        self.ax_raw.set_title("PRPD crudo"); self.ax_raw.set_xlim(0,360); self.ax_raw.set_xlabel("Fase (°)"); self.ax_raw.set_ylabel("Amplitud")

        # Alineado/filtrado con overlay de ruido
        self.ax_filtered.clear()
        ph_al = np.asarray(r.get("aligned", {}).get("phase_deg", []), dtype=float)
        amp_al = np.asarray(r.get("aligned", {}).get("amplitude", []), dtype=float)
        qty_vals = np.asarray(r.get("aligned", {}).get("quantity", []), dtype=float)
        quint_idx_raw = np.asarray(r.get("aligned", {}).get("qty_quintiles", []), dtype=int)
        quint_idx = quint_idx_raw if quint_idx_raw.size == ph_al.size else self._equal_frequency_bucket(qty_vals, groups=5)
        qty_deciles = np.asarray(r.get("aligned", {}).get("qty_deciles", []), dtype=int)
        if qty_deciles.size == quint_idx.size:
            self.last_qty_deciles = qty_deciles
        else:
            self.last_qty_deciles = None
        quint_colors = {
            1: "#0066CC",  # Azul
            2: "#009900",  # Verde
            3: "#FFCC00",  # Amarillo
            4: "#FF8000",  # Naranja
            5: "#CC0000",  # Rojo
        }
        try:
            use_hist2d = self.chk_hist2d.isChecked() and ph_al.size and amp_al.size
            if use_hist2d:
                H2, xe2, ye2 = np.histogram2d(ph_al, amp_al, bins=[72,50], range=[[0,360],[0,100]])
                self.ax_filtered.imshow(H2.T + 1e-9, origin='lower', aspect='auto', extent=[xe2[0], xe2[-1], ye2[0], ye2[-1]])
            colored = False
            if quint_idx.size == ph_al.size and ph_al.size:
                for q_idx in range(1, 6):
                    mask = (quint_idx == q_idx)
                    if not np.any(mask):
                        continue
                    colored = True
                    self.ax_filtered.scatter(
                        ph_al[mask], amp_al[mask], s=6,
                        color=quint_colors.get(q_idx, "#999999"),
                        alpha=0.85, label=f"Q{q_idx}"
                    )
                if colored:
                    try:
                        self.ax_filtered.legend(loc="upper right", fontsize=8)
                    except Exception:
                        pass
            if not colored and not use_hist2d:
                self.ax_filtered.scatter(ph_al, amp_al, s=3, alpha=0.85)
            # Overlay de ruido gris tenue cuando hay heatmap
            labels = np.asarray(r.get('labels', []))
            keep = np.asarray(r.get('keep_mask', []), dtype=bool)
            raw = r.get('raw', {})
            ph_raw = np.asarray(raw.get('phase_deg', []), dtype=float)
            amp_raw = np.asarray(raw.get('amplitude', []), dtype=float)
            if labels.size and keep.size and ph_raw.size == labels.size and amp_raw.size == labels.size:
                raw_keep = keep & (labels >= 0)
                noise_idx = ~raw_keep
                self.ax_filtered.scatter(ph_raw[noise_idx], amp_raw[noise_idx], s=2, alpha=0.15, color='#888888')
        except Exception:
            pass
        self.ax_filtered.set_title(f"Alineado/filtrado (offset={r.get('phase_offset',0)}°)")
        self.ax_filtered.set_xlim(0,360); self.ax_filtered.set_xlabel("Fase (°)"); self.ax_filtered.set_ylabel("Amplitud")

        # Fijar escala Y de 0..100 para la vista alineada/filtrada
        try:
            self.ax_filtered.set_ylim(0, 100)
        except Exception:
            pass
        # Panel inferior izquierdo: Probabilidades / ANGPD / Nubes (S3/S4/S5)
        self.ax_probs.clear()
        # Determinar modo de vista según el texto del combo. Convertir a minúsculas
        view_mode = self.cmb_plot.currentText().strip().lower()
        classes = ["cavidad","superficial","corona","flotante"]
        proba_dict = r.get("probs", {})

        if view_mode.startswith("histogramas"):
            # En la vista "Histogramas" se dibujan los histogramas por semiciclo y
            # se muestran las curvas ANGPD y N‑ANGPD (densidad) en el panel superior izquierdo.
            try:
                # Histograma de amplitud (panel superior derecho) y fase (panel inferior derecho)
                self._draw_histograms_semiciclo(r)
            except Exception:
                pass

            # Panel superior izquierdo: mostrar curvas ANGPD y N‑ANGPD (densidad, no quantity).
            ang = r.get("angpd", {})
            x = np.asarray(ang.get("phi_centers", []), dtype=float)
            y_ang = np.asarray(ang.get("angpd", []), dtype=float) * 100.0
            y_n   = np.asarray(ang.get("n_angpd", []), dtype=float)
            self.ax_raw.clear()
            if x.size and (y_ang.size or y_n.size):
                twin = self._prepare_twin_axis(self.ax_raw, "ax_raw_twin")
                if y_n.size:
                    self.ax_raw.plot(x, y_n, label="N-ANGPD (max=1)", color="#d62728", linewidth=2.0)
                    try:
                        self.ax_raw.fill_between(x, 0, y_n, color="#d62728", alpha=0.18, linewidth=0)
                    except Exception:
                        pass
                    self.ax_raw.set_ylim(0, max(1.0, float(y_n.max()) * 1.2))
                if y_ang.size:
                    twin.plot(x, y_ang, label="ANGPD (sum=1) x100", color="#1f77b4")
                    twin.set_ylim(0, max(1.0, float(y_ang.max()) * 1.05))
                self.ax_raw.set_xlim(0, 360)
                self.ax_raw.set_xlabel("Fase (°)")
                self.ax_raw.set_ylabel("N-ANGPD (max=1)")
                twin.set_ylabel("ANGPD (sum=1) x100")
                self.ax_raw.set_title("ANGPD / N-ANGPD")
                try:
                    handles, labels = self.ax_raw.get_legend_handles_labels()
                    h2, l2 = twin.get_legend_handles_labels()
                    self.ax_raw.legend(handles + h2, labels + l2, loc="upper right", fontsize=8)
                except Exception:
                    pass
            else:
                self.ax_raw.text(0.5, 0.5, "Sin datos", ha="center", va="center")
                self.ax_raw.set_xticks([])
                self.ax_raw.set_yticks([])

            # Panel inferior izquierdo: graficar ANGPD/N‑ANGPD ponderado por quantity como antes
            ang = r.get("angpd", {})
            x = np.asarray(ang.get("phi_centers", []), dtype=float)
            y_q_ang = np.asarray(ang.get("angpd_qty", []), dtype=float) * 100.0
            y_q_n   = np.asarray(ang.get("n_angpd_qty", []), dtype=float)
            self.ax_probs.clear()
            if x.size and (y_q_ang.size or y_q_n.size):
                twin_q = self._prepare_twin_axis(self.ax_probs, "ax_probs_twin")
                if y_q_n.size:
                    self.ax_probs.plot(x, y_q_n, label="N-ANGPD qty (max=1)", color="#d62728", linewidth=2.0)
                    try:
                        self.ax_probs.fill_between(x, 0, y_q_n, color="#d62728", alpha=0.18, linewidth=0)
                    except Exception:
                        pass
                    self.ax_probs.set_ylim(0, max(1.0, float(y_q_n.max()) * 1.2))
                if y_q_ang.size:
                    twin_q.plot(x, y_q_ang, label="ANGPD qty (sum=1) x100", color="#2ca02c")
                    twin_q.set_ylim(0, max(1.0, float(y_q_ang.max()) * 1.05))
                self.ax_probs.set_xlim(0, 360)
                self.ax_probs.set_xlabel("Fase (°)")
                self.ax_probs.set_ylabel("N-ANGPD qty (max=1)")
                twin_q.set_ylabel("ANGPD qty (sum=1) x100")
                self.ax_probs.set_title("ANGPD / N-ANGPD (quantity)")
                try:
                    handles, labels = self.ax_probs.get_legend_handles_labels()
                    h2, l2 = twin_q.get_legend_handles_labels()
                    self.ax_probs.legend(handles + h2, labels + l2, loc="upper right", fontsize=8)
                except Exception:
                    pass
            else:
                self.ax_probs.text(0.5, 0.5, "Sin quantity en XML", ha="center", va="center")
                self.ax_probs.set_xticks([])
                self.ax_probs.set_yticks([])
        elif view_mode.startswith("combinada"):
            try:
                self._draw_histograms_semiciclo(r)
            except Exception:
                pass
            self._draw_combined_panels(r, ph_al, amp_al)
        elif view_mode.startswith("nubes"):
            # Preparar datos y clusters S3/S4/S5
            ph = ph_al; amp = amp_al
            variants, combined_s3, clouds_s4, clouds_s5 = self._build_cluster_variants(ph, amp)
            filter_label = self._get_filter_label().lower()
            variant_key = "S2 Strong" if "strong" in filter_label else "S1 Weak"
            selected_s3 = variants.get(variant_key, combined_s3)

            mode = view_mode.replace("nubes", "").strip()
            if mode.startswith("(s4)"):
                clouds = clouds_s4
                title = "Nubes combinadas (S4)"
                # En S4 también colorear los puntos según su cluster para ver densidad
                color_points = True
                include_k = True
            elif mode.startswith("(s5)"):
                clouds = clouds_s5
                title = "Nubes dominantes (S5)"
                # En S5 colorear los puntos para distinguir mejor los clusters dominantes
                color_points = True
                include_k = True
            else:
                clouds = selected_s3
                title = f"Nubes crudas (S3) — {variant_key}"
                color_points = True
                include_k = False

            if ph.size and amp.size and clouds:
                # Utilizar la función estática para trazar clusters con un estilo coherente
                PRPDWindow._plot_clusters_on_ax(self.ax_probs, ph, amp, clouds,
                                                 title=title,
                                                 color_points=color_points,
                                                 include_k=include_k,
                                                 max_labels=10)
            else:
                self.ax_probs.text(0.5, 0.5, "Sin nubes", ha="center", va="center")
        else:
            self._render_conclusions(r)

        # Mantener 0..100 en vistas de Nubes (S3/S4/S5)
        try:
            if view_mode.startswith("nubes"):
                self.ax_probs.set_ylim(0, 100)
        except Exception:
            pass

        try:
            self.canvas.figure.tight_layout()
        except Exception:
            pass


        self.canvas.draw_idle()
        # La función termina aquí. No se ejecuta procesamiento adicional ni exportes desde render_result.
        return


    def on_btnLoadANN_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, 'Seleccionar modelo ANN', '', 'Modelos (*.pkl *.joblib)')
        if not path:
            return
        ok = False
        if _load_ann_model is not None:
            try:
                self.ann_model, self.ann_classes = _load_ann_model(path)
                ok = True
            except Exception:
                ok = False
        if (not ok) and hasattr(self, 'ann'):
            try:
                self.ann.load_model(path)
                ok = True
            except Exception:
                ok = False
        if ok:
            QMessageBox.information(self, 'ANN', f'Modelo cargado.\n{path}')
            if self.last_result:
                self.render_result(self.last_result)
        else:
            QMessageBox.warning(self, 'ANN', f'No se pudo cargar el modelo\n{path}')

    def on_export_all_clicked(self) -> None:
        """Exporta S3/S4/S5 y histogramas para los filtros activos (weak/strong)
        y ANGPD con columnas qty al CSV."""
        if not self.current_path or not self.last_result:
            QMessageBox.information(self, "Exportar", "Primero carga y procesa un archivo.")
            return
        try:
            current_profile = self._collect_current_profile()
            if not self.last_run_profile or self.last_run_profile != current_profile:
                QMessageBox.warning(
                    self,
                    "Exportar",
                    "Los parámetros actuales no coinciden con el último procesamiento.\n"
                    "Pulsa 'Procesar' antes de exportar para garantizar coherencia.",
                )
                return
            outdir = self._get_output_dir()
            profile_tag = self._build_profile_tag(current_profile)
            session_dir = self._create_session_dir(outdir, profile_tag)
            self._export_session_outputs(self.last_result, session_dir, profile_tag)
            QMessageBox.information(
                self,
                "Exportar",
                f"Resultados guardados en:\n{session_dir}",
            )
            self._open_folder(session_dir)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(self, "Exportar", f"Error en exportación total:\n{e}")

    # -------------------------------------------------------------------------
    # Funciones auxiliares de exportación
    #
    def _export_session_outputs(self, result: dict, session_dir: Path, profile_tag: str) -> None:
        session_dir = Path(session_dir)
        stem = self.current_path.stem if self.current_path else "session"
        raw = result.get("raw", {})
        aligned = result.get("aligned", {})
        ph_raw = np.asarray(raw.get("phase_deg", []), dtype=float)
        amp_raw = np.asarray(raw.get("amplitude", []), dtype=float)
        ph_al = np.asarray(aligned.get("phase_deg", []), dtype=float)
        amp_al = np.asarray(aligned.get("amplitude", []), dtype=float)
        quint = np.asarray(aligned.get("qty_quintiles", []), dtype=int)
        self._save_prpd_plot(
            ph_raw,
            amp_raw,
            session_dir / f"{stem}_raw_{profile_tag}.png",
            title="PRPD crudo",
            hist2d=False,
        )
        self._save_prpd_plot(
            ph_raw,
            amp_raw,
            session_dir / f"{stem}_raw_hist2d_{profile_tag}.png",
            title="PRPD crudo (Hist2D)",
            hist2d=True,
        )
        self._save_prpd_plot(
            ph_al,
            amp_al,
            session_dir / f"{stem}_aligned_{profile_tag}.png",
            title="Alineado / filtrado",
            hist2d=False,
            quintiles=quint,
        )
        self._save_prpd_plot(
            ph_al,
            amp_al,
            session_dir / f"{stem}_aligned_hist2d_{profile_tag}.png",
            title="Alineado / filtrado (Hist2D)",
            hist2d=True,
        )
        variants, combined_s3, clouds_s4, clouds_s5 = self._build_cluster_variants(ph_al, amp_al)
        self._save_cluster_images(
            ph_al,
            amp_al,
            variants,
            clouds_s4,
            clouds_s5,
            session_dir,
            stem,
            profile_tag,
            subfolder=None,
        )
        self._save_histograms(ph_al, amp_al, session_dir, stem, profile_tag, subfolder=None)
        ang = result.get("angpd", {})
        self._save_angpd_csv(ang, session_dir, stem, profile_tag, subfolder=None)
        self._save_angpd_plots(ang, session_dir, stem, profile_tag)
        self._save_combined_overlays(ph_al, amp_al, ang, session_dir / f"{stem}_combined_{profile_tag}.png")
        self._save_conclusions(result, session_dir, stem, profile_tag)

    def _save_prpd_plot(self, ph: np.ndarray, amp: np.ndarray, path: Path, *, title: str, hist2d: bool, quintiles: np.ndarray | None = None) -> None:
        import matplotlib.pyplot as _plt
        fig, ax = _plt.subplots(figsize=(6, 4), dpi=130)
        if ph.size and amp.size:
            if hist2d:
                H, xedges, yedges = np.histogram2d(ph, amp, bins=[72, 50], range=[[0, 360], [0, 100]])
                ax.imshow(H.T + 1e-9, origin="lower", aspect="auto", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="viridis")
            else:
                if quintiles is not None and quintiles.size == ph.size:
                    colors = {
                        1: "#0066CC",
                        2: "#009900",
                        3: "#FFCC00",
                        4: "#FF8000",
                        5: "#CC0000",
                    }
                    used = False
                    for q in range(1, 6):
                        mask = (quintiles == q)
                        if np.any(mask):
                            used = True
                            ax.scatter(ph[mask], amp[mask], s=8, color=colors.get(q, "#999999"), alpha=0.9, label=f"Q{q}")
                    if used:
                        ax.legend(loc="upper right", fontsize=8)
                else:
                    ax.scatter(ph, amp, s=6, alpha=0.7)
        else:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Fase (°)")
        ax.set_ylabel("Amplitud")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        _plt.close(fig)

    def _save_angpd_plots(self, ang: dict, session_dir: Path, stem: str, suffix: str) -> None:
        import matplotlib.pyplot as _plt
        x = np.asarray(ang.get("phi_centers", []), dtype=float)
        if not x.size:
            return
        y_n = np.asarray(ang.get("n_angpd", []), dtype=float)
        y_ang = np.asarray(ang.get("angpd", []), dtype=float) * 100.0
        y_q_n = np.asarray(ang.get("n_angpd_qty", []), dtype=float)
        y_q_ang = np.asarray(ang.get("angpd_qty", []), dtype=float) * 100.0
        fig = _plt.figure(figsize=(10, 5), dpi=130)
        gs = fig.add_gridspec(2, 1, hspace=0.35)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        self._plot_angpd_curves(ax1, x, y_n, y_ang, quantity=False, title="ANGPD / N-ANGPD")
        self._plot_angpd_curves(ax2, x, y_q_n, y_q_ang, quantity=True, title="ANGPD / N-ANGPD (quantity)")
        fig.tight_layout()
        fig.savefig(session_dir / f"{stem}_angpd_{suffix}.png", bbox_inches="tight")
        _plt.close(fig)

    def _plot_angpd_curves(self, ax, x: np.ndarray, y_primary: np.ndarray, y_secondary: np.ndarray, *, quantity: bool, title: str) -> None:
        twin = ax.twinx()
        if quantity:
            ax.set_ylabel("N-ANGPD qty (max=1)")
            twin.set_ylabel("ANGPD qty (sum=1) x100")
            sec_label = "ANGPD qty (sum=1) x100"
            primary_label = "N-ANGPD qty (max=1)"
            color_secondary = "#2ca02c"
        else:
            ax.set_ylabel("N-ANGPD (max=1)")
            twin.set_ylabel("ANGPD (sum=1) x100")
            sec_label = "ANGPD (sum=1) x100"
            primary_label = "N-ANGPD (max=1)"
            color_secondary = "#1f77b4"
        ax.fill_between(x, 0, y_primary, color="#f4c2c2", alpha=0.12)
        ax.plot(x, y_primary, color="#d62728", linewidth=2.0, label=primary_label)
        twin.plot(x, y_secondary, color=color_secondary, label=sec_label)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, max(1.0, float(y_primary.max()) * 1.2 if y_primary.size else 1.0))
        twin.set_ylim(0, max(1.0, float(y_secondary.max()) * 1.2 if y_secondary.size else 1.0))
        ax.set_xlabel("Fase (°)")
        ax.set_title(title)
        try:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = twin.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
        except Exception:
            pass

    def _save_combined_overlays(self, ph: np.ndarray, amp: np.ndarray, ang: dict, path: Path) -> None:
        import matplotlib.pyplot as _plt
        if not ph.size or not amp.size:
            return
        x = np.asarray(ang.get("phi_centers", []), dtype=float)
        if not x.size:
            return
        fig, axes = _plt.subplots(2, 1, figsize=(7, 6), dpi=130, sharex=True)
        self._plot_combined_panel(axes[0], ph, amp, x, ang, quantity=False, title="Nubes + ANGPD")
        self._plot_combined_panel(axes[1], ph, amp, x, ang, quantity=True, title="Nubes + ANGPD qty")
        axes[1].set_xlabel("Fase (°)")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        _plt.close(fig)

    def _save_conclusions(self, result: dict, session_dir: Path, stem: str, suffix: str) -> None:
        try:
            text, payload = self._get_conclusion_insight(result)
            if not text:
                return
            txt_path = session_dir / f"{stem}_conclusiones_{suffix}.txt"
            txt_path.write_text(text, encoding="utf-8")
            if payload:
                json_path = session_dir / f"{stem}_conclusiones_{suffix}.json"
                json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            traceback.print_exc()

    def _plot_combined_panel(self, ax, ph, amp, x, ang, *, quantity: bool, title: str) -> None:
        x = np.asarray(ang.get("phi_centers", []), dtype=float)
        if not ph.size or not amp.size or not x.size:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            return
        ax.scatter(ph, amp, s=4, alpha=0.25, color="#bfbfbf", label="Nubes S3")
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Fase (°)")
        ax.set_ylabel("Amplitud")
        twin = ax.twinx()
        if quantity:
            y_primary = np.asarray(ang.get("n_angpd_qty", []), dtype=float)
            y_secondary = np.asarray(ang.get("angpd_qty", []), dtype=float) * 100.0
            primary_label = "N-ANGPD qty (max=1)"
            secondary_label = "ANGPD qty"
            color_secondary = "#2ca02c"
        else:
            y_primary = np.asarray(ang.get("n_angpd", []), dtype=float)
            y_secondary = np.asarray(ang.get("angpd", []), dtype=float) * 100.0
            primary_label = "N-ANGPD (max=1)"
            secondary_label = "ANGPD"
            color_secondary = "#1f77b4"
        twin.fill_between(x, 0, y_primary, color="#f4c2c2", alpha=0.25, zorder=1)
        twin.plot(x, y_primary, color="#d62728", linewidth=2.0, label=primary_label, zorder=2)
        twin.plot(x, y_secondary, color=color_secondary, linewidth=2.0, label=f"{secondary_label} x100", zorder=2)
        twin.set_ylim(0, max(1.0, float(y_secondary.max()) * 1.2 if y_secondary.size else 1.0))
        ax.set_title(title)
        try:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = twin.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
        except Exception:
            pass

    def _save_angpd_csv(self, ang: dict, outdir: Path, stem: str, suffix: str, subfolder: str | None = "reports") -> None:
        """Escribe los datos ANGPD con columns qty en un CSV en la carpeta reports."""
        try:
            x = np.asarray(ang.get("phi_centers", []), dtype=float)
            y1 = np.asarray(ang.get("angpd", []), dtype=float)
            y2 = np.asarray(ang.get("n_angpd", []), dtype=float)
            y3 = np.asarray(ang.get("angpd_qty", []), dtype=float)
            y4 = np.asarray(ang.get("n_angpd_qty", []), dtype=float)
            if x.size:
                out_reports = self._resolve_export_dir(outdir, subfolder)
                file_path = out_reports / f"{stem}_angpd_{suffix}.csv"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('phi_deg,angpd,n_angpd,angpd_qty,n_angpd_qty\n')
                    for i in range(x.size):
                        a1 = float(y1[i]) if i < y1.size else 0.0
                        a2 = float(y2[i]) if i < y2.size else 0.0
                        a3 = float(y3[i]) if i < y3.size else 0.0
                        a4 = float(y4[i]) if i < y4.size else 0.0
                        f.write(f"{float(x[i]):.3f},{a1:.6f},{a2:.6f},{a3:.6f},{a4:.6f}\n")
        except Exception:
            # silencioso para evitar detener la exportación completa
            traceback.print_exc()

    def _save_histograms(self, ph: np.ndarray, amp: np.ndarray, outdir: Path, stem: str, suffix: str, subfolder: str | None = "reports") -> None:
        """Guarda los histogramas de amplitud y fase en formato PNG."""
        try:
            if not (ph.size and amp.size):
                return
            import matplotlib.pyplot as _plt
            import numpy as _np
            N = 16
            phi = ph % 360.0
            pos = (phi < 180.0)
            neg = ~pos
            xi = _np.arange(1, N + 1)
            # Histogramas de amplitud
            a_pos, _ = _np.histogram(amp[pos], bins=N, range=(0.0, 100.0))
            a_neg, _ = _np.histogram(amp[neg], bins=N, range=(0.0, 100.0))
            Ha_pos = _np.log10(1.0 + a_pos.astype(float))
            Ha_neg = _np.log10(1.0 + a_neg.astype(float))
            m = float(max(Ha_pos.max() if Ha_pos.size else 0.0, Ha_neg.max() if Ha_neg.size else 0.0, 1.0))
            Ha_pos /= m
            Ha_neg /= m
            fig, ax = _plt.subplots(figsize=(5, 3), dpi=120)
            ax.plot(xi, Ha_pos, '-o', label='H_amp+')
            ax.plot(xi, Ha_neg, '-o', label='H_amp-')
            ax.set_xlabel('Indice (N=16)')
            ax.set_ylabel('H_amp (norm)')
            ax.set_title('Histograma de Amplitud')
            ax.legend()
            out_reports = self._resolve_export_dir(outdir, subfolder)
            fig.tight_layout()
            fig.savefig(out_reports / f"{stem}_hist_amp_{suffix}.png", bbox_inches='tight')
            _plt.close(fig)
            # Histogramas de fase
            phi_pos = phi[pos]
            phi_neg = (phi[neg] - 180.0)
            p_pos, _ = _np.histogram(phi_pos, bins=N, range=(0.0, 180.0))
            p_neg, _ = _np.histogram(phi_neg, bins=N, range=(0.0, 180.0))
            Hp_pos = _np.log10(1.0 + p_pos.astype(float))
            Hp_neg = _np.log10(1.0 + p_neg.astype(float))
            m2 = float(max(Hp_pos.max() if Hp_pos.size else 0.0, Hp_neg.max() if Hp_neg.size else 0.0, 1.0))
            Hp_pos /= m2
            Hp_neg /= m2
            fig, ax = _plt.subplots(figsize=(5, 3), dpi=120)
            ax.plot(xi, Hp_pos, '-o', label='H_ph+')
            ax.plot(xi, Hp_neg, '-o', label='H_ph-')
            ax.set_xlabel('Indice (N=16)')
            ax.set_ylabel('H_ph (norm)')
            ax.set_title('Histograma de Fase')
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_reports / f"{stem}_hist_phase_{suffix}.png", bbox_inches='tight')
            _plt.close(fig)
        except Exception:
            traceback.print_exc()

    def _save_cluster_images(self, ph: np.ndarray, amp: np.ndarray,
                             variants: dict[str, list[dict]], clouds_s4: list[dict], clouds_s5: list[dict],
                             outdir: Path, stem: str, suffix: str, subfolder: str | None = "reports") -> None:
        """Guarda las imágenes de nubes S3, S4 y S5 con estilo consistente."""
        try:
            import matplotlib.pyplot as _plt
            out_reports = self._resolve_export_dir(outdir, subfolder)
            for name, clouds_s3 in variants.items():
                fig, ax = _plt.subplots(figsize=(5, 4), dpi=120)
                PRPDWindow._plot_clusters_on_ax(ax, ph, amp, clouds_s3,
                                                title=f'Nubes crudas (S3) — {name}',
                                                color_points=True,
                                                include_k=False,
                                                max_labels=10)
                fig.tight_layout()
                tag = f"{stem}_s3_{name.replace(' ', '_')}_{suffix}.png"
                fig.savefig(out_reports / tag, bbox_inches='tight')
                _plt.close(fig)
            # Nubes S4 (puntos grises, centros coloreados, mostrar k)
            fig, ax = _plt.subplots(figsize=(5, 4), dpi=120)
            PRPDWindow._plot_clusters_on_ax(ax, ph, amp, clouds_s4,
                                            title='Nubes combinadas (S4)',
                                            color_points=True,
                                            include_k=True,
                                            max_labels=10)
            fig.tight_layout()
            fig.savefig(out_reports / f"{stem}_s4_{suffix}.png", bbox_inches='tight')
            _plt.close(fig)
            # Nubes S5 (puntos grises, centros coloreados, mostrar k)
            fig, ax = _plt.subplots(figsize=(5, 4), dpi=120)
            PRPDWindow._plot_clusters_on_ax(ax, ph, amp, clouds_s5,
                                            title='Nubes dominantes (S5)',
                                            color_points=True,
                                            include_k=True,
                                            max_labels=10)
            fig.tight_layout()
            fig.savefig(out_reports / f"{stem}_s5_{suffix}.png", bbox_inches='tight')
            _plt.close(fig)
        except Exception:
            traceback.print_exc()


    def on_btnBatch_clicked(self) -> None:
        root = QFileDialog.getExistingDirectory(self, "Selecciona raíz (p.ej. EntrenamientoPatron)")
        if not root:
            return
        try:
            # Ejecutar procesamiento de carpeta y obtener texto resumen y directorio de salida
            text, out_dir = self._run_folder_batch(Path(root))
            # Mostrar resumen en un cuadro de mensaje informativo
            QMessageBox.information(self, "Resumen Batch", text)
            # Informar la ubicación de salida
            QMessageBox.information(self, "Batch", f"Salida:\n{out_dir}")
        except RuntimeError as e:
            # Errores esperados como ausencia de XML
            QMessageBox.warning(self, "Batch", str(e))
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(self, "Batch", f"Error procesando carpeta:\n{e}")

    def _run_folder_batch(self, root_dir: Path):
        from utils_io import ensure_out_dirs, time_tag, write_json
        root_dir = Path(root_dir)
        xmls = list(root_dir.rglob('*.xml'))
        if not xmls:
            raise RuntimeError("No se encontraron .xml en la carpeta seleccionada.")
        out_dir = Path('out') / 'batch' / f"{root_dir.name}_{time_tag()}"
        ensure_out_dirs(out_dir)
        lines = [f"Carpeta: {root_dir}", f"Total XML: {len(xmls)}", ""]
        summary = []
        # Utilizar el filtro actual y los offsets de fase definidos en la interfaz
        fast_mode = False
        mask_ranges = self._get_phase_mask_ranges()
        pixel_deciles = self._get_pixel_deciles_selection()
        qty_deciles = self._get_qty_deciles_selection()
        # Guardar resultados completos por cada archivo y filtro
        for i, xp in enumerate(xmls, 1):
            try:
                fstem = Path(xp).stem
                # Calcular offsets de fase seleccionados para la carpeta (mismo para todos los archivos)
                force_offsets = self._get_force_offsets()
                # Iterar sobre filtros como en on_export_all_clicked
                filters = [
                    ("weak", "S1 Weak"),
                    ("strong", "S2 Strong"),
                ]
                # Procesar cada filtro y guardar resultados/exports
                for suf, flabel in filters:
                    try:
                        res = core.process_prpd(
                            path=Path(xp),
                            out_root=out_dir,
                            force_phase_offsets=force_offsets,
                        fast_mode=fast_mode,
                        filter_level=flabel,
                        phase_mask=mask_ranges,
                        pixel_deciles_keep=pixel_deciles,
                        qty_deciles_keep=qty_deciles,
                    )
                        # Guardar resultados básicos para el summary
                        if suf == "weak":
                            line = f"[{i:03d}/{len(xmls):03d}] {xp.name} -> {res.get('predicted','?')} | sev={res.get('severity_score',0):.1f} | clusters={res.get('n_clusters',0)} | ruido={'Sí' if res.get('has_noise') else 'No'}"
                            lines.append(line)
                            summary.append({
                                'file': str(xp),
                                'predicted': res.get('predicted'),
                                'severity': res.get('severity_score'),
                                'n_clusters': res.get('n_clusters'),
                                'has_noise': res.get('has_noise'),
                                'phase_offset': res.get('phase_offset'),
                            })
                        # Exportar CSV, histogramas y nubes para cada filtro
                        # Guardar ANGPD con qty
                        self._save_angpd_csv(res.get('angpd', {}), out_dir, fstem, f"{suf}")
                        # Obtener datos alineados
                        ph = np.asarray(res.get('aligned', {}).get('phase_deg', []), dtype=float)
                        amp = np.asarray(res.get('aligned', {}).get('amplitude', []), dtype=float)
                        # Calcular nubes
                        variants, combined_s3, clouds_s4, clouds_s5 = self._build_cluster_variants(ph, amp)
                        # Guardar imágenes
                        self._save_cluster_images(ph, amp, variants, clouds_s4, clouds_s5, out_dir, fstem, suf)
                        # Guardar histogramas
                        self._save_histograms(ph, amp, out_dir, fstem, suf)
                    except Exception as filter_exc:
                        traceback.print_exc()
                        lines.append(f"[{i:03d}/{len(xmls):03d}] {xp.name} ({suf}) -> ERROR: {filter_exc}")
                # Separador entre archivos en el summary
                lines.append("")
            except Exception as e:
                traceback.print_exc()
                lines.append(f"[{i:03d}/{len(xmls):03d}] {xp.name} -> ERROR: {e}")
                summary.append({'file': str(xp), 'error': str(e)})
        text = "\n".join(lines)
        write_json(out_dir / 'batch_summary.json', {'root': str(root_dir), 'results': summary})
        (out_dir / 'batch_summary.txt').write_text(text, encoding='utf-8')
        return text, str(out_dir)


def compute_pd_metrics(result: dict, gap_stats: dict | None = None) -> dict:
    aligned = result.get("aligned", {}) or {}
    ph = np.asarray(aligned.get("phase_deg", []), dtype=float)
    amp = np.asarray(aligned.get("amplitude", []), dtype=float)
    sem = np.asarray(aligned.get("semicycle", []), dtype=int)
    if sem.size != amp.size and ph.size == amp.size:
        sem = (ph % 360.0 < 180.0).astype(int)
    amp_pos = amp[sem == 1] if amp.size and sem.size == amp.size else amp[ph % 360.0 < 180.0]
    amp_neg = amp[sem == 0] if amp.size and sem.size == amp.size else amp[ph % 360.0 >= 180.0]
    def _safe_p95(arr):
        return float(np.percentile(arr, 95)) if arr.size else 0.0
    p95_pos = _safe_p95(np.abs(amp_pos)) if amp_pos.size else 0.0
    p95_neg = _safe_p95(np.abs(amp_neg)) if amp_neg.size else 0.0
    amp_ratio = float(p95_pos / p95_neg) if p95_neg > 1e-6 else (float("inf") if p95_pos > 0 else 0.0)
    p95_mean = 0.5 * (p95_pos + p95_neg)
    ang = result.get("angpd", {}) or {}
    angpd = np.asarray(ang.get("angpd", []), dtype=float)
    n_angpd = np.asarray(ang.get("n_angpd", []), dtype=float)
    total_ang = float(np.sum(angpd)) if angpd.size else 0.0
    n_ang_ratio = float(np.sum(n_angpd) / total_ang) if total_ang > 0 else 0.0
    phase_center = _circ_mean_deg_array(ph)
    phase_width = _circ_width_deg_array(ph)
    hist, _ = np.histogram(ph, bins=16, range=(0, 360)) if ph.size else (np.array([]), None)
    peaks = int(np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))) if hist.size >= 3 else 0
    total_count = int(ph.size)
    snr = float(p95_mean / (float(np.mean(np.abs(amp))) + 1e-6)) if amp.size else 0.0
    gap_p50 = gap_stats.get("p50_ms") if isinstance(gap_stats, dict) else None
    gap_p5 = gap_stats.get("p5_ms") if isinstance(gap_stats, dict) else None
    return {
        "amp_p95_pos": round(p95_pos, 2),
        "amp_p95_neg": round(p95_neg, 2),
        "amp_ratio": round(amp_ratio, 2) if np.isfinite(amp_ratio) else None,
        "p95_mean": round(p95_mean, 2),
        "n_ang_ratio": round(n_ang_ratio, 3),
        "phase_center": None if phase_center is None else round(phase_center, 1),
        "phase_width": None if phase_width is None else round(phase_width, 1),
        "n_peaks": peaks,
        "total_count": total_count,
        "snr": round(snr, 3),
        "gap_p50": gap_p50,
        "gap_p5": gap_p5,
    }


def classify_pd(metrics: dict) -> dict:
    p95_mean = float(metrics.get("p95_mean", 0.0) or 0.0)
    amp_ratio = float(metrics.get("amp_ratio", 0.0) or 0.0)
    n_ang = float(metrics.get("n_ang_ratio", 0.0) or 0.0)
    n_peaks = int(metrics.get("n_peaks", 0) or 0)
    gap_p50 = metrics.get("gap_p50")
    gap_p5 = metrics.get("gap_p5")
    def _gap_weight(value, strong=False):
        if value is None:
            return 0.0
        if value < 3.0:
            return 2.0 if strong else 1.0
        if value < 7.0:
            return 1.0 if strong else 0.5
        return 0.0
    gap_penalty = _gap_weight(gap_p50, strong=True) + _gap_weight(gap_p5, strong=False)
    if n_peaks >= 3 or n_ang > 0.35:
        pd_type = "Superficial / Tracking"
        location = "Superficie de aislamiento e interfaces"
    elif amp_ratio < 0.8:
        pd_type = "Delaminación (DCI)"
        location = "Interfaz conductor-aislante"
    elif amp_ratio > 1.2 and n_peaks <= 1:
        pd_type = "Corona"
        location = "Puntos vivos expuestos al aire"
    else:
        pd_type = "Cavidad interna"
        location = "Volumen interno del aislamiento"
    if p95_mean > 85:
        risk = "Crítico"; stage = "Avanzada"; life = 0.7; actions = "Detener equipo y planear reparación inmediata."
    elif p95_mean > 75:
        risk = "Alto"; stage = "Evolucionada"; life = 2.0; actions = "Programar inspección off-line y análisis detallado."
    elif p95_mean > 60:
        risk = "Moderado"; stage = "Moderada"; life = 4.0; actions = "Monitorear con mayor frecuencia y revisar tendencias."
    else:
        risk = "Bajo"; stage = "Incipiente"; life = 7.0; actions = "Continuar con vigilancia rutinaria."
    risk_levels = ["Bajo", "Moderado", "Alto", "Crítico"]
    stage_levels = ["Incipiente", "Moderada", "Evolucionada", "Avanzada"]
    risk_idx = risk_levels.index(risk)
    stage_idx = stage_levels.index(stage)
    if gap_penalty >= 2.0:
        risk_idx = max(risk_idx, 3)
        stage_idx = max(stage_idx, 3)
    elif gap_penalty >= 1.5:
        risk_idx = max(risk_idx, 2)
        stage_idx = max(stage_idx, 2)
    elif gap_penalty >= 0.5:
        risk_idx = max(risk_idx, 1)
    risk = risk_levels[min(risk_idx, len(risk_levels)-1)]
    stage = stage_levels[min(stage_idx, len(stage_levels)-1)]
    life = max(0.5, life - gap_penalty * 0.8)
    if pd_type.startswith("Corona"):
        life *= 1.2
    if pd_type.startswith("Superficial") and n_ang > 0.5:
        life *= 0.5
    life_score = max(0.0, (100.0 - p95_mean) - gap_penalty * 12.0)
    if gap_penalty >= 2.0:
        actions += " Revisar gap-time: valores críticos indican riesgo alto de ruptura."
    elif gap_penalty >= 1.0:
        actions += " Gap-time bajo: limpiar inspeccionar aislamiento."
    return {
        "pd_type": pd_type,
        "location": location,
        "risk": risk,
        "stage": stage,
        "life_years": round(life, 1),
        "life_score": round(life_score, 1),
        "actions": actions,
    }


def _circ_mean_deg_array(ph: np.ndarray) -> float | None:
    try:
        if ph is None or ph.size == 0:
            return None
        sx = float(np.cos(np.deg2rad(ph)).sum())
        sy = float(np.sin(np.deg2rad(ph)).sum())
        ang = float((np.rad2deg(np.arctan2(sy, sx)) + 360.0) % 360.0)
        return ang
    except Exception:
        return None


def _circ_width_deg_array(ph: np.ndarray) -> float | None:
    try:
        if ph is None or ph.size == 0:
            return None
        C = float(np.cos(np.deg2rad(ph)).mean())
        S = float(np.sin(np.deg2rad(ph)).mean())
        R = (C * C + S * S) ** 0.5
        if R <= 0:
            return None
        std_rad = float((-2.0 * np.log(max(R, 1e-12))) ** 0.5)
        return float(np.rad2deg(std_rad) * 2.0)
    except Exception:
        return None


def main() -> None:
    app = QApplication(sys.argv)
    w = PRPDWindow()
    w.show()
    sys.exit(app.exec())




# --- CLI opcional: followup ---
def _cli_followup(argv=None):
    try:
        import sys as _sys, pathlib as _pl
        _pkg_dir = _pl.Path(__file__).resolve().parent  # PRPDapp
        _parent = _pkg_dir.parent
        if str(_parent) not in _sys.path:
            _sys.path.insert(0, str(_parent))
        from PRPDapp.tools.run_followup import main as _run
        _run(argv)
    except Exception as e:
        print('[followup] Error:', e)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].lower() == "followup":
        _cli_followup(sys.argv[2:])
    else:
        main()
