#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

 

import sys, os, json, traceback, re
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
        self.auto_phase = True
        self.pixel_deciles_enabled: set[int] = set(range(1, 11))
        self.qty_deciles_enabled: set[int] = set(range(1, 11))
        self.last_qty_deciles = None

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
        self.ax_raw      = fig.add_subplot(2, 2, 1)
        self.ax_filtered = fig.add_subplot(2, 2, 2)
        self.ax_probs    = fig.add_subplot(2, 2, 3)
        self.ax_text     = fig.add_subplot(2, 2, 4)
        for a in [self.ax_raw, self.ax_filtered, self.ax_probs, self.ax_text]:
            a.set_facecolor("#fafafa")
        v.addWidget(self.canvas)
        self.ax_raw_twin = None
        self.ax_probs_twin = None

        # Estado panel inferior izquierdo
        # Cambiar la opción "ANGPD" por "Histogramas" para reflejar que esta vista
        # muestra histogramas de amplitud/fase y curvas ANGPD/N‑ANGPD.
        self.cmb_plot = QComboBox();
        self.cmb_plot.addItems(["Probabilidades", "Histogramas", "Combinada", "Nubes (S3)", "Nubes (S4)", "Nubes (S5)"])
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
        self.signature_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.signature_label.setMinimumHeight(int(self.banner_max_height * 0.6))
        self.signature_label.setMaximumHeight(self.banner_max_height)
        self.signature_label.setScaledContents(True)
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
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", " "))
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
                if color_points:
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
                            label = f"C{j + 1}" if j < max_labels else None
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
                label = f"C{j + 1}" if j < max_labels else None
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


    def _get_output_dir(self) -> Path:
        """Ensure that an output directory exists and return it.
        Prompts the user to select a directory the first time; caches it in self.out_root."""
        try:
            # If out_root not yet defined or empty, prompt the user once
            if not hasattr(self, "out_root") or self.out_root is None:
                directory = QFileDialog.getExistingDirectory(self, "Selecciona carpeta de salida", str(Path.cwd()))
                self.out_root = Path(directory) if directory else Path("out")
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
        available_w = getattr(self.canvas, "width", lambda: self.signature_label.width())()
        target_w = max(int(available_w), 1)
        current_h = self.signature_label.height() if self.signature_label.height() > 0 else int(self.banner_max_height * 0.8)
        target_h = min(self.banner_max_height, max(current_h, int(self.banner_max_height * 0.6)))
        scaled = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.signature_label.setPixmap(scaled)

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
            self.last_result = result
            self._apply_qty_decile_filter(result, qty_deciles)

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
        try:
            import xml.etree.ElementTree as ET
            root = ET.parse(xml_path).getroot()
            times_text = None
            for tag in root.iter():
                t = (tag.tag or '').lower()
                if t.endswith('times'):
                    times_text = (tag.text or '').strip(); break
            if not times_text:
                return {}
            parts = [p for p in times_text.replace("\n"," ").replace("\t"," ").split(' ') if p]
            times = np.asarray(list(map(float, parts)), dtype=float)
            if times.size < 2:
                return {}
            diffs = np.diff(np.sort(times)); diffs = diffs[diffs > 0]
            if diffs.size == 0:
                return {}
            return {'p50_ms': float(np.percentile(diffs, 50)), 'p5_ms': float(np.percentile(diffs, 5))}
        except Exception:
            return {}

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

    def render_result(self, r: dict) -> None:
        self._clear_twin_axis("ax_raw_twin")
        self._clear_twin_axis("ax_probs_twin")
        # Crudo con overlay de ruido
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
                    self.ax_raw.plot(x, y_n, label="N-ANGPD (max=1)", color="#d62728")
                    self.ax_raw.set_ylim(0, max(1.0, float(y_n.max()) * 1.05))
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
                    self.ax_probs.plot(x, y_q_n, label="N-ANGPD qty (max=1)", color="#d62728")
                    self.ax_probs.set_ylim(0, max(1.0, float(y_q_n.max()) * 1.05))
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
            s3_eps = float(r.get("s3_eps", 0.045))
            s3_min_samples = int(r.get("s3_min_samples", 10))
            s3_force_multi = bool(r.get("s3_force_multi", False))
            clouds_s3 = pixel_cluster_clouds(ph, amp, eps=s3_eps, min_samples=s3_min_samples, force_multi=s3_force_multi) if (ph.size and amp.size) else []
            clouds_s4 = combine_clouds(clouds_s3) if clouds_s3 else []
            clouds_s5 = select_dominant_clouds(clouds_s3) if clouds_s3 else []

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
                clouds = clouds_s3
                title = "Nubes crudas (S3)"
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
            probs = [proba_dict.get(k,0.0) for k in classes]
            self.ax_probs.bar(classes, probs)
            self.ax_probs.set_ylim(0,1)
            self.ax_probs.set_title("Probabilidades")

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
        if not self.current_path:
            QMessageBox.information(self, "Exportar", "Primero carga y procesa un archivo.")
            return
        try:
            # Obtener directorio de salida usando el helper
            outdir = self._get_output_dir()
            stem = self.current_path.stem
            # Calcular offsets de fase seleccionados
            force_offsets = self._get_force_offsets()
            mask_ranges = self._get_phase_mask_ranges()
            pixel_deciles = self._get_pixel_deciles_selection()
            qty_deciles = self._get_qty_deciles_selection()
            # Lista de filtros: sufijo para archivos y etiqueta de filtro utilizada por core.process_prpd
            filters = [
                ("weak", "S1 Weak"),
                ("strong", "S2 Strong"),
            ]
            # Procesar y exportar por cada filtro
            for suf, flabel in filters:
                try:
                    res = core.process_prpd(
                        path=self.current_path,
                        out_root=outdir,
                        force_phase_offsets=force_offsets,
                        fast_mode=False,
                        filter_level=flabel,
                        phase_mask=mask_ranges,
                        pixel_deciles_keep=pixel_deciles,
                        qty_deciles_keep=qty_deciles,
                    )
                    # Exportar ANGPD al CSV con qty
                    self._save_angpd_csv(res.get('angpd', {}), outdir, stem, suf)
                    # Obtener datos alineados
                    ph = np.asarray(res.get('aligned', {}).get('phase_deg', []), dtype=float)
                    amp = np.asarray(res.get('aligned', {}).get('amplitude', []), dtype=float)
                    # Calcular nubes S3/S4/S5 usando las mismas utilidades
                    s3_eps = float(res.get("s3_eps", 0.045))
                    s3_min_samples = int(res.get("s3_min_samples", 10))
                    s3_force_multi = bool(res.get("s3_force_multi", False))
                    clouds_s3 = pixel_cluster_clouds(ph, amp, eps=s3_eps, min_samples=s3_min_samples, force_multi=s3_force_multi) if (ph.size and amp.size) else []
                    clouds_s4 = combine_clouds(clouds_s3) if clouds_s3 else []
                    clouds_s5 = select_dominant_clouds(clouds_s3) if clouds_s3 else []
                    # Exportar imágenes de nubes S3/S4/S5
                    self._save_cluster_images(ph, amp, clouds_s3, clouds_s4, clouds_s5, outdir, stem, suf)
                    # Exportar histogramas de amplitud y fase
                    self._save_histograms(ph, amp, outdir, stem, suf)
                except Exception as sub_e:
                    traceback.print_exc()
                    # Continuar con siguiente filtro si hay error específico
                    QMessageBox.warning(self, 'Exportar', f'Error exportando para filtro {suf}:\n{sub_e}')
            QMessageBox.information(self, 'Exportar', 'Exportes completos generados en la carpeta seleccionada')
        except Exception as e:
            traceback.print_exc(); QMessageBox.warning(self,'Exportar', f'Error en exportación total:\n{e}')

    # -------------------------------------------------------------------------
    # Funciones auxiliares de exportación
    #
    def _save_angpd_csv(self, ang: dict, outdir: Path, stem: str, suffix: str) -> None:
        """Escribe los datos ANGPD con columns qty en un CSV en la carpeta reports."""
        try:
            x = np.asarray(ang.get("phi_centers", []), dtype=float)
            y1 = np.asarray(ang.get("angpd", []), dtype=float)
            y2 = np.asarray(ang.get("n_angpd", []), dtype=float)
            y3 = np.asarray(ang.get("angpd_qty", []), dtype=float)
            y4 = np.asarray(ang.get("n_angpd_qty", []), dtype=float)
            if x.size:
                out_reports = Path(outdir) / 'reports'
                out_reports.mkdir(parents=True, exist_ok=True)
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

    def _save_histograms(self, ph: np.ndarray, amp: np.ndarray, outdir: Path, stem: str, suffix: str) -> None:
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
            ax.set_title(f'Histograma Amplitud ({suffix})')
            ax.legend()
            out_reports = Path(outdir) / 'reports'
            out_reports.mkdir(parents=True, exist_ok=True)
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
            ax.set_title(f'Histograma Fase ({suffix})')
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_reports / f"{stem}_hist_phase_{suffix}.png", bbox_inches='tight')
            _plt.close(fig)
        except Exception:
            traceback.print_exc()

    def _save_cluster_images(self, ph: np.ndarray, amp: np.ndarray,
                             clouds_s3: list[dict], clouds_s4: list[dict], clouds_s5: list[dict],
                             outdir: Path, stem: str, suffix: str) -> None:
        """Guarda las imágenes de nubes S3, S4 y S5 con estilo consistente."""
        try:
            import matplotlib.pyplot as _plt
            out_reports = Path(outdir) / 'reports'
            out_reports.mkdir(parents=True, exist_ok=True)
            # Nubes S3 (puntos coloreados)
            fig, ax = _plt.subplots(figsize=(5, 4), dpi=120)
            PRPDWindow._plot_clusters_on_ax(ax, ph, amp, clouds_s3,
                                            title=f'Nubes crudas (S3) - {suffix}',
                                            color_points=True,
                                            include_k=False,
                                            max_labels=10)
            fig.tight_layout()
            fig.savefig(out_reports / f"{stem}_s3_{suffix}.png", bbox_inches='tight')
            _plt.close(fig)
            # Nubes S4 (puntos grises, centros coloreados, mostrar k)
            fig, ax = _plt.subplots(figsize=(5, 4), dpi=120)
            PRPDWindow._plot_clusters_on_ax(ax, ph, amp, clouds_s4,
                                            title=f'Nubes combinadas (S4) - {suffix}',
                                            color_points=False,
                                            include_k=True,
                                            max_labels=10)
            fig.tight_layout()
            fig.savefig(out_reports / f"{stem}_s4_{suffix}.png", bbox_inches='tight')
            _plt.close(fig)
            # Nubes S5 (puntos grises, centros coloreados, mostrar k)
            fig, ax = _plt.subplots(figsize=(5, 4), dpi=120)
            PRPDWindow._plot_clusters_on_ax(ax, ph, amp, clouds_s5,
                                            title=f'Nubes dominantes (S5) - {suffix}',
                                            color_points=False,
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
                        s3_eps = float(res.get("s3_eps", 0.045))
                        s3_min_samples = int(res.get("s3_min_samples", 10))
                        s3_force_multi = bool(res.get("s3_force_multi", False))
                        clouds_s3 = pixel_cluster_clouds(ph, amp, eps=s3_eps, min_samples=s3_min_samples, force_multi=s3_force_multi) if (ph.size and amp.size) else []
                        clouds_s4 = combine_clouds(clouds_s3) if clouds_s3 else []
                        clouds_s5 = select_dominant_clouds(clouds_s3) if clouds_s3 else []
                        # Guardar imágenes
                        self._save_cluster_images(ph, amp, clouds_s3, clouds_s4, clouds_s5, out_dir, fstem, suf)
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
