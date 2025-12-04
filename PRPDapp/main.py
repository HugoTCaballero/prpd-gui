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
    QSizePolicy, QFrame, QMenu, QInputDialog, QToolButton, QColorDialog, QFormLayout
)
from PySide6.QtGui import QPixmap, QFont, QAction, QGuiApplication
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
from PRPDapp.metrics.advanced_kpi import compute_advanced_metrics
from PRPDapp.config_pd import CLASS_NAMES, CLASS_INFO
from PRPDapp.clouds import pixel_cluster_clouds, combine_clouds, select_dominant_clouds
from PRPDapp.logic import compute_pd_metrics as logic_compute_pd_metrics, classify_pd as logic_classify_pd
from PRPDapp import ui_dialogs, ui_draw, ui_render, ui_layout, ui_events
from PRPDapp.logic_hist import compute_semicycle_histograms_from_aligned



# ANN loader opcional (models/ann_loader)
try:
    from models.ann_loader import (
        load_ann_model as _load_ann_model,
        predict_proba as _ann_predict_proba,
    )
except Exception:
    _load_ann_model = None
    _ann_predict_proba = None


APP_TITLE = "PRPD GUI - Unificada (exports v2)"

class PRPDWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self._apply_default_size()
        self.setMinimumSize(1024, 640)

        # Estado
        self.current_path: Path | None = None
        self.last_result: dict | None = None
        self.last_run_profile: dict | None = None
        self.out_root: Path | None = None
        self.auto_phase = True
        self._phase_prev_index = 0
        self.manual_phase_offset: int | None = None
        self.pixel_deciles_enabled: set[int] = set(range(1, 11))
        self.qty_quintiles_enabled: set[int] = set(range(1, 6))   # Q1..Q5
        self.qty_deciles_enabled: set[int] = set(range(1, 11))    # Qt1..Qt10 (mitades de cada Q)
        self.last_qty_deciles = None
        self.last_conclusion_text: str = ""
        self.last_conclusion_payload: dict = {}
        self._ann_history_written: bool = False
        self._ann_history_recent: list[dict] = []
        self._last_ann_probs: dict[str, float] = {}
        self.manual_override: dict = {"enabled": False}
        self.gap_ext_files: list[Path] = []
        self.banner_dark_mode: bool = False
        self._hist_bins_phase = 32
        self._hist_bins_amp = 32
        self._dark_stylesheet = """
        QMainWindow, QWidget {
            background-color: #05070d;
            color: #e5e7eb;
        }
        QLabel, QCheckBox, QRadioButton, QGroupBox {
            color: #e5e7eb;
        }
        QPushButton {
            background-color: #0f172a;
            color: #e5e7eb;
            border: 1px solid #233044;
            border-radius: 4px;
            padding: 4px 8px;
        }
        QPushButton:hover {
            background-color: #1d4ed8;
        }
        QComboBox, QLineEdit, QPlainTextEdit, QTextEdit {
            background-color: #0b1020;
            color: #e5e7eb;
            border: 1px solid #233044;
            selection-background-color: #1d4ed8;
        }
        QMenu {
            background-color: #0b1020;
            color: #e5e7eb;
            border: 1px solid #233044;
        }
        QToolTip {
            background-color: #0f172a;
            color: #e5e7eb;
            border: 1px solid #1d4ed8;
        }
        """

        # ANN (fallback PRPDANN si no hay loader)
        self.pd_classes = CLASS_NAMES[:]  # referencia centralizada
        self.ann = PRPDANN(class_names=self.pd_classes)
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

        # Barra superior (construida desde ui_layout)
        top = ui_layout.build_top_bar(self)
        v.addLayout(top)

                # Figuras
        fig, self.canvas = ui_layout.build_figures(self)
        v.addWidget(self.canvas)

# Estado panel inferior izquierdo
        sub, batch_bar = ui_layout.build_bottom_area(self)
        v.addLayout(sub)
        v.addLayout(batch_bar)

        # Eventos
        ui_events.connect_events(self)
        self._refresh_gap_ext_buttons()
        # Tema inicial (claro)
        self._apply_theme_stylesheet(self.banner_dark_mode)


    def _apply_default_size(self) -> None:
        """Ajusta la ventana al 90% del area disponible, con minimos seguros."""
        try:
            screen = self.screen() or QGuiApplication.primaryScreen()
            if screen is None:
                self.resize(1200, 740)
                return
            avail = screen.availableGeometry()
            w = max(1100, int(avail.width() * 0.9))
            h = max(700, int(avail.height() * 0.9))
            self.resize(w, h)
            self.move(
                avail.x() + max(0, (avail.width() - w) // 2),
                avail.y() + max(0, (avail.height() - h) // 2),
            )
        except Exception:
            self.resize(1200, 740)

    def _precompute_qty_buckets(self, result: dict) -> None:
        """Calcula cortes globales sobre quantity, etiqueta Q/Qt y alinea arrays completos."""
        aligned = result.get("aligned", {}) or {}
        qty_vals = np.asarray(aligned.get("quantity", []), dtype=float)
        phase = np.asarray(aligned.get("phase_deg", []), dtype=float)
        amp = np.asarray(aligned.get("amplitude", []), dtype=float)
        if qty_vals.size == 0 or phase.size == 0 or amp.size == 0:
            return

        edges_q = np.percentile(qty_vals, [20, 40, 60, 80])
        edges_d = np.percentile(qty_vals, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        quint_idx = np.digitize(qty_vals, edges_q, right=True) + 1
        quint_idx = np.clip(quint_idx, 1, 5)
        dec_idx = np.digitize(qty_vals, edges_d, right=True) + 1
        dec_idx = np.clip(dec_idx, 1, 10)

        aligned["qty_quintiles"] = quint_idx
        aligned["qty_deciles"] = dec_idx
        result["qty_quintiles_meta"] = {"edges": edges_q.tolist()}
        result["qty_deciles_meta"] = {"edges": edges_d.tolist()}
        self._align_full_arrays(aligned)

    def _align_full_arrays(self, aligned: dict) -> None:
        """Alinea phase/amp/qty y etiquetas Q/Qt al mismo largo y guarda _full_* coherentes."""
        phase = np.asarray(aligned.get("phase_deg", []), dtype=float)
        amp = np.asarray(aligned.get("amplitude", []), dtype=float)
        qty = np.asarray(aligned.get("quantity", []), dtype=float)
        quint = np.asarray(aligned.get("qty_quintiles", []), dtype=int)
        dec = np.asarray(aligned.get("qty_deciles", []), dtype=int)
        pixel = np.asarray(aligned.get("pixel", []), dtype=float)
        labels = np.asarray(aligned.get("labels_aligned", []))

        lengths = [arr.size for arr in (phase, amp, qty, quint, dec) if arr.size]
        n = min(lengths) if lengths else 0
        if n <= 0:
            return

        aligned["phase_deg"] = phase[:n]
        aligned["amplitude"] = amp[:n]
        aligned["quantity"] = qty[:n]
        aligned["qty_quintiles"] = quint[:n]
        aligned["qty_deciles"] = dec[:n]

        aligned["_full_phase_deg"] = aligned["phase_deg"]
        aligned["_full_amplitude"] = aligned["amplitude"]
        aligned["_full_quantity"] = aligned["quantity"]
        aligned["_full_qty_quintiles"] = aligned["qty_quintiles"]
        aligned["_full_qty_deciles"] = aligned["qty_deciles"]
        if pixel.size:
            aligned["_full_pixel"] = pixel[:n]
        if labels.size:
            aligned["_full_labels_aligned"] = labels[:n]

    def _on_resolution_action(self, action) -> None:
        """Ajusta el tamano de la ventana/canvas segun el preset del boton Resolucion."""
        try:
            size = action.data() if action is not None else None
        except Exception:
            size = None
        if not size:
            self._apply_default_size()
        else:
            try:
                w, h = int(size[0]), int(size[1])
                self.resize(max(w, self.minimumWidth()), max(h, self.minimumHeight()))
                screen = self.screen() or QGuiApplication.primaryScreen()
                if screen is not None:
                    avail = screen.availableGeometry()
                    x = avail.x() + max(0, (avail.width() - self.width()) // 2)
                    y = avail.y() + max(0, (avail.height() - self.height()) // 2)
                    self.move(x, y)
                fig = self.canvas.figure
                dpi = float(fig.get_dpi() or 100.0)
                target_w = max(600.0, float(self.width()))
                target_h = max(400.0, float(self.height()) - 140.0)
                fig.set_size_inches(target_w / dpi, target_h / dpi, forward=True)
            except Exception:
                self._apply_default_size()
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_phase_changed(self, idx: int) -> None:
        """Actualizar auto_phase y soportar fase manual ingresada por el usuario."""
        manual_idx = self.cmb_phase.count() - 1
        if idx == manual_idx:
            if not self._open_manual_phase_dialog(force_dialog=True):
                # Revertir al valor previo si se cancela
                self.cmb_phase.blockSignals(True)
                self.cmb_phase.setCurrentIndex(self._phase_prev_index)
                self.cmb_phase.blockSignals(False)
                return
        else:
            self._phase_prev_index = idx
            self.auto_phase = (idx == 0)
            # Restaurar texto base de manual si se elige otro
            try:
                self.cmb_phase.setItemText(manual_idx, "Manual…")
            except Exception:
                pass
        # Si hay un resultado procesado, volver a renderizar con la nueva fase seleccionada
        try:
            if self.last_result:
                self.render_result(self.last_result)
        except Exception:
            pass

    def _open_manual_phase_dialog(self, force_dialog: bool = False) -> bool:
        """Wrapper a ui_dialogs para fase manual."""
        try:
            manual_idx = self.cmb_phase.count() - 1
        except Exception:
            return False
        if not force_dialog and self.cmb_phase.currentIndex() != manual_idx:
            return False
        val = ui_dialogs.open_manual_phase_dialog(self, self.manual_phase_offset)
        if val is None:
            return False
        self.manual_phase_offset = val
        self.cmb_phase.setItemText(manual_idx, f"Manual ({self.manual_phase_offset}°)")
        self.auto_phase = False
        self._phase_prev_index = manual_idx
        try:
            self.cmb_phase.blockSignals(True)
            self.cmb_phase.setCurrentIndex(manual_idx)
        finally:
            self.cmb_phase.blockSignals(False)
        return True

    def _open_manual_override_dialog(self) -> bool:
        """Wrapper a ui_dialogs para override manual de conclusiones."""
        new_override = ui_dialogs.open_manual_override_dialog(self, self.manual_override)
        if new_override is None:
            return False
        self.manual_override = new_override
        return True

    def _on_view_changed(self, text: str) -> None:
        """Abrir dialogo manual si se selecciona la vista Manual."""
        try:
            if text.strip().lower().startswith("manual"):
                if self._open_manual_override_dialog():
                    self.manual_override["enabled"] = True
                    self.cmb_plot.blockSignals(True)
                    self.cmb_plot.setCurrentText("Conclusiones")
                    self.cmb_plot.blockSignals(False)
                    if self.last_result:
                        self.render_result(self.last_result)
                else:
                    self.cmb_plot.blockSignals(True)
                    self.cmb_plot.setCurrentText("Conclusiones")
                    self.cmb_plot.blockSignals(False)
            else:
                if text.strip().lower().startswith("histogramas"):
                    self._hist_bins_phase = 32
                    self._hist_bins_amp = 32
                if self.last_result:
                    self.render_result(self.last_result)
        except Exception:
            pass

    def _on_hist_bins_changed(self, text: str) -> None:
        """Cambia bins de histograma (32/64) y recalcula metricas avanzadas si hay resultado."""
        t = (text or "").lower()
        bins = 32
        if "64" in t:
            bins = 64
        self._hist_bins_phase = bins
        self._hist_bins_amp = bins
        try:
            if self.last_result:
                self.last_result["metrics_advanced"] = compute_advanced_metrics(
                    {"aligned": self.last_result.get("aligned", {}), "angpd": self.last_result.get("angpd", {})},
                    bins_amp=bins,
                    bins_phase=bins,
                )
                self.render_result(self.last_result)
        except Exception:
            pass

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
        manual_idx = max(0, self.cmb_phase.count() - 1)
        # Índice 0 => auto_phase (usar todos los offsets cand)
        if idx == 0:
            return None
        # Manual
        if idx == manual_idx:
            if self.manual_phase_offset is None:
                return None
            return [int(self.manual_phase_offset) % 360]
        # Convertir índice a valor de fase fijo
        try:
            phase_val = {1: 0, 2: 120, 3: 240}.get(idx, 0)
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
            # Void: conservar rangos amplios alrededor de 0°/180° (envolviendo 360)
            "void": [(330.0, 360.0), (0.0, 60.0), (120.0, 240.0)],
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

    def _apply_auto_ylim(self, ax, amps: np.ndarray, default: tuple[float, float] = (0, 100)) -> None:
        """Ajusta el rango Y al percentil 1-99% de amplitud si está activado."""
        try:
            if not self.chk_auto_y.isChecked():
                ax.set_ylim(*default)
                return
            amps = np.asarray(amps, dtype=float)
            if amps.size == 0:
                ax.set_ylim(*default)
                return
            lo = np.nanpercentile(amps, 1)
            hi = np.nanpercentile(amps, 99)
            span = max(10.0, hi - lo)
            ymin = max(0.0, lo - 0.1 * span)
            ymax = min(100.0, hi + 0.2 * span)
            if ymin >= ymax or not np.isfinite([ymin, ymax]).all():
                ax.set_ylim(*default)
            else:
                ax.set_ylim(ymin, ymax)
        except Exception:
            try:
                ax.set_ylim(*default)
            except Exception:
                pass

    def _on_centers_combined_toggle(self, *_):
        """Refresca la vista combinada si está activa al cambiar el toggle de centros S3."""
        try:
            view = self.cmb_plot.currentText().strip().lower()
        except Exception:
            view = ""
        if view.startswith("combinada") and self.last_result:
            self.render_result(self.last_result)

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

    def _draw_prpd_scatter_base(self, ax, result: dict) -> None:
        """Scatter básico de PRPD alineado (o heatmap si está activo)."""
        aligned = result.get("aligned", {}) if isinstance(result, dict) else {}
        ph_al = np.asarray(aligned.get("phase_deg", []), dtype=float)
        amp_al = np.asarray(aligned.get("amplitude", []), dtype=float)
        if ph_al.size and amp_al.size:
            try:
                if self.chk_hist2d.isChecked():
                    H2, xe2, ye2 = np.histogram2d(ph_al, amp_al, bins=[72,50], range=[[0,360],[0,100]])
                    ax.imshow(H2.T + 1e-9, origin='lower', aspect='auto', extent=[xe2[0], xe2[-1], ye2[0], ye2[-1]])
                ax.scatter(ph_al, amp_al, s=3, alpha=0.5, color="#1f77b4", edgecolors="none")
            except Exception:
                ax.scatter(ph_al, amp_al, s=3, alpha=0.5, color="#1f77b4", edgecolors="none")

    # --- Vista FA profile --------------------------------------------------
    def _draw_fa_profile_left(self, ax, result: dict) -> None:
        fa_profile = result.get("fa_profile") if isinstance(result, dict) else None
        if not fa_profile:
            ax.clear()
            ax.text(0.5, 0.5, "FA profile no disponible", ha="center", va="center")
            ax.set_axis_off()
            return
        phase = np.asarray(fa_profile.get("phase_bins_deg", []), dtype=float)
        max_amp = np.asarray(fa_profile.get("max_amp_by_bin", []), dtype=float)
        max_smooth = np.asarray(fa_profile.get("max_amp_smooth", []), dtype=float)
        count = np.asarray(fa_profile.get("count_by_bin", []), dtype=float)
        if phase.size == 0:
            ax.clear()
            ax.text(0.5, 0.5, "FA profile vacío", ha="center", va="center")
            ax.set_axis_off()
            return
        max_amp_plot = np.copy(max_amp)
        max_amp_plot[np.isnan(max_amp_plot)] = 0.0
        if np.any(count > 0):
            count_norm = count / np.max(count)
        else:
            count_norm = count

        ax.clear()
        ax.plot(phase, max_amp_plot, marker="o", linestyle="none", alpha=0.4, label="Max amp por bin")
        if max_smooth.size:
            ax.plot(phase, max_smooth, linestyle="-", linewidth=2.0, label="Envolvente suavizada")
        ax.set_xlabel("Fase (°)")
        ax.set_ylabel("Amplitud (sensor)")
        ax.set_xlim(0, 360)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.fill_between(phase, 0, count_norm, alpha=0.15, step="mid", label="Conteo (norm)")
        ax2.set_ylabel("Conteo normalizado")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)
        ax.set_title("Perfil fase–amplitud (FA profile)")

    def _draw_fa_profile_right(self, ax, result: dict, ph_al: np.ndarray, amp_al: np.ndarray) -> None:
        fa_profile = result.get("fa_profile") if isinstance(result, dict) else None
        ax.clear()
        ax.set_facecolor("#fffdf5")
        if ph_al.size and amp_al.size:
            self._draw_prpd_scatter_base(ax, result)
        else:
            ax.text(0.5, 0.5, "Sin PRPD alineado", ha="center", va="center")
        if fa_profile:
            phase = np.asarray(fa_profile.get("phase_bins_deg", []), dtype=float)
            env = np.asarray(fa_profile.get("max_amp_smooth", []), dtype=float)
            if phase.size and env.size:
                ax.plot(phase, env, color="#ff7f0e", linewidth=2.4, label="Envolvente FA")
                try:
                    ax.legend(loc="upper right", fontsize=8)
                except Exception:
                    pass
        ax.set_xlim(0, 360)
        ax.set_xlabel("Fase (°)")
        ax.set_ylabel("Amplitud")
        self._apply_auto_ylim(ax, amp_al)
        ax.set_title("PRPD alineado + envolvente FA")

    def _draw_fa_profile_view(self, r: dict, ph_al: np.ndarray, amp_al: np.ndarray) -> None:
        """Vista 'FA profile': perfil 1D y overlay sobre PRPD alineado."""
        self._restore_standard_axes()
        for ax in (self.ax_raw, self.ax_filtered):
            ax.set_visible(True)
            ax.set_axis_on()
        for ax in (self.ax_probs, self.ax_text, self.ax_gap_wide):
            ax.set_visible(False)
            ax.clear()
        self._draw_fa_profile_left(self.ax_raw, r)
        self._draw_fa_profile_right(self.ax_filtered, r, ph_al, amp_al)

    def _ensure_conclusion_axis(self):
        """Devuelve el eje que ocupa todo el rectángulo inferior."""
        if self.ax_conclusion_box is None:
            # Eje ya creado al inicializar (fila inferior del gridspec principal)
            self.ax_conclusion_box = self.canvas.figure.add_subplot(self._gs_main[1, :])
            self.ax_conclusion_box.set_visible(False)
        self.ax_conclusion_box.set_facecolor("#fffdf5")
        return self.ax_conclusion_box

    def _set_conclusion_mode(self, enable: bool) -> None:
        """Oculta/muestra los ejes inferiores y el rectángulo de conclusiones."""
        if enable:
            for ax in (self.ax_raw, self.ax_filtered, self.ax_probs, self.ax_text, self.ax_gap_wide):
                ax.set_visible(False)
            self.ax_probs.set_visible(False)
            self.ax_text.set_visible(False)
            ax = self._ensure_conclusion_axis()
            ax.set_position([0.02, 0.02, 0.96, 0.94])
            ax.set_visible(True)
            try:
                ax.set_in_layout(True)
            except Exception:
                pass
        else:
            for ax in (self.ax_raw, self.ax_filtered, self.ax_gap_wide):
                ax.set_visible(True)
            self.ax_probs.set_visible(True)
            self.ax_probs.set_axis_on()
            self.ax_text.set_visible(True)
            self.ax_text.set_axis_on()
            if self.ax_conclusion_box is not None:
                self._clear_conclusion_artists()
                self.ax_conclusion_box.set_visible(False)
                try:
                    self.ax_conclusion_box.set_in_layout(False)
                except Exception:
                    pass

    @staticmethod
    def _draw_status_tag(ax, text: str, x: float, y: float, *, color: str, text_color: str = "#ffffff", size: int = 11):
        """Wrapper hacia helpers de dibujo (ui_draw)."""
        return ui_draw.draw_status_tag(ax, text, x, y, color=color, text_color=text_color, size=size)

    def _draw_conclusion_header(self, ax, title: str, subtitle: str | None = None) -> None:
        """Encabezado principal de la vista Conclusiones."""
        title_art = ax.text(0.03, 0.98, title.upper(), fontsize=18, fontweight="bold", ha="left", va="center", color="#0f172a")
        self._register_conclusion_artist(title_art)
        if subtitle:
            sub = ax.text(0.03, 0.91, subtitle, fontsize=11, fontweight="bold", ha="left", va="center", color="#5f6c7b")
            self._register_conclusion_artist(sub)
        line = ax.plot([0.03, 0.97], [0.91, 0.91], color="#d0d7de", linewidth=1.6)
        self._register_conclusion_artist(line[0])

    def _append_ann_history(self, probs: dict | None) -> None:
        if self._ann_history_written:
            return
        stem = self.current_path.stem if self.current_path else None
        if not stem or not probs:
            return
        norm = {}
        for key, value in probs.items():
            try:
                norm[str(key).lower()] = float(value)
            except Exception:
                continue
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

    def _draw_section_title(self, ax, text: str, *, y: float = 0.95, x: float = 0.02, register=None) -> float:
        """Wrapper para t�tulos de secci�n; mantiene registro de artistas para limpieza."""
        reg = register or self._register_conclusion_artist
        return ui_draw.draw_section_title(ax, text, y=y, x=x, register=reg)

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
        """Selector de quintiles y sub-quintiles (Qt1-Qt10) para quantity."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Filtros de quantity (Q / Qt)")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Cortes globales p20/p40/p60/p80 (id?ntico a Excel). Q1-Q5 gobiernan Qt1-Qt10."))

        row_q = QHBoxLayout()
        q_checks = []
        for q in range(1, 6):
            cb = QCheckBox(f"Q{q}")
            cb.setChecked(q in self.qty_quintiles_enabled)
            row_q.addWidget(cb)
            q_checks.append((q, cb))
        layout.addLayout(row_q)

        rows = [QHBoxLayout(), QHBoxLayout()]
        qt_checks = []
        for idx, dec in enumerate(range(1, 11)):
            cb = QCheckBox(f"Qt{dec}")
            cb.setChecked(dec in self.qty_deciles_enabled)
            rows[0 if idx < 5 else 1].addWidget(cb)
            qt_checks.append((dec, cb))
        for row in rows:
            layout.addLayout(row)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_all = btn_box.addButton("Todos", QDialogButtonBox.ActionRole)
        btn_none = btn_box.addButton("Vaciar", QDialogButtonBox.ActionRole)

        def _set_all(state: bool) -> None:
            for _, cb in q_checks:
                cb.setChecked(state)
            for _, cb in qt_checks:
                cb.setChecked(state)

        btn_all.clicked.connect(lambda: _set_all(True))
        btn_none.clicked.connect(lambda: _set_all(False))
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)

        if dlg.exec() == QDialog.Accepted:
            selected_q = {q for q, cb in q_checks if cb.isChecked()}
            selected_qt = {dec for dec, cb in qt_checks if cb.isChecked()}
            pairs = {1: {1, 2}, 2: {3, 4}, 3: {5, 6}, 4: {7, 8}, 5: {9, 10}}
            clean_qt = set()
            for q, subs in pairs.items():
                if q in selected_q:
                    clean_qt.update(s for s in subs if s in selected_qt)
            self.qty_quintiles_enabled = selected_q
            self.qty_deciles_enabled = clean_qt
            self._update_qty_button_text()

    def _update_qty_button_text(self) -> None:
        if not hasattr(self, "btn_qty"):
            return
        qs = sorted(self.qty_quintiles_enabled)
        qts = sorted(self.qty_deciles_enabled)

        def _fmt(vals, prefix):
            if not vals:
                return ""
            parts = []
            start = prev = vals[0]
            for v in vals[1:]:
                if v == prev + 1:
                    prev = v
                    continue
                parts.append(self._format_decile_range(start, prev).replace("D", prefix))
                start = prev = v
            parts.append(self._format_decile_range(start, prev).replace("D", prefix))
            return ",".join(parts)

        txt_q = _fmt(qs, "Q")
        txt_qt = _fmt(qts, "Qt")
        if not qs:
            text_lbl = "Qty: (Q apagados)"
        else:
            text_lbl = f"Qty: {txt_q or 'Q1-Q5'}"
            if qts:
                text_lbl += f" / {txt_qt}"
        self.btn_qty.setText(text_lbl)

    def _get_qty_deciles_selection(self) -> list[int]:
        return sorted(self.qty_deciles_enabled)

    def _get_qty_filters(self) -> tuple[list[int], list[int]]:
        """Devuelve (deciles_keep, quints_keep) respetando la lógica Q/Qt."""
        quints = sorted(self.qty_quintiles_enabled) or []
        subs = sorted(self.qty_deciles_enabled) or []
        pairs = {1: (1, 2), 2: (3, 4), 3: (5, 6), 4: (7, 8), 5: (9, 10)}
        dec_keep: list[int] = []
        for q in quints:
            if q not in pairs:
                continue
            allowed = [d for d in pairs[q] if d in subs] if subs else list(pairs[q])
            dec_keep.extend(allowed)
        return sorted(set(dec_keep)), quints

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
            "qty": tuple(sorted(self.qty_deciles_enabled)),          # Qt1-Qt10
            "qty_quints": tuple(sorted(self.qty_quintiles_enabled)), # Q1-Q5
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
        qty_quints = profile.get("qty_quints", tuple())
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
        qt_tag = self._format_decile_token(qty_vals)
        qq_tag = self._format_decile_token(qty_quints if qty_quints else tuple(range(1,6)))
        qty_tag = f"QQ{qq_tag}QT{qt_tag}"
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

    def _apply_qty_filters(self, result: dict, keep_quints: list[int], keep_deciles: list[int]) -> None:
        """Aplica Q/Qt como m?scara booleana sobre arrays ya alineados y recorta la vista/export."""
        aligned = result.get("aligned", {}) or {}
        phase_full = np.asarray(aligned.get("_full_phase_deg", aligned.get("phase_deg", [])), dtype=float)
        amp_full = np.asarray(aligned.get("_full_amplitude", aligned.get("amplitude", [])), dtype=float)
        qty_full = np.asarray(aligned.get("_full_quantity", aligned.get("quantity", [])), dtype=float)
        dec_full = np.asarray(aligned.get("_full_qty_deciles", aligned.get("qty_deciles", [])), dtype=int)
        quint_full = np.asarray(aligned.get("_full_qty_quintiles", aligned.get("qty_quintiles", [])), dtype=int)
        pixel_full = np.asarray(aligned.get("_full_pixel", aligned.get("pixel", [])), dtype=float)
        labels_full = np.asarray(aligned.get("_full_labels_aligned", aligned.get("labels_aligned", [])))

        n = phase_full.size
        if n == 0:
            return

        keep_q_set = {int(q) for q in keep_quints if 1 <= int(q) <= 5}
        keep_sub_set = {int(k) for k in keep_deciles if 1 <= int(k) <= 10}
        pairs = {1: {1, 2}, 2: {3, 4}, 3: {5, 6}, 4: {7, 8}, 5: {9, 10}}

        all_quints = keep_q_set.issuperset({1, 2, 3, 4, 5})
        all_deciles = (not keep_sub_set) or keep_sub_set == set(range(1, 11))

        if all_quints and all_deciles:
            mask_keep = np.ones(n, dtype=bool)
        elif not keep_q_set:
            mask_keep = np.zeros(n, dtype=bool)
        else:
            mask_keep = np.zeros(n, dtype=bool)
            for q in range(1, 6):
                if q not in keep_q_set:
                    continue
                subs = pairs[q]
                subs_on = {s for s in subs if s in keep_sub_set} if keep_sub_set else subs
                if dec_full.size == 0:
                    mask_keep |= (quint_full == q)
                else:
                    if not subs_on:
                        continue
                    mask_keep |= (quint_full == q) & np.isin(dec_full, list(subs_on))

        aligned["phase_deg"] = phase_full[mask_keep]
        aligned["amplitude"] = amp_full[mask_keep]
        aligned["quantity"] = qty_full[mask_keep]
        aligned["qty_quintiles"] = quint_full[mask_keep]
        aligned["qty_deciles"] = dec_full[mask_keep]
        if pixel_full.size:
            aligned["pixel"] = pixel_full[mask_keep]
        if labels_full.size:
            aligned["labels_aligned"] = labels_full[mask_keep]

        if pixel_full.size:
            aligned["pixel"] = pixel_full[mask_keep]
        if labels_full.size:
            aligned["labels_aligned"] = labels_full[mask_keep]

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

    @staticmethod
    def _quintiles_from_deciles(deciles: np.ndarray) -> np.ndarray:
        """Mapea deciles (1-10) a quintiles (1-5) de forma determinista."""
        arr = np.asarray(deciles, dtype=int)
        out = np.zeros(arr.shape, dtype=int)
        if arr.size == 0:
            return out
        mapping = {
            1: 1, 2: 1,
            3: 2, 4: 2,
            5: 3, 6: 3,
            7: 4, 8: 4,
            9: 5, 10: 5,
        }
        for d, q in mapping.items():
            out[arr == d] = q
        return out

    @staticmethod
    def _compute_deciles(values: np.ndarray) -> np.ndarray:
        """Calcula deciles 1-10 a partir de quantity (sin agrupar por fase)."""
        arr = np.asarray(values, dtype=float)
        dec = np.zeros(arr.shape, dtype=int)
        finite = np.isfinite(arr)
        if not finite.any():
            return dec
        vals = arr[finite]
        try:
            thresholds = np.nanpercentile(vals, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        except Exception:
            thresholds = []
        dec[finite] = 1
        for i, thr in enumerate(thresholds):
            dec[finite & (arr >= thr)] = i + 2  # 2..10
        return dec

    @staticmethod
    def _deciles_from_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Asigna deciles usando edges precomputados (percentiles 10..90)."""
        arr = np.asarray(values, dtype=float)
        idx = np.zeros(arr.shape, dtype=int)
        if arr.size == 0:
            return idx
        edges = np.asarray(edges, dtype=float)
        if edges.size != 9 or not np.all(np.isfinite(edges)):
            return idx
        bins = np.digitize(arr, edges, right=True)  # 0..9
        idx = (bins + 1).astype(int)  # 1..10
        idx[~np.isfinite(arr)] = 0
        return idx

    def _resolve_quintiles(self, ph_al: np.ndarray, result: dict, aligned: dict) -> np.ndarray:
        """Determina quintiles Q1-Q5 para aligned usando edges globales de quantity si existen."""
        qty_vals = np.asarray(aligned.get("quantity", []), dtype=float)
        quint_idx = np.zeros_like(ph_al, dtype=int)
        if ph_al.size == 0 or qty_vals.size != ph_al.size:
            return quint_idx
        # 1) edges globales (qty_deciles_meta) sobre quantity completo
        edges = (result.get("qty_deciles_meta") or {}).get("edges", [])
        edges_arr = np.asarray(edges, dtype=float)
        if edges_arr.size == 9 and np.all(np.isfinite(edges_arr)):
            dec_from_edges = self._deciles_from_edges(qty_vals, edges_arr)
            if dec_from_edges.size == ph_al.size:
                return self._quintiles_from_deciles(dec_from_edges)
        # 2) deciles ya precalculados en aligned (respetan DQ) -> map a quintiles
        deciles_aligned = np.asarray(aligned.get("qty_deciles", []), dtype=int)
        if deciles_aligned.size == ph_al.size and deciles_aligned.size:
            return self._quintiles_from_deciles(deciles_aligned)
        # 3) quintiles precalculados en aligned
        quint_from_aligned = np.asarray(aligned.get("qty_quintiles", []), dtype=int)
        if quint_from_aligned.size == ph_al.size and quint_from_aligned.size:
            return quint_from_aligned
        # 4) deciles recalculados sobre quantity alineado
        deciles_calc = self._compute_deciles(qty_vals)
        if deciles_calc.size == ph_al.size and deciles_calc.any():
            return self._quintiles_from_deciles(deciles_calc)
        # 5) fallback equal-frequency
        return self._equal_frequency_bucket(qty_vals, groups=5)

    def _cluster_profiles(self) -> dict[str, dict]:
        return {
            "S1 Weak": {"eps": 0.055, "min_samples": 8, "force_multi": False},
            "S2 Strong": {"eps": 0.030, "min_samples": 14, "force_multi": True},
        }

    def _build_cluster_variants(self, ph: np.ndarray, amp: np.ndarray, current_filter: str | None = None) -> tuple[dict[str, list[dict]], list[dict], list[dict], list[dict]]:
        """Genera nubes S3 (por perfil), S4 (combinadas) y S5 (dominantes).

        Si current_filter se provee (S1 Weak / S2 Strong), S4 y S5 se derivan
        solo de las nubes del perfil actual para mantener el flujo
        Alineado/filtrado -> S3 -> S4 -> S5 sin mezclar filtros.
        """
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
        combined_all = [dict(c) for arr in variants.values() for c in arr]
        if not combined_all:
            return variants, [], [], []

        # Escoger la base seg�n filtro activo (o todas si no se especifica)
        filt = (current_filter or "").lower()
        if "strong" in filt or "s2" in filt:
            base_s3 = variants.get("S2 Strong", []) or combined_all
        elif "weak" in filt or "s1" in filt:
            base_s3 = variants.get("S1 Weak", []) or combined_all
        else:
            base_s3 = combined_all

        # S4: combinar nubes del perfil actual; S5: dominantes sobre S4 (o S3 si no hay combinaci�n)
        total_base = max(1.0, sum(int(c.get("count", 0)) for c in base_s3))
        clouds_s4 = combine_clouds(base_s3)
        for idx, c in enumerate(clouds_s4):
            c["legend"] = c.get("legend") or f"C{idx + 1}"
            c["frac"] = c.get("frac", c.get("count", 0) / total_base)
        base_for_s5 = clouds_s4 if clouds_s4 else base_s3
        clouds_s5 = select_dominant_clouds(base_for_s5, min_frac=0.10)
        clouds_s5 = sorted(clouds_s5, key=lambda x: x.get("frac", 0), reverse=True)[:3]
        for idx, c in enumerate(clouds_s5):
            c["legend"] = c.get("legend") or f"C{idx + 1}"
            c["frac"] = c.get("frac", c.get("count", 0) / total_base)
        return variants, base_s3, clouds_s4, clouds_s5

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
                        # Distancia angular mínima (0..180) sin plegar semiciclo para evitar mezclar + y -
                        dp = _np.abs(((ph[i] - centers[:, 0] + 180.0) % 360.0) - 180.0)
                        dy = _np.abs(amp[i] - centers[:, 1])
                        j = int(_np.argmin(0.6 * dp + 0.4 * dy))
                        lbl[i] = j
                    # Colorear puntos según su cluster
                    for j in range(n_clusters):
                        m = (lbl == j)
                        if np.any(m):
                            color = palette[j % len(palette)]
                            legend = clouds[j].get("legend") if j < len(clouds) else None
                            label = legend or (f"C{j + 1}" if j < max_labels else None)
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
        # Configurar títulos y ejes (límites Y los define el caller: auto rango o fijo)
        ax.set_title(title)
        ax.set_xlim(0, 360)
        ax.set_xlabel("Fase (°)")
        ax.set_ylabel("Amplitud")


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

    def on_toggle_banner_mode(self, checked: bool) -> None:
        """Alterna el banner claro/oscuro y aplica tema."""
        try:
            self.banner_dark_mode = bool(checked)
            self._refresh_signature_banner()
            self._apply_theme_stylesheet(self.banner_dark_mode)
        except Exception:
            pass

    def _refresh_signature_banner(self) -> None:
        """Carga/actualiza el banner inferior (claro u oscuro)."""
        default_path = Path("firma_banner.png")
        dark_path = Path("phaseflux_dark_mode.png")
        sig_path = dark_path if getattr(self, "banner_dark_mode", False) and dark_path.exists() else default_path
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
                self.signature_label.setText("\nFirma/Autoría no encontrada.\nColoque 'firma_banner.png' o 'phaseflux_dark_mode.png' en la carpeta del programa.")
        except Exception:
            self._signature_pixmap = None
            self.signature_label.setText("\nFirma/Autoría no encontrada.\nColoque 'firma_banner.png' o 'phaseflux_dark_mode.png' en la carpeta del programa.")
        self.signature_label.setAlignment(Qt.AlignCenter)

    def _apply_theme_stylesheet(self, dark: bool) -> None:
        """Aplica un tema oscuro/claro a la GUI principal."""
        try:
            if dark:
                self.setStyleSheet(self._dark_stylesheet)
            else:
                self.setStyleSheet("")
            self._apply_axes_theme(dark)
        except Exception:
            pass

    def _apply_axes_theme(self, dark: bool) -> None:
        """Ajusta el fondo de los ejes para que coincidan con el modo (claro/oscuro)."""
        face = "#0b1020" if dark else "#ffffff"
        axes = [
            getattr(self, "ax_raw", None),
            getattr(self, "ax_filtered", None),
            getattr(self, "ax_probs", None),
            getattr(self, "ax_text", None),
            getattr(self, "ax_gap_wide", None),
        ]
        for ax in axes:
            try:
                if ax:
                    ax.set_facecolor(face)
                    if hasattr(ax, "fig"):
                        ax.figure.patch.set_facecolor(face)
            except Exception:
                pass
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

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
            qty_deciles, qty_quints = self._get_qty_filters()

            # Procesar PRPD utilizando el offset de fase, filtro y máscara/pixel
            result = core.process_prpd(
                path=self.current_path,
                out_root=outdir,
                force_phase_offsets=force_offsets,
                fast_mode=False,
                filter_level=filt_label,
                phase_mask=mask_ranges,
                pixel_deciles_keep=pixel_deciles,
            )
            if self.chk_gap.isChecked():
                gx = getattr(self, "_gap_xml_path", None)
                if gx:
                    result["gap_stats"] = self._compute_gap(gx) or {}
            # KPIs básicos (métricas clásicas)
            try:
                result["metrics"] = logic_compute_pd_metrics(result, gap_stats=result.get("gap_stats"))
            except Exception:
                result["metrics"] = {}
            # Calcular cortes globales de quantity (Q/Qt) una sola vez
            try:
                self._precompute_qty_buckets(result)
            except Exception:
                pass
            # KPIs avanzados y clasificador heurístico
            try:
                bins_phase = getattr(self, "_hist_bins_phase", 32)
                bins_amp = getattr(self, "_hist_bins_amp", 32)
                result["metrics_advanced"] = compute_advanced_metrics(
                    {"aligned": result.get("aligned", {}), "angpd": result.get("angpd", {})},
                    bins_amp=bins_amp,
                    bins_phase=bins_phase,
                )
            except Exception:
                result["metrics_advanced"] = {}
            self.last_result = result
            self.last_run_profile = self._collect_current_profile()
            self._apply_qty_filters(result, qty_quints, qty_deciles)
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
            threshold_high = base + 5.0
            threshold_low = base - 5.0
            mask = (amp > threshold_high) | (amp < threshold_low)

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
                "threshold_high": threshold_high,
                "threshold_low": threshold_low,
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
        if value_ms is not None and value_ms >= 500.0:
            return {
                "code": "sin_dp_500",
                "label": "No se detectaron descargas parciales",
                "level_name": "Sin descargas",
                "color": "#00B050",
                "threshold": 500.0,
                "action": "Sin descargas parciales detectadas.",
                "action_short": "Sin descargas parciales detectadas.",
            }
        table = [
            {
                "code": "sin_dp",
                "label": "Sin gap time",
                "level_name": "Sin descargas",
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
    def _percentile_inc(values: list[float], frac: float) -> float | None:
        """Percentil tipo Excel PERCENTIL.INC (interp lineal, 0<=frac<=1)."""
        if not values:
            return None
        vals = sorted(values)
        n = len(vals)
        if n == 1:
            return vals[0]
        if frac <= 0:
            return vals[0]
        if frac >= 1:
            return vals[-1]
        k = (n - 1) * frac
        lo = int(k)
        hi = min(lo + 1, n - 1)
        f = k - lo
        return vals[lo] * (1 - f) + vals[hi] * f

    def _refresh_gap_ext_buttons(self) -> None:
        """Actualiza el texto de los botones de gap extenso con el conteo."""
        try:
            n = len(self.gap_ext_files)
            self.btn_gap_ext_add.setText(f"Gap extenso ({n}/5)")
        except Exception:
            pass

    def on_add_gap_ext(self) -> None:
        """Agrega hasta 5 XML de gap-time para la vista extenso."""
        try:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Selecciona hasta 5 XML de gap-time",
                "",
                "XML (*.xml);;Todos (*.*)",
            )
        except Exception:
            files = []
        if not files:
            return
        updated = list(self.gap_ext_files)
        for f in files:
            pf = Path(f)
            if pf in updated:
                continue
            if len(updated) >= 5:
                break
            updated.append(pf)
        self.gap_ext_files = updated[:5]
        self._refresh_gap_ext_buttons()
        QMessageBox.information(self, "Gap-time extenso",
                                f"Usando {len(self.gap_ext_files)} archivo(s) para Gap-time extenso.")
        if self.last_result and self.cmb_plot.currentText().strip().lower().startswith("gap-time extenso"):
            try:
                self.render_result(self.last_result)
            except Exception:
                pass

    def on_clear_gap_ext(self) -> None:
        """Limpia la selección de gap-time extenso."""
        self.gap_ext_files = []
        self._refresh_gap_ext_buttons()
        if self.last_result and self.cmb_plot.currentText().strip().lower().startswith("gap-time extenso"):
            try:
                self.render_result(self.last_result)
            except Exception:
                pass



    def _render_gap_time_extenso(self) -> None:
        """Renderiza Gap-time extenso como serie combinada (max 5 XML)."""
        self._set_conclusion_mode(False)
        self._restore_standard_axes()
        fig = self.canvas.figure
        fig.texts.clear()

        for ax in (self.ax_raw, self.ax_filtered):
            ax.clear()
            ax.set_visible(False)

        # Usar ax_probs como panel de tabla y ocultar ax_text para evitar textos duplicados
        self.ax_text.clear()
        self.ax_text.set_visible(False)
        table_ax = self.ax_probs
        table_ax.clear()
        table_ax.set_visible(True)
        table_ax.set_facecolor("#ffffff")
        table_ax.axis("off")
        try:
            table_ax.set_xlim(0, 1)
            table_ax.set_ylim(0, 1)
        except Exception:
            pass

        self.ax_gap_wide.set_visible(True)
        ax_series = self.ax_gap_wide
        ax_series.clear()
        ax_series.set_facecolor("#f7f9fb")

        header_txt = "Gap-time extenso (max 5 XML)"

        if not self.gap_ext_files:
            ax_series.axis("off")
            table_ax.text(0.02, 0.90, header_txt, fontsize=12, fontweight="bold", ha="left", va="top")
            table_ax.text(0.02, 0.80, "Agrega XML con el boton 'Gap extenso (0/5)'.", fontsize=11, ha="left", va="top")
            return

        series_t = []
        series_y = []
        spikes_t = []
        spikes_y = []
        gaps_all = []
        rows = []
        offset_ms = 0.0
        base_line = None
        threshold_line_high = None
        threshold_line_low = None

        for idx, path in enumerate(self.gap_ext_files, 1):
            stats: dict = {}
            try:
                stats = self._compute_gap(str(path)) or {}
            except Exception:
                stats = {}

            amp = np.asarray(stats.get("series", []), dtype=float)
            p50 = stats.get("p50_ms")
            p5 = stats.get("p5_ms")
            cls = stats.get("classification") or {}
            label = cls.get("level_name") or "N/D"
            gaps = stats.get("gaps_ms") or []

            dt_ms = float(stats.get("dt_ms") or (0.5 / max(1, amp.size) * 1000.0))
            mask = np.asarray(stats.get("mask", []), dtype=bool)
            if mask.size != amp.size:
                base_tmp = float(stats.get("base") or (np.median(amp) if amp.size else 0.0))
                th_hi = float(stats.get("threshold_high") or (base_tmp + 5.0))
                th_lo = float(stats.get("threshold_low") or (base_tmp - 5.0))
                mask = (amp > th_hi) | (amp < th_lo)

            if amp.size:
                t_local = np.arange(amp.size, dtype=float) * dt_ms + offset_ms
                series_t.extend(t_local.tolist())
                series_y.extend(amp.tolist())
                if mask.any():
                    spikes_t.extend(t_local[mask].tolist())
                    spikes_y.extend(amp[mask].tolist())
                offset_ms += dt_ms * (amp.size + 2)
            else:
                offset_ms += 2.0

            try:
                gaps_all.extend([float(g) for g in gaps])
            except Exception:
                pass

            if base_line is None and stats.get("base") is not None:
                base_line = float(stats["base"])
            th_hi = stats.get("threshold_high")
            th_lo = stats.get("threshold_low")
            if threshold_line_high is None and th_hi is not None:
                threshold_line_high = float(th_hi)
            if threshold_line_low is None and th_lo is not None:
                threshold_line_low = float(th_lo)

            rows.append((idx, path.name, p50, p5, label))

        if not series_t:
            ax_series.axis("off")
            table_ax.text(0.02, 0.90, header_txt, fontsize=12, fontweight="bold", ha="left", va="top")
            table_ax.text(0.02, 0.80, "No se encontraron muestras validas en los XML seleccionados.", fontsize=11, ha="left", va="top")
            return

        p50_all = self._percentile_inc(gaps_all, 0.5) if gaps_all else None
        p5_all = self._percentile_inc(gaps_all, 0.05) if gaps_all else None
        cls_all = self._gap_condition(p50_all, has_activity=bool(spikes_t))

        arr_t = np.asarray(series_t, dtype=float)
        arr_y = np.asarray(series_y, dtype=float)
        ax_series.plot(arr_t, arr_y, color="#0a4fa6", linewidth=1.2, label="Magnitud <max>")
        if spikes_t:
            ax_series.scatter(spikes_t, spikes_y, color="#e65100", s=14, zorder=5, label="Descarga detectada")
        if base_line is not None:
            ax_series.axhline(base_line, color="#1b5e20", linestyle="--", linewidth=1.0, alpha=0.7, label="Mediana")
        if threshold_line_high is not None:
            ax_series.axhline(threshold_line_high, color="#2e7d32", linestyle=":", linewidth=1.2, alpha=0.9, label="Umbral base+5")
        if threshold_line_low is not None:
            ax_series.axhline(threshold_line_low, color="#2e7d32", linestyle=":", linewidth=1.2, alpha=0.9)
        if base_line is not None and threshold_line_high is not None and threshold_line_low is not None:
            ax_series.axhspan(threshold_line_low, threshold_line_high, color="#d0e8d0", alpha=0.2, zorder=0)

        ax_series.grid(True, alpha=0.25, linestyle="--", color="#778")
        ax_series.set_xlabel("Tiempo combinado (ms)")
        ax_series.set_ylabel("Magnitud (dBm)")
        ax_series.set_title(f"Serie combinada de gap-time ({len(gaps_all)} gaps de {len(self.gap_ext_files)} XML)")

        if arr_t.size:
            xpad = max(5.0, (arr_t.max() - arr_t.min()) * 0.02)
            ax_series.set_xlim(left=0.0, right=arr_t.max() + xpad)
        if arr_y.size:
            yr = arr_y.max() - arr_y.min()
            ypad = 1.5 if yr < 5 else yr * 0.08
            ax_series.set_ylim(arr_y.min() - ypad, arr_y.max() + ypad)

        if p50_all is not None:
            ax_series.text(0.02, 0.92, f"P50 = {p50_all:.1f} ms",
                           transform=ax_series.transAxes, fontsize=11, fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.35", fc="#ffe0b2", ec="#f08c00", alpha=0.9))
        if p5_all is not None:
            ax_series.text(0.02, 0.84, f"P5 = {p5_all:.1f} ms",
                           transform=ax_series.transAxes, fontsize=11, fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.35", fc="#ffcdd2", ec="#c62828", alpha=0.9))

        ax_series.legend(loc="upper right", fontsize=8, frameon=True)
        # Badge de severidad siempre visible
        label_txt = ""
        color_txt = "#1f2937"
        if cls_all:
            label_txt = cls_all.get("level_name") or cls_all.get("label") or ""
            color_txt = cls_all.get("color", color_txt)
        else:
            label_txt = "Severidad N/D"

        # Texto en caja coloreada para que resalte
        try:
            # Pequeño contraste: blanco si el color es oscuro
            import matplotlib.colors as mcolors
            r, g, b = mcolors.to_rgb(color_txt)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            txt_color = "#ffffff" if luminance < 0.5 else "#0f172a"
        except Exception:
            txt_color = "#0f172a"
        ax_series.text(
            0.82, 0.08,
            f"Severidad: {label_txt}",
            transform=ax_series.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc=color_txt, ec=color_txt, alpha=0.85),
            color=txt_color,
        )

        table_ax.text(0.02, 0.95, header_txt, fontsize=11, fontweight="bold", ha="left", va="top")
        headers = ["#", "Archivo", "P50 (ms)", "P5 (ms)", "Severidad"]
        xcols = [0.02, 0.10, 0.52, 0.70, 0.88]
        for x, h in zip(xcols, headers):
            table_ax.text(x, 0.87, h, fontsize=9, fontweight="bold", ha="left", va="top")
        y_txt = 0.80
        for idx, name, p50, p5, label in rows:
            name_short = name if len(name) <= 32 else name[:29] + "..."
            table_ax.text(xcols[0], y_txt, f"{idx}", fontsize=9, fontweight="bold", ha="left", va="top")
            table_ax.text(xcols[1], y_txt, name_short, fontsize=9, ha="left", va="top")
            table_ax.text(xcols[2], y_txt, f"{p50:.3f}" if p50 is not None else "N/D", fontsize=9, ha="left", va="top")
            table_ax.text(xcols[3], y_txt, f"{p5:.3f}" if p5 is not None else "N/D", fontsize=9, ha="left", va="top")
            table_ax.text(xcols[4], y_txt, label, fontsize=9, ha="left", va="top")
            y_txt -= 0.07
            if y_txt < 0.20:
                break
        table_ax.text(xcols[0], y_txt - 0.02, f"Total ({len(gaps_all)} gaps de {len(self.gap_ext_files)} XML)", fontsize=9, fontweight="bold", ha="left", va="top")
        table_ax.text(xcols[2], y_txt - 0.02, f"{p50_all:.3f}" if p50_all is not None else "N/D", fontsize=9, fontweight="bold", ha="left", va="top")
        table_ax.text(xcols[3], y_txt - 0.02, f"{p5_all:.3f}" if p5_all is not None else "N/D", fontsize=9, fontweight="bold", ha="left", va="top")
        table_ax.text(xcols[4], y_txt - 0.02, cls_all.get("level_name", "N/D") if cls_all else "N/D", fontsize=9, fontweight="bold", ha="left", va="top")
        table_ax.text(0.02, 0.10, "Etiqueta basada en p50 (tabla _gap_condition).", fontsize=9, style="italic", ha="left", va="top")

    def _render_kpi_avanzados(self, r: dict, payload: dict | None = None) -> None:
        """Vista sencilla para KPIs avanzados / histogramas."""
        self._set_conclusion_mode(False)
        self._restore_standard_axes()
        for ax in (self.ax_raw, self.ax_filtered, self.ax_probs, self.ax_text):
            ax.clear()
            ax.set_facecolor("#fafafa")
            try:
                ax.set_xticks([]); ax.set_yticks([])
            except Exception:
                pass

        metrics = (payload or {}).get("metrics") if isinstance(payload, dict) else None
        if metrics is None:
            metrics = r.get("kpi", {}) if isinstance(r, dict) else {}
        hist_kpi = {}
        try:
            hist_kpi = r.get("kpi", {}).get("hist", {}) if isinstance(r, dict) else {}
        except Exception:
            hist_kpi = {}
        adv = r.get("metrics_advanced", {}) if isinstance(r, dict) else {}

        ax_top = self.ax_raw
        ax_top.set_visible(True)
        txt_lines = ["KPI avanzados", ""]
        if hist_kpi:
            txt_lines.append("Histogramas (N=16):")
            for k, v in hist_kpi.items():
                txt_lines.append(f" - {k}: {v}")
        else:
            txt_lines.append("Sin KPIs de histogramas disponibles.")
        ax_top.text(0.02, 0.98, "\n".join(txt_lines), va="top", ha="left", fontsize=10)

        ax_bottom = self.ax_filtered
        ax_bottom.set_visible(True)
        lines2 = ["Metrics advanced (resumen)", ""]
        if isinstance(adv, dict) and adv:
            for k, v in adv.items():
                if isinstance(v, dict):
                    lines2.append(f"{k}: { {kk: vv for kk,vv in list(v.items())[:5]} }")
                else:
                    lines2.append(f"{k}: {v}")
                if len(lines2) > 18:
                    break
        else:
            lines2.append("Sin metrics_advanced.")
        ax_bottom.text(0.02, 0.98, "\n".join(lines2), va="top", ha="left", fontsize=10)
        self.canvas.draw_idle()

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
        # Histogramas H_amp/H_ph:
        # - Se calculan on-the-fly desde result["aligned"] (phase_deg, amplitude).
        # - N=16 bins (log10(1+conteos)), no se guardan en result.
        # - Semiciclo+: fase<180; Semiciclo-: fase>=180 (fase-180 para H_ph-).
        ax = self.ax_filtered
        ax.clear()
        try:
            aligned = r.get("aligned", {}) or {}
            ph_arr = np.asarray(aligned.get("phase_deg", []), dtype=float)
            amp_arr = np.asarray(aligned.get("amplitude", []), dtype=float)
            if ph_arr.size == 0 or amp_arr.size == 0:
                ax.text(0.5,0.5,'Sin datos',ha='center',va='center');
                self.ax_text.clear(); self.ax_text.text(0.5,0.5,'Sin datos',ha='center',va='center');
                return
            N = 16
            hist = compute_semicycle_histograms_from_aligned({"phase_deg": ph_arr, "amplitude": amp_arr}, N=N)
            Ha_pos = hist.get("H_amp_pos", np.zeros(N))
            Ha_neg = hist.get("H_amp_neg", np.zeros(N))
            Hp_pos = hist.get("H_ph_pos", np.zeros(N))
            Hp_neg = hist.get("H_ph_neg", np.zeros(N))
            xi = np.arange(1, N+1)
            ax.plot(xi, Ha_pos, '-o', color='#1f77b4', label='H_amp+')
            ax.plot(xi, Ha_neg, '-o', color='#d62728', label='H_amp-')

            # Overlay con bins avanzados (quantity) si existen. Fallos aquí no deben ocultar el histograma base.
            try:
                m_adv = r.get("metrics_advanced", {}) if isinstance(r, dict) else {}
                hist_adv = m_adv.get("hist", {}) if isinstance(m_adv, dict) else {}
                amp_pos_adv = np.asarray(hist_adv.get("amp_hist_pos", []), dtype=float)
                amp_edges_pos = np.asarray(hist_adv.get("amp_edges_pos", []), dtype=float)
                amp_neg_adv = np.asarray(hist_adv.get("amp_hist_neg", []), dtype=float)
                amp_edges_neg = np.asarray(hist_adv.get("amp_edges_neg", []), dtype=float)
                if amp_pos_adv.size and amp_edges_pos.size == amp_pos_adv.size + 1:
                    centers_pos = (amp_edges_pos[:-1] + amp_edges_pos[1:]) / 2.0
                    max_pos = np.max(amp_pos_adv) if amp_pos_adv.size else 0.0
                    amp_pos_norm = amp_pos_adv / max_pos if max_pos > 0 else amp_pos_adv
                    ax.plot(centers_pos, amp_pos_norm, '--', color='#1f77b4', alpha=0.7, label='H_amp+ (bins qty)')
                if amp_neg_adv.size and amp_edges_neg.size == amp_neg_adv.size + 1:
                    centers_neg = (amp_edges_neg[:-1] + amp_edges_neg[1:]) / 2.0
                    max_neg = np.max(amp_neg_adv) if amp_neg_adv.size else 0.0
                    amp_neg_norm = amp_neg_adv / max_neg if max_neg > 0 else amp_neg_adv
                    ax.plot(centers_neg, amp_neg_norm, '--', color='#d62728', alpha=0.7, label='H_amp- (bins qty)')
            except Exception:
                pass

            ax.set_xlabel('Indice de ventana (N=16) / centros bins'); ax.set_ylabel('H_amp (norm)')
            ax.set_title('Histograma de Amplitud (N=16 + qty)')
            ax.legend(loc='upper right', fontsize=8)

            # H_ph en panel inferior derecho
            bx = self.ax_text
            bx.clear()
            bx.plot(xi, Hp_pos, '-o', color='#1f77b4', label='H_ph+')
            bx.plot(xi, Hp_neg, '-o', color='#d62728', label='H_ph-')

            try:
                hist_adv = r.get("metrics_advanced", {}).get("hist", {}) if isinstance(r, dict) else {}
                ph_edges_adv = np.asarray(hist_adv.get("ph_edges", []), dtype=float)
                phase_hist_pos = np.asarray(hist_adv.get("phase_hist_pos", []), dtype=float)
                phase_hist_neg = np.asarray(hist_adv.get("phase_hist_neg", []), dtype=float)
                if ph_edges_adv.size and phase_hist_pos.size:
                    mid = len(ph_edges_adv) // 2
                    if mid > 0:
                        centers_pos = (ph_edges_adv[:mid] + ph_edges_adv[1:mid+1]) / 2.0
                        max_ph_pos = np.max(phase_hist_pos) if phase_hist_pos.size else 0.0
                        ph_pos_norm = phase_hist_pos / max_ph_pos if max_ph_pos > 0 else phase_hist_pos
                        bx.plot(centers_pos, ph_pos_norm, '--', color='#1f77b4', alpha=0.7, label='H_ph+ (bins)')
                if ph_edges_adv.size and phase_hist_neg.size:
                    mid = len(ph_edges_adv) // 2
                    if mid > 0:
                        edges_neg = ph_edges_adv[mid:]
                        if edges_neg.size >= 2:
                            centers_neg = (edges_neg[:-1] + edges_neg[1:]) / 2.0
                            centers_neg = centers_neg - 180.0
                            max_ph_neg = np.max(phase_hist_neg) if phase_hist_neg.size else 0.0
                            ph_neg_norm = phase_hist_neg / max_ph_neg if max_ph_neg > 0 else phase_hist_neg
                            bx.plot(centers_neg, ph_neg_norm, '--', color='#d62728', alpha=0.7, label='H_ph- (bins)')
            except Exception:
                pass

            bx.set_xlabel('Indice de ventana (N=16) / centros bins'); bx.set_ylabel('H_ph (norm)')
            bx.set_title('Histograma de Fase (N=16 + bins)')
            bx.legend(loc='upper right', fontsize=8)
        except Exception:
            ax.text(0.5,0.5,'Error histogramas',ha='center',va='center');
            try:
                self.ax_text.text(0.5,0.5,'Error histogramas',ha='center',va='center')
            except Exception:
                pass

    def _draw_combined_panels(self, r: dict, ph_al: np.ndarray, amp_al: np.ndarray, centers_s3: list[dict] | None = None) -> None:
        self._draw_combined_overlay(
            ax=self.ax_raw,
            twin_attr="ax_raw_twin",
            ph=ph_al,
            amp=amp_al,
            ang=r.get("angpd", {}),
            centers=centers_s3,
            quantity=False,
        )
        self._draw_combined_overlay(
            ax=self.ax_probs,
            twin_attr="ax_probs_twin",
            ph=ph_al,
            amp=amp_al,
            ang=r.get("angpd", {}),
            centers=centers_s3,
            quantity=True,
        )

    def _render_nubes_grid(self, ph_al: np.ndarray, amp_al: np.ndarray, quint_idx: np.ndarray,
                           clouds_s3: list[dict], clouds_s4: list[dict], clouds_s5: list[dict]) -> None:
        ui_render.render_nubes_grid(self, ph_al, amp_al, quint_idx, clouds_s3, clouds_s4, clouds_s5)

    def _render_conclusions(self, result: dict, payload: dict | None = None) -> None:
        ui_render.render_conclusions(self, result, payload)

    def _render_ann_gap_view(self, result: dict, payload: dict | None = None) -> None:
        ui_render.render_ann_gap_view(self, result, payload)

    def _render_gap_time_full(self, result: dict, payload: dict | None = None) -> None:
        ui_render.render_gap_time_full(self, result, payload)

    def _draw_gap_summary_panel(self, ax, metrics: dict, gap_stats: dict | None) -> None:
        """Wrapper: delega el render del panel de gap-time al helper ui_draw."""
        return ui_draw.draw_gap_summary_panel(
            ax,
            metrics,
            gap_stats,
            draw_tag=self._draw_status_tag,
            draw_title=self._draw_section_title,
            register=self._register_conclusion_artist,
        )

    def _draw_gap_summary_split(self, ax, metrics: dict, gap_stats: dict | None, side: str) -> None:
        """Wrapper: usa helper compartido para paneles P50/P5."""
        return ui_draw.draw_gap_summary_split(
            ax,
            metrics,
            gap_stats,
            side,
            draw_tag=self._draw_status_tag,
            draw_title=self._draw_section_title,
            register=self._register_conclusion_artist,
        )

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
        for target in (self.ax_probs, self.ax_text):
            target.clear()
            target.set_facecolor("#fafafa")

    def _infer_ann_probabilities(self, metrics: dict | None, summary: dict | None, mask_label: str | None) -> dict:
        metrics = metrics or {}
        summary = summary or {}
        mask = (mask_label or "").strip().lower()
        probs = {"corona": 0.0, "superficial": 0.0, "cavidad": 0.0}

        def _to_float(value):
            try:
                return float(value)
            except Exception:
                return None

        def _pct_diff(a, b):
            a = _to_float(a)
            b = _to_float(b)
            if a is None or b is None:
                return 100.0
            denom = max((abs(a) + abs(b)) * 0.5, 1e-6)
            return abs(a - b) / denom * 100.0

        if "corona" in mask:
            probs["corona"] = 1.0
            return probs
        if "void" in mask or "cavidad" in mask:
            probs["cavidad"] += 0.5
        if "superficial" in mask:
            probs["superficial"] += 0.4

        count_pos = _to_float(metrics.get("count_pos"))
        count_neg = _to_float(metrics.get("count_neg"))
        width_pos = _to_float(metrics.get("phase_width_pos"))
        width_neg = _to_float(metrics.get("phase_width_neg"))
        amp_pos = _to_float(metrics.get("amp_p95_pos"))
        amp_neg = _to_float(metrics.get("amp_p95_neg"))
        ratio = _to_float(metrics.get("n_ang_ratio")) or 0.0
        n_peaks_tot = int(metrics.get("n_peaks") or 0)
        n_peaks_pos = int(metrics.get("n_peaks_pos") or 0)
        n_peaks_neg = int(metrics.get("n_peaks_neg") or 0)

        diff_counts = _pct_diff(count_pos, count_neg)
        diff_width = _pct_diff(width_pos, width_neg)
        diff_amp = _pct_diff(amp_pos, amp_neg)
        symmetry_hits = sum(1 for d in (diff_counts, diff_width, diff_amp) if d <= 4.0)
        asym_hits = sum(1 for d in (diff_counts, diff_width, diff_amp) if d > 4.0)

        cavity_score = max(0.0, symmetry_hits)
        if symmetry_hits >= 2:
            cavity_score += 1.5
        elif symmetry_hits == 1:
            cavity_score += 0.5

        superficial_score = float(asym_hits)
        if n_peaks_tot > 2 or max(n_peaks_pos, n_peaks_neg) >= 2:
            superficial_score += 1.5
        if ratio > 3.0:
            superficial_score += 1.2
        elif ratio > 1.0:
            superficial_score += 0.7

        if asym_hits == 0 and ratio <= 1.0:
            cavity_score += 0.8

        probs["cavidad"] += max(0.0, cavity_score)
        probs["superficial"] += max(0.0, superficial_score)

        inferred = (summary.get("pd_type") or "").lower()
        if not any(probs.values()):
            if "superficial" in inferred or "tracking" in inferred:
                probs["superficial"] = 1.0
            elif "corona" in inferred:
                probs["corona"] = 1.0
            else:
                probs["cavidad"] = 1.0

        total = sum(probs.values())
        if total <= 0:
            probs["cavidad"] = 1.0
            total = 1.0
        for key in probs:
            probs[key] = max(0.0, float(probs[key]) / total)
        return probs

    def _draw_ann_prediction_panel(self, result: dict, summary: dict, metrics: dict | None = None, metrics_adv: dict | None = None) -> None:
        ax = self.ax_raw
        ax.clear()
        ax.set_facecolor("#f3f6fb")
        metrics = metrics or {}
        metrics_adv = metrics_adv or result.get("metrics_advanced") or {}
        mask_label = (self.last_run_profile or {}).get("mask", self.cmb_masks.currentText())
        # Usar rule_pd si existe, si no, fallback a heurística previa
        rule = result.get("rule_pd", {}) if isinstance(result, dict) else {}
        rule_probs = rule.get("class_probs", {}) if isinstance(rule, dict) else {}
        if rule_probs:
            p_corona = float(rule_probs.get("corona", 0.0))
            p_sup = float(rule_probs.get("superficial_tracking", 0.0)) + float(rule_probs.get("flotante", 0.0))
            p_cav = float(rule_probs.get("cavidad_interna", 0.0))
            total_rule = p_corona + p_sup + p_cav
            if total_rule > 0:
                p_corona /= total_rule
                p_sup /= total_rule
                p_cav /= total_rule
            values = [p_corona, p_sup, p_cav]
            labels = ["Corona +/-", "Superficial / Tracking", "Cavidad"]
            colors = [
                CLASS_INFO.get("corona", {}).get("color", "#888888"),
                CLASS_INFO.get("superficial", {}).get("color", "#888888") or CLASS_INFO.get("superficial_tracking", {}).get("color", "#888888"),
                CLASS_INFO.get("cavidad", {}).get("color", "#888888") or CLASS_INFO.get("cavidad_interna", {}).get("color", "#888888"),
            ]
            bar_keys = ["corona", "superficial_tracking", "cavidad_interna"]
        else:
            heur_probs = {}
            try:
                heur_probs = metrics_adv.get("heuristic_probs") or {}
            except Exception:
                heur_probs = {}

            ann_probs = {}
            if heur_probs:
                ann_probs = {str(k).lower(): float(v) for k, v in heur_probs.items() if v is not None}
            else:
                ann_probs = self._last_ann_probs or self._infer_ann_probabilities(metrics, summary, mask_label)
            ann_norm = {str(k).lower(): float(v) for k, v in ann_probs.items() if v is not None}

            # Orden principal de barras (sin duplicar corona si viene agregada)
            bar_keys = list(self.pd_classes or CLASS_NAMES)
            labels = []
            colors = []
            values = []
            for k in bar_keys:
                info = CLASS_INFO.get(k, {"name": k, "color": "#888888"})
                labels.append(info.get("name", k))
                colors.append(info.get("color", "#888888"))
                values.append(ann_norm.get(k, 0.0))

            # Fallback si todo está en cero
            if not any(values):
                inferred = (summary.get("pd_type") or "").lower()
                if "superficial" in inferred or "tracking" in inferred:
                    if "superficial" in bar_keys:
                        values[bar_keys.index("superficial")] = 1.0
                elif "corona" in inferred:
                    values[bar_keys.index("corona")] = 1.0
                elif "cavidad" in inferred:
                    if "cavidad" in bar_keys:
                        values[bar_keys.index("cavidad")] = 1.0
                else:
                    if "ruido" in bar_keys:
                        values[bar_keys.index("ruido")] = 0.5

            values = [max(0.0, float(v)) for v in values]
            total = sum(values)
            if total <= 0:
                total = 1.0
            values = [min(1.0, max(0.0, v / total)) for v in values]
            try:
                self._last_ann_probs = {k: v for k, v in zip(bar_keys, values)}
            except Exception:
                pass

        bars = ax.bar(labels, values, color=colors, edgecolor="#37474f", linewidth=1.0)
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
            # Ubicar dentro del panel sin recortar el texto
            self._draw_status_tag(ax, label, 0.72, 0.08, color=color)
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

    def _draw_combined_overlay(self, ax, twin_attr: str, ph: np.ndarray, amp: np.ndarray, ang: dict, *, quantity: bool, centers: list[dict] | None = None, centers_max_labels: int = 4) -> None:
        ax.clear()
        if ph.size and amp.size:
            ax.scatter(ph, amp, s=4, alpha=0.25, color="#bdbdbd", label="Nubes S3")
            if centers:
                palette = [
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                ]
                for idx, c in enumerate(centers):
                    cx = float(c.get("phase_mean", 0.0))
                    cy = float(c.get("y_mean", 0.0))
                    color = palette[idx % len(palette)]
                    lbl = c.get("legend") or f"C{idx+1}"
                    label = f"Centro {lbl}" if idx < centers_max_labels else None
                    ax.scatter([cx], [cy], s=50, color=color, edgecolors="black", zorder=5, label=label)
                # Si hay más centros que el límite, añadir un ítem “…” a la leyenda sin dibujar punto extra
                if len(centers) > centers_max_labels:
                    ax.scatter([], [], color="#7f7f7f", edgecolors="black", s=40, label="…")
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
            # Reusar métricas si ya existen; si no, calcúlalas
            metrics = result.get("metrics") if isinstance(result, dict) else None
            if not metrics:
                metrics = logic_compute_pd_metrics(result, gap_stats=gap_stats)
                if isinstance(result, dict):
                    result["metrics"] = metrics
        except Exception:
            metrics = {}
        if not metrics.get("total_count"):
            return ("Sin datos alineados para analizar conclusiones.", {})
        summary = logic_classify_pd(metrics)
        payload = {
            "metrics": metrics,
            "summary": summary,
            "phase_offset": result.get("phase_offset"),
            "filter": self._get_filter_label(),
            "gap": gap_stats or {},
            "metrics_advanced": result.get("metrics_advanced", {}),
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
        is_gap_ext = view_mode.startswith("gap-time extenso")
        is_gap_full = view_mode.startswith("gap-time") and not is_gap_ext
        is_kpi_adv = view_mode.startswith("kpi avanzados")
        is_fa_profile = view_mode.startswith("fa profile")

        text, payload = self._get_conclusion_insight(r)
        payload = payload or {}
        self.last_conclusion_text = text
        self.last_conclusion_payload = payload
        stem_for_history = self.current_path.stem if self.current_path else None
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        manual = self.manual_override if getattr(self, "manual_override", {}).get("enabled") else None
        mask_label = (self.last_run_profile or {}).get("mask", self.cmb_masks.currentText())
        self._last_ann_probs = self._infer_ann_probabilities(metrics, summary, mask_label)
        self._append_ann_history(self._last_ann_probs)
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
        if is_gap_ext:
            self._render_gap_time_extenso()
            self.canvas.draw_idle()
            return
        if is_kpi_adv:
            self._render_kpi_avanzados(r, payload)
            self.canvas.draw_idle()
            return
        if is_fa_profile:
            ph_al = np.asarray(r.get("aligned", {}).get("phase_deg", []), dtype=float)
            amp_al = np.asarray(r.get("aligned", {}).get("amplitude", []), dtype=float)
            self._draw_fa_profile_view(r, ph_al, amp_al)
            self.canvas.draw_idle()
            return
        if is_fa_profile:
            ph_al = np.asarray(r.get("aligned", {}).get("phase_deg", []), dtype=float)
            amp_al = np.asarray(r.get("aligned", {}).get("amplitude", []), dtype=float)
            self._draw_fa_profile_view(r, ph_al, amp_al)
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
        self._apply_auto_ylim(self.ax_raw, a0)

        # Alineado/filtrado con overlay de ruido
        self.ax_filtered.clear()
        ph_al = np.asarray(r.get("aligned", {}).get("phase_deg", []), dtype=float)
        amp_al = np.asarray(r.get("aligned", {}).get("amplitude", []), dtype=float)
        aligned_dict = r.get("aligned", {}) or {}
        quint_idx = self._resolve_quintiles(ph_al, r, aligned_dict)
        qty_deciles = np.asarray(aligned_dict.get("qty_deciles", []), dtype=int)
        if qty_deciles.size == quint_idx.size:
            self.last_qty_deciles = qty_deciles
        else:
            self.last_qty_deciles = None
        # Paleta fija para Q1..Q5 (coincide con la macro de Excel)
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
                        alpha=0.8, label=f"Q{q_idx}"
                    )
                if colored:
                    try:
                        self.ax_filtered.legend(loc="lower right", fontsize=8, framealpha=0.7)
                    except Exception:
                        pass
            if not colored and not use_hist2d:
                self.ax_filtered.scatter(ph_al, amp_al, s=3, alpha=0.7)
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
        self._apply_auto_ylim(self.ax_filtered, amp_al)

        # Fijar escala Y de 0..100 para la vista alineada/filtrada
        # (auto rango opcional ya aplicado)
        # Panel inferior izquierdo: Probabilidades / ANGPD / Nubes (S3/S4/S5)
        self.ax_probs.clear()
        # Determinar modo de vista según el texto del combo. Convertir a minúsculas
        view_mode = self.cmb_plot.currentText().strip().lower()
        variants: dict[str, list[dict]] = {}
        selected_s3: list[dict] = []
        clouds_s4: list[dict] = []
        clouds_s5: list[dict] = []
        if ph_al.size and amp_al.size:
            variants, selected_s3, clouds_s4, clouds_s5 = self._build_cluster_variants(
                ph_al,
                amp_al,
                current_filter=self._get_filter_label(),
            )

        if view_mode.startswith("manual"):
            # Abrir diálogo de override manual y volver a conclusiones
            opened = self._open_manual_override_dialog()
            # Si canceló, deshabilitar override manual
            if not opened:
                self.manual_override["enabled"] = False
            try:
                self.cmb_plot.blockSignals(True)
                self.cmb_plot.setCurrentText("Conclusiones")
            finally:
                self.cmb_plot.blockSignals(False)
            # Renderizar conclusiones con (o sin) override y salir
            self._render_conclusions(r, payload)
            self.canvas.draw_idle()
            return

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
        elif view_mode == "nubes":
            # Vista unificada: Alineado/filtrado + S3 + S4 + S5 en 2x2
            self._render_nubes_grid(ph_al, amp_al, quint_idx, selected_s3, clouds_s4, clouds_s5)
        elif view_mode.startswith("combinada"):
            try:
                self._draw_histograms_semiciclo(r)
            except Exception:
                pass
            centers_for_combined = selected_s3 if self.chk_centers_combined.isChecked() else None
            self._draw_combined_panels(r, ph_al, amp_al, centers_for_combined)
        elif view_mode == "nubes":
            # Vista unificada ya renderizada arriba (nubes_grid)
            pass
        else:
            self._render_conclusions(r)

        # Mantener 0..100 en vistas de Nubes (S3/S4/S5)
        try:
            if view_mode.startswith("nubes"):
                self.ax_probs.set_ylim(0, 100)
        except Exception:
            pass

        try:
            manual_active = getattr(self, "manual_override", {}).get("enabled", False) and view_mode.startswith("manual")
            if not manual_active:
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
        gap_stats = result.get("gap_stats") or {}
        metrics = logic_compute_pd_metrics(result, gap_stats=gap_stats)
        summary = logic_classify_pd(metrics)
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
        variants, selected_s3, clouds_s4, clouds_s5 = self._build_cluster_variants(
            ph_al,
            amp_al,
            current_filter=self._get_filter_label(),
        )
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
        centers_for_save = selected_s3 if self.chk_centers_combined.isChecked() else None
        self._save_combined_overlays(ph_al, amp_al, ang, session_dir / f"{stem}_combined_{profile_tag}.png", centers=centers_for_save)
        self._save_gap_outputs(gap_stats, metrics, session_dir, stem, profile_tag)
        self._save_ann_prediction_plot(metrics, summary, session_dir, stem, profile_tag, metrics_adv=result.get("metrics_advanced"))
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

    def _save_combined_overlays(self, ph: np.ndarray, amp: np.ndarray, ang: dict, path: Path, centers: list[dict] | None = None) -> None:
        import matplotlib.pyplot as _plt
        if not ph.size or not amp.size:
            return
        x = np.asarray(ang.get("phi_centers", []), dtype=float)
        if not x.size:
            return
        fig, axes = _plt.subplots(2, 1, figsize=(7, 6), dpi=130, sharex=True)
        self._plot_combined_panel(axes[0], ph, amp, x, ang, quantity=False, title="Nubes + ANGPD", centers=centers)
        self._plot_combined_panel(axes[1], ph, amp, x, ang, quantity=True, title="Nubes + ANGPD qty", centers=centers)
        axes[1].set_xlabel("Fase (°)")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        _plt.close(fig)

    def _save_gap_outputs(self, gap_stats: dict, metrics: dict, session_dir: Path, stem: str, suffix: str) -> None:
        """Guarda la gráfica de gap-time, paneles P50/P5 y JSON."""
        if not gap_stats:
            return
        import matplotlib.pyplot as _plt
        session_dir = Path(session_dir)
        try:
            fig, ax = _plt.subplots(figsize=(7, 4), dpi=130)
            self._draw_gap_chart(ax, gap_stats)
            fig.tight_layout()
            fig.savefig(session_dir / f"{stem}_gaptime_{suffix}.png", bbox_inches="tight")
            _plt.close(fig)
        except Exception:
            traceback.print_exc()
        try:
            fig, axes = _plt.subplots(1, 2, figsize=(10, 4), dpi=130)
            self._draw_gap_summary_split(axes[0], metrics, gap_stats, side="p50")
            self._draw_gap_summary_split(axes[1], metrics, gap_stats, side="p5")
            fig.tight_layout()
            fig.savefig(session_dir / f"{stem}_gaptime_resumen_{suffix}.png", bbox_inches="tight")
            _plt.close(fig)
        except Exception:
            traceback.print_exc()
        try:
            (session_dir / f"{stem}_gaptime_{suffix}.json").write_text(json.dumps(gap_stats, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            traceback.print_exc()

    def _save_conclusions_png(self, payload: dict, session_dir: Path, stem: str, suffix: str) -> None:
        """Renderiza conclusiones a PNG con formato similar a la GUI."""
        import matplotlib.pyplot as _plt
        from matplotlib.patches import FancyBboxPatch
        session_dir = Path(session_dir)
        metrics = payload.get("metrics", {})
        summary = payload.get("summary", {})
        risk = (summary.get("risk", "N/D") or "N/D")
        stage = summary.get("stage", "N/D")
        life_score = summary.get("life_score", "N/D")
        life_years = summary.get("life_years", "N/D")
        pd_type = summary.get("pd_type", "N/D")
        location = summary.get("location", "N/D")
        palette = {
            "bajo": "#00B050",
            "aceptable": "#00B050",
            "moderado": "#1565c0",
            "grave": "#FF8C00",
            "alto": "#FF8C00",
            "critico": "#B00000",
            "crítico": "#B00000",
            "sin descargas": "#00B050",
            "descargas parciales no detectadas": "#00B050",
        }
        risk_color = palette.get(str(risk).lower(), "#1565c0")
        estado_map = {
            "bajo": "Aceptable",
            "aceptable": "Aceptable",
            "moderado": "Moderado",
            "grave": "Grave",
            "alto": "Grave",
            "critico": "Crítico",
            "crítico": "Crítico",
            "sin descargas": "Sin descargas",
            "descargas parciales no detectadas": "Sin descargas",
        }
        estado_map = {
            "bajo": "Aceptable",
            "aceptable": "Aceptable",
            "moderado": "Moderado",
            "grave": "Grave",
            "alto": "Grave",
            "critico": "Crítico",
            "crítico": "Crítico",
            "sin descargas": "Sin descargas",
            "descargas parciales no detectadas": "Sin descargas",
        }
        risk_label = estado_map.get(str(risk).lower(), risk)

        fig, ax = _plt.subplots(figsize=(9.5, 5.5), dpi=140)
        ax.axis("off")
        # tarjeta general
        card = FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.015", linewidth=1.2, facecolor="#fffdf5", edgecolor="#c8c6c3", transform=ax.transAxes)
        ax.add_patch(card)
        # tarjetas internas
        left = FancyBboxPatch((0.02, 0.08), 0.46, 0.86, boxstyle="round,pad=0.02", linewidth=1.0, facecolor="#ffffff", edgecolor="#d9d4c7", transform=ax.transAxes)
        right = FancyBboxPatch((0.52, 0.08), 0.46, 0.86, boxstyle="round,pad=0.02", linewidth=1.0, facecolor="#ffffff", edgecolor="#d9d4c7", transform=ax.transAxes)
        for patch in (left, right):
            ax.add_patch(patch)

        def header(txt, x, y):
            ax.text(x, y, txt, transform=ax.transAxes, fontsize=13, fontweight="bold", color="#0f172a", bbox=dict(boxstyle="round,pad=0.25", facecolor="#e8ecf7", edgecolor="none"))

        header("RESULTADOS DE LOS PRINCIPALES KPI", 0.03, 0.95)
        header("Seguimiento y criticidad", 0.54, 0.91)
        header("Indicadores clave", 0.03, 0.91)

        def row(x, y, label, value, color="#111827"):
            ax.text(x, y, label, transform=ax.transAxes, fontsize=11.5, fontweight="bold", color="#0f172a", ha="left", va="center")
            ax.text(x + 0.22, y, value, transform=ax.transAxes, fontsize=11.5, color=color, ha="left", va="center")

        y = 0.85
        row(0.03, y, "Total pulsos útiles", f"{metrics.get('total_count','N/D')}"); y -= 0.05
        row(0.03, y, "Anchura fase", f"{metrics.get('phase_width','N/D')}°"); y -= 0.05
        row(0.03, y, "Centro", f"{metrics.get('phase_center','N/D')}°"); y -= 0.05
        row(0.03, y, "Número de picos de fase", f"{metrics.get('n_peaks','N/D')}"); y -= 0.05
        row(0.03, y, "P95 amplitud", f"{metrics.get('p95_mean','N/D')}"); y -= 0.05
        row(0.03, y, "Gap-Time P50", f"{metrics.get('gap_p50','N/D')} ms"); y -= 0.05
        row(0.03, y, "Gap-Time P5", f"{metrics.get('gap_p5','N/D')} ms"); y -= 0.05
        row(0.03, y, "Relación N-ANGPD/ANGPD", f"{metrics.get('n_ang_ratio','N/D')}"); y -= 0.05

        y_r = 0.85
        life_txt = life_score if isinstance(life_score, (int, float)) else "N/A"
        vida_txt = life_years if isinstance(life_years, (int, float)) else "N/A"
        badge_text = f"{risk_label}   |   LifeScore: {life_txt}   |   Vida remanente: {vida_txt} años"
        ax.text(0.54, y_r, badge_text, transform=ax.transAxes, ha="left", va="center", fontsize=12, color="#ffffff", fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", facecolor=risk_color, edgecolor="none"))
        y_r -= 0.10
        ax.text(0.54, y_r, "ACCIÓN RECOMENDADA", transform=ax.transAxes, fontsize=11.5, fontweight="bold", ha="left", va="center")
        y_r -= 0.06
        actions = summary.get("actions", "Sin acciones registradas.")
        for line in [p.strip() for p in str(actions).split(".") if p.strip()]:
            ax.text(0.54, y_r, line, transform=ax.transAxes, fontsize=11, color="#0d47a1", fontweight="bold", ha="left", va="center", bbox=dict(boxstyle="round,pad=0.20", facecolor="#e0ebff", edgecolor="none"))
            y_r -= 0.055
        y_r -= 0.05
        ax.text(0.54, y_r, "ETAPA PROBABLE", transform=ax.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="center")
        ax.text(0.74, y_r, stage, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color="#1e88e5", ha="left", va="center", bbox=dict(boxstyle="round,pad=0.15", facecolor="#e9f3ff", edgecolor="none"))
        y_r -= 0.06
        ax.text(0.54, y_r, "MODO DOMINANTE", transform=ax.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="center")
        ax.text(0.74, y_r, pd_type, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color="#1e88e5", ha="left", va="center", bbox=dict(boxstyle="round,pad=0.15", facecolor="#e9f3ff", edgecolor="none"))
        y_r -= 0.06
        ax.text(0.54, y_r, "UBICACIÓN PROBABLE", transform=ax.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="center")
        ax.text(0.74, y_r, location, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color="#1e88e5", ha="left", va="center", bbox=dict(boxstyle="round,pad=0.15", facecolor="#e9f3ff", edgecolor="none"))
        y_r -= 0.06
        ax.text(0.54, y_r, "RIESGO", transform=ax.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="center")
        ax.text(0.74, y_r, risk_label, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color="#ffffff", ha="left", va="center", bbox=dict(boxstyle="round,pad=0.20", facecolor=risk_color, edgecolor="none"))
        fig.savefig(session_dir / f"{stem}_conclusiones_{suffix}.png", bbox_inches="tight")
        _plt.close(fig)

    def _save_ann_prediction_plot(self, metrics: dict, summary: dict, session_dir: Path, stem: str, suffix: str, *, metrics_adv: dict | None = None) -> None:
        """Guarda la barra ANN/heurística de predicción de fuente PD."""
        import matplotlib.pyplot as _plt
        session_dir = Path(session_dir)
        mask_label = (self.last_run_profile or {}).get("mask", self.cmb_masks.currentText())
        heur_probs = (metrics_adv or {}).get("heuristic_probs") if isinstance(metrics_adv, dict) else {}
        ann_probs = {}
        if heur_probs:
            ann_probs = {str(k).lower(): float(v) for k, v in heur_probs.items() if v is not None}
        else:
            ann_probs = self._last_ann_probs or self._infer_ann_probabilities(metrics, summary, mask_label)
        ann_norm = {str(k).lower(): float(v) for k, v in ann_probs.items() if v is not None}

        bar_keys = list(self.pd_classes or CLASS_NAMES)
        labels = []
        color_map = []
        values = []
        for k in bar_keys:
            info = CLASS_INFO.get(k, {"name": k, "color": "#888888"})
            labels.append(info.get("name", k))
            color_map.append(info.get("color", "#888888"))
            values.append(ann_norm.get(k, 0.0))

        values = [max(0.0, float(v)) for v in values]
        total = sum(values)
        if total <= 0:
            # fallback con summary
            inferred = (summary.get("pd_type") or "").lower()
            if "superficial" in inferred or "tracking" in inferred:
                values[bar_keys.index("superficial")] = 1.0
            elif "corona" in inferred:
                values[bar_keys.index("corona")] = 1.0
            elif "cavidad" in inferred:
                values[bar_keys.index("cavidad")] = 1.0
            else:
                values[bar_keys.index("ruido")] = 0.5
            total = sum(values)
        if total <= 0:
            total = 1.0
        values = [min(1.0, max(0.0, v / total)) for v in values]

        fig, ax = _plt.subplots(figsize=(7.5, 4.0), dpi=130)
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
        ax.text(0.5, -0.25, text, transform=ax.transAxes, ha="center", va="top", fontsize=12, fontweight="bold", color="#1f2933")
        fig.tight_layout()
        fig.savefig(session_dir / f"{stem}_ann_{suffix}.png", bbox_inches="tight")
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
                try:
                    self._save_conclusions_png(payload, session_dir, stem, suffix)
                except Exception:
                    traceback.print_exc()
        except Exception:
            traceback.print_exc()

    def _plot_combined_panel(self, ax, ph, amp, x, ang, *, quantity: bool, title: str, centers: list[dict] | None = None, centers_max_labels: int = 4) -> None:
        x = np.asarray(ang.get("phi_centers", []), dtype=float)
        if not ph.size or not amp.size or not x.size:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            return
        ax.scatter(ph, amp, s=4, alpha=0.25, color="#bfbfbf", label="Nubes S3")
        if centers:
            palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            ]
            for idx, c in enumerate(centers):
                cx = float(c.get("phase_mean", 0.0))
                cy = float(c.get("y_mean", 0.0))
                color = palette[idx % len(palette)]
                lbl = c.get("legend") or f"C{idx+1}"
                label = f"Centro {lbl}" if idx < centers_max_labels else None
                ax.scatter([cx], [cy], s=55, color=color, edgecolors="black", zorder=4, label=label)
            if len(centers) > centers_max_labels:
                ax.scatter([], [], color="#7f7f7f", edgecolors="black", s=45, label="…")
        ax.set_xlim(0, 360)
        self._apply_auto_ylim(ax, amp)
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
            hist = compute_semicycle_histograms_from_aligned({"phase_deg": ph, "amplitude": amp}, N=N)
            xi = _np.arange(1, N + 1)
            Ha_pos = hist.get("H_amp_pos", _np.zeros(N))
            Ha_neg = hist.get("H_amp_neg", _np.zeros(N))
            Hp_pos = hist.get("H_ph_pos", _np.zeros(N))
            Hp_neg = hist.get("H_ph_neg", _np.zeros(N))
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
            try:
                data_amp = _np.column_stack([xi, Ha_pos, Ha_neg])
                header_amp = "bin,H_amp_pos,H_amp_neg"
                _np.savetxt(out_reports / f"{stem}_hist_amp_{suffix}.csv", data_amp, delimiter=",", header=header_amp, comments="")
            except Exception:
                traceback.print_exc()
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
            try:
                data_phase = _np.column_stack([xi, Hp_pos, Hp_neg])
                header_phase = "bin,H_ph_pos,H_ph_neg"
                _np.savetxt(out_reports / f"{stem}_hist_phase_{suffix}.csv", data_phase, delimiter=",", header=header_phase, comments="")
            except Exception:
                traceback.print_exc()
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
                                                include_k=True,
                                                max_labels=10)
                ax.set_ylim(0, 100)
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
            ax.set_ylim(0, 100)
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
            ax.set_ylim(0, 100)
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
                        )
                        try:
                            self._precompute_qty_buckets(res)
                            self._apply_qty_filters(res, qty_quints, qty_deciles)
                            bins_phase = getattr(self, "_hist_bins_phase", 32)
                            bins_amp = getattr(self, "_hist_bins_amp", 32)
                            res["metrics_advanced"] = compute_advanced_metrics(
                                {"aligned": res.get("aligned", {}), "angpd": res.get("angpd", {})},
                                bins_amp=bins_amp,
                                bins_phase=bins_phase,
                            )
                            res["metrics"] = logic_compute_pd_metrics(res, gap_stats=res.get("gap_stats"))
                        except Exception:
                            pass
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
                        variants, combined_s3, clouds_s4, clouds_s5 = self._build_cluster_variants(
                            ph,
                            amp,
                            current_filter=flabel,
                        )
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



