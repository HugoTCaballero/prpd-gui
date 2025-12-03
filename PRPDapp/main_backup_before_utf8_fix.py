# -*- coding: utf-8 -*-

# main.py
# PRPD GUI (Windows) ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ MVP con carga, auto-fase/ manual, clustering+prune, ANN (heurÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â­stica),
# severidad y exporte a PDF. Requiere: PySide6, numpy, matplotlib, scikit-learn.
import sys, os, json, traceback
from pathlib import Path
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QSlider, QCheckBox,
    QPlainTextEdit
)
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtCore import Qt, QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- mÃƒÂ³dulos locales ---
import pathlib as _pl
_THIS = _pl.Path(__file__).resolve(); _ROOT = _THIS.parents[1]; _PKG = _THIS.parent
for _p in (str(_ROOT), str(_PKG)):
    (_p in sys.path) or sys.path.insert(0, _p)

import prpd_core as core
from report import export_pdf_report
from utils_io import ensure_out_dirs, time_tag
from prpd_ann import PRPDANN
from three_d import plot_prpd_3d
from batch_bridge import run_batch
# --- ANN loader (opcional) ---
try:
    from models.ann_loader import load_ann_model as _load_ann_model, predict_proba as _ann_predict_proba, align_features_or_fallback as _align_feats
except Exception:
    _load_ann_model = None
    _ann_predict_proba = None
    _align_feats = None

APP_TITLE = "PRPD GUI ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ MVP"

class PRPDWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 740)

        # estado
        self.current_path: Path | None = None
        self.last_result: dict | None = None
        self.auto_phase = True
        self.current_points = None
        self.current_labels = None

        # Clasificador supervisado (opcional): intenta cargar ./modelos/prpd_ann.pkl
        self.ann = PRPDANN(class_names=["cavidad","superficial","corona","flotante","ruido"])
        try:
        # ANN loader (models/ann_loader) auto-carga si disponible
        self.ann_model = None; self.ann_classes = []
        try:
            if _load_ann_model is not None:
                self.ann_model, self.ann_classes = _load_ann_model(None)
        except Exception:
            pass
            self.ann.load_model(os.path.join("modelos","prpd_ann.pkl"))
        except Exception:
            # Si no hay modelo o joblib no estÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡, continuar con heurÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â­stica
            pass

        # UI
        central = QWidget(self); self.setCentralWidget(central)
        v = QVBoxLayout(central)

        # barra superior
        top = QHBoxLayout()
        self.btn_open = QPushButton("Abrir PRPDÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦")
        self.btn_run  = QPushButton("Procesar")
        self.btn_pdf  = QPushButton("Exportar PDF")
        self.btn_load_ann = QPushButton("Cargar ANN")
        self.btn_3d = QPushButton("3D")
        self.btn_batch = QPushButton("Procesar carpeta")
        self.btn_pdf.setEnabled(False)

        self.cmb_phase = QComboBox()
        self.cmb_phase.addItems(["Auto (0/120/240)", "0ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°", "120ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°", "240ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°"])
        self.cmb_phase.setCurrentIndex(0)

        self.chk_quiet = QCheckBox("Modo rÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡pido (menos figuras)")
        self.chk_quiet.setChecked(False)

        top.addWidget(self.btn_open)
        top.addWidget(self.btn_run)
        top.addWidget(QLabel("Fase:"))
        top.addWidget(self.cmb_phase)
        # Selector de filtro (S1/S2)
        self.cmb_filter = QComboBox(); self.cmb_filter.addItems(["S1 Weak","S2 Strong"]); self.cmb_filter.setCurrentIndex(0)
        top.addWidget(QLabel("Filtro:")); top.addWidget(self.cmb_filter)
        top.addStretch(1)
        top.addWidget(self.chk_quiet)
        top.addWidget(self.btn_load_ann)
        top.addWidget(self.btn_3d)
        top.addWidget(self.btn_batch)
        top.addWidget(self.btn_pdf)
        # Toggle densidad (histograma 2D)
        self.chk_hist2d = QCheckBox("Densidad (hist2D)")
        self.chk_hist2d.setChecked(True)
        top.addWidget(self.chk_hist2d)
        # Selector de vista inferior izquierda
        self.cmb_plot = QComboBox(); self.cmb_plot.addItems(["Probabilidades","ANGPD","Nubes"]); self.cmb_plot.setCurrentIndex(0)
        top.addWidget(QLabel("Vista:")); top.addWidget(self.cmb_plot)
        v.addLayout(top)

        # figuras matplotlib
        fig = Figure(figsize=(10,6), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvas(fig)
        self.ax_raw      = fig.add_subplot(2, 2, 1)
        self.ax_filtered = fig.add_subplot(2, 2, 2)
        self.ax_probs    = fig.add_subplot(2, 2, 3)
        self.ax_text     = fig.add_subplot(2, 2, 4)
        for a in [self.ax_raw, self.ax_filtered, self.ax_probs, self.ax_text]:
            a.set_facecolor("#fafafa")
        v.addWidget(self.canvas)

        # Panel simple para ANN (proba/severidad)
        ann_bar = QHBoxLayout()
        ann_bar.addWidget(QLabel("Severidad (ANN):"))
        self.lblSeveridad = QLabel("")
        ann_bar.addWidget(self.lblSeveridad)
        ann_bar.addStretch(1)
        ann_bar.addWidget(QLabel("Proba (ANN):"))
        self.lblProba = QLabel("")
        self.lblProba.setWordWrap(True)
        self.lblProba.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ann_bar.addWidget(self.lblProba)
        v.addLayout(ann_bar)

        # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Ârea para resumen de batch
        batch_bar = QHBoxLayout()
        batch_bar.addWidget(QLabel("Resumen Batch:"))
        self.txtBatchSummary = QPlainTextEdit("")
        self.txtBatchSummary.setReadOnly(True)
        self.txtBatchSummary.setMaximumHeight(120)
        batch_bar.addWidget(self.txtBatchSummary)
        v.addLayout(batch_bar)

        # eventos
        self.btn_open.clicked.connect(self.open_file_dialog)
        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_pdf.clicked.connect(self.export_pdf_clicked)
        self.cmb_phase.currentIndexChanged.connect(self.phase_mode_changed)
        self.btn_load_ann.clicked.connect(self.on_btnLoadANN_clicked)
        self.btn_3d.clicked.connect(self.on_btn3D_clicked)
        self.btn_batch.clicked.connect(self.on_btnBatch_clicked)

        # al arrancar, abrir diÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡logo automÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ticamente
        QTimer.singleShot(200, self.open_file_dialog)

    # -------- UI handlers ----------
    def phase_mode_changed(self):
        self.auto_phase = (self.cmb_phase.currentIndex() == 0)

    def open_file_dialog(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Selecciona archivo PRPD (CSV/XML)", "",
            "PRPD (*.csv *.xml);;CSV (*.csv);;XML (*.xml);;Todos (*.*)")
        if not fn:
            return
        self.current_path = Path(fn)
        # carga preliminar para preview
        try:
            data = core.load_prpd(self.current_path)
            self.plot_raw(data)
        except Exception as e:
            QMessageBox.critical(self, "Error al cargar", str(e))

    def run_pipeline(self):
        if not self.current_path:
            QMessageBox.warning(self, "Falta archivo", "Primero carga un CSV de PRPD.")
            return
        try:
            outdir = ensure_out_dirs(Path("out"))
            # configuraciÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³n de fase
            if self.auto_phase:
                force_offsets = None  # auto {0,120,240}
            else:
                idx = self.cmb_phase.currentIndex()
                offsets = [0,0,120,240]
                force_offsets = [offsets[idx]]

            filt_label = self.cmb_filter.currentText().strip() if hasattr(self, 'cmb_filter') else 'S1 Weak'
            result = core.process_prpd(
                path=self.current_path,
                out_root=outdir,
                force_phase_offsets=force_offsets,
                fast_mode=self.chk_quiet.isChecked(),
                filter_level=filt_label
            )
            self.last_result = result
            # Exports: ANGPD, Clouds, Metrics, PNGs
            try:
                out_reports = (Path(outdir) / 'reports')
                out_reports.mkdir(parents=True, exist_ok=True)
                stem = self.current_path.stem if self.current_path else 'session'
                # ANGPD CSV + PNG
                ang = result.get('angpd', {})
                x = np.asarray(ang.get('phi_centers', []), dtype=float)
                y1 = np.asarray(ang.get('angpd', []), dtype=float)
                y2 = np.asarray(ang.get('n_angpd', []), dtype=float)
                if x.size:
                    with open(out_reports / f"{stem}_angpd.csv", 'w', encoding='utf-8') as f:
                        f.write('phi_deg,angpd,n_angpd\n')
                        for i in range(x.size):
                            a1 = float(y1[i]) if i < y1.size else 0.0
                            a2 = float(y2[i]) if i < y2.size else 0.0
                            f.write(f"{float(x[i]):.3f},{a1:.6f},{a2:.6f}\n")
                    try:
                        import matplotlib.pyplot as _plt
                        fig,_ax = _plt.subplots(figsize=(5,3), dpi=120)
                        if y1.size: _ax.plot(x, y1, label='ANGPD (sum=1)')
                        if y2.size: _ax.plot(x, y2, label='N-ANGPD (max=1)')
                        _ax.set_xlim(0,360); _ax.set_xlabel('Fase (deg)'); _ax.set_ylabel('Densidad')
                        _ax.set_title('ANGPD / N-ANGPD'); _ax.legend(loc='upper right', fontsize=8)
                        fig.tight_layout(); fig.savefig(out_reports / f"{stem}_angpd.png", bbox_inches='tight'); _plt.close(fig)
                    except Exception:
                        pass
                # Clouds CSVs + PNG
                ph = np.asarray(result.get('aligned', {}).get('phase_deg', []), dtype=float)
                amp = np.asarray(result.get('aligned', {}).get('amplitude', []), dtype=float)
                if ph.size and amp.size:
                    clouds_all = pixel_cluster_clouds(ph, amp)
                    clouds_comb = combine_clouds(clouds_all)
                    clouds_sel = select_dominant_clouds(clouds_all)
                    with open(out_reports / f"{stem}_clouds_raw.csv", 'w', encoding='utf-8') as f:
                        f.write('id,count,frac,phase_mean,y_mean\n')
                        for c in clouds_all:
                            f.write(f"{int(c.get('id',0))},{int(c.get('count',0))},{float(c.get('frac',0.0)):.6f},{float(c.get('phase_mean',0.0)):.2f},{float(c.get('y_mean',0.0)):.2f}\n")
                    with open(out_reports / f"{stem}_clouds_combined.csv", 'w', encoding='utf-8') as f:
                        f.write('ids,count,phase_mean,y_mean\n')
                        for c in clouds_comb:
                            ids = ';'.join(map(str, c.get('ids', [])))
                            f.write(f"{ids},{int(c.get('count',0))},{float(c.get('phase_mean',0.0)):.2f},{float(c.get('y_mean',0.0)):.2f}\n")
                    with open(out_reports / f"{stem}_clouds_selected.csv", 'w', encoding='utf-8') as f:
                        f.write('id,count,frac,phase_mean,y_mean\n')
                        for c in clouds_sel:
                            f.write(f"{int(c.get('id',0))},{int(c.get('count',0))},{float(c.get('frac',0.0)):.6f},{float(c.get('phase_mean',0.0)):.2f},{float(c.get('y_mean',0.0)):.2f}\n")
                    try:
                        import matplotlib.pyplot as _plt, numpy as _np
                        cols = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
                        fig,_ax = _plt.subplots(figsize=(5,4), dpi=120)
                        centers = _np.array([[c.get('phase_mean',0.0), c.get('y_mean',0.0)] for c in clouds_sel], dtype=float) if clouds_sel else _np.zeros((0,2))
                        if centers.size:
                            lbl = _np.zeros(ph.shape[0], dtype=int)
                            for i in range(ph.shape[0]):
                                dp = _np.minimum.reduce([_np.abs(ph[i]-centers[:,0]), _np.abs(ph[i]-centers[:,0]-180), _np.abs(ph[i]-centers[:,0]+180)])
                                dy = _np.abs(amp[i]-centers[:,1])
                                j = int(_np.argmin(0.6*dp + 0.4*dy)); lbl[i] = j
                            for j in range(centers.shape[0]):
                                m = (lbl==j)
                                if _np.any(m): _ax.scatter(ph[m], amp[m], s=5, alpha=0.7, color=cols[j%len(cols)], label=f'C{j+1}')
                            for j,c in enumerate(clouds_sel):
                                _ax.scatter([c.get('phase_mean',0.0)], [c.get('y_mean',0.0)], s=60, color=cols[j%len(cols)], edgecolors='black')
                        else:
                            _ax.scatter(ph, amp, s=3, alpha=0.6, color='#1f77b4')
                        _ax.set_xlim(0,360); _ax.set_xlabel('Fase (deg)'); _ax.set_ylabel('Amplitud'); _ax.set_title('Nubes dominantes')
                        if centers.size: _ax.legend(loc='upper right', fontsize=8)
                        fig.tight_layout(); fig.savefig(out_reports / f"{stem}_clouds.png", bbox_inches='tight'); _plt.close(fig)
                    except Exception:
                        pass
                # Baseline JSON (creaci n si no existe)
                try:
                    import json as _json
                    base_path = out_reports / f"{stem}_baseline.json"
                    bd = result.get('severity_breakdown', {})
                    cur = {
                        'p95_amp': float(bd.get('p95_amp', 0.0)),
                        'dens': float(bd.get('dens', 0.0)),
                        'R_phase': float(bd.get('R_phase', 0.0)),
                        'std_circ_deg': float(bd.get('std_circ_deg', 0.0)),
                        'severity': float(result.get('severity_score', 0.0)),
                    }
                    if not base_path.exists():
                        base_path.write_text(_json.dumps(cur, ensure_ascii=False, indent=2), encoding='utf-8')
                except Exception:
                    pass
                # Metrics CSV (+ new_severity)
                bd = result.get('severity_breakdown', {})
                # Compute new_severity: base + ANN influence + band deviation
                try:
                    base_sev = float(result.get('severity_score', 0.0))
                    proba = result.get('probs', {})
                    # weight ANN toward 'cavidad' as example target; adjust if needed
                    p_target = float(proba.get('cavidad', 0.0))
                    ann_term = 100.0 * p_target
                    # band penalty/bonus using baseline if present
                    import json as _json
                    base_path = out_reports / f"{stem}_baseline.json"
                    band_penalty = 0.0
                    if base_path.exists():
                        base = _json.loads(base_path.read_text(encoding='utf-8'))
                        def out_of_band(cur, b):
                            lo = 0.8*b; hi = 1.2*b
                            return (cur < lo) or (cur > hi)
                        if out_of_band(float(bd.get('p95_amp',0.0)), float(base.get('p95_amp',0.0))): band_penalty += 5.0
                        if out_of_band(float(bd.get('dens',0.0)), float(base.get('dens',0.0))): band_penalty += 3.0
                        if out_of_band(float(bd.get('R_phase',0.0)), float(base.get('R_phase',0.0))): band_penalty += 2.0
                    new_sev = max(0.0, min(100.0, 0.7*base_sev + 0.3*ann_term - band_penalty))
                except Exception:
                    new_sev = float(result.get('severity_score', 0.0))
                with open(out_reports / f"{stem}_metrics.csv", 'w', encoding='utf-8') as f:
                    f.write('predicted,severity,new_severity,p95_amp,dens,R_phase,std_circ_deg,phase_offset,filter_level\n')
                    f.write(f"{result.get('predicted','')},{float(result.get('severity_score',0.0)):.2f},{new_sev:.2f},{float(bd.get('p95_amp',0.0)):.4f},{float(bd.get('dens',0.0)):.6f},{float(bd.get('R_phase',0.0)):.6f},{float(bd.get('std_circ_deg',0.0)):.2f},{int(result.get('phase_offset',0))},{filt_label}\n")
                # Tracking PNG (bandas Â±20%)
                try:
                    import matplotlib.pyplot as _plt
                    import numpy as _np
                    base_path = out_reports / f"{stem}_baseline.json"
                    if base_path.exists():
                        base = _json.loads(base_path.read_text(encoding='utf-8'))
                        keys = ['p95_amp','dens','R_phase','std_circ_deg','severity']
                        curv = [float(bd.get('p95_amp',0.0)), float(bd.get('dens',0.0)), float(bd.get('R_phase',0.0)), float(bd.get('std_circ_deg',0.0)), float(result.get('severity_score',0.0))]
                        basev = [float(base.get('p95_amp',0.0)), float(base.get('dens',0.0)), float(base.get('R_phase',0.0)), float(base.get('std_circ_deg',0.0)), float(base.get('severity',0.0))]
                        fig, ax = _plt.subplots(figsize=(6,3), dpi=120)
                        x = _np.arange(len(keys))
                        lo = [0.8*b for b in basev]; hi = [1.2*b for b in basev]
                        for i in range(len(keys)):
                            ax.fill_between([i-0.3,i+0.3], [lo[i],lo[i]], [hi[i],hi[i]], color='#cfe8ff', alpha=0.7)
                        ax.plot(x, basev, 'o-', color='#1f77b4', label='base')
                        ax.plot(x, curv, 'x-', color='#d62728', label='actual')
                        ax.set_xticks(x, keys, rotation=20)
                        ax.set_title('Tracking (Â±20% bandas)'); ax.grid(True, alpha=0.3)
                        ax.legend(loc='upper left', fontsize=8)
                        fig.tight_layout(); fig.savefig(out_reports / f"{stem}_tracking.png", bbox_inches='tight'); _plt.close(fig)
                except Exception:
                    pass
            except Exception:
                pass
            self.render_result(result)
            self.btn_pdf.setEnabled(True)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error en pipeline", str(e))

    def export_pdf_clicked(self):
        if not self.last_result:
            QMessageBox.warning(self, "Sin resultados", "Ejecuta primero el procesamiento.")
            return
        try:
            pdf_path = export_pdf_report(self.last_result, Path("out"))
            QMessageBox.information(self, "PDF exportado", f"Guardado en:\n{pdf_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error exportando PDF", str(e))

    # --------- plots ----------
    def plot_raw(self, data: dict):
        self.ax_raw.clear()
        try:
            import numpy as _np
            from matplotlib.colors import LogNorm as _LogNorm
            ph = _np.asarray(data.get("phase_deg", []), dtype=float)
            amp = _np.asarray(data.get("amplitude", []), dtype=float)
            if getattr(self, 'chk_hist2d', None) is not None and self.chk_hist2d.isChecked() and ph.size and amp.size:
                H, xedges, yedges = _np.histogram2d(ph, amp, bins=[72, 50], range=[[0,360],[0,100]])
                self.ax_raw.imshow(H.T + 1e-9, origin='lower', aspect='auto',
                                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                   norm=_LogNorm(vmin=1e-9, vmax=max(1.0, H.max())))
            else:
                self.ax_raw.scatter(ph, amp, s=4, alpha=0.6)
        except Exception:
            try:
                ph = data.get("phase_deg", [])
                amp = data.get("amplitude", [])
                self.ax_raw.scatter(ph, amp, s=4, alpha=0.6)
            except Exception:
                pass
        self.ax_raw.set_title("PRPD crudo")
        self.ax_raw.set_xlabel("Fase (deg)"); self.ax_raw.set_xlim(0, 360)
        self.ax_raw.set_ylabel("Amplitud")
        self.canvas.draw_idle()


    def render_result(self, r: dict):
        # raw
        self.ax_raw.clear()
        try:
            import numpy as _np
            from matplotlib.colors import LogNorm as _LogNorm
            ph0 = _np.asarray(r.get("raw", {}).get("phase_deg", []), dtype=float)
            a0 = _np.asarray(r.get("raw", {}).get("amplitude", []), dtype=float)
            if getattr(self, 'chk_hist2d', None) is not None and self.chk_hist2d.isChecked() and ph0.size and a0.size:
                H0, xe0, ye0 = _np.histogram2d(ph0, a0, bins=[72,50], range=[[0,360],[0,100]])
                self.ax_raw.imshow(H0.T + 1e-9, origin='lower', aspect='auto',
                                   extent=[xe0[0], xe0[-1], ye0[0], ye0[-1]],
                                   norm=_LogNorm(vmin=1e-9, vmax=max(1.0, H0.max())))
                # Overlay de ruido (gris tenue) para crudo
                try:
                    labels = np.asarray(r.get('labels', []))
                    keep = np.asarray(r.get('keep_mask', []), dtype=bool)
                    if labels.size and keep.size and ph0.size == labels.size and a0.size == labels.size:
                        raw_keep = keep & (labels >= 0)
                        noise_idx = ~raw_keep
                        self.ax_raw.scatter(ph0[noise_idx], a0[noise_idx], s=2, alpha=0.12, color='#888888')
                except Exception:
                    pass
        except Exception:
            pass
        if not (hasattr(self,'chk_hist2d') and self.chk_hist2d.isChecked()):
            self.ax_raw.scatter(r["raw"]["phase_deg"], r["raw"]["amplitude"], s=3, alpha=0.4)
        self.ax_raw.set_title("PRPD crudo")
        self.ax_raw.set_xlim(0,360); self.ax_raw.set_xlabel("Fase (deg)"); self.ax_raw.set_ylabel("Amplitud")
        try:
            self.canvas.figure.tight_layout()
            self.canvas.figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)
        except Exception:
            pass

        # filtrado/alineado
        self.ax_filtered.clear()
        # Heatmap para alineado si el toggle esta activo
        try:
            import numpy as _np
            from matplotlib.colors import LogNorm as _LogNorm
            ph_al = _np.asarray(r.get("aligned", {}).get("phase_deg", []), dtype=float)
            amp_al = _np.asarray(r.get("aligned", {}).get("amplitude", []), dtype=float)
            if getattr(self, 'chk_hist2d', None) is not None and self.chk_hist2d.isChecked() and ph_al.size and amp_al.size:
                H2, xe2, ye2 = _np.histogram2d(ph_al, amp_al, bins=[72,50], range=[[0,360],[0,100]])
                self.ax_filtered.imshow(H2.T + 1e-9, origin='lower', aspect='auto',
                                        extent=[xe2[0], xe2[-1], ye2[0], ye2[-1]],
                                        norm=_LogNorm(vmin=1e-9, vmax=max(1.0, H2.max())))
        except Exception:
            pass
        if not (hasattr(self,'chk_hist2d') and self.chk_hist2d.isChecked()):
            self.ax_filtered.scatter(r["aligned"]["phase_deg"], r["aligned"]["amplitude"], s=3, alpha=0.8)
        else:
            # Overlay de ruido gris tenue cuando hay heatmap
            try:
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
        self.ax_filtered.set_title(f"Alineado/filtrado (offset={r['phase_offset']} deg)")
        self.ax_filtered.set_xlim(0,360); self.ax_filtered.set_xlabel("Fase (deg)"); self.ax_filtered.set_ylabel("Amplitud")

        # barras de prob (ANN o heurÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â­stica)
        self.ax_probs.clear()
        classes = ["cavidad","superficial","corona","flotante"]
        proba_dict = r.get("probs", {})
        # Autocargar ANN por defecto si no estÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¿ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ cargada
        try:
            if hasattr(self, 'ann') and not getattr(self.ann, 'is_loaded', False):
                default_model = os.path.join("models", "ann.pkl")
                fallback_model = os.path.join("modelos", "prpd_ann.pkl")
                if os.path.exists(default_model):
                    self.ann.load_model(default_model)
                elif os.path.exists(fallback_model):
                    self.ann.load_model(fallback_model)
        except Exception:
            pass
        try:
            if hasattr(self, "ann") and getattr(self.ann, "is_loaded", False):
                ph = np.asarray(r.get("aligned", {}).get("phase_deg", []), dtype=float)
                amp = np.asarray(r.get("aligned", {}).get("amplitude", []), dtype=float)
                feats_core = r.get("features", {})
                features_for_ann = {
                    "amp_mean": float(np.mean(np.abs(amp))) if amp.size else 0.0,
                    "amp_std": float(np.std(amp)) if amp.size else 0.0,
                    "amp_p95": float(feats_core.get("p95_amp", 0.0)),
                    "density": float(feats_core.get("dens", 0.0)),
                    "phase_std_deg": float(np.std(ph)) if ph.size else 0.0,
                    "phase_entropy": 0.0,
                    "rep_rate": 0.0,
                    "rep_entropy": 0.0,
                    "cluster_compactness": 0.0,
                    "cluster_separation": 0.0,
                    "lobes_count": 0.0,
                    "area_ratio": 0.0,
                }
                proba_ann = self.ann.predict_proba(features_for_ann)
                if isinstance(proba_ann, dict) and proba_ann:
                    proba_dict = proba_ann
        except Exception:
            pass
        # Mostrar ANGPD o Probabilidades segÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¿ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½n selector
        view_mode = self.cmb_plot.currentText().strip().lower() if hasattr(self, 'cmb_plot') else 'probabilidades'
        if view_mode.startswith("angpd"):
            ang = r.get("angpd", {})
            x = np.asarray(ang.get("phi_centers", []), dtype=float)
            y1 = np.asarray(ang.get("angpd", []), dtype=float)
            y2 = np.asarray(ang.get("n_angpd", []), dtype=float)
            if x.size and (y1.size or y2.size):
                if y1.size:
                    self.ax_probs.plot(x, y1, label="ANGPD (sum=1)", color="#1f77b4")
                if y2.size:
                    self.ax_probs.plot(x, y2, label="N-ANGPD (max=1)", color="#ff7f0e", alpha=0.85)
                self.ax_probs.set_xlim(0,360); self.ax_probs.set_xlabel("Fase (deg)")
                self.ax_probs.set_ylabel("Densidad")
                self.ax_probs.set_title("ANGPD / N-ANGPD")
                self.ax_probs.legend(loc="upper right", fontsize=8)
            else:
                self.ax_probs.text(0.5, 0.5, "Sin datos ANGPD", ha="center", va="center")
        else:
            probs = [proba_dict.get(k,0.0) for k in classes]
            self.ax_probs.bar(classes, probs)
            self.ax_probs.set_ylim(0,1); self.ax_probs.set_title("Probabilidades")

        # texto con severidad y resumen
        self.ax_text.clear(); self.ax_text.axis("off")
        summ = [
            f"Archivo: {self.current_path.name}",
            f"Ruido detectado: {'SÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â­' if r['has_noise'] else 'No'}",
            f"Clase: {r['predicted']}",
            f"Severidad: {r['severity_score']:.1f}/100",
            f"Clusters: {r['n_clusters']}  |  Puntos: {len(r['aligned']['phase_deg'])}",
        ]
        # Anexar estado ANN, meta de ruido y desglose de severidad
        ann_state = 'cargado' if (hasattr(self,'ann') and getattr(self.ann,'is_loaded',False)) else 'heuristica'
        sev_bd = r.get('severity_breakdown', {})
        nm = r.get('noise_meta', {})
        summ += [
            f"ANN: {ann_state}",
            f"Ruido meta: occ={nm.get('occupied','?')} strong={nm.get('strong','?')} ratio={nm.get('ratio','?')}",
            f"Severidad det.: p95={sev_bd.get('p95_amp','?')}, dens={sev_bd.get('dens','?')}, R={sev_bd.get('R_phase','?')}, std_circ={sev_bd.get('std_circ_deg','?')} deg",
        ]
        self.ax_text.text(0.01, 0.98, "\n".join(summ), va="top", fontsize=10)
        try:
            self.canvas.figure.tight_layout()
            self.canvas.figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)
        except Exception:
            pass
        self.canvas.draw_idle()

        # preparar datos para 3D: (fase_deg, amplitud, orden)
        try:
            ph = np.asarray(r["aligned"]["phase_deg"], dtype=float)
            amp = np.asarray(r["aligned"]["amplitude"], dtype=float)
            npt = len(ph)
            # Escalar Z a 0..100 para mejor relieve visual
            order = np.linspace(0.0, 100.0, npt) if npt else np.array([])
            self.current_points = np.column_stack([ph, amp, order]) if npt else None
            # Etiquetado por subcluster (estable) con ANN si disponible
            self.current_labels = self._labels_by_subcluster(r, ph, amp)
        except Exception:
            self.current_points = None
            self.current_labels = None

        # Proba/Severidad ANN en etiquetas
        try:
            import json as _json
            if proba_dict:
                self.lblProba.setText(_json.dumps(proba_dict, ensure_ascii=False, indent=2))
                sev = int(100.0 * (
                    proba_dict.get("cavidad",0)*0.45 +
                    proba_dict.get("corona",0)*0.25 +
                    proba_dict.get("superficial",0)*0.20 +
                    proba_dict.get("flotante",0)*0.10
                ))
                self.lblSeveridad.setText(f"{sev} / 100")
            else:
                self.lblProba.setText("")
                self.lblSeveridad.setText("")
        except Exception:
            self.lblProba.setText("")
            self.lblSeveridad.setText("")

        # Etiquetas por punto para 3D usando ANN si estÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ cargado
        try:
            if hasattr(self, "ann") and getattr(self.ann, "is_loaded", False):
                self.current_labels = self._ann_pointwise_labels(r)
            else:
                lbl = r.get("predicted", "ruido")
                ph = np.asarray(r["aligned"]["phase_deg"], dtype=float)
                self.current_labels = [lbl] * len(ph)
        except Exception:
            lbl = r.get("predicted", "ruido")
            ph = np.asarray(r["aligned"]["phase_deg"], dtype=float)
            self.current_labels = [lbl] * len(ph)

    def on_btnLoadANN_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Seleccionar modelo ANN', '', 'Modelos (*.pkl *.joblib)')
        if not path:
            return
        ok=False
        if _load_ann_model is not None:
            try:
                self.ann_model, self.ann_classes = _load_ann_model(path)
                ok=True
            except Exception:
                ok=False
        if (not ok) and hasattr(self,'ann'):
            try:
                self.ann.load_model(path)
                ok=True
            except Exception:
                ok=False
        if ok:
            QMessageBox.information(self, 'ANN', f'Modelo cargado.\n{path}')
            if self.last_result:
                self.render_result(self.last_result)
        else:
            QMessageBox.warning(self, 'ANN', f'No se pudo cargar el modelo\n{path}')

