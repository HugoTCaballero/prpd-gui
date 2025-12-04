from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QCheckBox,
    QLineEdit,
    QToolButton,
    QSizePolicy,
)
from PySide6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


def build_top_bar(wnd):
    """Construye la barra superior (dos filas) y devuelve el layout."""
    top = QVBoxLayout()

    # Botones principales
    wnd.btn_open = QPushButton("Abrir PRPD...")
    wnd.btn_run = QPushButton("Procesar")
    wnd.btn_pdf = QPushButton("Exportar PDF")
    wnd.btn_load_ann = QPushButton("Cargar ANN")
    wnd.btn_batch = QPushButton("Procesar carpeta")
    wnd.btn_help = QPushButton("Ayuda/README")
    wnd.btn_reset_base = QPushButton("Reset baseline")
    wnd.btn_pdf.setEnabled(True)

    # Gap-time opcional
    wnd.chk_gap = QCheckBox("Gap-time XML")
    wnd.chk_gap.setToolTip("Habilita la carga de serie gap-time y su análisis en Conclusiones/Gap-time.")
    wnd.btn_gap_pick = QPushButton("...")
    wnd.btn_compare = QPushButton("Comparar vs base")
    wnd.btn_gap_ext_add = QPushButton("Gap extenso (0/5)")
    wnd.btn_gap_ext_add.setToolTip("Agregar hasta 5 XML de gap-time para la vista Gap-time extenso.")
    wnd.btn_gap_ext_clear = QPushButton("Limpiar gap extenso")

    # Ocultar botones no utilizados
    for _btn in (wnd.btn_batch, wnd.btn_compare, wnd.btn_reset_base):
        try:
            _btn.setVisible(False)
        except Exception:
            pass

    # Centros S3 en combinada
    wnd.chk_centers_combined = QCheckBox("Centros S3 en combinada")
    wnd.chk_centers_combined.setChecked(True)
    wnd.chk_centers_combined.setToolTip("Solo afecta la vista combinada (Nubes + ANGPD); desactiva los centros S3 en ambos gráficos.")
    wnd.chk_centers_combined.stateChanged.connect(wnd._on_centers_combined_toggle)

    # Fase / filtros / máscaras
    wnd.cmb_phase = QComboBox()
    wnd.cmb_phase.addItems(["Auto (0/120/240)", "0°", "120°", "240°", "Manual..."])
    wnd.cmb_phase.setCurrentIndex(0)
    wnd.btn_phase_manual = QToolButton()
    wnd.btn_phase_manual.setText("...")
    wnd.btn_phase_manual.setToolTip("Ingresar fase manual (°)")

    wnd.cmb_filter = QComboBox()
    wnd.cmb_filter.addItems(["S1 Weak", "S2 Strong"])
    wnd.cmb_filter.setCurrentIndex(0)

    wnd.cmb_masks = QComboBox()
    wnd.cmb_masks.addItems(["Ninguna", "Corona +", "Corona -", "Superficial", "Void", "Manual"])
    wnd.cmb_masks.setCurrentIndex(0)
    wnd.cmb_masks.currentTextChanged.connect(wnd._on_mask_mode_changed)

    wnd.mask_manual_label = QLabel("Intervalos (°):")
    wnd.mask_manual_label.setVisible(False)
    wnd.mask_interval_1 = QLineEdit()
    wnd.mask_interval_1.setPlaceholderText("Int1 (ej: 45-135)")
    wnd.mask_interval_1.setMaximumWidth(120)
    wnd.mask_interval_1.setVisible(False)
    wnd.mask_interval_2 = QLineEdit()
    wnd.mask_interval_2.setPlaceholderText("Int2 (ej: 225-315)")
    wnd.mask_interval_2.setMaximumWidth(120)
    wnd.mask_interval_2.setVisible(False)
    wnd._mask_manual_widgets = [wnd.mask_manual_label, wnd.mask_interval_1, wnd.mask_interval_2]
    wnd._on_mask_mode_changed(wnd.cmb_masks.currentText())

    # Pixel / Qty
    wnd.btn_pixel = QPushButton("Pixel: D1-D10")
    wnd.btn_pixel.setToolTip("Selecciona los deciles (D1-D10) que deseas conservar.")
    wnd.btn_pixel.clicked.connect(wnd._open_pixel_dialog)
    wnd._update_pixel_button_text()

    wnd.btn_qty = QPushButton("Qty: DQ1-DQ10")
    wnd.btn_qty.setToolTip("Selecciona los deciles (DQ1-DQ10) de quantity que deseas conservar.")
    wnd.btn_qty.clicked.connect(wnd._open_qty_dialog)
    wnd._update_qty_button_text()

    # Opciones
    wnd.chk_hist2d = QCheckBox("Densidad (hist2D)")
    wnd.chk_hist2d.setChecked(True)
    wnd.chk_hist2d.setToolTip("Muestra histograma 2D en las vistas de nubes/filtrado.")
    wnd.chk_auto_y = QCheckBox("Auto rango Y")
    wnd.chk_auto_y.setChecked(False)
    wnd.chk_auto_y.setToolTip("Ajusta el rango Y al percentil 1-99% de amplitud en nubes/filtrado.")
    wnd.btn_resolution = build_resolution_button(wnd)

    # Layout en dos filas
    top_layout = top  # reutilizar el QVBoxLayout creado arriba
    row1 = QHBoxLayout()
    row2 = QHBoxLayout()

    row1.addWidget(wnd.btn_open)
    row1.addWidget(wnd.btn_run)
    row1.addWidget(wnd.btn_pdf)
    row1.addWidget(wnd.btn_load_ann)
    row1.addWidget(wnd.btn_gap_ext_add)
    row1.addWidget(wnd.btn_gap_ext_clear)
    row1.addWidget(wnd.chk_gap)
    row1.addWidget(wnd.btn_gap_pick)
    row1.addStretch(1)
    row1.addWidget(wnd.btn_help)

    row2.addWidget(QLabel("Fase:")); row2.addWidget(wnd.cmb_phase); row2.addWidget(wnd.btn_phase_manual)
    row2.addWidget(QLabel("Filtro:")); row2.addWidget(wnd.cmb_filter)
    row2.addWidget(QLabel("Máscara:")); row2.addWidget(wnd.cmb_masks)
    for w in wnd._mask_manual_widgets:
        row2.addWidget(w)
    row2.addWidget(QLabel("Pixel:")); row2.addWidget(wnd.btn_pixel)
    row2.addWidget(QLabel("Qty:")); row2.addWidget(wnd.btn_qty)
    row2.addWidget(wnd.btn_resolution)
    row2.addWidget(wnd.chk_hist2d)
    row2.addWidget(wnd.chk_auto_y)
    row2.addWidget(wnd.chk_centers_combined)
    row2.addStretch(1)

    top_layout.addLayout(row1)
    top_layout.addLayout(row2)

    return top_layout


def build_bottom_area(wnd):
    """Construye selector de vista y banner inferior."""
    wnd.cmb_plot = QComboBox()
    wnd.cmb_plot.addItems([
        "Conclusiones",
        "ANN / Gap-time",
        "Gap-time",
        "Gap-time extenso",
        "Histogramas",
        "ANGPD avanzado",
        "KPI avanzados",
        "FA profile",
        "Combinada",
        "Nubes",
        "Manual",
    ])
    wnd.cmb_plot.setCurrentIndex(0)

    sub = QHBoxLayout()
    sub.addWidget(QLabel("Vista:"))
    sub.addWidget(wnd.cmb_plot)
    wnd.cmb_hist_bins = QComboBox()
    wnd.cmb_hist_bins.addItems(["32 bins", "64 bins"])
    wnd.cmb_hist_bins.setCurrentIndex(0)
    sub.addWidget(QLabel("Bins:"))
    sub.addWidget(wnd.cmb_hist_bins)
    wnd.chk_banner_dark = QCheckBox("Modo nocturno")
    wnd.chk_banner_dark.setToolTip("Alterna el banner inferior a la versión oscura (phaseflux_dark_mode.png).")
    sub.addWidget(wnd.chk_banner_dark)
    sub.addStretch(1)
    wnd.btn_export_all = QPushButton("Exportar todo los resultados")
    sub.addWidget(wnd.btn_export_all)

    batch_bar = QHBoxLayout()
    wnd.banner_max_height = 150
    wnd.signature_label = QLabel()
    wnd.signature_label.setAlignment(Qt.AlignCenter)
    wnd.signature_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
    wnd.signature_label.setMinimumHeight(int(wnd.banner_max_height * 0.5))
    wnd.signature_label.setMaximumHeight(wnd.banner_max_height)
    wnd.signature_label.setScaledContents(False)
    wnd._signature_pixmap = None
    wnd._refresh_signature_banner()
    batch_bar.addWidget(wnd.signature_label)
    return sub, batch_bar


def build_figures(wnd):
    """Crea la figura principal y ejes; retorna (fig, canvas)."""
    fig = Figure(figsize=(10, 6), dpi=100, constrained_layout=True)
    canvas = FigureCanvas(fig)
    wnd._gs_main = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    wnd.ax_raw = fig.add_subplot(wnd._gs_main[0, 0])
    wnd.ax_filtered = fig.add_subplot(wnd._gs_main[0, 1])
    wnd.ax_probs = fig.add_subplot(wnd._gs_main[1, 0])
    wnd.ax_text = fig.add_subplot(wnd._gs_main[1, 1])
    wnd.ax_gap_wide = fig.add_subplot(wnd._gs_main[0, :])
    wnd.ax_gap_wide.set_visible(False)
    wnd.ax_gap_wide.set_facecolor("#f5f6fb")
    for a in [wnd.ax_raw, wnd.ax_filtered, wnd.ax_probs, wnd.ax_text]:
        a.set_facecolor("#fafafa")
    wnd.ax_raw_twin = None
    wnd.ax_probs_twin = None
    wnd.ax_conclusion_box = fig.add_subplot(wnd._gs_main[1, :])
    wnd.ax_conclusion_box.set_visible(False)
    try:
        wnd.ax_conclusion_box.set_in_layout(False)
    except Exception:
        pass
    wnd._conclusion_artists = []
    wnd._conclusion_subaxes = []
    return fig, canvas


def build_resolution_button(wnd):
    """Botón con presets de resolución."""
    from PySide6.QtWidgets import QMenu, QToolButton

    btn = QToolButton()
    btn.setText("Resolución")
    btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
    btn.setPopupMode(QToolButton.InstantPopup)
    menu = QMenu(btn)
    presets = [
        ("Auto (90% pantalla)", None),
        ("1920x1080 — completo", (1920, 1080)),
        ("1920x1080 — 1/2 ancho", (960, 1080)),
        ("1920x1080 — 1/4 (960x540)", (960, 540)),
        ("5120x1440 — completo", (5120, 1440)),
        ("5120x1440 — 1/2 ancho", (2560, 1440)),
        ("5120x1440 — 1/4 (1280x720)", (1280, 720)),
    ]
    for label, size in presets:
        act = menu.addAction(label)
        act.setData(size)
    btn.setMenu(menu)
    return btn
