def connect_events(wnd) -> None:
    """Conecta las señales de la ventana principal."""
    wnd.btn_open.clicked.connect(wnd.open_file_dialog)
    wnd.btn_run.clicked.connect(wnd.run_pipeline)
    wnd.btn_pdf.clicked.connect(wnd.export_pdf_clicked)
    wnd.btn_export_all.clicked.connect(wnd.on_export_all_clicked)
    wnd.btn_clear_cache.clicked.connect(wnd.on_clear_cache_clicked)
    wnd.btn_load_ann.clicked.connect(wnd.on_btnLoadANN_clicked)
    wnd.btn_batch.clicked.connect(wnd.on_btnBatch_clicked)
    wnd.btn_help.clicked.connect(wnd.on_open_readme)
    wnd.btn_reset_base.clicked.connect(wnd.on_reset_baseline)
    wnd.btn_dash3d.clicked.connect(wnd.on_open_dash_3d)
    wnd.btn_gap_pick.clicked.connect(wnd.on_pick_gap_xml)
    wnd.btn_gap_ext_add.clicked.connect(wnd.on_add_gap_ext)
    wnd.btn_gap_ext_clear.clicked.connect(wnd.on_clear_gap_ext)
    wnd.btn_compare.clicked.connect(wnd.on_compare_vs_base)
    wnd.btn_phase_manual.clicked.connect(lambda: wnd._open_manual_phase_dialog(force_dialog=True))
    wnd.chk_banner_dark.toggled.connect(wnd.on_toggle_banner_mode)
    try:
        for action in wnd.btn_resolution.menu().actions():
            action.triggered.connect(lambda checked=False, a=action: wnd._on_resolution_action(a))
    except Exception:
        pass
    wnd.cmb_plot.currentTextChanged.connect(wnd._on_view_changed)
    wnd.cmb_phase.currentIndexChanged.connect(wnd._on_phase_changed)
    wnd.cmb_hist_bins.currentTextChanged.connect(wnd._on_hist_bins_changed)
    if hasattr(wnd, "cmb_asset"):
        wnd.cmb_asset.currentTextChanged.connect(wnd._on_asset_changed)
    if hasattr(wnd, "chk_ann_hide_sr"):
        wnd.chk_ann_hide_sr.toggled.connect(wnd._on_ann_display_config_changed)
