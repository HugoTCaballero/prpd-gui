#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interfaz gráfica para ejecutar el pipeline PRPD (Bloques 3–6).
- Fase: auto | manual con grados (0–360, decimales)
- K: k_use visible; si Modo K=manual usa --k-use en B3 aligned y B5 (no en B6)
- Emparejamiento: parámetros + toggles 'K por pares (avanzada)' y 'Mostrar líneas de unión (pares)'
- Batch: procesa carpeta -> out/batch/NNN_<stem>
- Reporte: abre HTML automáticamente al terminar (archivo único)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import json
import os
import sys
from pathlib import Path
import threading
import queue

APP_TITLE = "Ejecutor de Pipeline PRPD"
UI_STATE_FILE = "ui_last.json"


class PipelineGUI:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        master.title(APP_TITLE)
        master.geometry("900x720")
        master.resizable(False, False)

        # ==== Variables ====
        self.xml_file = tk.StringVar()
        self.batch_folder = tk.StringVar()

        self.sensor = tk.StringVar(value="auto")
        self.tipo_tx = tk.StringVar(value="seco")

        # Fase: auto|manual + grados (double)
        self.align_mode = tk.StringVar(value="auto")
        self.phase_deg = tk.DoubleVar(value=0.0)

        # K: auto|manual; k_use siempre visible
        self.k_mode = tk.StringVar(value="auto")  # 'auto' | 'manual'
        self.k_manual = tk.IntVar(value=3)
        self.k_use = tk.IntVar(value=3)

        # Paletas / alfas
        self.palette = tk.StringVar(value="paper")
        self.alpha_base = tk.DoubleVar(value=0.25)
        self.alpha_clusters = tk.DoubleVar(value=0.85)

        # Otras / Subclusters
        self.allow_otras = tk.BooleanVar(value=True)
        self.otras_min_score = tk.DoubleVar(value=0.12)
        self.otras_cap = tk.DoubleVar(value=0.25)
        self.subclusters = tk.BooleanVar(value=True)
        self.sub_min_pct = tk.DoubleVar(value=0.02)

        # Pairing (numéricos)
        self.pair_max_phase_deg = tk.DoubleVar(value=25.0)
        self.pair_max_y_ks = tk.DoubleVar(value=0.25)
        self.pair_min_weight_ratio = tk.DoubleVar(value=0.4)
        self.pair_miss_penalty = tk.DoubleVar(value=0.15)

        # Pairing (toggles)
        self.pair_enforce_same_k = tk.BooleanVar(value=False)  # K por pares (avanzada)
        self.pairs_show_lines = tk.BooleanVar(value=False)     # Mostrar líneas (pares)

        # Reporte
        self.embed_assets = tk.BooleanVar(value=True)
        self.no_pdf = tk.BooleanVar(value=True)  # Solo HTML por defecto

        # Internos
        self.proc_thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.log_queue: "queue.Queue[str]" = queue.Queue()

        # ==== UI ====
        self._build_file_frame(master)
        self._build_params_frame(master)
        self._build_k_frame(master)
        self._build_palette_frame(master)
        self._build_adv_frame(master)
        self._build_actions_frame(master)
        self._build_log_frame(master)

        self._refresh_phase_mode()
        self._refresh_k_mode()
        self._load_ui_state()

        self.master.after(120, self._pump_logs)

    # ---------- UI builders ----------
    def _build_file_frame(self, master: tk.Tk) -> None:
        frm = ttk.LabelFrame(master, text="Archivo o carpeta XML")
        frm.pack(fill="x", padx=10, pady=10)

        ttk.Label(frm, text="Archivo XML:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frm, textvariable=self.xml_file, width=70).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frm, text="Seleccionar...", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(frm, text="Carpeta de XMLs:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frm, textvariable=self.batch_folder, width=70).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(frm, text="Seleccionar...", command=self.browse_folder).grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(frm, text="Si se selecciona una carpeta se procesarán todos los *.xml en ella.").grid(
            row=2, column=0, columnspan=3, sticky="w", padx=5
        )

    def _build_params_frame(self, master: tk.Tk) -> None:
        frm = ttk.LabelFrame(master, text="Parámetros")
        frm.pack(fill="x", padx=10, pady=5)

        ttk.Label(frm, text="Sensor:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Combobox(frm, textvariable=self.sensor, values=["auto", "UHF", "HFCT", "TEV"],
                     state="readonly", width=10).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(frm, text="Tipo TX:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        ttk.Combobox(frm, textvariable=self.tipo_tx, values=["seco", "aceite"], state="readonly", width=10)\
            .grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(frm, text="Alineación fase:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        combo = ttk.Combobox(frm, textvariable=self.align_mode, values=["auto", "manual"],
                             state="readonly", width=10)
        combo.grid(row=1, column=1, padx=5, pady=5)
        combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_phase_mode())

        ttk.Label(frm, text="Grados:").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.ent_phase = ttk.Entry(frm, textvariable=self.phase_deg, width=10)
        self.ent_phase.grid(row=1, column=3, padx=5, pady=5)

    def _build_k_frame(self, master: tk.Tk) -> None:
        frm = ttk.LabelFrame(master, text="Segmentación (K)")
        frm.pack(fill="x", padx=10, pady=5)

        ttk.Label(frm, text="Modo K:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        c2 = ttk.Combobox(frm, textvariable=self.k_mode, values=["auto", "manual"], state="readonly", width=10)
        c2.grid(row=0, column=1, padx=5, pady=5)
        c2.bind("<<ComboboxSelected>>", lambda e: self._refresh_k_mode())

        ttk.Label(frm, text="k_use:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.ent_kuse = ttk.Entry(frm, textvariable=self.k_use, width=8, state="readonly")
        self.ent_kuse.grid(row=0, column=3, padx=5, pady=5)

        ttk.Button(frm, text="Calcular k_auto (B4)", command=self.on_calc_k_auto)\
            .grid(row=0, column=4, padx=5, pady=5)

        ttk.Label(frm, text="k_manual:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.ent_kmanual = ttk.Entry(frm, textvariable=self.k_manual, width=8)
        self.ent_kmanual.grid(row=1, column=1, padx=5, pady=5)

    def _build_palette_frame(self, master: tk.Tk) -> None:
        frm = ttk.LabelFrame(master, text="Colores y transparencias")
        frm.pack(fill="x", padx=10, pady=5)

        ttk.Label(frm, text="Paleta:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Combobox(frm, textvariable=self.palette,
                     values=["paper", "pastel", "viridis", "tab20", "warm", "cool"],
                     state="readonly", width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frm, text="alpha base:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        ttk.Entry(frm, textvariable=self.alpha_base, width=10).grid(row=0, column=3, padx=5, pady=5)
        ttk.Label(frm, text="alpha clusters:").grid(row=0, column=4, sticky="w", padx=5, pady=5)
        ttk.Entry(frm, textvariable=self.alpha_clusters, width=10).grid(row=0, column=5, padx=5, pady=5)

    def _build_adv_frame(self, master: tk.Tk) -> None:
        frm = ttk.LabelFrame(master, text="Opciones avanzadas")
        frm.pack(fill="x", padx=10, pady=5)

        ttk.Checkbutton(frm, text="Permitir 'Otras'", variable=self.allow_otras)\
            .grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Label(frm, text="Otras min score:").grid(row=0, column=1, sticky="w", padx=5, pady=5)
        ttk.Entry(frm, textvariable=self.otras_min_score, width=8).grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(frm, text="Otras cap:").grid(row=0, column=3, sticky="w", padx=5, pady=5)
        ttk.Entry(frm, textvariable=self.otras_cap, width=8).grid(row=0, column=4, padx=5, pady=5)

        ttk.Checkbutton(frm, text="Subclusters", variable=self.subclusters)\
            .grid(row=0, column=5, sticky="w", padx=5, pady=5)
        ttk.Label(frm, text="Sub min pct:").grid(row=0, column=6, sticky="w", padx=5, pady=5)
        ttk.Entry(frm, textvariable=self.sub_min_pct, width=8).grid(row=0, column=7, padx=5, pady=5)

        # Pairing numéricos
        ttk.Label(frm, text="Pair Δφ (°):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frm, textvariable=self.pair_max_phase_deg, width=8).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(frm, text="Pair Δy ks:").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        ttk.Entry(frm, textvariable=self.pair_max_y_ks, width=8).grid(row=1, column=3, padx=5, pady=5)
        ttk.Label(frm, text="Pair w ratio:").grid(row=1, column=4, sticky="w", padx=5, pady=5)
        ttk.Entry(frm, textvariable=self.pair_min_weight_ratio, width=8).grid(row=1, column=5, padx=5, pady=5)
        ttk.Label(frm, text="Pair miss pen:").grid(row=1, column=6, sticky="w", padx=5, pady=5)
        ttk.Entry(frm, textvariable=self.pair_miss_penalty, width=8).grid(row=1, column=7, padx=5, pady=5)

        # Pairing toggles
        ttk.Checkbutton(frm, text="K por pares (avanzada)", variable=self.pair_enforce_same_k)\
            .grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(frm, text="Mostrar líneas de unión (pares)", variable=self.pairs_show_lines)\
            .grid(row=2, column=1, sticky="w", padx=5, pady=5)

        # Reporte
        ttk.Checkbutton(frm, text="Embed assets", variable=self.embed_assets)\
            .grid(row=2, column=6, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(frm, text="Solo HTML (sin PDF)", variable=self.no_pdf)\
            .grid(row=2, column=7, sticky="w", padx=5, pady=5)

    def _build_actions_frame(self, master: tk.Tk) -> None:
        frm = ttk.Frame(master)
        frm.pack(fill="x", padx=10, pady=10)

        # >>> ESTE es el botón de Ejecutar <<<
        self.btn_run = ttk.Button(frm, text="Ejecutar", command=self.on_run, width=18)
        self.btn_run.pack(side="left", padx=5)

        self.btn_open_out = ttk.Button(frm, text="Abrir out", command=self.open_out, width=12)
        self.btn_open_out.pack(side="left", padx=5)

        ttk.Label(frm, text="Progreso:").pack(side="left", padx=10)
        self.pb = ttk.Progressbar(frm, mode="determinate", length=350)
        self.pb.pack(side="left", padx=5)

    def _build_log_frame(self, master: tk.Tk) -> None:
        frm = ttk.LabelFrame(master, text="Consola")
        frm.pack(fill="both", expand=True, padx=10, pady=5)
        self.txt = tk.Text(frm, height=14, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=6, pady=6)

    # ---------- Events / helpers ----------
    def browse_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("XML", "*.xml")])
        if path:
            self.xml_file.set(path)

    def browse_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.batch_folder.set(path)

    def _refresh_phase_mode(self) -> None:
        manual = (self.align_mode.get() == "manual")
        self.ent_phase.configure(state="normal" if manual else "disabled")

    def _refresh_k_mode(self) -> None:
        manual = (self.k_mode.get() == "manual")
        self.ent_kmanual.configure(state="normal" if manual else "disabled")

    def _validate_phase(self) -> float | None:
        if self.align_mode.get() == "auto":
            return None
        try:
            val = float(self.phase_deg.get())
        except Exception:
            messagebox.showerror("Fase inválida", "Introduce grados (0–360).")
            return None
        if not (0.0 <= val <= 360.0):
            messagebox.showerror("Fase inválida", "El valor debe estar entre 0 y 360.")
            return None
        return val

    def log(self, s: str) -> None:
        self.log_queue.put(s)

    def _pump_logs(self) -> None:
        while True:
            try:
                s = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.txt.insert("end", s + "\n")
            self.txt.see("end")
        self.master.after(120, self._pump_logs)

    def _save_ui_state(self) -> None:
        state = {
            "xml_file": self.xml_file.get(),
            "batch_folder": self.batch_folder.get(),
            "sensor": self.sensor.get(),
            "tipo_tx": self.tipo_tx.get(),
            "align_mode": self.align_mode.get(),
            "phase_deg": self.phase_deg.get(),
            "k_mode": self.k_mode.get(),
            "k_manual": self.k_manual.get(),
            "k_use": self.k_use.get(),
            "palette": self.palette.get(),
            "alpha_base": self.alpha_base.get(),
            "alpha_clusters": self.alpha_clusters.get(),
            "allow_otras": self.allow_otras.get(),
            "otras_min_score": self.otras_min_score.get(),
            "otras_cap": self.otras_cap.get(),
            "subclusters": self.subclusters.get(),
            "sub_min_pct": self.sub_min_pct.get(),
            "pair_max_phase_deg": self.pair_max_phase_deg.get(),
            "pair_max_y_ks": self.pair_max_y_ks.get(),
            "pair_min_weight_ratio": self.pair_min_weight_ratio.get(),
            "pair_miss_penalty": self.pair_miss_penalty.get(),
            "pair_enforce_same_k": self.pair_enforce_same_k.get(),
            "pairs_show_lines": self.pairs_show_lines.get(),
            "embed_assets": self.embed_assets.get(),
            "no_pdf": self.no_pdf.get(),
        }
        try:
            with open(UI_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _load_ui_state(self) -> None:
        p = Path(UI_STATE_FILE)
        if not p.exists():
            return
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        for key, val in state.items():
            if hasattr(self, key):
                var = getattr(self, key)
                try:
                    if isinstance(var, tk.Variable):
                        var.set(val)
                except Exception:
                    pass
        self._refresh_phase_mode()
        self._refresh_k_mode()

    # ---------- Actions ----------
    def on_calc_k_auto(self) -> None:
        """Ejecuta B4 para precalcular k_auto y llenar k_use."""
        xml = self.xml_file.get().strip()
        if not xml:
            messagebox.showwarning("Falta archivo", "Selecciona un XML.")
            return

        out_dir = Path("out"); out_dir.mkdir(exist_ok=True)
        stem = Path(xml).stem
        # usar el mismo prefix que B3/B5/B6 -> out/stem
        out_prefix = out_dir / stem
        phase_flag = ["--phase-align", "auto"] if self.align_mode.get() == "auto" \
            else ["--phase-align", str(self.phase_deg.get())]

        cmd = [sys.executable, "bloque4.py", "--xml", xml, "--recluster-after-align",
               "--out-prefix", str(out_prefix), *phase_flag]
        self.log("[B4] " + " ".join(cmd))
        try:
            out = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if out.stdout.strip():
                self.log(out.stdout.strip())
            if out.stderr.strip():
                self.log(out.stderr.strip())
        except subprocess.CalledProcessError as e:
            self.log(e.stdout.strip())
            self.log(e.stderr.strip())
            messagebox.showerror("B4 falló", "No se pudo calcular k_auto. Revisa la consola.")
            return

        # Leer JSON k_auto
        kjson = out_dir / f"{stem}_b4_kauto.json"
        if kjson.exists():
            try:
                data = json.loads(kjson.read_text(encoding="utf-8"))
                k_opt_auto = int(data.get("k_opt_auto", 0))
                if k_opt_auto > 0:
                    self.k_use.set(k_opt_auto)
                    self.k_mode.set("auto")
                    self._refresh_k_mode()
                    self.log(f"[B4] k_opt_auto = {k_opt_auto}")
                else:
                    messagebox.showwarning("k_auto no encontrado",
                                           "No se encontró k_opt_auto en el JSON.")
            except Exception:
                messagebox.showwarning("k_auto no legible", "No se pudo leer el JSON.")
        else:
            messagebox.showwarning("Sin JSON", f"No existe: {kjson}")

    def on_run(self) -> None:
        """Botón Ejecutar — lanza el pipeline en un hilo."""
        if self.proc_thread and self.proc_thread.is_alive():
            messagebox.showinfo("En curso", "Ya hay un proceso ejecutándose.")
            return
        xml = self.xml_file.get().strip()
        folder = self.batch_folder.get().strip()
        if not xml and not folder:
            messagebox.showwarning("Entrada requerida", "Elige un XML o una carpeta de XMLs.")
            return
        if self.align_mode.get() == "manual":
            if self._validate_phase() is None:
                return
        if self.k_mode.get() == "manual":
            try:
                self.k_use.set(int(self.k_manual.get()))
            except Exception:
                messagebox.showwarning("K manual inválido", "k_manual debe ser entero positivo.")
                return

        self.btn_run.configure(state="disabled")
        self.pb.configure(value=0, maximum=100)
        self.stop_flag.clear()
        self._save_ui_state()
        self.proc_thread = threading.Thread(target=self.run_pipeline, daemon=True)
        self.proc_thread.start()

    # --------- Core pipeline ----------
    def _iter_xmls(self) -> list[Path]:
        if self.batch_folder.get().strip():
            d = Path(self.batch_folder.get().strip())
            xmls = sorted([p for p in d.glob("*.xml") if p.is_file()])
        elif self.xml_file.get().strip():
            xmls = [Path(self.xml_file.get().strip())]
        else:
            xmls = []
        return xmls

    def run_pipeline(self) -> None:
        try:
            xmls = self._iter_xmls()
            if not xmls:
                self.log("No hay XMLs para procesar.")
                return

            out_dir = Path("out"); out_dir.mkdir(exist_ok=True)
            total_steps = len(xmls) * 5  # B3 nat + B3 al + B4 + B5 + B6
            step = 0

            for i, xml_path in enumerate(xmls, 1):
                is_batch = bool(self.batch_folder.get().strip())
                stem = xml_path.stem
                # Carpeta destino
                if is_batch:
                    batch_dir = out_dir / "batch"; batch_dir.mkdir(exist_ok=True)
                    subdir = batch_dir / f"{i:03d}_{stem}"
                    subdir.mkdir(parents=True, exist_ok=True)
                    base_prefix = subdir / stem
                else:
                    base_prefix = out_dir / stem

                # Phase flag
                phase_flag = ["--phase-align", "auto"] if self.align_mode.get() == "auto" \
                    else ["--phase-align", str(self.phase_deg.get())]

                # ===== B3 natural =====
                cmd_b3_nat = [
                    sys.executable, "bloque3.py", str(xml_path),
                    "--mode", "natural",
                    "--palette", self.palette.get(),
                    "--alpha-base", str(self.alpha_base.get()),
                    "--alpha-clusters", str(self.alpha_clusters.get()),
                    "--sub-min-pct", str(self.sub_min_pct.get()),
                    "--out-prefix", str(base_prefix), *phase_flag, "--plotly-3d"
                ]
                self._run_and_log(cmd_b3_nat); step += 1; self._progress(step, total_steps)

                # ===== B3 aligned =====
                k_use_val = int(self.k_use.get())
                cmd_b3_al = [
                    sys.executable, "bloque3.py", str(xml_path),
                    "--mode", "aligned",
                    "--k-use", str(k_use_val),
                    "--palette", self.palette.get(),
                    "--alpha-base", str(self.alpha_base.get()),
                    "--alpha-clusters", str(self.alpha_clusters.get()),
                    "--sub-min-pct", str(self.sub_min_pct.get()),
                    "--out-prefix", str(base_prefix), *phase_flag, "--plotly-3d"
                ]
                self._run_and_log(cmd_b3_al); step += 1; self._progress(step, total_steps)

                # ===== B4 (k_auto auxiliar) =====
                cmd_b4 = [
                    sys.executable, "bloque4.py",
                    "--xml", str(xml_path),
                    "--recluster-after-align",
                    "--out-prefix", str(base_prefix), *phase_flag
                ]
                self._run_and_log(cmd_b4); step += 1; self._progress(step, total_steps)

                # ===== B5 (con pairing, políticas y toggles) =====
                cmd_b5 = [
                    sys.executable, "bloque5.py", str(xml_path),
                    "--sensor", self.sensor.get(),
                    "--tipo-tx", self.tipo_tx.get(),
                    "--k-use", str(k_use_val),
                    "--allow-otras", "true" if self.allow_otras.get() else "false",
                    "--otras-min-score", str(self.otras_min_score.get()),
                    "--otras-cap", str(self.otras_cap.get()),
                    "--subclusters" if self.subclusters.get() else "",
                    "--sub-min-pct", str(self.sub_min_pct.get()),
                    "--palette", self.palette.get(),
                    "--alpha-base", str(self.alpha_base.get()),
                    "--alpha-clusters", str(self.alpha_clusters.get()),
                    "--pair-max-phase-deg", str(self.pair_max_phase_deg.get()),
                    "--pair-max-y-ks", str(self.pair_max_y_ks.get()),
                    "--pair-min-weight-ratio", str(self.pair_min_weight_ratio.get()),
                    "--pair-miss-penalty", str(self.pair_miss_penalty.get()),
                    "--pair-enforce-same-k" if self.pair_enforce_same_k.get() else "",
                    "--pairs-show-lines" if self.pairs_show_lines.get() else "",
                    "--out-prefix", str(base_prefix), *phase_flag,
                    "--recluster-after-align",
                ]
                cmd_b5 = [c for c in cmd_b5 if c != ""]
                self._run_and_log(cmd_b5); step += 1; self._progress(step, total_steps)

                # ===== B6 (reporte) =====
                # Preparar comandos para Bloque 6.  No se pasa --k-use; en su lugar, cuando
                # k_mode es manual se utiliza --k-manual para mostrar la comparativa en
                # el reporte.  Si k_mode es auto, se omite k_manual.
                cmd_b6 = [
                    sys.executable, "bloque6.py",
                    "--xml", str(xml_path),
                    "--sensor", self.sensor.get(),
                    "--tipo-tx", self.tipo_tx.get(),
                    "--allow-otras", "true" if self.allow_otras.get() else "false",
                    "--otras-min-score", str(self.otras_min_score.get()),
                    "--otras-cap", str(self.otras_cap.get()),
                    "--subclusters" if self.subclusters.get() else "",
                    "--sub-min-pct", str(self.sub_min_pct.get()),
                    "--embed-assets" if self.embed_assets.get() else "",
                    "--no-pdf" if self.no_pdf.get() else "",
                    "--out-prefix", str(base_prefix),
                ] + list(phase_flag)
                # Agregar k_manual si procede
                if self.k_mode.get() == "manual":
                    try:
                        kman = int(self.k_manual.get())
                        if kman > 0:
                            cmd_b6 += ["--k-manual", str(kman)]
                    except Exception:
                        pass
                cmd_b6 += ["--recluster-after-align"]
                cmd_b6 = [c for c in cmd_b6 if c != ""]
                self._run_and_log(cmd_b6); step += 1; self._progress(step, total_steps)

                # Abrir HTML automáticamente en archivo único
                if not is_batch:
                    html_path = Path(f"{str(base_prefix)}_reporte.html")
                    if html_path.exists():
                        try:
                            if os.name == "nt":
                                os.startfile(str(html_path))  # type: ignore[attr-defined]
                            elif sys.platform == "darwin":
                                subprocess.run(["open", str(html_path)], check=False)
                            else:
                                subprocess.run(["xdg-open", str(html_path)], check=False)
                            self.log(f"[OPEN] {html_path}")
                        except Exception:
                            self.log(f"[WARN] No pude abrir {html_path}")

                self.log(f"[OK] {i}/{len(xmls)} completado → {base_prefix}")

            self.log("Proceso finalizado.")
        finally:
            self.btn_run.configure(state="normal")
            self.pb.configure(value=self.pb["maximum"])

    def _run_and_log(self, cmd: list[str]) -> None:
        self.log(" ".join(cmd))
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if out.stdout.strip():
                self.log(out.stdout.strip())
            if out.stderr.strip():
                self.log(out.stderr.strip())
        except subprocess.CalledProcessError as e:
            if e.stdout:
                self.log(e.stdout.strip())
            if e.stderr:
                self.log(e.stderr.strip())
            self.log("[WARN] Comando falló; se continúa con el pipeline.")

    def _progress(self, step: int, total: int) -> None:
        frac = max(0.0, min(1.0, step / max(1, total)))
        self.pb.configure(value=int(frac * 100))
        self.master.update_idletasks()

    def open_out(self) -> None:
        out_dir = Path("out")
        out_dir.mkdir(exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(out_dir))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(out_dir)], check=False)
            else:
                subprocess.run(["xdg-open", str(out_dir)], check=False)
        except Exception:
            messagebox.showinfo("Abrir out", f"Abre manualmente: {out_dir.resolve()}")


def main() -> None:
    root = tk.Tk()
    PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
