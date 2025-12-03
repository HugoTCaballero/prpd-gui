# batch_bridge.py
# -*- coding: utf-8 -*-
"""
Invoca batch_metrics.py desde la GUI (Windows) y devuelve resúmenes útiles.
Respeta: manifest existente + criterios PCT/MIN/WILSON.
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

def run_batch(root_dir: str, manifest_path: str, out_dir: str = "out") -> Dict[str, Any]:
    """
    Ejecuta: python batch_metrics.py <root> --manifest <manifest> --out-dir <out>
    Devuelve dict con paths y resúmenes (filters_selected + summary).
    """
    py = sys.executable or "python"
    cmd = [py, "batch_metrics.py", root_dir, "--manifest", manifest_path, "--out-dir", out_dir]
    subprocess.run(cmd, check=True)

    metrics_dir = Path(out_dir, "metrics")
    res = {
        "filters_selected_path": str(metrics_dir / "filters_selected.json"),
        "summary_path": str(metrics_dir / "summary.txt"),
        "sens_spec": {
            "cavidad": str(metrics_dir / "sens_spec_cavidad.csv"),
            "superficial": str(metrics_dir / "sens_spec_superficial.csv"),
            "corona": str(metrics_dir / "sens_spec_corona.csv"),
            "flotante": str(metrics_dir / "sens_spec_flotante.csv"),
        },
    }
    # Cargar resúmenes si existen
    try:
        with open(res["filters_selected_path"], "r", encoding="utf-8") as f:
            res["filters_selected"] = json.load(f)
    except Exception:
        res["filters_selected"] = {}

    try:
        with open(res["summary_path"], "r", encoding="utf-8") as f:
            res["summary"] = f.read()
    except Exception:
        res["summary"] = ""

    return res
