# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import xml.etree.ElementTree as ET
import numpy as np


def _to_float_array(text: str) -> np.ndarray:
    vals = text.strip().split()
    return np.asarray([float(x) for x in vals], dtype=np.float32)


def _to_int_array(text: str) -> np.ndarray:
    vals = text.strip().split()
    return np.asarray([int(float(x)) for x in vals], dtype=np.int32)


def parse_prpd_events(path: str | Path, mains_hz: float = 60.0) -> Dict[str, Any]:
    """
    Parser de PRPD EVENTOS (<time> root) -> dict con:
      phase_deg[N] (float32), time_in_cycle_ms[N] (float32), quantity[N] (uint32),
      pixel[N] (float32, magnitud normalizada 0-100),
      semiciclo[N] (uint8 {0=L,1=H}), Q (int), meta {sensor, date, time, folder}.
    Reglas:
      - pixel invertido -> phase_deg = ((Q-1 - pixel) * 360)/Q
      - sample ignorado
    """
    p = Path(path)
    root = ET.parse(p).getroot()
    # Campos básicos
    sensor = (root.findtext('antenna') or '').strip() or (root.findtext('sensor') or '').strip()
    date = (root.findtext('date') or '').strip()
    timetxt = (root.findtext('time') or '').strip()
    folder = (root.findtext('folder') or '').strip()

    times_txt = root.findtext('times')
    pixel_txt = root.findtext('pixel')
    qty_txt = root.findtext('quantity')
    if times_txt is None or pixel_txt is None or qty_txt is None:
        raise ValueError('XML PRPD sin <times>/<pixel>/<quantity>')

    times = _to_float_array(times_txt)  # en ms dentro del ciclo
    pixel_raw = _to_int_array(pixel_txt)
    qty = _to_int_array(qty_txt).astype(np.uint32)

    if not (len(times) == len(pixel_raw) == len(qty)):
        nmin = min(len(times), len(pixel_raw), len(qty))
        times = times[:nmin]; pixel_raw = pixel_raw[:nmin]; qty = qty[:nmin]

    Q = int(np.nanmax(pixel_raw)) + 1 if pixel_raw.size else 360
    phase_deg = ((Q - 1 - pixel_raw.astype(np.float32)) * 360.0) / float(Q)
    phase_deg = np.mod(phase_deg, 360.0).astype(np.float32)
    semiciclo = (phase_deg >= 180.0).astype(np.uint8)

    pixel_vals = pixel_raw.astype(np.float32)

    out = {
        'phase_deg': phase_deg,
        'time_in_cycle_ms': times.astype(np.float32),
        'quantity': qty,
        'pixel': pixel_vals,
        'semiciclo': semiciclo,
        'Q': int(Q),
        'meta': {
            'sensor': sensor,
            'date': date,
            'time': timetxt,
            'folder': folder,
            'mains_hz': float(mains_hz),
            'source': str(p),
        }
    }
    return out
