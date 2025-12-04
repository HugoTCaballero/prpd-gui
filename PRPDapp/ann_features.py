from __future__ import annotations

import numpy as np

# Orden fijo de características para ANN futura
ANN_FEATURE_ORDER = [
    # FA-profile
    "fa_phase_width_deg",
    "fa_phase_center_deg",
    "fa_symmetry_index",
    "fa_concentration_index",
    "fa_p95_amplitude",
    "fa_n_pulses_total",
    # ANGPD ratio
    "n_angpd_angpd_ratio",
    # Gap-time
    "gap_p50_ms",
    "gap_p5_ms",
    # ANGPD avanzado
    "ang_phase_width_deg",
    "ang_phase_symmetry",
    "ang_phase_peaks",
    "ang_amp_concentration",
    "ang_amp_peaks",
]


def build_ann_feature_vector(result: dict) -> np.ndarray:
    """Construye vector X para ANN (hoy sin usar, futuro-ready)."""
    kpis = result.get("kpis", {}) if isinstance(result, dict) else {}
    vals = []
    for key in ANN_FEATURE_ORDER:
        try:
            val = kpis.get(key)
        except Exception:
            val = None
        if val is None:
            val = 0.0
        try:
            vals.append(float(val))
        except Exception:
            vals.append(0.0)
    return np.asarray(vals, dtype=float)
