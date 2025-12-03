# prpd_ann.py
# -*- coding: utf-8 -*-
"""
ANN / Clasificador supervisado para PRPD basado en scikit-learn.
- Carga .pkl/.joblib (LogReg, MLP, XGB scikit-compatible, etc.)
- Vectoriza features por nombre (orden estable) + fallback heurístico.
- Devuelve probabilidades por clase: cavidad/superficial/corona/flotante/ruido.

Uso:
    ann = PRPDANN(class_names=["cavidad","superficial","corona","flotante","ruido"])
    ann.load_model("modelos/prpd_ann.pkl")
    proba = ann.predict_proba(features_dict)  # dict -> dict
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Optional
import numpy as np

try:
    import joblib
except Exception:
    joblib = None


DEFAULT_CLASSES = ["cavidad", "superficial", "corona", "flotante", "ruido"]

# Orden canónico de features (ajústalo a tu prpd_features.py si difiere)
FEATURE_ORDER: List[str] = [
    # amplitud / densidad
    "amp_mean", "amp_std", "amp_p95", "density",
    # fase y circularidad
    "phase_std_deg", "phase_entropy",
    # repetitividad / tasa de pulsos
    "rep_rate", "rep_entropy",
    # compacidad del cluster
    "cluster_compactness", "cluster_separation",
    # morfología global
    "lobes_count", "area_ratio",
]

class PRPDANN:
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names: List[str] = class_names or list(DEFAULT_CLASSES)
        self.model = None
        self.is_loaded = False

    def load_model(self, model_path: str) -> None:
        """Carga un clasificador sklearn serializado con joblib/pickle."""
        if joblib is None:
            raise RuntimeError("joblib no disponible. Instala scikit-learn y joblib.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        self.model = joblib.load(model_path)  # sklearn estimator con predict_proba
        # Si el modelo tiene clases específicas, respétalas (si son texto)
        try:
            if hasattr(self.model, "classes_"):
                # mapear a texto si venían como ints
                if all(isinstance(c, str) for c in self.model.classes_):
                    self.class_names = list(self.model.classes_)
        except Exception:
            pass
        self.is_loaded = True

    def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
        """Convierte el dict de features en vector ordenado y estable. Falta -> 0.0."""
        vec = [float(features.get(k, 0.0) or 0.0) for k in FEATURE_ORDER]
        return np.asarray([vec], dtype=np.float64)

    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        """Devuelve probabilidades por clase. Si no hay modelo, usa heurístico estable."""
        x = self._vectorize(features)
        if self.is_loaded and hasattr(self.model, "predict_proba"):
            try:
                p = self.model.predict_proba(x)[0]
                # Asegurar longitudes
                if len(p) != len(self.class_names):
                    # normaliza y recorta/expande
                    p = np.asarray(p, dtype=np.float64)
                    p = p[: len(self.class_names)]
                    p = p / max(p.sum(), 1e-8)
                return {c: float(v) for c, v in zip(self.class_names, p)}
            except Exception:
                pass

        # ---- Fallback heurístico (robusto cuando no hay modelo) ----
        # Señales: cavidad -> amp alta + compacidad alta; superficial -> rep_rate alto; 
        # corona -> phase_std alta; flotante -> density baja + rep_entropy alta; ruido -> catch-all.
        f = features
        amp = float(f.get("amp_p95", 0.0))
        compact = float(f.get("cluster_compactness", 0.0))
        rep = float(f.get("rep_rate", 0.0))
        phase_std = float(f.get("phase_std_deg", 0.0))
        density = float(f.get("density", 0.0))
        rep_ent = float(f.get("rep_entropy", 0.0))

        s_cav = 0.4 * self._norm(amp, 0, 1) + 0.6 * self._norm(compact, 0, 1)
        s_sup = 0.7 * self._norm(rep, 0, 1) + 0.3 * self._norm(density, 0, 1)
        s_cor = 0.8 * self._norm(phase_std, 0, 180) + 0.2 * self._norm(rep_ent, 0, 1)
        s_flo = 0.6 * (1 - self._norm(density, 0, 1)) + 0.4 * self._norm(rep_ent, 0, 1)

        scores = {
            "cavidad": s_cav,
            "superficial": s_sup,
            "corona": s_cor,
            "flotante": s_flo,
        }
        # ruido = lo que no encaja
        noise = max(0.0, 0.8 - max(scores.values()))
        scores["ruido"] = noise

        v = np.array(list(scores.values()), dtype=np.float64)
        v = v / max(v.sum(), 1e-8)
        return {k: float(p) for k, p in zip(scores.keys(), v)}

    @staticmethod
    def _norm(x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        y = (x - lo) / (hi - lo)
        return float(max(0.0, min(1.0, y)))
