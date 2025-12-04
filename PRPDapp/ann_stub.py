from __future__ import annotations

import numpy as np


def predict_pd_source_ann(result: dict) -> dict:
    """
    Fachada para el gráfico 'ANN Predicted PD Source'.
    Hoy usa rule_pd como backend; mañana se puede reemplazar por ANN real.
    """
    rule = result.get("rule_pd", {}) if isinstance(result, dict) else {}
    classes = rule.get("classes") or [
        "corona",
        "superficial_tracking",
        "cavidad_interna",
        "flotante",
        "ruido_baja",
    ]
    probs = rule.get("class_probs")
    scores = rule.get("class_scores") or rule.get("scores")
    if probs is None and scores is not None:
        arr = np.asarray(list(scores.values()) if isinstance(scores, dict) else scores, dtype=float)
        if arr.size and arr.sum() > 0:
            probs = (arr / arr.sum()).tolist()
    if probs is None:
        probs = [0.0] * len(classes)

    return {
        "classes": classes,
        "probs": probs,
        "backend": "rule_based",
    }
