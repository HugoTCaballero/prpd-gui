# three_d.py
# -*- coding: utf-8 -*-
"""
Visualización 3D de PRPD:
- points: ndarray (N,3) -> ejes sugeridos: [fase_deg, amplitud, orden/tiempo]
- labels: lista o array (N,) con clases (cavidad/superficial/corona/flotante/ruido).
- Crea una ventana Matplotlib 3D standalone (funciona bien en Windows).

Uso:
    plot_prpd_3d(points, labels, title="PRPD 3D")
"""

from __future__ import annotations
from typing import Iterable, Dict, Optional
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")  # Para Windows + PyQt/Qt5
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

COLOR_BY_CLASS: Dict[str, str] = {
    "cavidad": "#e91e63",
    "superficial": "#03a9f4",
    "corona": "#ff9800",
    "flotante": "#4caf50",
    "ruido": "#9e9e9e",
}

def plot_prpd_3d(points: np.ndarray, labels: Iterable, title: str = "PRPD 3D") -> None:
    if points is None or len(points) == 0:
        return
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("Se esperaban puntos de forma (N,3).")

    labels = list(labels) if labels is not None else ["ruido"] * len(P)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    classes = list(sorted(set(labels)))
    for cls in classes:
        idx = [i for i, c in enumerate(labels) if c == cls]
        if not idx:
            continue
        c = COLOR_BY_CLASS.get(cls, "#607d8b")
        ax.scatter(P[idx, 0], P[idx, 1], P[idx, 2], s=14, depthshade=True, alpha=0.85, label=cls, c=c)

    ax.set_xlabel("Fase (°)")
    ax.set_ylabel("Amplitud (u)")
    ax.set_zlabel("Orden/tiempo")
    ax.legend(loc="upper right", title="Clase")
    ax.view_init(elev=22, azim=38)
    ax.grid(True)
    plt.tight_layout()
    plt.show()
