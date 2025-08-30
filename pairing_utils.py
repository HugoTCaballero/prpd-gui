"""
pairing_utils.py
================

Funciones utilitarias para emparejar subclusters detectados en el análisis PRPD.
El objetivo es agrupar lóbulos pertenecientes a la misma fuente física pero que
aparecen en semiciclos opuestos del periodo eléctrico.  Se toman en cuenta
distancias de fase cíclicas, diferencias de amplitud y proporciones de peso.

Se utilizan métricas normalizadas para calcular un coste de emparejamiento.  Si
está disponible SciPy, se aplica el algoritmo húngaro para encontrar la
asignación óptima; de lo contrario se utiliza un método greedy.  Los tipos
cavidad, superficial y flotante deben emparejar sus lóbulos entre semiciclos,
mientras que corona puede aparecer como mono‑semiciclo.

La función principal devuelve una lista de pares con campos detallados y un
diccionario de mapeo subcluster→pair_id.

"""

from __future__ import annotations

from typing import List, Dict, Tuple, Any, Optional
import math

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _scipy_available = True
except Exception:
    _scipy_available = False


def _phase_diff_semi(phi1: float, phi2: float) -> float:
    """Calcula la mínima diferencia de fase considerando semiciclos.

    Se toman tres diferencias: |φ1−φ2−180|, |φ1−φ2+180| y |φ1−φ2|,
    retornando la mínima en grados.

    Parameters
    ----------
    phi1, phi2 : float
        Ángulos en grados (0–360).

    Returns
    -------
    float
        La menor diferencia en grados.
    """
    d1 = abs((phi1 - phi2) - 180.0)
    d2 = abs((phi1 - phi2) + 180.0)
    d3 = abs(phi1 - phi2)
    return min(d1, d2, d3)


def pair_subclusters(
    subclusters: List[Dict[str, Any]],
    pair_max_phase_deg: float = 25.0,
    pair_max_y_ks: float = 0.25,
    pair_min_weight_ratio: float = 0.4,
    pair_miss_penalty: float = 0.15,
    *,
    enforce_same_k: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Empareja subclusters en posibles fuentes de descarga.

    Para cada tipo (cavidad, superficial, flotante) se separan los subclusters
    en dos semiciclos (fase <180° y ≥180°) y se calcula una matriz de costes
    para emparejarlos.  El coste combina la distancia de fase, la diferencia
    de amplitud y una penalización por diferencia de pesos.  Se aplica el
    algoritmo húngaro si está disponible; en caso contrario se utiliza un
    emparejamiento greedy.  Los subclusters de tipo corona no se emparejan y
    se tratan como fuentes mono‑semiciclo.

    Parameters
    ----------
    subclusters : list of dict
        Lista de subclusters con claves al menos 'id', 'type', 'phase', 'y' y 'weight'.
    pair_max_phase_deg : float
        Umbral máximo de desfase aceptable para un emparejamiento (grados).  Diferencias
        mayores se normalizan pero se pueden penalizar.
    pair_max_y_ks : float
        Umbral máximo de diferencia de amplitud (en unidades de "k-s", aproximado por
        fracción del rango 0–100).  Diferencias mayores se normalizan pero se pueden
        penalizar.
    pair_min_weight_ratio : float
        Relación mínima de pesos entre subclusters (peso menor / mayor).  Si la relación
        es menor, se añade una penalización.
    pair_miss_penalty : float
        Penalización al coste para pares no asignados o emparejamientos muy malos.

    Returns
    -------
    list of dict, dict
        Una lista de pares (fuentes) y un diccionario que asigna cada id de subcluster
        a un identificador de par.  Cada par tiene las claves: 'pair_id', 'type',
        'id_pos', 'id_neg', 'phi_pos', 'phi_neg', 'dy', 'weight_pos', 'weight_neg',
        'weight_sum' y 'score'.  Los campos con datos no aplicables se dejan vacíos
        o con NaN.
    """
    pairs: List[Dict[str, Any]] = []
    pair_map: Dict[str, int] = {}
    next_pair_id = 1
    # Normalizar pair_max_y_ks para usarlo como factor de escalado: se asume que los
    # valores de 'y' están en [0,100]; por lo tanto dy_norm = |y1-y2| / (pair_max_y_ks*100).
    y_scale = max(pair_max_y_ks * 100.0, 1e-6)
    # Si se solicita, agrupar también por k_label para emparejar sólo dentro del mismo K
    # 'k_label' debe estar presente en cada subcluster cuando enforce_same_k=True.
    grouped: Dict[Tuple[str, Optional[Any]], List[Dict[str, Any]]] = {}
    for sc in subclusters:
        t = str(sc.get('type', '')).lower()
        if enforce_same_k:
            k_key = sc.get('k_label')
        else:
            k_key = None
        grouped.setdefault((t, k_key), []).append(sc)
    # Procesar cada grupo (tipo y k_label si aplica)
    for (t, k_key), sublist in grouped.items():
        if t == 'corona':
            # No emparejar corona: cada subcluster es una fuente individual
            for sc in sublist:
                pair_id = next_pair_id
                next_pair_id += 1
                pair_map[sc['id']] = pair_id
                phi = float(sc.get('phase', 0.0))
                is_pos = (phi < 180.0)
                pairs.append({
                    'pair_id': pair_id,
                    'type': t,
                    'id_pos': sc['id'] if is_pos else '',
                    'id_neg': sc['id'] if not is_pos else '',
                    'phi_pos': phi if is_pos else float('nan'),
                    'phi_neg': phi if not is_pos else float('nan'),
                    'dy': 0.0,
                    'weight_pos': sc.get('weight', 0.0) if is_pos else 0.0,
                    'weight_neg': sc.get('weight', 0.0) if not is_pos else 0.0,
                    'weight_sum': sc.get('weight', 0.0),
                    'score': 0.0,
                })
            continue
        # Para tipos a emparejar: cavidad, superficial, flotante
        left: List[Dict[str, Any]] = []  # φ in [0,180)
        right: List[Dict[str, Any]] = []  # φ in [180,360)
        for sc in sublist:
            phi = float(sc.get('phase', 0.0))
            if 0.0 <= phi < 180.0:
                left.append(sc)
            else:
                right.append(sc)
        n_left = len(left)
        n_right = len(right)
        # Si una de las mitades está vacía, tratar cada subcluster como fuente individual
        if n_left == 0 or n_right == 0:
            for sc in left + right:
                pair_id = next_pair_id
                next_pair_id += 1
                pair_map[sc['id']] = pair_id
                phi = float(sc.get('phase', 0.0))
                is_pos = (phi < 180.0)
                pairs.append({
                    'pair_id': pair_id,
                    'type': t,
                    'id_pos': sc['id'] if is_pos else '',
                    'id_neg': sc['id'] if not is_pos else '',
                    'phi_pos': phi if is_pos else float('nan'),
                    'phi_neg': phi if not is_pos else float('nan'),
                    'dy': 0.0,
                    'weight_pos': sc.get('weight', 0.0) if is_pos else 0.0,
                    'weight_neg': sc.get('weight', 0.0) if not is_pos else 0.0,
                    'weight_sum': sc.get('weight', 0.0),
                    'score': 1.0,
                })
            continue
        # Construir matriz de costes n_left x n_right
        cost_mat: List[List[float]] = [[0.0 for _ in range(n_right)] for _ in range(n_left)]
        for i, sc_l in enumerate(left):
            for j, sc_r in enumerate(right):
                phi_l = float(sc_l.get('phase', 0.0))
                phi_r = float(sc_r.get('phase', 0.0))
                # Distancia de fase semicíclica normalizada
                dphi = _phase_diff_semi(phi_l, phi_r)
                phi_norm = dphi / max(pair_max_phase_deg, 1e-6)
                if phi_norm > 1.0:
                    phi_norm = 1.0
                # Diferencia de amplitud normalizada
                y_l = float(sc_l.get('y', 0.0))
                y_r = float(sc_r.get('y', 0.0))
                dy_norm = abs(y_l - y_r) / y_scale
                if dy_norm > 1.0:
                    dy_norm = 1.0
                # Penalización por relación de pesos
                w_l = float(sc_l.get('weight', 0.0))
                w_r = float(sc_r.get('weight', 0.0))
                if w_l <= 0.0 or w_r <= 0.0:
                    ratio = 0.0
                else:
                    ratio = min(w_l, w_r) / max(w_l, w_r)
                weight_penalty = 0.0
                if ratio < pair_min_weight_ratio:
                    weight_penalty = (pair_min_weight_ratio - ratio)
                # Coste total (ponderaciones fijas 0.6 y 0.4)
                cost = 0.6 * phi_norm + 0.4 * dy_norm + weight_penalty
                # Penalizar costes fuera de umbrales
                if dphi > pair_max_phase_deg or abs(y_l - y_r) / 100.0 > pair_max_y_ks:
                    cost += pair_miss_penalty
                if cost > 1.0:
                    cost = 1.0
                cost_mat[i][j] = cost
        # Emparejar mediante Hungarian o método greedy
        matched_left: List[Optional[int]] = [None] * n_left
        matched_right: List[Optional[int]] = [None] * n_right
        if _scipy_available:
            # linear_sum_assignment funciona con matrices rectangulares y devuelve
            # tantas asignaciones como min(n_left, n_right)
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            for i_idx, j_idx in zip(row_ind, col_ind):
                matched_left[i_idx] = j_idx
                matched_right[j_idx] = i_idx
        else:
            # Greedy: para cada izquierda, elegir el mejor derecho disponible
            used_right = [False] * n_right
            for i in range(n_left):
                best_j = None
                best_cost = None
                for j in range(n_right):
                    if used_right[j]:
                        continue
                    c = cost_mat[i][j]
                    if best_cost is None or c < best_cost:
                        best_cost = c
                        best_j = j
                if best_j is not None:
                    matched_left[i] = best_j
                    matched_right[best_j] = i
                    used_right[best_j] = True
        # Construir pares emparejados
        for i, sc_l in enumerate(left):
            j_idx = matched_left[i]
            if j_idx is not None:
                sc_r = right[j_idx]
                # Coste de esta pareja
                phi_l = float(sc_l.get('phase', 0.0))
                phi_r = float(sc_r.get('phase', 0.0))
                dphi = _phase_diff_semi(phi_l, phi_r)
                phi_norm = dphi / max(pair_max_phase_deg, 1e-6)
                if phi_norm > 1.0:
                    phi_norm = 1.0
                dy_norm = abs(sc_l.get('y', 0.0) - sc_r.get('y', 0.0)) / y_scale
                if dy_norm > 1.0:
                    dy_norm = 1.0
                w_l = float(sc_l.get('weight', 0.0))
                w_r = float(sc_r.get('weight', 0.0))
                ratio = 0.0
                if w_l > 0.0 and w_r > 0.0:
                    ratio = min(w_l, w_r) / max(w_l, w_r)
                weight_penalty = 0.0
                if ratio < pair_min_weight_ratio:
                    weight_penalty = (pair_min_weight_ratio - ratio)
                score = 0.6 * phi_norm + 0.4 * dy_norm + weight_penalty
                if dphi > pair_max_phase_deg or abs(sc_l.get('y', 0.0) - sc_r.get('y', 0.0)) / 100.0 > pair_max_y_ks:
                    score += pair_miss_penalty
                if score > 1.0:
                    score = 1.0
                pair_id = next_pair_id
                next_pair_id += 1
                pair_map[sc_l['id']] = pair_id
                pair_map[sc_r['id']] = pair_id
                pairs.append({
                    'pair_id': pair_id,
                    'type': t,
                    'id_pos': sc_l['id'],
                    'id_neg': sc_r['id'],
                    'phi_pos': phi_l,
                    'phi_neg': phi_r,
                    'dy': abs(sc_l.get('y', 0.0) - sc_r.get('y', 0.0)),
                    'weight_pos': w_l,
                    'weight_neg': w_r,
                    'weight_sum': w_l + w_r,
                    'score': score,
                })
        # Añadir subclusters no emparejados como pares individuales
        for i, sc_l in enumerate(left):
            if matched_left[i] is None:
                pair_id = next_pair_id
                next_pair_id += 1
                pair_map[sc_l['id']] = pair_id
                phi_l = float(sc_l.get('phase', 0.0))
                w_l = float(sc_l.get('weight', 0.0))
                pairs.append({
                    'pair_id': pair_id,
                    'type': t,
                    'id_pos': sc_l['id'],
                    'id_neg': '',
                    'phi_pos': phi_l,
                    'phi_neg': float('nan'),
                    'dy': 0.0,
                    'weight_pos': w_l,
                    'weight_neg': 0.0,
                    'weight_sum': w_l,
                    'score': 1.0,
                })
        for j, sc_r in enumerate(right):
            if matched_right[j] is None:
                pair_id = next_pair_id
                next_pair_id += 1
                pair_map[sc_r['id']] = pair_id
                phi_r = float(sc_r.get('phase', 0.0))
                w_r = float(sc_r.get('weight', 0.0))
                pairs.append({
                    'pair_id': pair_id,
                    'type': t,
                    'id_pos': '',
                    'id_neg': sc_r['id'],
                    'phi_pos': float('nan'),
                    'phi_neg': phi_r,
                    'dy': 0.0,
                    'weight_pos': 0.0,
                    'weight_neg': w_r,
                    'weight_sum': w_r,
                    'score': 1.0,
                })
    return pairs, pair_map