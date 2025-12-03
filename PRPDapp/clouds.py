from __future__ import annotations
import numpy as np
from sklearn.cluster import DBSCAN

def circular_mean_deg(ph: np.ndarray, w: np.ndarray | None = None) -> float:
    if ph.size == 0:
        return 0.0
    th = np.deg2rad(ph)
    if w is None:
        C = np.mean(np.cos(th)); S = np.mean(np.sin(th))
    else:
        w = np.asarray(w, dtype=float)
        W = float(np.sum(w)) if np.sum(w) > 0 else 1.0
        C = float(np.sum(w * np.cos(th))) / W
        S = float(np.sum(w * np.sin(th))) / W
    return float(np.rad2deg(np.arctan2(S, C))) % 360.0


def pixel_cluster_clouds(
    phase_deg: np.ndarray,
    amp: np.ndarray,
    eps: float = 0.045,
    min_samples: int = 10,
    force_multi: bool = False,
) -> list[dict]:
    if phase_deg.size == 0:
        return []
    # Normalize space: phase in [0,1], amplitude robust-scaled to tanh range
    med = float(np.median(amp))
    mad = float(np.median(np.abs(amp - med)) + 1e-9)
    a_n = np.tanh((amp - med) / (1.4826 * mad))
    X = np.column_stack([(phase_deg / 360.0), a_n])
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    mask = labels >= 0
    unique_labels = set(labels[mask])
    if force_multi and len(unique_labels) <= 1 and phase_deg.size >= 4:
        segments = np.floor((phase_deg % 360.0) / 360.0 * 3).astype(int)
        segments = np.clip(segments, 0, 2)
        if np.unique(segments).size <= 1:
            segments = np.zeros_like(segments)
            segments[phase_deg >= 180.0] = 1
        labels = segments
        mask = labels >= 0
        unique_labels = set(labels[mask])
    clouds: list[dict] = []
    n_total = len(labels)
    for k in sorted(set(labels)):
        if k < 0:
            continue
        idx = labels == k
        n = int(np.sum(idx))
        frac = n / max(1, n_total)
        ph = phase_deg[idx]
        a = amp[idx]
        w = np.abs(a)
        clouds.append({
            'id': int(k),
            'count': n,
            'frac': float(frac),
            'phase_mean': float(circular_mean_deg(ph, w)),
            'y_mean': float(np.average(a, weights=w)) if np.sum(w) > 0 else float(np.mean(a)),
        })
    clouds.sort(key=lambda c: (-c['count'], -c['frac']))
    return clouds


def combine_clouds(clouds: list[dict], max_dphi: float = 20.0, max_dy: float = 5.0) -> list[dict]:
    def dphi_semi(a: float, b: float) -> float:
        d1 = abs((a - b) - 180.0); d2 = abs((a - b) + 180.0); d3 = abs(a - b)
        return min(d1, d2, d3)
    remaining = list(clouds)
    merged: list[dict] = []
    used = [False] * len(remaining)
    for i, ci in enumerate(remaining):
        if used[i]:
            continue
        acc_ids = [ci['id']]
        total = ci['count']
        w_phase = ci['phase_mean'] * ci['count']
        w_y = ci['y_mean'] * ci['count']
        used[i] = True
        for j, cj in enumerate(remaining):
            if used[j]:
                continue
            if dphi_semi(ci['phase_mean'], cj['phase_mean']) <= max_dphi and abs(ci['y_mean'] - cj['y_mean']) <= max_dy:
                used[j] = True
                acc_ids.append(cj['id'])
                total += cj['count']
                w_phase += cj['phase_mean'] * cj['count']
                w_y += cj['y_mean'] * cj['count']
        merged.append({
            'ids': acc_ids,
            'count': int(total),
            'phase_mean': float((w_phase / max(1, total)) % 360.0),
            'y_mean': float(w_y / max(1, total)),
        })
    merged.sort(key=lambda c: -c['count'])
    return merged


def select_dominant_clouds(clouds: list[dict], min_frac: float = 0.05) -> list[dict]:
    if not clouds:
        return []
    total = sum(int(c.get('count', 0)) for c in clouds)
    sel: list[dict] = []
    for c in clouds:
        frac = float(c.get('frac')) if 'frac' in c else (c.get('count', 0) / max(1, total))
        if frac >= min_frac:
            cc = dict(c); cc['frac'] = float(frac)
            sel.append(cc)
    if not sel:
        top = max(clouds, key=lambda x: x.get('count', 0))
        tt = dict(top); tt['frac'] = float(top.get('count', 0) / max(1, total))
        sel = [tt]
    return sel
