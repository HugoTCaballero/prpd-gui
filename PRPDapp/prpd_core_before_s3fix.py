# prpd_core.py
# Núcleo de procesamiento: carga CSV, detección de ruido, auto-fase (0/120/240),
# clustering DBSCAN + prune, features, heurística de clase y severidad.

from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import xml.etree.ElementTree as ET
import math

SN_FILTER_CONFIGS = {
    "sn1": {"eps": 0.045, "min_samples": 8, "min_share": 0.02, "fallback_eps": [0.035, 0.03], "force_multi": True, "keep_all": True, "s5_min_frac": 0.02, "s3_eps": 0.065, "s3_min_samples": 6, "s3_force_multi": True},
    "sn2": {"eps": 0.055, "min_samples": 8, "min_share": 0.018, "fallback_eps": [0.04, 0.032], "force_multi": True, "keep_all": True, "s5_min_frac": 0.018, "s3_eps": 0.07, "s3_min_samples": 5, "s3_force_multi": True},
    "sn3": {"eps": 0.065, "min_samples": 6, "min_share": 0.015, "fallback_eps": [0.045, 0.035], "force_multi": True, "keep_all": True, "s5_min_frac": 0.015, "s3_eps": 0.075, "s3_min_samples": 5, "s3_force_multi": True},
    "sn4": {"eps": 0.05, "min_samples": 6, "min_share": 0.02, "fallback_eps": [0.035, 0.03], "force_multi": True, "keep_all": True, "s5_min_frac": 0.02, "s3_eps": 0.06, "s3_min_samples": 5, "s3_force_multi": True},
    "sn5": {"eps": 0.05, "min_samples": 6, "min_share": 0.01, "fallback_eps": [0.035, 0.03], "force_multi": True, "keep_all": True, "s5_min_frac": 0.01, "s3_eps": 0.06, "s3_min_samples": 5, "s3_force_multi": True},
}

SN_NOISE_OVERRIDES = {
    "sn4": {"bins_phase": 80, "bins_amp": 40, "min_count": 2, "ratio_thresh": 0.85},
}

# ---------- I/O ----------
def load_csv_prpd(path: Path) -> dict:
    import csv
    phase_keys = {"phase_deg","phase","phi"}
    amp_keys   = {"amplitude","amp","a"}
    ph, amp = [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        hdr = {k.strip().lower(): k for k in rdr.fieldnames or []}
        k_phase = next((hdr[k] for k in hdr if k.strip().lower() in phase_keys), None)
        k_amp   = next((hdr[k] for k in hdr if k.strip().lower() in amp_keys), None)
        if not k_phase or not k_amp:
            raise ValueError("CSV debe contener columnas de fase y amplitud (e.g. phase_deg, amplitude).")
        for row in rdr:
            try:
                ph.append(float(row[k_phase]))
                amp.append(float(row[k_amp]))
            except Exception:
                continue
    ph = np.asarray(ph, dtype=float)
    amp = np.asarray(amp, dtype=float)
    # normalización ligera de amplitud para algoritmos (no afecta reporte)
    amp_norm = robust_scale(amp)
    return {"phase_deg": wrap360(ph), "amplitude": amp, "amp_norm": amp_norm}


# ---------- XML loader ----------
PHASE_KEYS = {"phase", "fase", "angle", "phi", "phase_deg"}
AMP_KEYS = {"amplitude", "amp", "value", "v", "a"}


def _to_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _norm_phase_deg(x):
    if x is None:
        return None
    xv = float(x)
    if -2 * math.pi <= xv <= 2 * math.pi:
        xv = math.degrees(xv)
    xv = xv % 360.0
    if xv < 0:
        xv += 360.0
    return xv


def _find_by_keys(elem, keys):
    # 1) attributes
    for k, v in elem.attrib.items():
        if (k or "").strip().lower() in keys:
            val = _to_float(v)
            if val is not None:
                return val
    # 2) child tags
    for child in list(elem):
        tag = (child.tag or "").split("}")[-1].lower()
        if tag in keys and (child.text or "").strip():
            val = _to_float(child.text)
            if val is not None:
                return val
    return None


def _parse_points_xml_auto(root):
    pts = []
    for e in root.iter():
        ph = _find_by_keys(e, PHASE_KEYS)
        amp = _find_by_keys(e, AMP_KEYS)
        if ph is not None and amp is not None:
            phv = _norm_phase_deg(ph)
            if phv is not None:
                pts.append((phv, float(amp)))
    return pts


def _parse_megger_lists(root):
    """
    Fallback para XMLs de entrenamiento: extrae listas 'pixel'|'sample', 'times', 'quantity'.
    - Prioriza 'pixel' si existe; si no, usa 'sample'.
    - Convierte 'times' (ms, 0..16.6) a fase 0..360.
    - Amplitud = 'pixel' invertido y normalizado a 0..100; si no hay pixel, 'sample' min-max a 0..100.
    Devuelve (phase_deg, amplitude) o (None, None) si no se encuentran listas válidas.
    """
    def to_float_vec(s: str):
        s = (s or "").replace("\n", " ").replace("\t", " ")
        parts = [p for p in s.split(" ") if p != ""]
        if not parts:
            return None
        try:
            return np.asarray(list(map(float, parts)), dtype=float)
        except Exception:
            return None

    sample_text = None
    pixel_text = None
    times_text = None
    quantity_text = None
    for tag in root.iter():
        tname = (tag.tag or "").split('}')[-1].lower()
        ttext = (tag.text or "").strip()
        if tname.endswith("pixel"):
            pixel_text = ttext
        elif tname.endswith("sample"):
            sample_text = ttext
        elif tname.endswith("times"):
            times_text = ttext
        elif tname.endswith("quantity"):
            quantity_text = ttext

    if times_text is None or (pixel_text is None and sample_text is None):
        return None, None, None, None
    y_src = pixel_text if pixel_text is not None else sample_text
    y = to_float_vec(y_src)
    t = to_float_vec(times_text)
    if y is None or t is None:
        return None, None, None, None
    n = min(len(y), len(t))
    if n == 0:
        return None, None, None, None
    y = y[:n]
    t = t[:n]
    # Amplitud normalizada
    if pixel_text is not None:
        ymax = float(np.nanmax(y)) if np.isfinite(y).any() else 1.0
        if ymax <= 0:
            ymax = 1.0
        amp = 100.0 - (y * 100.0 / ymax)
    else:
        ymin = float(np.nanmin(y))
        ymax = float(np.nanmax(y))
        den = (ymax - ymin) if (ymax - ymin) > 0 else 1.0
        amp = (y - ymin) / den * 100.0
    # Fase desde ms -> grados
    phase = (t / 16.6667) * 360.0
    phase = wrap360(phase)
    # Quantity (opcional): si existe, alinear a longitud n
    qty = None
    if quantity_text:
        try:
            parts = (quantity_text or "").replace("\n", " ").replace("\t", " ").split(" ")
            vals = [float(p) for p in parts if p]
            if len(vals) >= n:
                qty = np.asarray(vals[:n], dtype=float)
            elif len(vals) > 0:
                import math as _m
                rep = int(_m.ceil(n / len(vals)))
                qty = np.asarray((vals * rep)[:n], dtype=float)
        except Exception:
            qty = None
    pixel_vals = np.asarray(y, dtype=float) if pixel_text is not None else None
    return phase, amp, qty, pixel_vals


def load_xml_prpd(path: Path) -> dict:
    tree = ET.parse(str(path))
    root = tree.getroot()
    # 1) Intentar formato genérico por pares phase/amplitude
    pts = _parse_points_xml_auto(root)
    pixel_vals = None
    if not pts:
        # 2) Fallback: formato Megger de listas (EntrenamientoPatron)
        phase, amp, qty, pixel_vals = _parse_megger_lists(root)
        if phase is None or amp is None:
            raise ValueError("XML sin puntos detectables (phase/amplitude).")
        ph = np.asarray(phase, dtype=float)
        am = np.asarray(amp, dtype=float)
        qn = None if qty is None else np.asarray(qty, dtype=float)
    else:
        ph = np.asarray([p[0] for p in pts], dtype=float)
        am = np.asarray([p[1] for p in pts], dtype=float)
        qn = None  # no quantity en este modo
    amp_norm = robust_scale(am)
    out = {"phase_deg": wrap360(ph), "amplitude": am, "amp_norm": amp_norm, "pixel_raw": pixel_vals}
    if qn is not None:
        out["quantity"] = qn
    return out


def load_prpd(path: Path) -> dict:
    """Carga PRPD desde CSV o XML, devolviendo dict con 'phase_deg','amplitude','amp_norm'."""
    p = Path(path)
    ext = (p.suffix or '').lower()
    if ext == ".csv":
        return load_csv_prpd(p)
    if ext == ".xml":
        return load_xml_prpd(p)
    # intento heurístico: si no hay extensión o es desconocida, probar CSV y luego XML
    try:
        return load_csv_prpd(p)
    except Exception:
        return load_xml_prpd(p)

def wrap360(x):
    y = np.mod(x, 360.0)
    y[y<0]+=360.0
    return y

def robust_scale(a):
    m = np.median(a)
    mad = np.median(np.abs(a-m)) + 1e-9
    return (a-m)/(1.4826*mad)

# ---------- ruido ----------
def grid_noise_gate(phase_deg, amp_norm, bins_phase=64, bins_amp=48, min_count=3, ratio_thresh=0.9):
    H,_,_ = np.histogram2d(phase_deg, amp_norm, bins=[bins_phase, bins_amp], range=[[0,360],[-4,4]])
    # celdas ocupadas y fuertes
    occupied = (H>0).sum()
    strong   = (H>=min_count).sum()
    if occupied == 0:
        return True, {"occupied":0,"strong":0,"ratio":0.0}
    ratio = strong/max(occupied,1)
    # heurística: si la mayoría de celdas ocupadas ya son fuertes → poco ruido
    has_noise = not (ratio >= ratio_thresh)
    return has_noise, {"occupied":int(occupied), "strong":int(strong), "ratio":float(ratio)}

# ---------- fase ----------
def choose_phase_offset(phase_deg, amp_norm, candidates=(0,120,240)):
    # elegir offset que maximiza concentración vectorial después de rotar
    best_off, best_R = 0, -1
    ang = np.deg2rad(phase_deg)
    for off in candidates:
        th = np.deg2rad((phase_deg + off) % 360.0)
        # pese amplitud (limitado)
        w = np.clip(np.abs(amp_norm), 0, 3) + 0.5
        R = np.abs(np.sum(w*np.exp(1j*th)))/np.sum(w)
        if R > best_R:
            best_R, best_off = R, off
    return int(best_off), float(best_R)

def apply_phase_offset(phase_deg, off):
    return wrap360(phase_deg + off)

# ---------- clustering + prune ----------
def cluster_and_prune(phase_deg, amp_norm, eps=0.065, min_samples=20, min_share=0.07,
                      force_multi=False, fallback_eps=None, keep_all=False):
    # espacio normalizado (fase a [0,1], amp_norm a [-1,1] aprox)
    X = np.column_stack([(phase_deg/360.0), np.tanh(amp_norm/2.0)])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    mask = labels>=0
    if force_multi and len(set(labels[mask])) <= 1 and fallback_eps:
        for alt_eps in fallback_eps:
            if alt_eps >= eps:
                continue
            db = DBSCAN(eps=alt_eps, min_samples=min_samples).fit(X)
            labels = db.labels_
            mask = labels>=0
            if len(set(labels[mask])) > 1:
                break
    if force_multi and len(set(labels[mask])) <= 1 and mask.sum() >= 2:
        base = labels.max() + 1 if labels.size else 0
        X_keep = X[mask]
        try:
            kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X_keep)
            labels_arr = labels.copy()
            labels_arr[mask] = kmeans.labels_ + base
            labels = labels_arr
            mask = labels>=0
        except Exception:
            pass
    # prune por participación mínima
    keep = np.zeros_like(labels, dtype=bool)
    n = len(labels)
    if keep_all:
        keep = mask.copy()
    else:
        for k in set(labels[mask]):
            idx = (labels == k)
            if idx.sum()/n >= min_share:
                keep |= idx
    return labels, keep

# ---------- features ----------
def compute_features(phase_deg, amp, amp_norm, labels, keep_mask):
    # considera solo kept
    idx = keep_mask & (labels>=0)
    ph = phase_deg[idx]; a = amp[idx]; an = amp_norm[idx]
    out = {
        "count": int(idx.sum()),
        "p95_amp": float(np.percentile(np.abs(a), 95)) if idx.any() else 0.0,
        "dens": float(idx.mean()),
        "R_phase": float(phase_concentration(ph)),
        "polarity_balance": float(polarity_balance(a)),
    }
    return out

def phase_concentration(phase_deg):
    if len(phase_deg)==0: return 0.0
    th = np.deg2rad(phase_deg)
    R = np.abs(np.mean(np.exp(1j*th)))
    return R  # 0..1

def polarity_balance(a):
    if len(a)==0: return 0.0
    pos = np.mean(a>=0); neg = 1-pos
    return 1.0 - abs(pos-neg)  # 1 = balanceado

# ---------- heurística de clase ----------
def heuristic_class(features):
    # Reglas simples (placeholder si no hay ANN entrenada)
    p95 = features["p95_amp"]
    R   = features["R_phase"]
    bal = features["polarity_balance"]
    # umbrales rápidos
    if p95>1.5 and R>0.65 and bal<0.6:
        cls = "corona"
    elif p95>1.0 and R>0.55 and bal>=0.6:
        cls = "cavidad"
    elif p95<=1.0 and R>0.50:
        cls = "superficial"
    else:
        cls = "flotante"
    # probs aproximadas
    probs = {"cavidad":0.0,"superficial":0.0,"corona":0.0,"flotante":0.0}
    probs[cls] = 0.82
    # repartir resto
    for k in probs:
        if k!=cls:
            probs[k] = 0.06
    return cls, probs

# ---------- severidad ----------
def severity_score(features):
    # mezcla simple (0..100) para wow-factor inicial
    p95 = min(features.get("p95_amp", 0.0)/2.5, 1.0)
    dens = min(features.get("dens", 0.0)/0.35, 1.0)
    R    = float(features.get("R_phase", 0.0))  # 0..1
    s = 60*p95 + 25*dens + 15*R
    return float(np.clip(s, 0, 100))


def severity_with_breakdown(features):
    """Calcula severidad y devuelve desglose de contribuciones.

    - p95_amp (escala relativa)
    - dens (fraccion de puntos kept)
    - R_phase (concentracion vectorial; tambien se informa std circular)
    """
    p95_abs = float(features.get("p95_amp", 0.0))
    dens_abs = float(features.get("dens", 0.0))
    R = float(features.get("R_phase", 0.0))
    # componentes normalizadas
    p95 = min(p95_abs/2.5, 1.0)
    dens = min(dens_abs/0.35, 1.0)
    std_circ_deg = 180.0
    try:
        if R > 1e-12:
            std_circ_deg = float(np.degrees(np.sqrt(-2.0*np.log(R))))
    except Exception:
        pass
    score = float(np.clip(60*p95 + 25*dens + 15*R, 0, 100))
    breakdown = {
        'p95_amp': p95_abs,
        'dens': dens_abs,
        'R_phase': R,
        'std_circ_deg': std_circ_deg,
        'weights': {'p95': 0.60, 'dens': 0.25, 'R': 0.15},
    }
    return score, breakdown

# ---------- pipeline ----------
def _compute_angpd(phase_deg: np.ndarray, bins: int = 72, weights: np.ndarray | None = None) -> dict:
    """Calcula ANGPD y N-ANGPD en bins uniformes (0..360)."""
    try:
        phi = np.asarray(phase_deg, dtype=float) % 360.0
        w = None if weights is None else np.asarray(weights, dtype=float)
        H, edges = np.histogram(phi, bins=bins, range=(0.0, 360.0), weights=w)
        centers = 0.5 * (edges[:-1] + edges[1:])
        angpd = H.astype(float)
        s = float(angpd.sum())
        if s > 0:
            angpd = angpd / s
        n_angpd = H.astype(float)
        m = float(n_angpd.max())
        if m > 0:
            n_angpd = n_angpd / m
        return {"phi_centers": centers, "angpd": angpd, "n_angpd": n_angpd}
    except Exception:
        return {"phi_centers": np.zeros(0), "angpd": np.zeros(0), "n_angpd": np.zeros(0)}


def _quintile_index(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Igual que antes: quintiles por frecuencia (equal-frequency)."""
    return _ef_group_index(v, n_groups=5)


def _ef_group_index(v: np.ndarray, n_groups: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Asigna grupos equal-frequency 1..n_groups por ranking estable.

    Devuelve (idx, edges_percentiles). edges son percentiles inter-grupo
    solo informativos.
    """
    v = np.asarray(v, dtype=float)
    finite = np.isfinite(v)
    idx_all = np.zeros_like(v, dtype=int)
    if not finite.any():
        return idx_all, np.array([np.nan] * max(n_groups - 1, 0), dtype=float)
    vals = v[finite]
    m = vals.size
    # percentiles informativos
    if n_groups > 1:
        percs = np.linspace(100.0 / n_groups, 100.0 - 100.0 / n_groups, n_groups - 1)
        edges = np.percentile(vals, percs)
    else:
        edges = np.array([], dtype=float)
    # ranking estable y asignacin de grupos
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(m)
    grp = (ranks * n_groups) // max(m, 1)  # 0..n_groups-1
    grp = np.clip(grp, 0, n_groups - 1)
    idx = (grp + 1).astype(int)            # 1..n_groups
    idx_all[finite] = idx
    return idx_all, edges


def _normalize_phase_mask(phase_mask) -> list[tuple[float, float]]:
    """Normaliza la lista de intervalos de fase a conservar."""
    normalized: list[tuple[float, float]] = []
    if not phase_mask:
        return normalized
    for pair in phase_mask:
        if pair is None:
            continue
        try:
            start, end = pair
            start = float(start) % 360.0
            end = float(end) % 360.0
        except Exception:
            continue
        if abs(start - end) < 1e-9:
            continue
        normalized.append((start, end))
    return normalized


def _phase_mask_bool(phases: np.ndarray, mask_ranges: list[tuple[float, float]]) -> np.ndarray:
    """Devuelve una máscara booleana indicando los puntos dentro de los intervalos."""
    if not mask_ranges or phases.size == 0:
        return np.ones(phases.shape, dtype=bool)
    phi = np.mod(phases, 360.0)
    keep = np.zeros(phi.shape, dtype=bool)
    for start, end in mask_ranges:
        if start <= end:
            keep |= (phi >= start) & (phi <= end)
        else:
            keep |= (phi >= start) | (phi <= end)
    return keep


def _percentile_buckets(values: np.ndarray, percentiles: list[float]) -> tuple[np.ndarray | None, list[float]]:
    """Asigna buckets basados en percentiles acumulados."""
    if values is None:
        return None, []
    arr = np.asarray(values, dtype=float)
    idx_all = np.zeros(arr.shape, dtype=int)
    finite = np.isfinite(arr)
    if not finite.any():
        return idx_all, []
    edges = np.percentile(arr[finite], percentiles)
    idx = np.searchsorted(edges, arr[finite], side="right") + 1
    idx_all[finite] = idx.astype(int)
    return idx_all, [float(x) for x in edges]


def _normalize_deciles_keep(values) -> list[int] | None:
    """Normaliza la lista de deciles solicitados (1..10)."""
    if values is None:
        return None
    vals = list(values)
    normalized: set[int] = set()
    try:
        for v in vals:
            iv = int(v)
            if 1 <= iv <= 10:
                normalized.add(iv)
    except Exception:
        return None
    if not normalized:
        return [] if len(vals) == 0 else None
    ordered = sorted(normalized)
    if ordered == list(range(1, 11)):
        return None
    return ordered


def _compute_pixel_deciles(pixel_vals) -> tuple[np.ndarray | None, dict]:
    """Calcula deciles de magnitud para los valores de pixel (normalizados e invertidos)."""
    meta = {"available": False}
    if pixel_vals is None:
        return None, meta
    arr = np.asarray(pixel_vals, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return None, meta
    min_val = float(np.nanmin(arr))
    max_val = float(np.nanmax(arr))
    span = max(max_val - min_val, 1e-12)
    scaled = np.full_like(arr, fill_value=np.nan, dtype=float)
    finite = np.isfinite(arr)
    scaled[finite] = (arr[finite] - min_val) / span
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled = (1.0 - scaled) * 100.0
    scaled_min = float(np.nanmin(scaled[finite]))
    scaled_max = float(np.nanmax(scaled[finite]))
    span = max(scaled_max - scaled_min, 1e-12)
    deciles = np.zeros_like(scaled, dtype=int)
    idx = np.floor(((scaled[finite] - scaled_min) / span) * 10.0)
    idx = np.clip(idx, 0, 9)
    deciles[finite] = (idx + 1).astype(int)
    edges = np.linspace(scaled_min, scaled_max, 11)
    meta = {
        "available": True,
        "min_raw": min_val,
        "max_raw": max_val,
        "edges": [float(x) for x in edges],
    }
    return deciles, meta


def process_prpd(path: Path, out_root: Path, force_phase_offsets=None, fast_mode=False,
                 filter_level: str = "weak",
                 phase_mask: list[tuple[float, float]] | None = None,
                 pixel_deciles_keep: list[int] | None = None,
                 qty_deciles_keep: list[int] | None = None) -> dict:
    out_root = Path(out_root)
    # Asegurar subdirectorios de auditoría
    try:
        (out_root/"aligned").mkdir(parents=True, exist_ok=True)
        (out_root/"filtered").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    data = load_prpd(path)
    mask_ranges = _normalize_phase_mask(phase_mask)
    pixel_deciles_selected = _normalize_deciles_keep(pixel_deciles_keep)
    qty_deciles_selected = _normalize_deciles_keep(qty_deciles_keep)
    fl_text = (filter_level or "S1 Weak").strip()
    fl = fl_text.lower()
    fl_key = fl.split()[0] if fl else ""
    special_cfg = SN_FILTER_CONFIGS.get(fl_key, None)
    noise_cfg = SN_NOISE_OVERRIDES.get(fl_key, {})
    has_noise, noise_meta = grid_noise_gate(
        data["phase_deg"],
        data["amp_norm"],
        bins_phase=noise_cfg.get("bins_phase", 64),
        bins_amp=noise_cfg.get("bins_amp", 48),
        min_count=noise_cfg.get("min_count", 3),
        ratio_thresh=noise_cfg.get("ratio_thresh", 0.90),
    )

    # fase: auto en {0,120,240} o forzada
    cand = (0,120,240) if force_phase_offsets is None else tuple(force_phase_offsets)
    off, R = choose_phase_offset(data["phase_deg"], data["amp_norm"], cand)
    ph_al = apply_phase_offset(data["phase_deg"], off)

    if special_cfg:
        fl = fl_key
    s5_min_frac = float(special_cfg.get("s5_min_frac", 0.05)) if special_cfg else 0.05
    s3_eps = float(special_cfg.get("s3_eps", 0.045)) if special_cfg else 0.045
    s3_min_samples = int(special_cfg.get("s3_min_samples", 10)) if special_cfg else 10
    s3_force_multi = bool(special_cfg.get("s3_force_multi", False)) if special_cfg else False
    amp0 = np.asarray(data.get("amplitude"), dtype=float)
    qty0 = data.get("quantity", None)
    qty_dec_all = None
    qty_quint_all = None
    qty_dec_meta = {"available": False}
    qty_quint_meta = {"available": False}
    if qty0 is not None:
        qty0 = np.asarray(qty0, dtype=float)
        qty_dec_all, _ = _ef_group_index(qty0, n_groups=10)
        qty_quint_all, _ = _ef_group_index(qty0, n_groups=5)
        finite_qty = np.isfinite(qty0)
        if finite_qty.any():
            dec_edges = np.percentile(qty0[finite_qty], np.linspace(10, 90, 9))
            quint_edges = np.percentile(qty0[finite_qty], [20, 40, 60, 80])
        else:
            dec_edges = []
            quint_edges = []
        qty_dec_meta = {
            "available": True,
            "edges": dec_edges if isinstance(dec_edges, list) else dec_edges.tolist(),
        }
        qty_quint_meta = {
            "available": True,
            "edges": quint_edges if isinstance(quint_edges, list) else quint_edges.tolist(),
        }
    pixel0 = data.get("pixel_raw", None)
    if pixel0 is not None:
        pixel0 = np.asarray(pixel0, dtype=float)
    pixel_dec_all, pixel_meta = _compute_pixel_deciles(pixel0)
    n0 = amp0.size
    gate_mask = np.ones(n0, dtype=bool)
    if qty_deciles_selected is not None:
        if len(qty_deciles_selected) == 0:
            gate_mask[:] = False
        elif qty_dec_all is not None and qty_dec_all.size == gate_mask.size:
            try:
                q_mask = np.isin(qty_dec_all, qty_deciles_selected)
                gate_mask &= q_mask
            except Exception:
                pass
    if pixel_deciles_selected is not None:
        if len(pixel_deciles_selected) == 0:
            gate_mask[:] = False
        elif pixel_dec_all is not None and pixel_dec_all.size == gate_mask.size:
            try:
                dec_mask = np.isin(pixel_dec_all, pixel_deciles_selected)
                gate_mask &= dec_mask
            except Exception:
                pass
    ph_g = ph_al[gate_mask]
    amp_g = amp0[gate_mask]
    qty_quint_g = None
    if qty_quint_all is not None and qty_quint_all.size == gate_mask.size:
        qty_quint_g = qty_quint_all[gate_mask]
    qty_g = None if qty0 is None else qty0[gate_mask]
    qty_dec_g = None
    if qty_dec_all is not None and qty_dec_all.size == gate_mask.size:
        qty_dec_g = qty_dec_all[gate_mask]
    pixel_g = None
    if pixel0 is not None:
        pixel_g = pixel0[gate_mask]
    amp_norm_g = robust_scale(amp_g) if amp_g.size else np.zeros(0, dtype=float)

    # clustering (si hay “mucho ruido”, DBSCAN más estricto)
    # clustering / filtrado seg�n nivel S1 (weak) o S2 (strong)
    # (par�metros iniciales ser�n sobreescritos abajo)
    # labels, keep se recalcula tras ajustar eps/min_samples/min_share
    if special_cfg:
        eps = special_cfg["eps"]
        min_samples = special_cfg["min_samples"]
        min_share = special_cfg["min_share"]
    elif fl.startswith("s2") or fl.startswith("strong"):
        eps = 0.055 if has_noise else 0.07
        min_samples = 22 if has_noise else 16
        min_share = 0.08
    else:
        eps = 0.07 if has_noise else 0.09
        min_samples = 16 if has_noise else 12
        min_share = 0.06

    # Ejecutar clustering sobre eventos ya filtrados por gating
    force_multi = bool(special_cfg and special_cfg.get("force_multi"))
    fallback_eps = special_cfg.get("fallback_eps") if special_cfg else None
    keep_all = bool(special_cfg and special_cfg.get("keep_all", False))
    if ph_g.size:
        labels, keep = cluster_and_prune(
            ph_g, amp_norm_g,
            eps=eps, min_samples=min_samples, min_share=min_share,
            force_multi=force_multi,
            fallback_eps=fallback_eps,
            keep_all=keep_all,
        )
    else:
        labels = np.zeros(0, dtype=int)
        keep = np.zeros(0, dtype=bool)
    if mask_ranges and ph_g.size and keep.size:
        try:
            inside = _phase_mask_bool(ph_g, mask_ranges)
            if inside.shape == keep.shape:
                keep = keep & inside
            else:
                keep = keep & inside.astype(bool)
        except Exception:
            pass
    # features
    feats = compute_features(ph_g, amp_g, amp_norm_g, labels, keep)
    cls, probs = heuristic_class(feats)
    sev, sev_bd = severity_with_breakdown(feats)

    # recolecta datos alineados/filtrados (+ etiquetas)
    kept_idx = keep & (labels>=0)
    aligned = {"phase_deg": ph_g[kept_idx], "amplitude": amp_g[kept_idx]}
    if qty_g is not None:
        aligned["quantity"] = qty_g[kept_idx]  # S5 hereda quantity para todos los histogramas
    if qty_dec_g is not None:
        aligned["qty_deciles"] = qty_dec_g[kept_idx]
    if qty_quint_g is not None:
        aligned["qty_quintiles"] = qty_quint_g[kept_idx]
    if pixel_g is not None:
        aligned["pixel"] = pixel_g[kept_idx]
    labels_aligned = labels[kept_idx]
    raw = {"phase_deg": data["phase_deg"], "amplitude": data["amplitude"]}
    if qty0 is not None:
        raw["quantity"] = np.asarray(qty0, dtype=float)

    # ANGPD (|amplitud|) y N-ANGPD + variante por quantity si existe
    try:
        weights_amp = np.abs(aligned.get("amplitude", np.zeros(0)))
        weights_amp = weights_amp if weights_amp.size else None
        angpd = _compute_angpd(aligned.get("phase_deg", np.zeros(0)), bins=72, weights=weights_amp)
        # Variante por repeticin (quantity) si existe
        if "quantity" in aligned:
            wq = np.asarray(aligned["quantity"], dtype=float)
            wq = wq if wq.size else None
            ang_q = _compute_angpd(aligned.get("phase_deg", np.zeros(0)), bins=72, weights=wq)
            # Insertar curvas adicionales sin romper claves existentes
            angpd["angpd_qty"] = ang_q.get("angpd", np.zeros(0))
            angpd["n_angpd_qty"] = ang_q.get("n_angpd", np.zeros(0))
        else:
            angpd["angpd_qty"] = np.zeros_like(angpd.get("angpd", np.zeros(0)))
            angpd["n_angpd_qty"] = np.zeros_like(angpd.get("n_angpd", np.zeros(0)))
    except Exception:
        angpd = {"phi_centers": np.zeros(0), "angpd": np.zeros(0), "n_angpd": np.zeros(0), "angpd_qty": np.zeros(0), "n_angpd_qty": np.zeros(0)}

    # guardar auditoría mínima
    run_id = path.stem
    np.savez(out_root/"aligned"/f"{run_id}_aligned.npz", phase_deg=ph_al, amp=data["amplitude"])
    np.savez(out_root/"filtered"/f"{run_id}_kept.npz", phase_deg=aligned.get("phase_deg", np.zeros(0)), amp=aligned.get("amplitude", np.zeros(0)))

    if pixel_deciles_selected is None:
        selected_meta = list(range(1, 11))
    else:
        selected_meta = list(pixel_deciles_selected)
    pixel_meta["selected"] = selected_meta
    if qty_deciles_selected is None:
        qty_selected_meta = list(range(1, 11))
    else:
        qty_selected_meta = list(qty_deciles_selected)
    qty_dec_meta["selected"] = qty_selected_meta

    return {
        "raw": raw,
        "aligned": aligned,
        "labels": labels,                 # etiquetas DBSCAN para todos los puntos (incl. ruido)
        "keep_mask": keep,                # máscara de puntos conservados en pruning
        "labels_aligned": labels_aligned, # etiquetas sólo para puntos alineados/filtrados (>=0)
        "phase_offset": off,
        "phase_vector_R": R,
        "has_noise": bool(has_noise),
        "noise_meta": noise_meta,
        "n_clusters": int(len(set(labels[labels>=0]))),
        "features": feats,
        "predicted": cls,
        "probs": probs,
        "severity_score": sev,
        "severity_breakdown": sev_bd,
        "angpd": angpd,
        "s5_min_frac": s5_min_frac,
        "s3_eps": s3_eps,
        "s3_min_samples": s3_min_samples,
        "s3_force_multi": s3_force_multi,
        "run_id": run_id,
        "source_path": str(path),
        "phase_mask_ranges": mask_ranges,
        "pixel_deciles_meta": pixel_meta,
        "qty_deciles_meta": qty_dec_meta,
        "qty_quintiles_meta": qty_quint_meta,
    }





