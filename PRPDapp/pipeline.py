from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from PRPDapp import prpd_core as core
from PRPDapp.conclusion_rules import build_conclusion_block
from PRPDapp.logic import classify_pd, compute_pd_metrics
from PRPDapp.metrics.advanced_kpi import compute_advanced_metrics
from PRPDapp.ang_proj import compute_ang_proj, compute_ang_proj_kpis
from PRPDapp.pd_rules import build_rule_features, infer_pd_summary, rule_based_scores


ANN_CLASSES = ["corona", "superficial", "cavidad", "flotante", "ruido"]
ANN_DISPLAY = {
    "corona": {"label": "Corona", "color": "#5a6c80"},
    "superficial": {"label": "Superficial / Tracking", "color": "#1f77b4"},
    "cavidad": {"label": "Cavidad", "color": "#8e44ad"},
    "flotante": {"label": "Flotante", "color": "#ff9800"},
    "ruido": {"label": "Ruido", "color": "#7f7f7f"},
}


def compute_all(state: dict[str, Any], raw_data: dict[str, Any]) -> dict:
    """Compute pipeline result from raw PRPD data and UI state."""
    state = state or {}
    path = Path(state.get("path") or "run")
    out_root = Path(state.get("out_root") or "out")
    force_offsets = state.get("force_phase_offsets")
    filter_level = state.get("filter_level") or "S1 Weak"
    mask_ranges = core._normalize_phase_mask(state.get("phase_mask") or [])

    force_offsets_run = force_offsets
    mask_offset = None
    if mask_ranges:
        mask_offset = _resolve_mask_offset(raw_data, force_offsets)
        if mask_offset is not None and (not force_offsets or len(force_offsets) != 1):
            force_offsets_run = [int(mask_offset) % 360]
    masked_raw, mask_applied = _apply_phase_mask(raw_data, mask_ranges, phase_offset=mask_offset)

    result = core.process_prpd(
        path=path,
        out_root=out_root,
        force_phase_offsets=force_offsets_run,
        fast_mode=False,
        filter_level=filter_level,
        phase_mask=mask_ranges,
        pixel_deciles_keep=None,
        qty_deciles_keep=None,
        raw_data=raw_data,
    )
    result["filter_level"] = filter_level

    if mask_applied:
        raw_view = {
            "phase_deg": np.asarray(masked_raw.get("phase_deg", []), dtype=float),
            "amplitude": np.asarray(masked_raw.get("amplitude", []), dtype=float),
        }
        if "quantity" in masked_raw:
            raw_view["quantity"] = np.asarray(masked_raw.get("quantity", []), dtype=float)
        result["raw"] = raw_view

    result["phase_mask_ranges"] = mask_ranges
    result["mask_applied"] = bool(mask_applied)
    if mask_offset is not None:
        result["mask_offset"] = float(mask_offset)
    asset_type = state.get("asset_type")
    if asset_type:
        result["asset_type"] = asset_type

    gap_stats = state.get("gap_stats") or {}
    if gap_stats:
        result["gap_stats"] = gap_stats
    gap_stats_raw = state.get("gap_stats_raw") or {}
    if gap_stats_raw:
        result["gap_stats_raw"] = gap_stats_raw
    gap_stats_total = state.get("gap_stats_total") or {}
    if gap_stats_total:
        result["gap_stats_total"] = gap_stats_total

    _precompute_qty_buckets(result)
    refresh_view_filters(result, state)

    return result


def refresh_view_filters(result: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Apply view filters (pixel/qty) without recomputing alignment/clustering."""
    if not isinstance(result, dict):
        return result
    state = state or {}

    pixel_keep = state.get("pixel_deciles_keep")
    if pixel_keep is None:
        pixel_keep = list(range(1, 11))
    qty_quints_keep = state.get("qty_quints_keep")
    if qty_quints_keep is None:
        qty_quints_keep = list(range(1, 6))
    qty_deciles_keep = state.get("qty_deciles_keep")
    if qty_deciles_keep is None:
        qty_deciles_keep = list(range(1, 11))

    _apply_view_filters(
        result,
        pixel_deciles_keep=pixel_keep,
        qty_quints_keep=qty_quints_keep,
        qty_deciles_keep=qty_deciles_keep,
    )
    _compute_derived_blocks(result, state)
    return result


def _align_full_arrays(aligned: dict[str, Any]) -> None:
    """Store aligned arrays as the stable baseline for view filters."""
    phase = np.asarray(aligned.get("phase_deg", []), dtype=float)
    amp = np.asarray(aligned.get("amplitude", []), dtype=float)
    qty = np.asarray(aligned.get("quantity", []), dtype=float)
    quint = np.asarray(aligned.get("qty_quintiles", []), dtype=int)
    dec = np.asarray(aligned.get("qty_deciles", []), dtype=int)
    pixel = np.asarray(aligned.get("pixel", []), dtype=float)
    labels = np.asarray(aligned.get("labels_aligned", []))

    lengths = [arr.size for arr in (phase, amp, qty, quint, dec) if arr.size]
    n = min(lengths) if lengths else 0
    if n <= 0:
        return

    aligned["phase_deg"] = phase[:n]
    aligned["amplitude"] = amp[:n]
    aligned["quantity"] = qty[:n]
    aligned["qty_quintiles"] = quint[:n]
    aligned["qty_deciles"] = dec[:n]

    aligned["_full_phase_deg"] = aligned["phase_deg"]
    aligned["_full_amplitude"] = aligned["amplitude"]
    aligned["_full_quantity"] = aligned["quantity"]
    aligned["_full_qty_quintiles"] = aligned["qty_quintiles"]
    aligned["_full_qty_deciles"] = aligned["qty_deciles"]
    if pixel.size:
        aligned["_full_pixel"] = pixel[:n]
    if labels.size:
        aligned["_full_labels_aligned"] = labels[:n]


def _precompute_qty_buckets(result: dict[str, Any]) -> None:
    aligned = result.get("aligned", {}) or {}
    qty_vals = np.asarray(aligned.get("quantity", []), dtype=float)
    phase = np.asarray(aligned.get("phase_deg", []), dtype=float)
    amp = np.asarray(aligned.get("amplitude", []), dtype=float)
    if qty_vals.size == 0 or phase.size == 0 or amp.size == 0:
        return

    finite = np.isfinite(qty_vals)
    if not np.any(finite):
        return

    edges_q = np.percentile(qty_vals[finite], [20, 40, 60, 80])
    edges_d = np.percentile(qty_vals[finite], [10, 20, 30, 40, 50, 60, 70, 80, 90])
    quint_idx = np.zeros_like(qty_vals, dtype=int)
    dec_idx = np.zeros_like(qty_vals, dtype=int)
    quint_idx[finite] = np.digitize(qty_vals[finite], edges_q, right=True) + 1
    dec_idx[finite] = np.digitize(qty_vals[finite], edges_d, right=True) + 1
    quint_idx[finite] = np.clip(quint_idx[finite], 1, 5)
    dec_idx[finite] = np.clip(dec_idx[finite], 1, 10)

    aligned["qty_quintiles"] = quint_idx
    aligned["qty_deciles"] = dec_idx
    result["qty_quintiles_meta"] = {"edges": edges_q.tolist()}
    result["qty_deciles_meta"] = {"edges": edges_d.tolist()}
    _align_full_arrays(aligned)


def _equal_frequency_bucket(values: np.ndarray, groups: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    idx_all = np.zeros(arr.shape, dtype=int)
    if groups <= 0:
        return idx_all
    finite = np.isfinite(arr)
    if not finite.any():
        return idx_all
    vals = arr[finite]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(vals.size)
    grp = (ranks * groups) // max(vals.size, 1)
    grp = np.clip(grp, 0, groups - 1)
    idx = (grp + 1).astype(int)
    idx_all[finite] = idx
    return idx_all


def _build_qty_keep_mask(
    quint_full: np.ndarray,
    dec_full: np.ndarray,
    keep_quints: list[int],
    keep_deciles: list[int],
) -> np.ndarray:
    n = int(quint_full.size)
    if n <= 0:
        return np.zeros(0, dtype=bool)

    keep_q_set = {int(q) for q in keep_quints if 1 <= int(q) <= 5}
    keep_sub_set = {int(k) for k in keep_deciles if 1 <= int(k) <= 10}
    pairs = {1: {1, 2}, 2: {3, 4}, 3: {5, 6}, 4: {7, 8}, 5: {9, 10}}

    all_quints = keep_q_set.issuperset({1, 2, 3, 4, 5})
    all_deciles = (not keep_sub_set) or keep_sub_set == set(range(1, 11))
    if all_quints and all_deciles:
        return np.ones(n, dtype=bool)
    if not keep_q_set:
        return np.zeros(n, dtype=bool)

    mask_keep = np.zeros(n, dtype=bool)
    for q in range(1, 6):
        if q not in keep_q_set:
            continue
        subs = pairs[q]
        subs_on = {s for s in subs if s in keep_sub_set} if keep_sub_set else subs
        if dec_full.size == 0:
            mask_keep |= (quint_full == q)
        else:
            if not subs_on:
                continue
            mask_keep |= (quint_full == q) & np.isin(dec_full, list(subs_on))
    return mask_keep


def _apply_view_filters(
    result: dict[str, Any],
    *,
    pixel_deciles_keep: list[int],
    qty_quints_keep: list[int],
    qty_deciles_keep: list[int],
) -> None:
    aligned = result.get("aligned", {}) or {}
    phase_full = np.asarray(aligned.get("_full_phase_deg", aligned.get("phase_deg", [])), dtype=float)
    amp_full = np.asarray(aligned.get("_full_amplitude", aligned.get("amplitude", [])), dtype=float)
    qty_full = np.asarray(aligned.get("_full_quantity", aligned.get("quantity", [])), dtype=float)
    dec_full = np.asarray(aligned.get("_full_qty_deciles", aligned.get("qty_deciles", [])), dtype=int)
    quint_full = np.asarray(aligned.get("_full_qty_quintiles", aligned.get("qty_quintiles", [])), dtype=int)
    pixel_full = np.asarray(aligned.get("_full_pixel", aligned.get("pixel", [])), dtype=float)
    labels_full = np.asarray(aligned.get("_full_labels_aligned", aligned.get("labels_aligned", [])))

    n = int(phase_full.size)
    if n <= 0:
        return

    keep_mask = np.ones(n, dtype=bool)

    keep_set = {int(d) for d in pixel_deciles_keep if 1 <= int(d) <= 10}
    if not keep_set:
        keep_mask &= False
    elif keep_set != set(range(1, 11)):
        pix_dec = np.asarray(aligned.get("_full_pixel_deciles", []), dtype=int)
        if pix_dec.size != n:
            mag = np.clip(np.abs(amp_full), 0.0, 100.0)
            pix_dec = _equal_frequency_bucket(mag, groups=10)
            aligned["_full_pixel_deciles"] = pix_dec
        keep_mask &= np.isin(pix_dec, list(keep_set))

    if quint_full.size == n:
        keep_mask &= _build_qty_keep_mask(quint_full, dec_full, qty_quints_keep, qty_deciles_keep)

    aligned["phase_deg"] = phase_full[keep_mask]
    aligned["amplitude"] = amp_full[keep_mask]
    if qty_full.size == n:
        aligned["quantity"] = qty_full[keep_mask]
    if quint_full.size == n:
        aligned["qty_quintiles"] = quint_full[keep_mask]
    if dec_full.size == n:
        aligned["qty_deciles"] = dec_full[keep_mask]
    if pixel_full.size == n:
        aligned["pixel"] = pixel_full[keep_mask]
    if labels_full.size == n:
        aligned["labels_aligned"] = labels_full[keep_mask]


def _compute_derived_blocks(result: dict[str, Any], state: dict[str, Any]) -> None:
    gap_stats = state.get("gap_stats") if isinstance(state, dict) else None
    if gap_stats is None:
        gap_stats = result.get("gap_stats") or {}
    if gap_stats:
        result["gap_stats"] = gap_stats

    asset_type = state.get("asset_type") if isinstance(state, dict) else None
    if asset_type:
        result["asset_type"] = asset_type

    _recompute_angpd_blocks(result)

    metrics = compute_pd_metrics(result, gap_stats=gap_stats or None)
    result["metrics"] = metrics

    bins_phase = int(state.get("bins_phase") or 32)
    bins_amp = int(state.get("bins_amp") or 32)
    metrics_adv = compute_advanced_metrics(
        {"aligned": result.get("aligned", {}), "angpd": result.get("angpd", {})},
        bins_amp=bins_amp,
        bins_phase=bins_phase,
    )
    _normalize_adv_heuristics(metrics_adv)
    result["metrics_advanced"] = metrics_adv

    summary = classify_pd(metrics or {})
    result["summary"] = summary

    try:
        rule_features = build_rule_features(result)
        rule_probs = rule_based_scores(rule_features)
        result["rule_pd"] = infer_pd_summary(rule_features, rule_probs)
    except Exception:
        pass

    ann_block = _compute_ann_block(
        result,
        metrics=metrics,
        metrics_adv=metrics_adv,
        summary=summary,
        state=state,
    )
    result["ann"] = ann_block

    result["kpis"] = _merge_kpis(result, metrics, gap_stats or {})

    visual_extended = bool(state.get("visual_extended", False))
    conclusion_block = build_conclusion_block(result, summary, visual_extended=visual_extended)
    ann_display = ann_block.get("dominant_display")
    if ann_display:
        conclusion_block["dominant_discharge"] = ann_display
    result["conclusion_block"] = conclusion_block


def _recompute_angpd_blocks(result: dict[str, Any]) -> None:
    aligned = result.get("aligned", {}) or {}
    ph = np.asarray(aligned.get("phase_deg", []), dtype=float)
    amp = np.asarray(aligned.get("amplitude", []), dtype=float)
    qty = np.asarray(aligned.get("quantity", []), dtype=float)

    if ph.size and amp.size:
        try:
            weights_amp = np.abs(amp) if amp.size else None
            angpd = core._compute_angpd(ph, bins=72, weights=weights_amp)  # type: ignore[attr-defined]
            if qty.size:
                ang_q = core._compute_angpd(ph, bins=72, weights=qty)  # type: ignore[attr-defined]
                angpd["angpd_qty"] = ang_q.get("angpd", np.zeros(0))
                angpd["n_angpd_qty"] = ang_q.get("n_angpd", np.zeros(0))
            else:
                angpd["angpd_qty"] = np.zeros_like(angpd.get("angpd", np.zeros(0)))
                angpd["n_angpd_qty"] = np.zeros_like(angpd.get("n_angpd", np.zeros(0)))
            result["angpd"] = angpd
        except Exception:
            pass

    try:
        ang_proj = compute_ang_proj(aligned, n_phase_bins=32, n_amp_bins=16, n_points=64)
        ang_proj_kpis = compute_ang_proj_kpis(ang_proj)
        result["ang_proj"] = ang_proj
        result["ang_proj_kpis"] = ang_proj_kpis
    except Exception:
        pass

    try:
        fa_profile = core.compute_fa_profile(aligned, bin_width_deg=6.0, smooth_window_bins=5)
        fa_kpis = core.compute_fa_kpis(aligned, fa_profile)
        result["fa_profile"] = fa_profile
        result["fa_kpis"] = fa_kpis
    except Exception:
        pass


def _resolve_mask_offset(raw_data: dict[str, Any], force_offsets: list[int] | None) -> float | None:
    if force_offsets and len(force_offsets) == 1:
        try:
            return float(force_offsets[0]) % 360.0
        except Exception:
            return None

    phase = np.asarray(raw_data.get("phase_deg", []), dtype=float)
    if phase.size == 0:
        return None

    amp_norm = np.asarray(raw_data.get("amp_norm", []), dtype=float)
    if amp_norm.size != phase.size:
        amp_vals = np.asarray(raw_data.get("amplitude", []), dtype=float)
        amp_norm = core.robust_scale(amp_vals) if amp_vals.size else np.zeros(phase.shape, dtype=float)
    try:
        offset, _ = core.choose_phase_offset(phase, amp_norm, (0, 120, 240))
        return float(offset) % 360.0
    except Exception:
        return None


def _apply_phase_mask(
    raw_data: dict[str, Any],
    mask_ranges: list[tuple[float, float]],
    *,
    phase_offset: float | None = None,
) -> tuple[dict[str, Any], bool]:
    if not mask_ranges:
        return raw_data, False
    phase = np.asarray(raw_data.get("phase_deg", []), dtype=float)
    if phase.size == 0:
        return raw_data, False
    if phase_offset is not None:
        phase_eval = core.apply_phase_offset(phase, phase_offset)
    else:
        phase_eval = phase
    mask = core._phase_mask_bool(phase_eval, mask_ranges)
    if mask.size == 0:
        return raw_data, False

    out: dict[str, Any] = {}
    for key, value in raw_data.items():
        try:
            arr = np.asarray(value)
        except Exception:
            out[key] = value
            continue
        if arr.ndim >= 1 and arr.shape[0] == mask.shape[0]:
            out[key] = arr[mask]
        else:
            out[key] = value

    try:
        amp_vals = np.asarray(out.get("amplitude", []), dtype=float)
        out["amp_norm"] = core.robust_scale(amp_vals) if amp_vals.size else np.zeros(0, dtype=float)
    except Exception:
        out["amp_norm"] = np.zeros(0, dtype=float)
    return out, True


def _compute_ann_block(
    result: dict[str, Any],
    *,
    metrics: dict[str, Any],
    metrics_adv: dict[str, Any],
    summary: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    features = _build_ann_features_from_result(result)
    raw_probs, source = _get_ann_probs(state, features, metrics_adv, result)
    raw_probs = _apply_manual_override(raw_probs, state.get("manual_override"))

    heuristic_top = _pick_heuristic_top(result, metrics_adv, summary)
    manual = state.get("manual_override") or {}
    mask_label = str(state.get("mask_label") or "").lower()
    force_corona = ("corona" in mask_label) and (not manual.get("ann_class"))
    if force_corona:
        canonical = {cls: 1.0 if cls == "corona" else 0.0 for cls in ANN_CLASSES}
        invalid = False
        source = "mask"
    else:
        canonical, invalid = _filter_ann_probs(raw_probs)
        if invalid:
            canonical = {cls: 1.0 if cls == heuristic_top else 0.0 for cls in ANN_CLASSES}

    top_classes, mixed, dominant_display, decimals = _ann_display_meta(canonical)
    display = _build_ann_display(canonical, dominant_display, top_classes, mixed, decimals)

    return {
        "source": source,
        "model_path": state.get("ann_model_path"),
        "classes": ANN_CLASSES[:],
        "probs": canonical,
        "probs_raw": raw_probs,
        "valid": not invalid,
        "dominant": top_classes[0] if top_classes else heuristic_top,
        "dominant_display": dominant_display,
        "mixed": mixed,
        "top_classes": top_classes,
        "display": display,
        "features": features,
        "heuristic_top": heuristic_top,
    }


def _build_ann_features_from_result(result: dict[str, Any]) -> dict[str, float]:
    aligned = result.get("aligned", {}) or {}
    raw = result.get("raw", {}) or {}
    ph = np.asarray(aligned.get("phase_deg", []), dtype=float)
    amp = np.asarray(aligned.get("amplitude", []), dtype=float)
    amp_abs = np.abs(amp)

    raw_ph = raw.get("phase_deg", [])
    try:
        raw_n = int(len(raw_ph))
    except Exception:
        raw_n = 0
    density = float(ph.size) / float(raw_n) if raw_n > 0 else 0.0

    phase_std_deg = 180.0
    if ph.size:
        try:
            th = np.deg2rad(np.mod(ph, 360.0))
            c = float(np.mean(np.cos(th)))
            s = float(np.mean(np.sin(th)))
            r_val = float(np.hypot(c, s))
            if r_val > 0:
                phase_std_deg = float(np.degrees(np.sqrt(max(0.0, -2.0 * np.log(max(r_val, 1e-12))))))
        except Exception:
            phase_std_deg = 180.0

    return {
        "amp_mean": float(np.mean(amp_abs)) if amp_abs.size else 0.0,
        "amp_std": float(np.std(amp_abs)) if amp_abs.size else 0.0,
        "amp_p95": float(np.percentile(amp_abs, 95)) if amp_abs.size else 0.0,
        "density": float(density),
        "phase_std_deg": float(phase_std_deg),
        "phase_entropy": 0.0,
        "rep_rate": 0.0,
        "rep_entropy": 0.0,
        "cluster_compactness": 0.0,
        "cluster_separation": 0.0,
        "lobes_count": 0.0,
        "area_ratio": 0.0,
    }


def _get_ann_probs(
    state: dict[str, Any],
    features: dict[str, float],
    metrics_adv: dict[str, Any],
    result: dict[str, Any],
) -> tuple[dict[str, float], str]:
    ann_model = state.get("ann_model")
    ann_predict: Callable[[Any, dict[str, float]], dict[str, float]] | None = state.get("ann_predict_proba")
    ann_fallback = state.get("ann_fallback")

    if ann_predict is not None and ann_model is not None:
        try:
            probs = ann_predict(ann_model, features)
            if isinstance(probs, dict):
                return probs, "model"
        except Exception:
            pass

    if ann_fallback is not None and getattr(ann_fallback, "is_loaded", False):
        try:
            probs = ann_fallback.predict_proba(features)
            if isinstance(probs, dict):
                return probs, "model"
        except Exception:
            pass

    heur = metrics_adv.get("heuristic_probs") if isinstance(metrics_adv, dict) else None
    if isinstance(heur, dict) and heur:
        return {str(k): float(v) for k, v in heur.items() if v is not None}, "heuristic_kpi"

    core_probs = result.get("probs") if isinstance(result, dict) else None
    if isinstance(core_probs, dict) and core_probs:
        return {str(k): float(v) for k, v in core_probs.items() if v is not None}, "heuristic_core"

    return {}, "none"


def _apply_manual_override(raw_probs: dict[str, float], manual: dict[str, Any] | None) -> dict[str, float]:
    if not manual or not manual.get("ann_class"):
        return raw_probs
    forced = _map_ann_class(str(manual.get("ann_class")))
    if not forced:
        return raw_probs
    try:
        bias = float(manual.get("ann_bias") or 1.2)
    except Exception:
        bias = 1.2
    bias = max(0.0, bias)
    updated = dict(raw_probs or {})
    updated[forced] = float(updated.get(forced, 0.0)) + bias
    return updated


def _map_ann_class(name: str | None) -> str | None:
    if not name:
        return None
    n = str(name).strip().lower()
    if not n:
        return None
    if "cav" in n or "void" in n or "intern" in n:
        return "cavidad"
    if "super" in n or "track" in n:
        return "superficial"
    if "corona" in n:
        return "corona"
    if "flot" in n:
        return "flotante"
    if "ruido" in n or "noise" in n or "indeterm" in n or "suspend" in n:
        return "ruido"
    return None


def _filter_ann_probs(raw_probs: dict[str, float]) -> tuple[dict[str, float], bool]:
    canonical = {cls: 0.0 for cls in ANN_CLASSES}
    mapped_keys: set[str] = set()
    invalid = False
    for key, val in (raw_probs or {}).items():
        cls = _map_ann_class(key)
        if not cls:
            continue
        mapped_keys.add(cls)
        try:
            canonical[cls] += float(val)
        except Exception:
            invalid = True

    values = np.asarray([canonical[c] for c in ANN_CLASSES], dtype=float)
    if not np.all(np.isfinite(values)):
        invalid = True
    total = float(np.sum(values))
    if total <= 0 or not np.isfinite(total):
        invalid = True

    if not invalid:
        canonical = {k: float(v) / total for k, v in canonical.items()}
        values = np.asarray([canonical[c] for c in ANN_CLASSES], dtype=float)
        order = np.argsort(values)[::-1]
        if order.size >= 3:
            top1 = float(values[order[0]])
            top3 = float(values[order[2]])
            if (top1 - top3) <= 0.02:
                invalid = True
    return canonical, invalid


def _pick_heuristic_top(
    result: dict[str, Any],
    metrics_adv: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    candidates = [
        metrics_adv.get("heuristic_top") if isinstance(metrics_adv, dict) else None,
        result.get("predicted") if isinstance(result, dict) else None,
        summary.get("pd_type") if isinstance(summary, dict) else None,
    ]
    for cand in candidates:
        mapped = _map_ann_class(cand)
        if mapped:
            return mapped
    return ANN_CLASSES[0]


def _ann_display_meta(probs: dict[str, float]) -> tuple[list[str], bool, str, int]:
    values = [float(probs.get(cls, 0.0)) for cls in ANN_CLASSES]
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    top1 = order[0] if order else 0
    top2 = order[1] if len(order) > 1 else top1
    p1 = values[top1] if values else 0.0
    p2 = values[top2] if values else 0.0
    mixed = (p1 - p2) <= 0.02
    decimals = 2 if (p1 - p2) < 0.01 else 1

    top_classes = [ANN_CLASSES[top1]]
    if mixed and top2 != top1:
        top_classes.append(ANN_CLASSES[top2])

    if mixed and len(top_classes) >= 2:
        a = ANN_DISPLAY.get(top_classes[0], {}).get("label", top_classes[0])
        b = ANN_DISPLAY.get(top_classes[1], {}).get("label", top_classes[1])
        dominant_display = f"Mixto: {a}/{b}"
    else:
        dominant_display = ANN_DISPLAY.get(top_classes[0], {}).get("label", top_classes[0])

    return top_classes, mixed, dominant_display, decimals


def _build_ann_display(
    probs: dict[str, float],
    dominant_display: str,
    top_classes: list[str],
    mixed: bool,
    decimals: int,
) -> dict[str, Any]:
    labels = [ANN_DISPLAY[c]["label"] for c in ANN_CLASSES]
    colors = [ANN_DISPLAY[c]["color"] for c in ANN_CLASSES]
    values = [float(probs.get(c, 0.0)) for c in ANN_CLASSES]
    top_indices = [ANN_CLASSES.index(c) for c in top_classes if c in ANN_CLASSES]
    return {
        "labels": labels,
        "colors": colors,
        "values": values,
        "decimals": int(decimals),
        "dominant_display": dominant_display,
        "mixed": mixed,
        "top_indices": top_indices,
    }


def _merge_kpis(result: dict[str, Any], metrics: dict[str, Any], gap_stats: dict[str, Any]) -> dict[str, Any]:
    kpis = dict(result.get("kpis") or {})

    n_ang_ratio = metrics.get("n_ang_ratio") if isinstance(metrics, dict) else None
    if n_ang_ratio is not None:
        kpis["n_angpd_angpd_ratio"] = n_ang_ratio

    gap_p50 = None
    gap_p5 = None
    if isinstance(gap_stats, dict) and gap_stats:
        gap_p50 = gap_stats.get("p50_ms")
        gap_p5 = gap_stats.get("p5_ms")
    if gap_p50 is None:
        gap_p50 = metrics.get("gap_p50") if isinstance(metrics, dict) else None
    if gap_p5 is None:
        gap_p5 = metrics.get("gap_p5") if isinstance(metrics, dict) else None
    if gap_p50 is not None:
        kpis["gap_p50_ms"] = gap_p50
    if gap_p5 is not None:
        kpis["gap_p5_ms"] = gap_p5

    return kpis


def _normalize_adv_heuristics(metrics_adv: dict[str, Any]) -> None:
    if not isinstance(metrics_adv, dict):
        return
    top = metrics_adv.get("heuristic_top")
    mapped = _map_ann_class(top)
    if mapped:
        metrics_adv["heuristic_top"] = mapped
    else:
        metrics_adv["heuristic_top"] = None
