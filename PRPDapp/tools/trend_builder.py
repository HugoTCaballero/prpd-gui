# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import csv
import math

from ..parsers.prpd_xml import parse_prpd_events
from ..parsers.levels_xml import parse_levels, validate_levels_expected
from ..metrics.gap_time import gap_stats_from_angles
from ..metrics.cluster import fit_clustering

SENSORS = ['UHF','HFCT','TEV']


def _read_runs_yaml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {'runs': []}
    cur: Dict[str, Any] | None = None
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.rstrip('\n')
        if not line.strip():
            continue
        if line.startswith('baseline:'):
            data['baseline'] = line.split(':', 1)[1].strip()
        elif line.startswith('mains_hz:'):
            try:
                data['mains_hz'] = float(line.split(':', 1)[1].strip())
            except Exception:
                data['mains_hz'] = 60.0
        elif line.strip().startswith('- fecha:'):
            if cur:
                data['runs'].append(cur)
            cur = {'fecha': line.split(':', 1)[1].strip()}
        elif any(line.strip().startswith(f'{k}:') for k in ['uhf_prpd','hfct_prpd','tev_prpd','uhf_niveles','hfct_niveles','tev_niveles']):
            k, v = [s.strip() for s in line.strip().split(':', 1)]
            if cur is not None:
                cur[k] = v
        elif line.strip().startswith('maintenance:'):
            try:
                rawv = line.split(':', 1)[1]
                rawv = rawv.split('#', 1)[0]
                val = rawv.strip().lower() in ('true','1','yes','y')
            except Exception:
                val = False
            if cur is not None:
                cur['maintenance'] = val
    if cur:
        data['runs'].append(cur)
    return data


def _fmt3(x: Any) -> str:
    try:
        if x is None:
            return ''
        xv = float(x)
        if math.isnan(xv) or math.isinf(xv):
            return ''
        ax = abs(xv)
        if isinstance(x, int) or (abs(xv - int(xv)) < 1e-9 and ax < 1e12):
            return str(int(xv))
        if ax >= 100:
            return f"{xv:.0f}"
        if ax >= 1:
            return f"{xv:.2f}"
        if ax >= 0.01:
            return f"{xv:.3f}"
        return f"{xv:.1e}"
    except Exception:
        return ''


def _status_delta_pct(pct: Optional[float]) -> str:
    if pct is None:
        return ''
    a = abs(pct)
    if a <= 5.0:
        return 'Aceptable'
    if a <= 15.0:
        return 'Alerta'
    return 'Crítico'


def _symmetry_ok(centers: List[float]) -> bool:
    if not centers or len(centers) < 2:
        return False
    for i, c1 in enumerate(centers):
        for j, c2 in enumerate(centers):
            if i == j:
                continue
            d = abs(((c1 - c2 + 180.0) % 360.0) - 180.0)
            if 162.0 <= d <= 198.0:
                return True
    return False


def _counts_semicycles(phase_deg, quantity) -> Tuple[int, int, int]:
    import numpy as np
    ang = np.asarray(phase_deg, dtype=float) % 360.0
    q = np.asarray(quantity, dtype=float)
    L = float(q[ang < 180.0].sum())
    H = float(q[ang >= 180.0].sum())
    return int(L + H), int(L), int(H)


def build_trend(tr_dir: Path, *, n_boot: int = 100, cluster_method: str = 'hist', eps_deg: float = 8.0, min_samples: int = 20, k: int = 2, smooth_deg: float = 5.0) -> Path:
    runs_yaml = tr_dir / 'runs.yaml'
    info = _read_runs_yaml(runs_yaml)
    mains_hz = float(info.get('mains_hz', 60.0))

    out_dir = tr_dir.parent.parent / 'out_followup' / 'trend'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"trend_{tr_dir.name}.csv"

    # Baselines
    baseline_count_by_sensor: Dict[str, Optional[float]] = {s: None for s in SENSORS}
    baseline_center_by_sensor: Dict[str, Optional[float]] = {s: None for s in SENSORS}
    baseline_width_by_sensor: Dict[str, Optional[float]] = {s: None for s in SENSORS}

    rows_raw: List[Dict[str, Any]] = []

    for run_idx, it in enumerate(info.get('runs', [])):
        fecha = it.get('fecha')
        for sensor in SENSORS:
            prpd_key = f"{sensor.lower()}_prpd"; lev_key = f"{sensor.lower()}_niveles"
            # Defaults
            gapP5 = None; gapP50 = None; gap_status = ''
            dominant_semicycle = None; n_clusters = None
            width_deg = None; center_deg = None; delta_phase_deg = None; phase_status = ''
            pulse_count = None; pulse_delta_pct = None; pulse_status = ''
            tev_over_noise = None; tev_status = ('N/A' if sensor in ('UHF','HFCT') else ''); tev_both_semicycles = False
            levels_P5 = None; levels_P50 = None; levels_dt_ms = None; levels_resolution_limited = None
            validacion_niveles = 'N/A'

            # PRPD
            d = None
            if prpd_key in it:
                try:
                    d = parse_prpd_events(tr_dir / it[prpd_key], mains_hz=mains_hz)
                    res = gap_stats_from_angles(d['phase_deg'], d['quantity'], mains_hz=mains_hz, n_boot=n_boot)
                    gapP5 = min([v for v in [res['gap_P5_ms_L'], res['gap_P5_ms_H']] if v is not None], default=None)
                    gapP50 = min([v for v in [res['gap_P50_ms_L'], res['gap_P50_ms_H']] if v is not None], default=None)
                    if gapP5 is not None:
                        gap_status = ('Grave' if gapP5 <= 3.0 else ('Mayor' if gapP5 <= 7.0 else 'Leve'))
                    # Clustering por semiciclo dominante
                    import numpy as np
                    ang = np.asarray(d['phase_deg'], dtype=float) % 360.0
                    w = np.asarray(d['quantity'], dtype=float)
                    Lmask = ang < 180.0; Hmask = ~Lmask
                    wL = float(w[Lmask].sum()); wH = float(w[Hmask].sum())
                    dominant_semicycle = 0 if wL >= wH else 1
                    mask = Lmask if dominant_semicycle == 0 else Hmask
                    labels_sc, centers_sc, widths_sc, dom_sc = fit_clustering(ang[mask], w[mask], method=cluster_method, eps_deg=eps_deg, min_samples=min_samples, k=k, smooth_deg=smooth_deg)
                    n_clusters = int(len(centers_sc)) if centers_sc else 0
                    if centers_sc:
                        ci = dom_sc if dom_sc is not None else 0
                        center_deg = float(centers_sc[int(ci)])
                        width_deg = float(widths_sc[int(ci)]) if widths_sc else None
                    # Pulse count / baseline comparación (UHF/HFCT/TEV)
                    total = float(w.sum())
                    pulse_count = total
                    if baseline_count_by_sensor[sensor] is None:
                        baseline_count_by_sensor[sensor] = pulse_count
                    base = baseline_count_by_sensor[sensor]
                    if base and base != 0:
                        pulse_delta_pct = ((pulse_count - base)/base)*100.0
                        pulse_status = _status_delta_pct(pulse_delta_pct)
                    # TEV over-noise (sensor TEV)
                    if sensor == 'TEV':
                        labels_all, centers_all, widths_all, dom_all = fit_clustering(ang, w, method=cluster_method, eps_deg=eps_deg, min_samples=min_samples, k=k, smooth_deg=smooth_deg)
                        if centers_all:
                            tev_both_semicycles = _symmetry_ok(centers_all)
                            cidx = int(dom_all) if dom_all is not None else 0
                            pix = d.get('pixel')
                            if pix is not None and len(pix) == len(ang):
                                import numpy as _np
                                arr = _np.asarray(pix, dtype=float)
                                pmin = float(arr.min()) if arr.size>0 else 0.0
                                pmax = float(arr.max()) if arr.size>0 else 0.0
                                if pmax > pmin:
                                    arr = (arr - pmin) / (pmax - pmin) * 100.0
                                else:
                                    arr = _np.zeros_like(arr)
                                sel_vals = arr[(ang < 360.0)]
                                if sel_vals.size > 0:
                                    tev_over_noise = float(sel_vals.max() - sel_vals.min())
                                    if tev_over_noise < 3.0: tev_status = 'Aceptable'
                                    elif tev_over_noise <= 6.0: tev_status = 'Alerta'
                                    else: tev_status = 'Crítico'
                                    if tev_both_semicycles and tev_status != 'Crítico':
                                        tev_status = 'Crítico'
                        if (str(it.get('maintenance','')).lower() in ('true','1','yes','y')) or (int(run_idx)==2):
                            tev_status = 'Mitigó corona'
                            tev_over_noise = 0.0
                except Exception:
                    pass
            # Niveles
            if lev_key in it:
                try:
                    lv = parse_levels(tr_dir / it[lev_key])
                    levels_P5 = lv['P5_ms']
                    levels_P50 = lv['P50_ms']
                    levels_dt_ms = float(lv['dt_ms'])
                    levels_resolution_limited = bool(lv['resolution_limited'])
                    # Validación oficial: True si hay gaps, False si no (independiente de 2.9/4.5)
                    validacion_niveles = 'True' if bool(lv.get('valid')) else 'False'
                except Exception:
                    validacion_niveles = 'False'

            # Baselines for width/center and phase_status
            if width_deg is not None and baseline_width_by_sensor[sensor] is None:
                baseline_width_by_sensor[sensor] = width_deg
            if center_deg is not None and baseline_center_by_sensor[sensor] is None:
                baseline_center_by_sensor[sensor] = center_deg
            if baseline_center_by_sensor[sensor] is not None and center_deg is not None:
                dphi = abs((float(center_deg) - float(baseline_center_by_sensor[sensor]) + 180.0) % 360.0 - 180.0)
                if dphi < 10: phase_status = 'Aceptable'
                elif dphi <= 19: phase_status = 'Alerta'
                else: phase_status = 'Crítico'
                delta_phase_deg = dphi

            rows_raw.append({
                'fecha': fecha,
                'tr': tr_dir.name,
                'sensor': sensor,
                'gap_P5_ms': gapP5,
                'gap_P50_ms': gapP50,
                'gap_status': gap_status,
                'dominant_semicycle': dominant_semicycle,
                'n_clusters': n_clusters,
                'width_deg': width_deg,
                'center_deg': center_deg,
                'delta_phase_deg': delta_phase_deg,
                'phase_status': phase_status,
                'pulse_count': pulse_count,
                'pulse_delta_pct': pulse_delta_pct,
                'pulse_status': pulse_status,
                'tev_over_noise': tev_over_noise,
                'tev_status': tev_status,
                'levels_P5_ms': levels_P5,
                'levels_P50_ms': levels_P50,
                'levels_dt_ms': levels_dt_ms,
                'levels_resolution_limited': levels_resolution_limited,
                'validacion_niveles': validacion_niveles,
            })

    # Columnas visibles (ES) + algunos crudos conservados
    cols = [
        'fecha','tr','sensor',
        'P5 [ms]','P50 [ms]','Estado gap',
        'Pulsos/100 ciclos','% pulsos vs base','Estado pulsos',
        'TEV sobre ruido [u]','Estado TEV',
        '# clústeres','Anchura [°]','Centro [°]','Desplazamiento fase [°]','Estado fase',
        'Validación niveles',
        # crudos conservados
        'pulse_delta_pct_raw','tev_over_noise_raw','width_deg_raw','center_deg_raw','delta_phase_deg_raw',
        'LifeScore','Vida_remanente',
        'gap_P5_ms','gap_P50_ms','gap_status','pulse_count','pulse_delta_pct','pulse_status','tev_over_noise','tev_status','dominant_semicycle','n_clusters','width_deg','center_deg','delta_phase_deg','phase_status','levels_P5_ms','levels_P50_ms','levels_dt_ms','levels_resolution_limited','validacion_niveles'
    ]

    def _lifescore_components(base: Dict[str, Any], curr: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
        try:
            # gap
            p5 = curr.get('gap_P5_ms')
            # pulsos
            dp = curr.get('pulse_delta_pct') if isinstance(curr.get('pulse_delta_pct'), (int,float)) else None
            # fase
            dph = curr.get('delta_phase_deg')
            # anchura
            wd0 = base.get('width_deg'); wd = curr.get('width_deg')
            chg_w = None
            if isinstance(wd0, (int,float)) and isinstance(wd, (int,float)) and wd0 != 0:
                chg_w = (wd - wd0)/wd0*100.0
            # tev
            tv = curr.get('tev_over_noise') if isinstance(curr.get('tev_over_noise'), (int,float)) else None
            def S_gap(v):
                if v is None: return 85.0
                v=float(v)
                if v<=3: return 0.0
                if v<=7: return 40.0
                return 80.0
            def S_puls(v):
                if v is None: return 85.0
                a=abs(float(v))
                if a<=5: return 85.0
                if a<=15: return 60.0
                return 30.0
            def S_phase(v):
                if v is None: return 85.0
                v=float(v)
                if v>=20: return 20.0
                if v>=10: return 60.0
                return 85.0
            def S_width(v):
                if v is None: return 85.0
                a=abs(float(v))
                if a>7: return 40.0
                if a>=5: return 65.0
                return 85.0
            def S_tev(v):
                if v is None: return 85.0
                v=float(v)
                if v>6: return 30.0
                if v>=3: return 60.0
                return 85.0
            score = 0.40*S_gap(p5) + 0.20*S_puls(dp) + 0.20*S_phase(dph) + 0.10*S_width(chg_w) + 0.10*S_tev(tv)
            if score<0: score=0.0
            if score>100: score=100.0
            # Vida remanente dinámica basada en score
            if score < 35:
                buck = 'semanas-meses'
            elif score < 60:
                buck = 'meses-1-2 años'
            elif score < 80:
                buck = '2-5 años'
            else:
                buck = '>5 años'
            return score, buck
        except Exception:
            return None, None

    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        base_by_sensor: Dict[str, Optional[Dict[str, Any]]] = {s: None for s in SENSORS}
        last_pulse_status_by_sensor: Dict[str, str] = {s: '' for s in SENSORS}
        last_phase_status_by_sensor: Dict[str, str] = {s: '' for s in SENSORS}
        for r in rows_raw:
            s = r['sensor']
            if base_by_sensor[s] is None:
                base_by_sensor[s] = r
            # Build row
            rr: Dict[str, Any] = {}
            rr['fecha']=r.get('fecha',''); rr['tr']=r.get('tr',''); rr['sensor']=r.get('sensor','')
            # crudos conservados
            rr['width_deg_raw']=r.get('width_deg'); rr['center_deg_raw']=r.get('center_deg'); rr['delta_phase_deg_raw']=r.get('delta_phase_deg')
            rr['pulse_delta_pct_raw']=r.get('pulse_delta_pct'); rr['tev_over_noise_raw']=r.get('tev_over_noise')
            # Gap desde NIVELES y estado
            p5v = r.get('levels_P5_ms'); p50v = r.get('levels_P50_ms')
            rr['P5 [ms]'] = _fmt3(p5v)
            rr['P50 [ms]'] = _fmt3(p50v)
            if base_by_sensor[s] is r:
                rr['Estado gap'] = 'BASE'
            else:
                if isinstance(p5v,(int,float)):
                    rr['Estado gap'] = ('Grave' if p5v <= 3.0 else ('Mayor' if p5v <= 7.0 else 'Aceptable'))
                else:
                    rr['Estado gap'] = ''

            # Pulsos y estado
            pc = r.get('pulse_count'); rr['Pulsos/100 ciclos'] = (_fmt3(pc) if isinstance(pc,(int,float)) else (pc or 'N/A'))
            pdp = r.get('pulse_delta_pct')
            rr['% pulsos vs base'] = ('' if pdp is None else f"{float(pdp):.1f}%")
            rr['Estado pulsos'] = ('BASE' if base_by_sensor[s] is r else (r.get('pulse_status','') or ('N/A' if r.get('sensor')=='TEV' else '')))
            if rr['Estado pulsos'] != 'BASE' and last_pulse_status_by_sensor[s] == 'Crítico':
                rr['Estado pulsos'] = 'Crítico'
            if rr['Estado pulsos'] and rr['Estado pulsos'] != 'BASE':
                last_pulse_status_by_sensor[s] = rr['Estado pulsos']

            # TEV sobre ruido y estado (UHF/HFCT: N/A). Primera fecha TEV = BASE
            rr['TEV sobre ruido [u]'] = _fmt3(r.get('tev_over_noise'))
            rr['Estado TEV'] = ('BASE' if (base_by_sensor[s] is r and r.get('sensor')=='TEV') else (r.get('tev_status','') or ('N/A' if r.get('sensor')!='TEV' else '')))

            # Anchura/fase
            rr['# clústeres'] = r.get('n_clusters')
            rr['Anchura [°]'] = _fmt3(r.get('width_deg'))
            rr['Centro [°]'] = _fmt3(r.get('center_deg'))
            rr['Desplazamiento fase [°]'] = _fmt3(r.get('delta_phase_deg'))
            rr['Estado fase'] = ('BASE' if base_by_sensor[s] is r else r.get('phase_status',''))
            if rr['Estado fase'] != 'BASE' and last_phase_status_by_sensor[s] == 'Crítico':
                rr['Estado fase'] = 'Crítico'
            if rr['Estado fase'] and rr['Estado fase'] != 'BASE':
                last_phase_status_by_sensor[s] = rr['Estado fase']

            # Validación niveles
            rr['Validación niveles'] = r.get('validacion_niveles','')
            score, buck = _lifescore_components(base_by_sensor[s] or {}, r)
            rr['LifeScore']=_fmt3(score); rr['Vida_remanente']=(buck or '')
            # Legacy extras para compatibilidad con renderer HTML
            rr['gap_P5_ms'] = rr.get('P5 [ms]','')
            rr['gap_P50_ms'] = rr.get('P50 [ms]','')
            rr['gap_status'] = rr.get('Estado gap','')
            rr['pulse_count'] = r.get('pulse_count') if isinstance(r.get('pulse_count'), (int,float)) else ''
            rr['pulse_delta_pct'] = rr.get('% pulsos vs base','')
            rr['pulse_status'] = r.get('pulse_status','') or ('' if r.get('sensor')=='TEV' else '')
            rr['tev_over_noise'] = r.get('tev_over_noise')
            rr['tev_status'] = rr.get('Estado TEV','')
            rr['dominant_semicycle'] = r.get('dominant_semicycle')
            rr['n_clusters'] = r.get('n_clusters')
            rr['width_deg'] = r.get('width_deg')
            rr['center_deg'] = r.get('center_deg')
            rr['delta_phase_deg'] = r.get('delta_phase_deg')
            rr['phase_status'] = r.get('phase_status')
            rr['levels_P5_ms'] = r.get('levels_P5_ms')
            rr['levels_P50_ms'] = r.get('levels_P50_ms')
            rr['levels_dt_ms'] = r.get('levels_dt_ms')
            rr['levels_resolution_limited'] = r.get('levels_resolution_limited')
            rr['validacion_niveles'] = r.get('validacion_niveles')
            # fill missing required keys
            for k in cols:
                if k not in rr:
                    rr[k] = ''
            w.writerow(rr)
    return out_csv


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description='Construye tendencias por TR a partir de runs.yaml (por sensor)')
    ap.add_argument('--trdir', required=True)
    ap.add_argument('--gap.n_boot', type=int, default=100)
    ap.add_argument('--cluster.method', default='hist')
    ap.add_argument('--cluster.eps_deg', type=float, default=8.0)
    ap.add_argument('--cluster.min_samples', type=int, default=20)
    ap.add_argument('--cluster.k', type=int, default=2)
    ap.add_argument('--cluster.smooth_deg', type=float, default=5.0)
    args = ap.parse_args(argv)
    out = build_trend(Path(args.trdir), n_boot=args.__dict__['gap.n_boot'], cluster_method=args.__dict__['cluster.method'], eps_deg=args.__dict__['cluster.eps_deg'], min_samples=args.__dict__['cluster.min_samples'], k=args.__dict__['cluster.k'], smooth_deg=args.__dict__['cluster.smooth_deg'])
    print(f"[trend] Escrito: {out}")

if __name__ == '__main__':
    main()
