#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PRPD GUI (PySimpleGUI) â€” Ensamblado mÃ­nimo plug-and-play

Pipeline (referencia):
  carga -> alineaciÃ³n de fase -> filtro de ruido (s1 weak, s2 strong,
  s3 pixel clustering [stub], s4 combinaciÃ³n [stub], s5 selecciÃ³n dominante)
  -> histogramas (angulares) -> ANN -> severidad (con desglose)
  -> seguimiento (comparar con base opcional) -> visualizaciÃ³n y exportaciÃ³n

Dependencias: PySimpleGUI (opcional). Si no estÃ¡, muestra instrucciones.
No rompe PRPDapp/main.py (PySide6); es una GUI alternativa ligera.
"""

from __future__ import annotations

import os
from pathlib import Path
import io
import sys
import json
import math
import numpy as np

try:
    import PySimpleGUI as sg  # type: ignore
except Exception:
    sg = None  # fallback: avisar mÃ¡s abajo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# NÃºcleo existente
# Soportar ejecuciÃ³n como script (python PRPDapp\psg_main.py) o mÃ³dulo (-m PRPDapp.psg_main)
import pathlib as _pl, sys as _sys
_THIS = _pl.Path(__file__).resolve(); _ROOT = _THIS.parents[1]; _PKG = _THIS.parent
for _p in (str(_ROOT), str(_PKG)):
    (_p in _sys.path) or _sys.path.insert(0, _p)

from PRPDapp import prpd_core as core
from PRPDapp.prpd_ann import PRPDANN


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_hist2d(phase_deg: np.ndarray, amp: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    if phase_deg.size and amp.size:
        H, xedges, yedges = np.histogram2d(phase_deg, amp, bins=[72, 50], range=[[0, 360], [0, 100]])
        ax.imshow(H.T + 1e-9, origin='lower', aspect='auto',
                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                  norm=LogNorm(vmin=1e-9, vmax=max(1.0, H.max())))
    ax.set_xlim(0, 360); ax.set_ylim(0, 100)
    ax.set_xlabel('Fase (deg)'); ax.set_ylabel('Amplitud (0â€“100)')
    ax.set_title(title)
    fig.tight_layout()
    return fig


def angular_hist(phase_deg: np.ndarray, weights: np.ndarray | None = None, bins: int = 72):
    w = weights if weights is not None else np.ones_like(phase_deg)
    H, edges = np.histogram(phase_deg % 360.0, bins=bins, range=(0, 360), weights=w)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, H


def run_pipeline(xml_path: Path,
                 density_view: bool,
                 s1_weak: bool,
                 s2_strong: bool,
                 s3_pixclus: bool,
                 s4_combine: bool,
                 s5_dominant: bool,
                 ann_model: PRPDANN | None,
                 base_path: Path | None) -> dict:
    out_root = Path('out_psg'); out_root.mkdir(exist_ok=True)
    result = core.process_prpd(path=xml_path, out_root=out_root, force_phase_offsets=None, fast_mode=True)

    # Derivar vistas y variantes de filtrado segÃºn toggles
    ph = np.asarray(result['raw']['phase_deg'], dtype=float)
    amp = np.asarray(result['raw']['amplitude'], dtype=float)
    ph_al = np.asarray(result['aligned']['phase_deg'], dtype=float)
    amp_al = np.asarray(result['aligned']['amplitude'], dtype=float)

    # s1/s2: aplicar DBSCAN con parÃ¡metros distintos y combinar
    labels = np.asarray(result['labels'])
    keep = np.asarray(result['keep_mask'])
    keep_combined = keep.copy()
    if s1_weak or s2_strong:
        eps_w, ms_w = 0.09, 10
        eps_s, ms_s = 0.055, 22
        # recomputar sobre fase alineada y amp_norm
        eps_use = []
        if s1_weak: eps_use.append(('weak', eps_w, ms_w))
        if s2_strong: eps_use.append(('strong', eps_s, ms_s))
        for _, eps, ms in eps_use:
            lab, km = core.cluster_and_prune(ph_al, result['raw']['amp_norm'], eps=eps, min_samples=ms)
            if s4_combine:
                keep_combined |= km
            else:
                keep_combined = km

    # s3 pixel clustering: stub (reutiliza keep existente)
    if s3_pixclus:
        # simple refuerzo: mantener bins de alta densidad
        centers, ah = angular_hist(ph_al, bins=72)
        thr = 0.02 * (ah.max() if ah.size else 1.0)
        hot = (ah >= thr)
        idx = np.floor((ph_al % 360.0) / (360.0 / 72)).astype(int)
        keep_pix = hot[np.clip(idx, 0, 71)]
        keep_combined &= keep_pix

    # s5 selecciÃ³n dominante: conservar top 2 clÃºsteres por tamaÃ±o
    if s5_dominant:
        # derivar labels sobre aligned (map sencillo por bin)
        labels_al = result.get('labels_aligned')
        if labels_al is not None and len(labels_al) == len(keep_combined):
            labs = np.asarray(labels_al)
        else:
            # fallback: todo uno
            labs = np.zeros_like(keep_combined, dtype=int)
        vals, counts = np.unique(labs[keep_combined], return_counts=True)
        order = list(vals[np.argsort(-counts)])[:2]
        mask_dom = np.isin(labs, order)
        keep_combined &= mask_dom

    # Probabilidades (ANN si cargado)
    proba = result.get('probs', {})
    if ann_model is not None and getattr(ann_model, 'is_loaded', False):
        try:
            # features simples a partir de aligned+keep_combined
            amp_k = amp_al[keep_combined]
            ph_k = ph_al[keep_combined]
            p95 = float(np.percentile(np.abs(amp_k), 95)) if amp_k.size else 0.0
            dens = float(keep_combined.mean()) if keep_combined.size else 0.0
            # std circular:
            if ph_k.size:
                th = np.deg2rad(ph_k)
                C = float(np.mean(np.cos(th))); S = float(np.mean(np.sin(th)))
                R = float(np.hypot(C, S))
            else:
                R = 0.0
            features_for_ann = {
                'amp_mean': float(np.mean(np.abs(amp_k))) if amp_k.size else 0.0,
                'amp_std': float(np.std(amp_k)) if amp_k.size else 0.0,
                'amp_p95': p95,
                'density': dens,
                'phase_std_deg': float(np.degrees(np.sqrt(-2.0*np.log(max(R,1e-12))))) if R>1e-12 else 180.0,
                'phase_entropy': 0.0,
                'rep_rate': 0.0,
                'rep_entropy': 0.0,
                'cluster_compactness': 0.0,
                'cluster_separation': 0.0,
                'lobes_count': 0.0,
                'area_ratio': 0.0,
            }
            proba = ann_model.predict_proba(features_for_ann)
        except Exception:
            pass

    # Seguimiento: comparar con base (JSON) si provista
    tracking = {}
    if base_path and base_path.exists():
        try:
            base = json.loads(base_path.read_text(encoding='utf-8', errors='ignore'))
            # esperar campos: severity_score, probs, etc.
            tracking['delta_severity'] = result.get('severity_score', 0.0) - float(base.get('severity_score', 0.0))
        except Exception:
            pass

    # Render fig
    fig_crudo = plot_hist2d(ph, amp, 'PRPD crudo') if density_view else None
    fig_al = plot_hist2d(ph_al[keep_combined], amp_al[keep_combined], 'Alineado/filtrado') if density_view else None

    # Exportar
    out = {
        'result': result,
        'keep_mask_combined': keep_combined,
        'proba': proba,
        'tracking': tracking,
        'fig_crudo_bytes': fig_to_bytes(fig_crudo) if fig_crudo else b'',
        'fig_al_bytes': fig_to_bytes(fig_al) if fig_al else b'',
    }
    # guardar json de salida
    try:
        (out_root / 'last_result.json').write_text(json.dumps({
            'source': str(xml_path),
            'predicted': result.get('predicted'),
            'severity_score': result.get('severity_score'),
            'severity_breakdown': result.get('severity_breakdown'),
            'proba': proba,
            'has_noise': result.get('has_noise'),
            'noise_meta': result.get('noise_meta'),
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass
    return out


def main():
    if sg is None:
        print('PySimpleGUI no estÃ¡ instalado. Instala con: pip install PySimpleGUI')
        return

    sg.theme('SystemDefault')
    ann = PRPDANN()
    ann_state = 'heurÃ­stica'

    col_controls = [
        [sg.Text('XML'), sg.Input(key='-XML-', size=(60,1)), sg.FileBrowse(file_types=(('XML','*.xml'),))],
        [sg.Checkbox('Densidad (hist2D)', key='-DENS-', default=True)],
        [sg.Text('Filtro ruido:'),
         sg.Checkbox('s1 weak', key='-S1-', default=True),
         sg.Checkbox('s2 strong', key='-S2-', default=False),
         sg.Checkbox('s3 pixel clustering', key='-S3-', default=False),
         sg.Checkbox('s4 combine', key='-S4-', default=True),
         sg.Checkbox('s5 dominant', key='-S5-', default=True)],
        [sg.Text('Seguimiento base (opcional)'), sg.Input(key='-BASE-', size=(48,1)), sg.FileBrowse(file_types=(('JSON','*.json'),))],
        [sg.Button('Cargar ANN'), sg.Text(f'ANN: {ann_state}', key='-ANNSTATE-')],
        [sg.Button('Ejecutar'), sg.Button('Salir')],
        [sg.Multiline(key='-LOG-', size=(76,12), autoscroll=True, write_only=True)]
    ]
    col_imgs = [
        [sg.Text('Crudo')],
        [sg.Image(key='-IMG0-')],
        [sg.Text('Alineado/Filtrado')],
        [sg.Image(key='-IMG1-')],
    ]

    layout = [
        [sg.Column(col_controls), sg.VSeparator(), sg.Column(col_imgs)]
    ]
    win = sg.Window('PRPD PSG GUI', layout, finalize=True, resizable=True)

    while True:
        ev, vals = win.read()
        if ev in (sg.WINDOW_CLOSED, 'Salir'):
            break
        if ev == 'Cargar ANN':
            path = sg.popup_get_file('Seleccionar modelo ANN', file_types=(('Modelo','*.pkl;*.joblib'),))
            if path:
                try:
                    ann.load_model(path)
                    win['-ANNSTATE-'].update('ANN: cargado')
                    win['-LOG-'].print(f'Modelo ANN cargado: {path}')
                except Exception as e:
                    win['-LOG-'].print(f'No se pudo cargar ANN: {e}')
        if ev == 'Ejecutar':
            xml = vals.get('-XML-','').strip()
            if not xml:
                sg.popup_error('Selecciona un XML')
                continue
            try:
                out = run_pipeline(
                    Path(xml),
                    density_view=bool(vals.get('-DENS-', True)),
                    s1_weak=bool(vals.get('-S1-', False)),
                    s2_strong=bool(vals.get('-S2-', False)),
                    s3_pixclus=bool(vals.get('-S3-', False)),
                    s4_combine=bool(vals.get('-S4-', False)),
                    s5_dominant=bool(vals.get('-S5-', False)),
                    ann_model=ann,
                    base_path=Path(vals['-BASE-']) if vals.get('-BASE-') else None,
                )
                res = out['result']
                sev = res.get('severity_score', 0.0)
                sev_bd = res.get('severity_breakdown', {})
                proba = out.get('proba', {})
                win['-LOG-'].print(f"OK: clase={res.get('predicted')} sev={sev:.1f} | noise={res.get('has_noise')} meta={res.get('noise_meta')}")
                win['-LOG-'].print(json.dumps({'proba': proba, 'sev_bd': sev_bd}, ensure_ascii=False, indent=2))
                if out.get('fig_crudo_bytes'):
                    win['-IMG0-'].update(data=out['fig_crudo_bytes'])
                if out.get('fig_al_bytes'):
                    win['-IMG1-'].update(data=out['fig_al_bytes'])
            except Exception as e:
                win['-LOG-'].print(f'ERROR: {e}')

    win.close()


if __name__ == '__main__':
    main()
