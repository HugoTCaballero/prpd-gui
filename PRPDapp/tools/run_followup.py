# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Optional

from .discover_runs import discover
from .trend_builder import build_trend
from ..followup.followup_report import render_tr_report, render_index


def _export_pdf(html_path: Path) -> Path | None:
    pdf_path = html_path.with_suffix('.pdf')
    try:
        try:
            from weasyprint import HTML
            HTML(filename=str(html_path)).write_pdf(str(pdf_path))
            return pdf_path
        except Exception:
            pass
        try:
            import pdfkit
            pdfkit.from_file(str(html_path), str(pdf_path))
            return pdf_path
        except Exception:
            pass
        # Fallback: placeholder PDF
        with open(pdf_path, 'wb') as f:
            f.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<<>>endobj\n2 0 obj<<>>endobj\ntrailer<<>>\n%%EOF')
        return pdf_path
    except Exception:
        return None


def run_all(root: Path, tr: str, auto: bool, force_discover: bool, **opts) -> Path:
    tr_dir = root / tr
    runs_yaml = tr_dir / 'runs.yaml'
    if auto and (force_discover or not runs_yaml.exists()):
        discover(root, tr)
    build_trend(tr_dir,
                n_boot=opts.get('gap_n_boot', 100),
                cluster_method=opts.get('cluster_method', 'hist'),
                eps_deg=opts.get('cluster_eps_deg', 8.0),
                min_samples=opts.get('cluster_min_samples', 20),
                k=opts.get('cluster_k', 2),
                smooth_deg=opts.get('cluster_smooth_deg', 5.0))
    html_path = render_tr_report(root, tr)
    _export_pdf(html_path)
    return html_path


def main(argv: Optional[list[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description='Orquestador Seguimiento y Criticidad (per TR)')
    ap.add_argument('--root', required=True)
    ap.add_argument('--tr', default='ALL', choices=['TR1','TR2','ALL'])
    ap.add_argument('--auto', action='store_true')
    ap.add_argument('--force-discover', action='store_true')
    ap.add_argument('--gap.n_boot', dest='gap_n_boot', type=int, default=100)
    ap.add_argument('--cluster.method', dest='cluster_method', default='hist')
    ap.add_argument('--cluster.eps_deg', dest='cluster_eps_deg', type=float, default=8.0)
    ap.add_argument('--cluster.min_samples', dest='cluster_min_samples', type=int, default=20)
    ap.add_argument('--cluster.k', dest='cluster_k', type=int, default=2)
    ap.add_argument('--cluster.smooth_deg', dest='cluster_smooth_deg', type=float, default=5.0)
    args = ap.parse_args(argv)

    root = Path(args.root)
    trs = ['TR1','TR2'] if args.tr == 'ALL' else [args.tr]
    last_by_tr: dict[str, Path] = {}
    for t in trs:
        print(f"[run] TR={t}")
        try:
            last_by_tr[t] = run_all(root, t, auto=args.auto, force_discover=args.force_discover,
                                    gap_n_boot=args.gap_n_boot,
                                    cluster_method=args.cluster_method,
                                    cluster_eps_deg=args.cluster_eps_deg,
                                    cluster_min_samples=args.cluster_min_samples,
                                    cluster_k=args.cluster_k,
                                    cluster_smooth_deg=args.cluster_smooth_deg)
            print(f"[run] Reporte: {last_by_tr[t]}")
        except Exception as e:
            print(f"[run] TR={t} error: {e}")
            continue
    if last_by_tr:
        idx = render_index(root, last_by_tr)
        print(f"[DONE] Índice: {idx}")

if __name__ == '__main__':
    main()
