# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import re, datetime as dt

DATE_PATTS = [re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")]
SENSORS = ['UHF','HFCT','TEV']


def _date_from_name_or_parent(p: Path) -> Optional[str]:
    for txt in (p.name, p.stem, p.parent.name):
        m = DATE_PATTS[0].search(txt)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _date_from_mtime(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d')


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    def _yaml_val(v):
        if v is None:
            return 'null'
        if isinstance(v, bool):
            return 'true' if v else 'false'
        if isinstance(v, (int, float)):
            return str(v)
        return str(v).replace('\\', '/')
    lines: List[str] = []
    lines.append(f"baseline: {_yaml_val(data.get('baseline'))}")
    lines.append(f"mains_hz: {_yaml_val(data.get('mains_hz', 60))}")
    lines.append("runs:")
    for it in data.get('runs', []):
        lines.append("  - fecha: " + _yaml_val(it.get('fecha')))
        for k in ['uhf_prpd','hfct_prpd','tev_prpd','uhf_niveles','hfct_niveles','tev_niveles']:
            v = it.get(k)
            if v:
                lines.append(f"    {k}: {v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding='utf-8')


def _rel(p: Path, base: Path) -> str:
    try:
        return str(p.relative_to(base)).replace('\\\\', '/').replace('\\', '/')
    except Exception:
        return str(p).replace('\\\\', '/').replace('\\', '/')


def discover(root: str | Path, tr: str) -> Path:
    root = Path(root)
    assert tr in ('TR1','TR2'), 'tr debe ser TR1 o TR2'
    tr_dir = root / tr
    runs: Dict[str, Dict[str, Any]] = {}
    for sub in ['segprpd','segniveles']:
        for sensor in SENSORS:
            for xp in (tr_dir / sub / sensor).glob('*.xml'):
                fecha = _date_from_name_or_parent(xp) or _date_from_mtime(xp)
                rec = runs.setdefault(fecha, {'fecha': fecha})
                key = f"{sensor.lower()}_{'prpd' if sub=='segprpd' else 'niveles'}"
                rec[key] = _rel(xp, tr_dir)
    if not runs:
        raise RuntimeError(f"No se hallaron .xml en {tr_dir}")
    fechas = sorted(runs.keys())
    data = {'baseline': fechas[0], 'mains_hz': 60, 'runs': [runs[f] for f in fechas]}
    out = tr_dir / 'runs.yaml'
    _dump_yaml(out, data)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description='Autodiscovery de runs.yaml por TR')
    ap.add_argument('--root', required=True)
    ap.add_argument('--tr', required=True, choices=['TR1','TR2'])
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)
    out = Path(args.root) / args.tr / 'runs.yaml'
    if out.exists() and not args.force:
        print(f"[discover] Ya existe {out}")
        return
    p = discover(args.root, args.tr)
    print(f"[discover] Escrito: {p}")

if __name__ == '__main__':
    main()
