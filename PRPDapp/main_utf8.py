#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compat: redirige a la interfaz unificada PRPDapp.main."""
from __future__ import annotations
import sys
import pathlib as _pl

def main() -> None:
    try:
        _THIS = _pl.Path(__file__).resolve(); _ROOT = _THIS.parents[1]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        from PRPDapp.main import main as _launch
        _launch()
    except Exception as e:
        print("No se pudo abrir la interfaz unificada:", e)
        raise

if __name__ == "__main__":
    main()
