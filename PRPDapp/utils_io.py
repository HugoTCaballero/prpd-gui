# utils_io.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json

def time_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_out_dirs(root: Path) -> Path:
    root = Path(root)
    (root / "aligned").mkdir(parents=True, exist_ok=True)
    (root / "filtered").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    return root

def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
