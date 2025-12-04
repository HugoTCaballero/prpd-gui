# -*- coding: utf-8 -*-
"""Validación en lote del clasificador por reglas.

Lee etiquetas verdaderas desde data/pd_cases_labels.json, procesa los XML
ubicados en data/xml_cases/ y genera un CSV con predicciones + KPIs clave.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PRPDapp.prpd_core import process_prpd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
XML_DIR = DATA_DIR / "xml_cases"
LABELS_PATH = DATA_DIR / "pd_cases_labels.json"
OUT_DIR = ROOT / "out" / "validation_rules"
OUT_CSV = OUT_DIR / "validation_rules_out.csv"


def load_labels() -> dict:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {LABELS_PATH}. Rellena el JSON con tus casos etiquetados."
        )
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def process_case(xml_path: Path) -> dict:
    # Usa fast_mode para acelerar; out_root independiente por caso
    out_root = OUT_DIR / xml_path.stem
    out_root.mkdir(parents=True, exist_ok=True)
    result = process_prpd(xml_path, out_root, fast_mode=True)
    return result


def summarize_confusion(rows: list[dict]) -> None:
    cm: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        true_cls = r.get("true_class") or "na"
        pred_cls = r.get("pred_class") or "na"
        cm[true_cls][pred_cls] += 1

    print("\n=== Matriz de confusión (verdadera vs predicha) ===")
    for true_cls, counter in cm.items():
        print(f"\nVerdadera: {true_cls}")
        for pred_cls, count in counter.items():
            print(f"  Predicha: {pred_cls} -> {count}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()

    rows: list[dict] = []
    for fname, meta in labels.items():
        xml_path = XML_DIR / fname
        if not xml_path.exists():
            print(f"[WARN] No se encontró {xml_path}, se omite.")
            continue
        print(f"[INFO] Procesando {xml_path}...")
        try:
            result = process_case(xml_path)
        except Exception as exc:  # pragma: no cover - script de depuración
            print(f"[ERROR] Falló {xml_path}: {exc}")
            continue

        rule = result.get("rule_pd", {}) or {}
        kpis = result.get("kpis", {}) or {}

        row = {
            "file": fname,
            "true_class": meta.get("true_class"),
            "true_stage": meta.get("true_stage"),
            "pred_class": rule.get("class_id"),
            "pred_stage": rule.get("stage"),
            "fa_phase_width": kpis.get("fa_phase_width_deg"),
            "fa_symmetry": kpis.get("fa_symmetry_index"),
            "fa_concentration": kpis.get("fa_concentration_index"),
            "fa_p95_amp": kpis.get("fa_p95_amplitude"),
            "n_angpd_angpd_ratio": kpis.get("n_angpd_angpd_ratio"),
            "gap_p50_ms": kpis.get("gap_p50_ms"),
            "gap_p5_ms": kpis.get("gap_p5_ms"),
            "ang_phase_peaks": kpis.get("ang_phase_peaks"),
            "ang_amp_concentration": kpis.get("ang_amp_concentration"),
        }
        rows.append(row)

    if not rows:
        print("[WARN] No hay filas para escribir; ¿faltan XML en data/xml_cases?")
        return

    # CSV
    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV guardado en {OUT_CSV}")

    summarize_confusion(rows)


if __name__ == "__main__":
    main()
