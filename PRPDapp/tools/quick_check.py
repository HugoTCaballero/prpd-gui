from pathlib import Path
import csv
root = Path(r"C:\Users\he_hu\Documents\PRPD")

def read_csv(p):
    with p.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def show(rows,tr):
    print(f"\n=== {tr} ===")
    for s in ["UHF","HFCT","TEV"]:
        rs=[r for r in rows if r.get("sensor")==s]
        if not rs:
            print(f" {s}: sin filas");
            continue
        p5 =[r.get("P5 [ms]") or r.get("levels_P5_ms") for r in rs]
        p50=[r.get("P50 [ms]") or r.get("levels_P50_ms") for r in rs]
        print(f" {s}: P5={p5} P50={p50} valid={ [r.get('Validación niveles') or r.get('validacion_niveles') for r in rs] }")
        print(f"    Pulsos={ [r.get('Pulsos/100 ciclos') or r.get('pulse_count') for r in rs] } Estado={ [r.get('Estado pulsos') or r.get('pulse_status') for r in rs] }")
        print(f"    TEV_over={ [r.get('TEV sobre ruido [u]') or r.get('tev_over_noise') for r in rs] } Estado={ [r.get('Estado TEV') or r.get('tev_status') for r in rs] }")
        print(f"    Anchura={ [r.get('Anchura [°]') or r.get('width_deg') for r in rs] }  dFase={ [r.get('Desplazamiento fase [°]') or r.get('delta_phase_deg') for r in rs] }  Estado_fase={ [r.get('Estado fase') or r.get('phase_status') for r in rs] }")
for p,tr in [(root/r'out_followup\\trend\\trend_TR1.csv','TR1'),(root/r'out_followup\\trend\\trend_TR2.csv','TR2')]:
    if p.exists(): show(read_csv(p),tr)
