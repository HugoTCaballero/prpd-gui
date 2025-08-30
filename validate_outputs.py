# tools/validate_outputs.py
import sys, glob, os, pandas as pd

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "out"

PAIRED_REQ = ["pair_id","id_pos","phi_pos","y_pos","w_pos",
              "id_neg","phi_neg","y_neg","w_neg","dphi","dy","w_ratio","type"]
PM_REQ     = ["type","count_sources","p_sources"]

def has_cols(df, req): 
    return all(c in df.columns for c in req)

errors = []
csvs = glob.glob(os.path.join(OUT_DIR, "*.csv"))

if not csvs:
    errors.append(f"Sin CSVs en {OUT_DIR}. ¿Ejecutaste el pipeline?")

for csv in csvs:
    name = os.path.basename(csv)
    try:
        df = pd.read_csv(csv)
    except Exception as e:
        errors.append(f"{name}: no se pudo leer ({e})"); 
        continue

    if name.startswith("paired_sources") and not has_cols(df, PAIRED_REQ):
        errors.append(f"{name}: faltan columnas de contrato en paired_sources")
    if name.startswith("p_multiplicity") and not has_cols(df, PM_REQ):
        errors.append(f"{name}: faltan columnas de contrato en p_multiplicity")
    if name.startswith("summary") and "p_global" not in df.columns:
        errors.append(f"{name}: falta p_global (debe recalcularse post-pair cuando aplique)")

if errors:
    print("ERRORES DE CONTRATO/CONTENIDO:")
    for e in errors: print(" -", e)
    sys.exit(1)
else:
    print(f"OK: contratos y p_global presentes en {OUT_DIR}")
