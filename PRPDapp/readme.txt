PRPD GUI – MVP (Windows)

1) Requisitos
   - Python 3.10/3.11 (64-bit)
   - pip actualizado

2) Instalar dependencias
   pip install -r requirements.txt

3) Estructura de CSV
   Encabezados: phase_deg, amplitude  (o equivalentes: phase|phi y amplitude|amp|a)

4) Ejecutar
   python main.py

5) Flujo
   - Abrir PRPD (CSV)
   - Elegir fase: Auto (0/120/240) o fija a 0/120/240
   - Procesar: alineado+filtros+clustering DBSCAN + poda (no dominantes)
   - Salida: clase heurística, severidad (0..100), PDF en out/reports/

6) Carpetas de salida (para auditoría)
   out/aligned/   → PRPD alineado (npz)
   out/filtered/  → PRPD filtrado/kept (npz)
   out/reports/   → PDF

7) Notas
   - ANN real: este MVP usa reglas heurísticas; integra tu modelo cuando esté listo.
   - Todo el código está modularizado para integrarse con tu “Codex”/pipeline existente.
