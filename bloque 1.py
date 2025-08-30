#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bloque 1 — Generador de mapa PRPD térmico tipo “grad-cam”.

Este script toma un archivo XML de mediciones de descargas parciales (formato
Megger UHF PDD o similar) y construye una visualización térmica (mapa de
densidad) de la fase vs. la magnitud normalizada de la señal. Se utiliza un
histograma 2D ponderado por la cantidad de pulsos y un colormap de
Matplotlib para mostrar la distribución de la energía de las descargas de forma
análoga a una cámara térmica. También detecta de forma heurística el tipo de
sensor (UHF, TEV, HFCT) en base al contenido del XML y permite aplicar un
corrimiento de fase de 0°, 120° o 240°.  Finalmente, genera un texto de
resumen que incorpora la ponderación por tipo de transformador (seco u en
aceite).

Uso (ejemplo):

    python bloque1.py time_20000101_022514.xml --phase-shift 120 \
        --transformer-type seco --colormap inferno --save-fig salida.png

El script imprimirá en pantalla el resumen textual y guardará la figura en
`salida.png` si se indica la opción `--save-fig`.  Si no se especifica
`--save-fig`, se mostrará la figura en una ventana interactiva.

Dependencias: requiere `prpd_mega.py` en el mismo directorio o en el
`PYTHONPATH`, ya que reutiliza sus funciones de parsing y normalización.
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker

# Importamos utilidades de prpd_mega.py.  Se asume que prpd_mega.py se
# encuentra en el mismo directorio o en el PYTHONPATH.  No modificamos
# directamente prpd_mega.py para preservar su funcionalidad original.
try:
    from prpd_mega import (
        parse_xml_points,
        identify_sensor_from_data,
        normalize_y,
        phase_from_times,
        prpd_hist2d,
    )
except ImportError as e:
    raise ImportError(
        "No se pudo importar prpd_mega. Asegúrate de que prpd_mega.py está en el mismo directorio."
    ) from e


def generate_prpd_thermal(
    xml_path: str,
    phase_shift_deg: float = 0.0,
    transformer_type: str = "seco",
    colormap: str = "inferno",
) -> Tuple[plt.Figure, str]:
    """Genera un mapa PRPD térmico y un resumen textual.

    Parameters
    ----------
    xml_path : str
        Ruta al archivo XML de medición.
    phase_shift_deg : float, optional
        Corrimiento de fase a aplicar (en grados).  Debe ser 0, 120 o 240,
        aunque no se fuerza programáticamente; se asume que el usuario
        proporciona un valor válido. Por defecto es 0.0.
    transformer_type : str, optional
        Tipo de transformador ("seco" o "en aceite"). Se utiliza para
        ponderar el recuento total de pulsos en el resumen. Por defecto es
        "seco".
    colormap : str, optional
        Nombre del colormap de Matplotlib a utilizar para la visualización
        térmica. Por defecto es "inferno".

    Returns
    -------
    fig : matplotlib.figure.Figure
        La figura generada con el mapa de densidad.
    summary : str
        Cadena de texto con el sensor detectado, el tipo de transformador,
        el factor de ponderación aplicado y el total de pulsos ponderado.

    Notes
    -----
    El mapa térmico se construye utilizando un histograma 2D de fase (0–360°)
    versus magnitud normalizada (0–100) ponderado por la cantidad de pulsos.
    Se emplea una escala logarítmica en los colores para resaltar las
    concentraciones de descargas, de manera similar a las visualizaciones
    tipo "grad‑cam" empleadas en redes neuronales.
    """
    # Parseo del XML para extraer vectores crudos
    data = parse_xml_points(xml_path)
    raw_y = data["raw_y"]
    times_ms = data["times"]
    quantity = data["quantity"]
    sample_name = data.get("sample_name")

    # Detectar sensor de forma heurística
    sensor = identify_sensor_from_data(raw_y, sample_name)

    # Normalizar la coordenada vertical (0–100).  El texto de la etiqueta se
    # ignora aquí, ya que la visualización especifica su propio eje.
    y_norm, _ = normalize_y(raw_y, sample_name)

    # Convertir los tiempos a fase, aplicando el corrimiento pedido
    phase = phase_from_times(times_ms, phase_shift_deg)

    # Construir el histograma 2D
    H, xedges, yedges = prpd_hist2d(phase, y_norm, quantity)

    # Calcular suma total de pulsos para el resumen
    total_counts = float(np.sum(quantity))

    # Ponderación según tipo de transformador.  Definimos un factor simple:
    # 1.0 para "seco" y 0.8 para "en aceite".  Si se indican otros valores,
    # se dejará la ponderación en 1.0 por defecto.
    ttype = (transformer_type or "").strip().lower()
    if ttype.startswith("seco"):
        weight_factor = 1.0
    elif "aceite" in ttype:
        weight_factor = 0.8
    else:
        weight_factor = 1.0

    weighted_total = total_counts * weight_factor

    # Crear la figura y la visualización
    fig, ax = plt.subplots(figsize=(9, 5))
    # Añadimos 1e-9 para evitar valores cero en la escala logarítmica
    im = ax.imshow(
        H.T + 1e-9,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        norm=LogNorm(vmin=1e-9, vmax=max(1.0, H.max())),
        cmap=colormap,
    )
    # Título con información básica
    title = (
        f"PRPD Térmico – Sensor {sensor} – {Path(xml_path).name} "
        f"(shift {phase_shift_deg}°)"
    )
    ax.set_title(title)
    ax.set_xlabel("Fase (°)")
    ax.set_ylabel("Muestra normalizada (0–100)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Recuento (log)")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 100)
    # Líneas de cuadrícula cada 30° para una mejor lectura de la fase
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    # Ajuste de diseño
    fig.tight_layout()

    # Construir el texto de resumen
    summary = (
        f"Sensor detectado: {sensor}. Tipo de transformador: {transformer_type}. "
        f"Factor de ponderación aplicado: {weight_factor:.2f}. "
        f"Recuento total de pulsos (sin ponderar): {total_counts:.0f}. "
        f"Recuento total ponderado: {weighted_total:.0f}."
    )

    return fig, summary


def main() -> None:
    """Punto de entrada para ejecución desde consola."""
    parser = argparse.ArgumentParser(
        description=(
            "Bloque 1: Genera un PRPD térmico tipo 'grad‑cam' a partir de un "
            "archivo XML de descargas parciales."
        )
    )
    parser.add_argument(
        "xml_file",
        help="Ruta al archivo XML de medición a procesar.",
    )
    parser.add_argument(
        "--phase-shift",
        type=float,
        default=0.0,
        help="Corrimiento de fase a aplicar (grados). Ej.: 0, 120, 240.",
    )
    parser.add_argument(
        "--transformer-type",
        type=str,
        default="seco",
        help="Tipo de transformador: 'seco' o 'en aceite'.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="inferno",
        help="Nombre del colormap de Matplotlib a usar en el mapa térmico.",
    )
    parser.add_argument(
        "--save-fig",
        type=str,
        default=None,
        help="Ruta de salida para guardar la figura en formato PNG. Si se omite, se mostrará en pantalla.",
    )
    args = parser.parse_args()

    # Generar figura y resumen
    fig, summary = generate_prpd_thermal(
        args.xml_file,
        phase_shift_deg=args.phase_shift,
        transformer_type=args.transformer_type,
        colormap=args.colormap,
    )

    # Mostrar o guardar la figura según corresponda
    if args.save_fig:
        # Asegurarse de que la ruta de salida existe
        out_path = Path(args.save_fig)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        print(f"Figura guardada en {out_path}")
    else:
        # Modo interactivo para entornos con interfaz gráfica
        plt.show()

    # Mostrar resumen en consola
    print(summary)


if __name__ == "__main__":
    main()
