CLASS_INFO = {
    "cavidad": {
        "name": "Cavidad / Descarga interna",
        "color": "#1f77b4",
        "order": 0,
    },
    "superficial": {
        "name": "Superficial / Tracking",
        "color": "#ff7f0e",
        "order": 1,
    },
    "corona": {
        "name": "Corona (aceite/aire)",
        "color": "#2ca02c",
        "order": 2,
    },
    "flotante": {
        "name": "Flotante",
        "color": "#d62728",
        "order": 3,
    },
    "suspendida": {
        "name": "Partículas suspendidas",
        "color": "#9467bd",
        "order": 4,
    },
    "ruido": {
        "name": "Ruido / Indeterminado",
        "color": "#7f7f7f",
        "order": 5,
    },
}

CLASS_NAMES = [
    k for k, _ in sorted(CLASS_INFO.items(), key=lambda kv: kv[1]["order"])
]
