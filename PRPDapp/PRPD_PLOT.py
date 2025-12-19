import argparse, xml.etree.ElementTree as ET
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio

PERIOD_MS = 1000/60  # 16.667 ms @60Hz

# =======================
# Parseo XML (sin tocar)
# =======================
def parse_xml_all(path):
    root = ET.parse(path).getroot()
    times_nodes   = root.findall(".//times")
    samples_nodes = root.findall(".//sample")
    q_nodes       = root.findall(".//quantity")
    T=[]; S=[]; Q=[]
    blocks = max(len(times_nodes), len(samples_nodes), len(q_nodes))
    for b in range(blocks):
        t_text = times_nodes[b].text if b < len(times_nodes) else ""
        s_text = samples_nodes[b].text if b < len(samples_nodes) else ""
        q_text = q_nodes[b].text if b < len(q_nodes) else None
        def to_nums(txt):
            if txt is None: return []
            for ch in ['\n','\r','\t',',']:
                txt = txt.replace(ch, ' ')
            txt = ' '.join(txt.strip().split())
            if not txt: return []
            out = []
            for tok in txt.split(' '):
                if tok=='': continue
                try:
                    out.append(float(tok.replace(',', '.')))
                except:
                    pass
            return out
        t = to_nums(t_text)
        s = to_nums(s_text)
        q = to_nums(q_text) if q_text is not None else [1.0]*len(s)
        n = min(len(t), len(s), len(q))
        if n>0:
            T.extend(t[:n]); S.extend(s[:n]); Q.extend(q[:n])
    return np.array(T), np.array(S), np.array(Q)

def normalize_to_phase_pixel(T, S):
    Tms = T*1000.0 if (len(T)>0 and np.nanmax(T)<=1.0) else T.copy()
    t_env = Tms - PERIOD_MS*np.floor(Tms/PERIOD_MS)
    phase = (t_env/PERIOD_MS)*360.0
    smin, smax = np.nanmin(S), np.nanmax(S)
    if smax==smin: smax = smin + 1.0
    pixel = np.clip(np.rint((S - smin)*300.0/(smax - smin)), 0, 300)
    return phase, pixel

def grid_density(phase, pixel, Q, time_bins=360, pixel_bins=301, log_scale=False):
    Z = np.zeros((pixel_bins, time_bins), dtype=float)
    i = np.clip(np.rint(phase*(time_bins-1)/360.0).astype(int), 0, time_bins-1)
    j = np.clip(pixel.astype(int), 0, pixel_bins-1)
    for ii, jj, qq in zip(i, j, np.maximum(Q,0.0)):
        Z[jj, ii] += qq
    if log_scale:
        Z = np.log10(Z + 1.0)  # evitar log(0)
    x = np.linspace(0, 360, time_bins)    # phase
    y = np.linspace(0, 300, pixel_bins)   # pixel
    return x, y, Z

def roll_phase(phase, deg):
    out = phase + deg
    out[out>=360.0] -= 360.0
    out[out<0.0]    += 360.0
    return out

# =======================
# Dash App
# =======================
def run_dash(phase, pixel, Q):
    app = Dash(_name_)

    colorscales = ['Viridis', 'Plasma', 'Cividis', 'Inferno', 'Magma', 'Turbo', 'Rainbow', 'Ice', 'Earth']

    app.layout = html.Div([
        html.H4('PRPD 3D interactivo'),
        dcc.Graph(id="graph"),

        html.Div([
            html.Label("Filtro por amplitud (pixel):"),
            dcc.RangeSlider(
                id='range-slider',
                min=0, max=300, step=10,
                marks={0: '0', 300: '300'},
                value=[0, 300]
            ),
        ], style={'marginBottom':'16px'}),

        html.Div([
            html.Div([
                html.Label("Tipo de gráfico:"),
                dcc.Dropdown(
                    id='graph-type',
                    options=[
                        {'label': 'Scatter 3D', 'value': 'scatter'},
                        {'label': 'Surface (densidad)', 'value': 'surface'}
                    ],
                    value='scatter', clearable=False
                ),
            ], style={'width':'24%', 'display':'inline-block', 'marginRight':'14px'}),

            html.Div([
                html.Label("Colores (para 1 fase / scatter):"),
                dcc.Dropdown(
                    id='colorscale',
                    options=[{'label': c, 'value': c} for c in colorscales],
                    value='Viridis', clearable=False
                ),
            ], style={'width':'28%', 'display':'inline-block', 'marginRight':'14px'}),

            html.Div([
                html.Label("Fases:"),
                dcc.RadioItems(
                    id='phase-select',
                    options=[
                        {'label': 'A (0°)', 'value': 'A'},
                        {'label': 'B (+120°)', 'value': 'B'},
                        {'label': 'C (+240°)', 'value': 'C'},
                        {'label': 'Todas', 'value': 'ALL'}
                    ],
                    value='A',
                    inline=True
                ),
            ], style={'width':'30%', 'display':'inline-block', 'marginRight':'14px'}),

            html.Div([
                dcc.Checklist(
                    id='showscale',
                    options=[{'label': ' Mostrar barra de colores', 'value': 'on'}],
                    value=[]
                )
            ], style={'width':'16%', 'display':'inline-block'}),
        ], style={'marginBottom':'10px'}),

        html.Div([
            dcc.Checklist(
                id='logz',
                options=[{'label': ' Escala log10 en Z (solo Surface)', 'value': 'log'}],
                value=[]
            )
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Resolución (bins superficie): "),
            dcc.Input(id='time-bins', type='number', value=360, min=90, max=1440, step=30, debounce=True, style={'width':'110px', 'marginRight':'10px'}),
            dcc.Input(id='pixel-bins', type='number', value=301, min=101, max=1201, step=50, debounce=True, style={'width':'110px'}),
        ], style={'marginBottom':'16px'}),

        html.Div([
            html.Button("Exportar PNG", id="btn-export", n_clicks=0),
            dcc.Download(id="download-png")
        ]),
    ], style={'padding':'18px'})

    # --------- Figura interactiva ----------
    @app.callback(
        Output("graph", "figure"),
        Input("range-slider", "value"),
        Input("graph-type", "value"),
        Input("colorscale", "value"),
        Input("phase-select", "value"),
        Input("showscale", "value"),
        Input("logz", "value"),
        Input("time-bins", "value"),
        Input("pixel-bins", "value"))
    def update_chart(slider_range, graph_type, colorscale, phase_choice, showscale_list, logz, time_bins, pixel_bins):
        low, high = slider_range
        mask = (pixel >= low) & (pixel <= high)

        show_scale = 'on' in showscale_list
        log_scale = 'log' in logz
        tb = int(time_bins) if time_bins else 360
        pb = int(pixel_bins) if pixel_bins else 301

        palette_map_surface = {'Fase A': 'Viridis', 'Fase B': 'Plasma', 'Fase C': 'Turbo'}

        def make_surface(ph, name, cscale, showbar=False, colorbar_pos=None):
            x, y, Z = grid_density(ph[mask], pixel[mask], Q[mask], tb, pb, log_scale=log_scale)
            kwargs = dict(x=x, y=y, z=Z, colorscale=cscale, name=name, showscale=showbar)
            if showbar and colorbar_pos:
                kwargs['colorbar'] = dict(x=colorbar_pos, len=0.8)
            return go.Surface(**kwargs)

        def make_scatter(ph, name, cscale, showbar=False):
            return go.Scatter3d(
                x=ph[mask], y=pixel[mask], z=Q[mask],
                mode='markers',
                marker=dict(size=3, opacity=0.7, color=Q[mask],
                            colorscale=cscale, showscale=showbar),
                name=name
            )

        # Fases seleccionadas
        phases = []
        if phase_choice == 'A':
            phases = [(phase, 'Fase A')]
        elif phase_choice == 'B':
            phases = [(roll_phase(phase,120), 'Fase B')]
        elif phase_choice == 'C':
            phases = [(roll_phase(phase,240), 'Fase C')]
        else:
            phases = [
                (phase, 'Fase A'),
                (roll_phase(phase,120), 'Fase B'),
                (roll_phase(phase,240), 'Fase C')
            ]

        data = []
        if graph_type == 'scatter':
            # En scatter usamos una sola escala (dropdown) para coherencia
            for ph, name in phases:
                data.append(make_scatter(ph, name, colorscale, showbar=show_scale))
        else:
            # En surface: si son todas las fases, cada una con su paleta
            if phase_choice == 'ALL':
                # Para no saturar de barras, si show_scale=True mostramos SOLO la de Fase A (a la derecha)
                for idx, (ph, name) in enumerate(phases):
                    cscale = palette_map_surface[name]
                    showbar = (show_scale and idx == 0)
                    data.append(make_surface(ph, name, cscale, showbar=showbar, colorbar_pos=1.05 if showbar else None))
            else:
                # Una sola fase → usa el dropdown y respeta showscale
                ph, name = phases[0]
                data.append(make_surface(ph, name, colorscale, showbar=show_scale, colorbar_pos=1.05 if show_scale else None))

        fig = go.Figure(data=data)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0,360], title='Phase (°)'),
                yaxis=dict(range=[0,300], title='Amplitude (pixel)'),
                zaxis=dict(title='Q' if graph_type=='scatter' else ('Count (log10)' if log_scale else 'Count'))
            ),
            template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        return fig

    # --------- Exportar PNG ----------
    @app.callback(
        Output("download-png", "data"),
        Input("btn-export", "n_clicks"),
        State("graph", "figure"),
        prevent_initial_call=True
    )
    def export_png(n_clicks, figure_json):
        # Usa kaleido para exportar el estado actual de la figura
        if n_clicks:
            fig = go.Figure(figure_json)
            img_bytes = pio.to_image(fig, format="png", width=1400, height=900, scale=2)
            return dcc.send_bytes(lambda b: b.write(img_bytes), "PRPD_3D.png")

    app.run(debug=True)

# =======================
# Main
# =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('xml', help='ruta al XML')
    args = ap.parse_args()

    T,S,Q = parse_xml_all(args.xml)
    phase, pixel = normalize_to_phase_pixel(T,S)

    run_dash(phase, pixel, Q)

if _name_ == "_main_":
    main()
