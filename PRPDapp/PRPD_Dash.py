import argparse, xml.etree.ElementTree as ET
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.io as pio

PERIOD_MS = 1000/60  # 16.667 ms @60Hz

# ------------------ Parseo XML ------------------
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
        t = to_nums(t_text); s = to_nums(s_text)
        q = to_nums(q_text) if q_text is not None else [1.0]*len(s)
        n = min(len(t), len(s), len(q))
        if n>0:
            T.extend(t[:n]); S.extend(s[:n]); Q.extend(q[:n])
    return np.array(T), np.array(S), np.array(Q)

def normalize_to_phase_pixel(T, S):
    Tms = T*1000.0 if (len(T)>0 and np.nanmax(T)<=1.0) else T.copy()
    t_env = Tms - PERIOD_MS*np.floor(Tms/PERIOD_MS)
    phase = (t_env/PERIOD_MS)*360.0            # X
    smin, smax = np.nanmin(S), np.nanmax(S)
    if smax==smin: smax = smin + 1.0
    pixel = np.clip(np.rint((S - smin)*300.0/(smax - smin)), 0, 300)  # Y
    return phase, pixel

# ------------------ Agregación (tu lógica original) ------------------
def grid_density(phase, pixel, Q, time_bins=360, pixel_bins=301, log_scale=False):
    # Z se construye sumando Q por celda (tu comportamiento original)
    Z = np.zeros((pixel_bins, time_bins), dtype=float)
    i = np.clip(np.rint(phase*(time_bins-1)/360.0).astype(int), 0, time_bins-1)
    j = np.clip(pixel.astype(int), 0, pixel_bins-1)
    for ii, jj, qq in zip(i, j, np.maximum(Q,0.0)):
        Z[jj, ii] += qq
    if log_scale:
        Z = np.log10(Z + 1.0)
    x = np.linspace(0, 360, time_bins)   # Phase (X)
    y = np.linspace(0, 300, pixel_bins)  # Amplitude (Y)
    return x, y, Z

def bin_counts(phase, pixel, Q, time_bins=90, pixel_bins=90, log_scale=False):
    x, y, Z = grid_density(phase, pixel, Q, time_bins, pixel_bins, log_scale)
    X, Y = np.meshgrid(x, y)  # X ↔ Phase, Y ↔ Amplitude
    return X, Y, Z

def roll_phase(phase, deg):
    out = phase + deg
    out[out>=360.0] -= 360.0
    out[out<0.0]    += 360.0
    return out

# ------------------ Dash ------------------
def run_dash(phase, pixel, Q, port=8050):
    app = Dash(__name__)
    colorscales = ['Viridis','Plasma','Cividis','Inferno','Magma','Turbo','Rainbow','Ice','Earth']
    marker_symbols = ['circle','square','diamond','cross','x','triangle-up','triangle-down']

    # Rango Q para slider
    qmin = float(np.nanmin(Q)) if len(Q) else 0.0
    qmax = float(np.nanmax(Q)) if len(Q) else 1.0
    qlo  = float(np.percentile(Q, 1)) if len(Q) else qmin
    qhi  = float(np.percentile(Q, 99)) if len(Q) else qmax

    app.layout = html.Div([
        html.H3('PRPD 3D interactivo'),

        # Gráfico grande y manipulable (sin tocar ejes)
        dcc.Graph(
            id="graph",
            style={"height": "82vh", "width": "100%"},
            config={
                "displayModeBar": True,
                "scrollZoom": True,
                "doubleClick": "autosize",
                "responsive": True
            }
        ),

        # Controles
        html.Div([
            html.Div([
                html.Label("Amplitud (pixel)"),
                dcc.RangeSlider(id='range-pixel', min=0, max=300, step=5,
                                marks={0:'0',300:'300'}, value=[0,300]),
            ], style={'flex':'2','minWidth':'260px','marginRight':'14px'}),

            html.Div([
                html.Label("Intensidad Q"),
                dcc.RangeSlider(id='range-q', min=qmin, max=qmax,
                                step=(qmax-qmin)/200 if qmax>qmin else 1,
                                value=[qlo, qhi]),
            ], style={'flex':'2','minWidth':'260px','marginRight':'14px'}),

            html.Div([
                html.Label("Tipo de gráfico"),
                dcc.Dropdown(id='plot-kind',
                    options=[
                        {'label':'Scatter 3D (color por Q)','value':'scatter_q'},
                        {'label':'Scatter 3D (color por Fase)','value':'scatter_phase'},
                        {'label':'Scatter 3D (conteo binned)','value':'scatter_count'},
                        {'label':'Barras 3D (conteo binned)','value':'bars_count'},
                        {'label':'Barras Mesh3D (cubos)','value':'bars_mesh'},
                        {'label':'Surface (1 fase)','value':'surface_1'},
                        {'label':'Surface (3 fases)','value':'surface_3'},
                    ],
                    value='surface_1', clearable=False),
            ], style={'flex':'2','minWidth':'260px','marginRight':'14px'}),

            html.Div([
                html.Label("Tema"),
                dcc.RadioItems(id='theme',
                    options=[{'label':'Oscuro','value':'dark'},
                             {'label':'Blanco','value':'light'}],
                    value='light', inline=True),
            ], style={'flex':'1','minWidth':'170px','marginRight':'14px'}),

            html.Div([
                html.Label("Paleta de color"),
                dcc.Dropdown(id='colorscale',
                    options=[{'label':c,'value':c} for c in colorscales],
                    value='Turbo', clearable=False),
                dcc.Checklist(id='showscale',
                    options=[{'label':' Mostrar barra de colores','value':'on'}],
                    value=['on'])
            ], style={'flex':'2','minWidth':'220px'}),

            # >>> NUEVO: símbolo para los modos de dispersión <<<
            html.Div([
                html.Label("Símbolo (solo scatter)"),
                dcc.Dropdown(
                    id='marker-symbol',
                    options=[{'label':s, 'value':s} for s in marker_symbols],
                    value='circle', clearable=False
                )
            ], style={'flex':'1','minWidth':'200px','marginLeft':'14px'}),
        ], style={'display':'flex','flexWrap':'wrap','marginBottom':'10px'}),

        # Exportar
        html.Div([
            html.Button("Exportar PNG", id="btn-export", n_clicks=0),
            dcc.Download(id="download-png")
        ]),
    ],
    style={"padding":"16px", "maxWidth":"1600px", "margin":"0 auto", "minHeight":"100vh"}
    )

    @app.callback(
        Output("graph","figure"),
        Input("range-pixel","value"),
        Input("range-q","value"),
        Input("plot-kind","value"),
        Input("theme","value"),
        Input("colorscale","value"),
        Input("showscale","value"),
        Input("marker-symbol","value"),   # << nuevo input
        prevent_initial_call=False
    )
    def update_fig(rpixel, rq, kind, theme, cscale, showscale_list, msym):
        low_px, high_px = rpixel
        low_q,  high_q  = rq
        mask = (pixel >= low_px) & (pixel <= high_px) & (Q >= low_q) & (Q <= high_q)

        show_scale = ('on' in showscale_list)
        fig = go.Figure()

        # ---- Scatter (raw) ----
        if kind == 'scatter_q':
            fig.add_trace(go.Scatter3d(
                x=phase[mask], y=pixel[mask], z=Q[mask],
                mode='markers',
                marker=dict(size=3, symbol=msym, opacity=0.75,
                            color=Q[mask], colorscale=cscale, showscale=show_scale),
                name="Pulsos"
            ))
        elif kind == 'scatter_phase':
            fig.add_trace(go.Scatter3d(
                x=phase[mask], y=pixel[mask], z=Q[mask],
                mode='markers',
                marker=dict(size=3, symbol=msym, opacity=0.75,
                            color=phase[mask], colorscale=cscale, showscale=show_scale),
                name="Fase"
            ))

        # ---- Binned como puntos ----
        elif kind == 'scatter_count':
            X, Y, Z = bin_counts(phase[mask], pixel[mask], Q[mask])
            m = Z > 0
            fig.add_trace(go.Scatter3d(
                x=X[m], y=Y[m], z=Z[m],
                mode='markers',
                marker=dict(size=3, symbol=msym, opacity=0.85,
                            color=Z[m], colorscale=cscale, showscale=show_scale),
                name="Binned"
            ))

        # ---- Barras / Mesh (sin cambios de símbolo) ----
        elif kind == 'bars_count':
            X, Y, Z = bin_counts(phase[mask], pixel[mask], Q[mask])
            m = Z > 0
            fig.add_trace(go.Scatter3d(
                x=X[m], y=Y[m], z=Z[m]/2.0,
                mode='markers',
                marker=dict(size=8, symbol='square', opacity=0.6,
                            color=Z[m], colorscale=cscale, showscale=show_scale),
                name="Barras"
            ))

        elif kind == 'bars_mesh':
            X, Y, Z = bin_counts(phase[mask], pixel[mask], Q[mask], time_bins=30, pixel_bins=30)
            m = Z > 0
            for xi, yi, zi in zip(X[m].flatten(), Y[m].flatten(), Z[m].flatten()):
                fig.add_trace(go.Mesh3d(
                    x=[xi, xi+5, xi+5, xi, xi, xi+5, xi+5, xi],
                    y=[yi, yi, yi+5, yi+5, yi, yi, yi+5, yi+5],
                    z=[0, 0, 0, 0, zi, zi, zi, zi],
                    color='blue', opacity=0.30, name="Cubo", showscale=False
                ))

        elif kind == 'surface_1':
            x,y,Z = grid_density(phase[mask], pixel[mask], Q[mask])
            fig.add_trace(go.Surface(
                x=x, y=y, z=Z,
                colorscale=cscale, showscale=show_scale, name="Surface"
            ))

        else:  # surface_3
            for shift, name in [(0,'A'), (120,'B'), (240,'C')]:
                ph = roll_phase(phase, shift)
                x,y,Z = grid_density(ph[mask], pixel[mask], Q[mask])
                fig.add_trace(go.Surface(
                    x=x, y=y, z=Z, colorscale=cscale, showscale=show_scale, name=f'Fase {name}'
                ))

        template = 'plotly_dark' if theme=='dark' else 'plotly'
        ztitle = 'Q' if kind.startswith('scatter') else 'Count'
        fig.update_layout(
            template=template,
            uirevision="keep",
            margin=dict(l=0, r=0, t=40, b=0),
            autosize=True,
            scene=dict(
                dragmode="orbit",
                xaxis=dict(title='Phase (°)',     range=[0, 360]),
                yaxis=dict(title='Amplitude (pixel)', range=[0, 300]),
                zaxis=dict(title=ztitle),
                aspectmode="manual",
                aspectratio=dict(x=1.2, y=1.1, z=0.7)
            ),
            scene_camera=dict(eye=dict(x=1.25, y=1.45, z=1.05))
        )
        return fig

    @app.callback(
        Output("download-png","data"),
        Input("btn-export","n_clicks"),
        State("graph","figure"),
        prevent_initial_call=True
    )
    def export_png(n, fig_json):
        if n:
            fig = go.Figure(fig_json)
            img = pio.to_image(fig, format="png", width=1600, height=900, scale=2)
            return dcc.send_bytes(lambda b: b.write(img), "PRPD_3D.png")

    # Evitar reloader/debug para que no se lance dos veces (y no choque el puerto).
    app.run(debug=False, port=port, host="127.0.0.1", use_reloader=False)

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('xml', help='ruta al XML')
    ap.add_argument('--port', type=int, default=8050)
    args = ap.parse_args()
    T,S,Q = parse_xml_all(args.xml)
    phase, pixel = normalize_to_phase_pixel(T,S)
    run_dash(phase, pixel, Q, port=args.port)

if __name__ == "__main__":
    main()
