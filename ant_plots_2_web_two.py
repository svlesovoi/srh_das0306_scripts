import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import matplotlib
import pylab as PL
import datetime

from astropy.io import fits
import numpy as NP

fig = go.Figure()

app = dash.Dash(__name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"])
app.layout = html.Div([
    html.Div([
    html.H1("SRH 0306 antennas"),
    dcc.Graph(id="graph", figure = figure, style={'width': '90vw', 'height': '45vh'}),
    dcc.Slider(0,7,1,value=0, id='dr_slider', marks={i: 'DR{}'.format(i+1) for i in range(8)}),
    dcc.Interval(id='refresh_timer', interval=10000, n_intervals=0)
    ]),

    html.Div([
    html.H1("SRH 0306 amplitudes"),
    dcc.Graph(id="amp_ant_plot", figure = fig, style={'width': '90vw', 'height': '40vh'}),
    dcc.Interval(id='refresh_timer', interval=10000, n_intervals=0)
    ])
])    
             
@callback(
    Output('graph', 'figure'),
    [Input('dr_slider', 'value'), Input('refresh_timer', 'n_intervals')]
    )

def update_figure(selected_dr, n_intervals):
    c_list = matplotlib.colors.LinearSegmentedColormap.from_list(PL.cm.datad['gist_rainbow'], colors=['r','g','b'], N = 16)
    dateName = datetime.datetime.now().strftime("%Y%m%d")
    ant_plot_fits = fits.open('SRH0306/antPlots/srh_0306_ant_' + dateName + '.fits')
    ant_time = ant_plot_fits[2].data['time']
    ant_time_format = []
    for tt in range(ant_time.shape[1]):
        t = ant_time[0,tt]
        hh = int(t // 3600)
        t -= hh*3600
        mm = int(t // 60)
        t -= mm*60
        ss = int(t)
        ant_time_format.append(datetime.datetime(datetime.datetime.now().year,datetime.datetime.now().month,datetime.datetime.now().day,hh,mm,ss))
        
    ant_names = ant_plot_fits[6].data['antenna_names']
    
    ant_lcp_data = NP.zeros((ant_time.shape[1], 16))
    fig = go.Figure()
    for ant in range(selected_dr*16,(selected_dr + 1)*16):
        dr_ant = ant - selected_dr*16
        for ff in range(16):
            data_ID = 'ant_LCP_%02d'%ff
            ant_lcp_data[:,dr_ant] += ant_plot_fits[5].data[data_ID][:,ant]
        cc = c_list(dr_ant)
        ant_data = ant_lcp_data[:,dr_ant] / 16
        fig.add_trace(go.Scatter(x=ant_time_format,y=ant_data-ant_data.mean() + 3e3*dr_ant, name=ant_names[ant], line_color='rgb(%d,%d,%d)'%(255*cc[0],255*cc[1],255*cc[2])))
    
    fig.update_layout(transition_duration=500, plot_bgcolor='black')
    
    ant_plot_fits.close()

    return fig

@callback(
    Output('amp_ant_plot', 'fig_amp_ant_plot'),
    [Input(Input('refresh_timer', 'n_intervals')]
    )

def update_figure(selected_dr, n_intervals):
    c_list = matplotlib.colors.LinearSegmentedColormap.from_list(PL.cm.datad['gist_rainbow'], colors=['r','g','b'], N = 16)
    dateName = datetime.datetime.now().strftime("%Y%m%d")
    ant_plot_fits = fits.open('SRH0306/antPlots/srh_0306_ant_' + dateName + '.fits')
    ant_time = ant_plot_fits[2].data['time']
    ant_time_format = []
    for tt in range(ant_time.shape[1]):
        t = ant_time[0,tt]
        hh = int(t // 3600)
        t -= hh*3600
        mm = int(t // 60)
        t -= mm*60
        ss = int(t)
        ant_time_format.append(datetime.datetime(datetime.datetime.now().year,datetime.datetime.now().month,datetime.datetime.now().day,hh,mm,ss))
        
    ant_names = ant_plot_fits[6].data['antenna_names']
    
    ant_lcp_data = NP.zeros((ant_names.shape[0]))
    fig = go.Figure()
    for ant in range(selected_dr*16,(selected_dr + 1)*16):
        dr_ant = ant - selected_dr*16
        for ff in range(16):
            data_ID = 'ant_LCP_%02d'%ff
            ant_lcp_data[:,dr_ant] += ant_plot_fits[5].data[data_ID][:,ant]
        cc = c_list(dr_ant)
        ant_data = ant_lcp_data[:,dr_ant] / 16
        fig.add_trace(go.Scatter(x=ant_time_format,y=ant_data-ant_data.mean(), name=ant_names[ant], line_color='rgb(255,0,0)')
    
    fig.update_layout(transition_duration=500, plot_bgcolor='black')
    
    ant_plot_fits.close()

    return fig
if __name__ == '__main__':
    app.run_server(debug=True, port=8055, host='0.0.0.0')
