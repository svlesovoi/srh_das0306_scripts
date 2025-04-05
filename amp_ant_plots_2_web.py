import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import matplotlib
import pylab as PL
import datetime

from astropy.io import fits
import numpy as NP

fig = go.Figure()
app = dash.Dash(__name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"])
app.layout = html.Div([
#    html.H1("SRH 0306 antennas"),
    dcc.Graph(id="graph", figure = fig, style={'width': '90vw', 'height': '80vh'}),
    dcc.Slider(0,15,1,value=0, id='frequency_slider', marks={i: 'f{}'.format(i+1) for i in range(16)}),
    dcc.Interval(id='refresh_timer', interval=10000, n_intervals=0)
    ])    
             
@callback(
    Output('graph', 'figure'),
    [Input('frequency_slider', 'value'), Input('refresh_timer', 'n_intervals')]
    )

def update_figure(selected_frequency, n_intervals):
    dateName = datetime.datetime.now().strftime("%Y%m%d")
    ant_plot_fits = fits.open('SRH0306/antPlots/srh_0306_ant_' + dateName + '.fits')
    ant_time = ant_plot_fits[2].data['time']
    t = ant_time[0,-1]
    hh = int(t // 3600)
    t -= hh*3600
    mm = int(t // 60)
    t -= mm*60
    ss = int(t)
    ant_time_format = datetime.datetime(datetime.datetime.now().year,datetime.datetime.now().month,datetime.datetime.now().day,hh,mm,ss)
    ant_names = ant_plot_fits[6].data['antenna_names']
    dig_rec_number = 8
    
    ant_lcp_data = NP.zeros(128)
    fig = go.Figure()
    data_ID = 'ant_LCP_%02d'%selected_frequency
    ant_lcp_data = ant_plot_fits[5].data[data_ID][-1]
    label = '{}, {:.2f} GHz'.format(ant_time_format, ant_plot_fits[1].data['frequencies'][selected_frequency]*1e-6)
    fig.add_trace(go.Scatter(y=ant_lcp_data, x=ant_names, line_color='blue', text=label))
    
    title_text = 'SRH 0306, ' + label
    fig.update_layout(transition_duration=500, plot_bgcolor='black', title_text=title_text)
    fig.update_xaxes(nticks=8, gridcolor='gray')
    fig.update_yaxes(gridcolor='gray')
    for dig_rec_idx in range(dig_rec_number):
        fig.add_annotation(x=8 + 16 * dig_rec_idx,y=0,text='DR%02d'%(dig_rec_idx+1),showarrow=False, bgcolor='gray', font=dict(size=18, color='white'))
    
    ant_plot_fits.close()

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8056, host='0.0.0.0')
