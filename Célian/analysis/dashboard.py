
#%%
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Exemple de données
df_rent = pd.read_csv('../../data/rent.csv', sep=';', encoding='utf-8')
df_rent = df_rent[df_rent['surface'].notna()]

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Carte des Locations à Paris"),
    dcc.RangeSlider(
        id='surface-slider',
        min=df_rent['surface'].min(),
        max=df_rent['surface'].max(),
        step=5,
        value=[df_rent['surface'].min(), df_rent['surface'].max()]
    ),
    dcc.Graph(id='map-graph')
])

@app.callback(
    Output('map-graph', 'figure'),
    Input('surface-slider', 'value')
)
def update_map(surface_range):
    filtered_df = df_rent[(df_rent['surface'] >= surface_range[0]) & (df_rent['surface'] <= surface_range[1])]
    fig = px.scatter_map(
        filtered_df,
        lat='latitude',
        lon='longitude',
        color='price',
        color_continuous_scale='Turbo',
        size_max=15,
        zoom=12
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)

# %%
