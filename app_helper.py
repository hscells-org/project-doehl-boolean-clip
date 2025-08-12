from dash import html, dcc

DEFAULT_CHAR_AMT = 50

header = html.H1(
    "Embedding Visualization",
    style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#333'}
)

query_controls = html.Div(
    style={'display': 'flex', 'gap': '10px', 'justifyContent': 'center',
           'flexWrap': 'wrap', 'marginBottom': '15px'},
    children=[
        dcc.Input(
            id='manual-query',
            placeholder="Type your query here...",
            debounce=1,
            style={'width': '60%', 'padding': '8px', 'fontSize': '16px',
                   'border': '1px solid #ccc', 'borderRadius': '6px'}
        ),
        dcc.Dropdown(
            id='query-dropdown',
            placeholder="Or select a query...",
            style={'width': '100%', 'fontSize': '16px'}
        )
    ]
)

settings_controls = html.Div(
    style={'display': 'flex', 'gap': '20px', 'justifyContent': 'center',
           'flexWrap': 'wrap', 'marginBottom': '20px'},
    children=[
        html.Div([
            html.Label("Top K matches", style={'fontWeight': 'bold'}),
            html.Div([
                dcc.Input(id='top-k', type='number', min=1, max=1000, value=100, step=1,
                        style={'width': '80px', 'textAlign': 'center'})
            ])
        ]),
        html.Div([
            html.Label("Default opacity", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='default-opacity', min=0, max=1, step=0.01, value=0.3,
                marks={0: '0', 0.5: '0.5', 1: '1'},
                tooltip={"placement": "bottom", "always_visible": False}
            )
        ], style={'width': '200px'}),
        html.Div([
            html.Label("Non-match opacity", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='non-match-opacity', min=0, max=0.2, step=0.01, value=0.05,
                marks={0: '0', 0.1: '0.1', 0.2: '0.2'},
                tooltip={"placement": "bottom", "always_visible": False}
            )
        ], style={'width': '200px'}),
        html.Div([
            html.Label("Dropoff strength", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='dropoff-strength',
                min=0, max=2, step=0.01, value=0.5,
                marks={0: '0', 0.5: '0.5', 1: '1', 2: '2'},
                tooltip={"placement": "bottom", "always_visible": False}
            )
        ], style={'width': '200px'}),
        html.Div([
            html.Label("Hover char amt.", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='char-amt', min=0, max=200, step=1, value=DEFAULT_CHAR_AMT,
                marks={0: '0', 100: '100', 200: '200'},
                tooltip={"placement": "bottom", "always_visible": False}
            )
        ], style={'width': '200px'}),
    ]
)

graph_section = dcc.Graph(
    id='embedding-graph',
    style={"height": "800px", 'border': '1px solid #eee',
           'borderRadius': '8px', 'padding': '5px', 'backgroundColor': '#fafafa'}
)

layout = html.Div(
    style={'maxWidth': '1100px', 'margin': '0 auto', 'padding': '20px',
           'fontFamily': 'Arial, sans-serif', 'textAlign': 'center'},
    children=[header, query_controls, settings_controls, graph_section]
)
