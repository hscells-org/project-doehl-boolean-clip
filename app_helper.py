from dash import html, dcc

header = html.H1(
    "Embedding Visualization Dashboard",
    style={
        'textAlign': 'center',
        'marginBottom': '20px',
        'color': '#333'
    }
)

query_controls = html.Div(
    style={
        'display': 'flex',
        'gap': '10px',
        'justifyContent': 'center',
        'flexWrap': 'wrap',
        'marginBottom': '15px'
    },
    children=[
        dcc.Input(
            id='manual-query',
            placeholder="Type your query here...",
            style={
                'width': '60%',
                'padding': '8px',
                'fontSize': '16px',
                'border': '1px solid #ccc',
                'borderRadius': '6px'
            }
        ),
        dcc.Dropdown(
            id='query-dropdown',
            placeholder="Or select a query...",
            style={
                'width': '60%',
                'fontSize': '16px'
            }
        )
    ]
)

settings_controls = html.Div(
    style={
        'display': 'flex',
        'gap': '20px',
        'justifyContent': 'center',
        'flexWrap': 'wrap',
        'marginBottom': '20px'
    },
    children=[
        html.Div([
            html.Label("Top K matches", style={'fontWeight': 'bold'}),
            dcc.Input(
                id='top-k',
                type='number',
                min=1,
                max=1000,
                value=100,
                step=1,
                style={'width': '100px', 'textAlign': 'center'}
            )
        ]),
        html.Div([
            html.Label("Non-match opacity", style={'fontWeight': 'bold'}),
            html.Div(
                dcc.Slider(
                    id='non-match-opacity',
                    min=0,
                    max=0.2,
                    step=0.01,
                    value=0.05,
                    marks={0: '0', 0.5: '0.5', 1: '1'},
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
                style={'width': '200px', 'margin': '0 auto'}
            )
        ])
    ]
)

graph_section = dcc.Graph(
    id='embedding-graph',
    style={
        "height": "800px",
        'border': '1px solid #eee',
        'borderRadius': '8px',
        'padding': '5px',
        'backgroundColor': '#fafafa'
    }
)

layout = html.Div(
    style={
        'maxWidth': '1100px',
        'margin': '0 auto',
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif',
        'textAlign': 'center'
    },
    children=[
        header,
        query_controls,
        settings_controls,
        graph_section
    ]
)
