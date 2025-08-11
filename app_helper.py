from dash import html, dcc

layout = html.Div(
    style={
        'maxWidth': '1100px',
        'margin': '0 auto',
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif',
        'textAlign': 'center'
    },
    children=[
        html.H1(
            "Embedding Visualization Dashboard",
            style={
                'textAlign': 'center',
                'marginBottom': '20px',
                'color': '#333'
            }
        ),
        html.Div(
            style={
                'display': 'flex',
                'gap': '10px',
                'justifyContent': 'center',
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
                        'width': '100%',
                        'fontSize': '16px'
                    }
                )
            ]
        ),
        dcc.Graph(
            id='embedding-graph',
            style={
                "height": "800px",
                'border': '1px solid #eee',
                'borderRadius': '8px',
                'padding': '5px',
                'backgroundColor': '#fafafa'
            }
        )
    ]
)
