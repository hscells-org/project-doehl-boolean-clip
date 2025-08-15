from dash import html, dcc
import os
import hashlib
import pickle
import pandas as pd
import torch

DEFAULT_CHAR_AMT = 50

def get_cache_path(model_name, paths):
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for p in paths:
        h.update(str(os.path.getmtime(p)).encode("utf-8"))
        h.update(str(os.path.getsize(p)).encode("utf-8"))
    return os.path.join("cache", f"embeddings_{h.hexdigest()[:16]}.pkl")

def load_or_create_embeddings(model, data_paths, in_key, out_key, data_amount=None):
    os.makedirs("cache", exist_ok=True)
    cache_file = get_cache_path(model.model_name, data_paths)

    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, "rb") as f:
            df, embeddings = pickle.load(f)
    else:
        print("Loading dataset and computing embeddings...")
        dataset = pd.concat([pd.read_json(p, lines=True) for p in data_paths])
        df = dataset[dataset[in_key] != ""]
        if data_amount is not None:
            df = df.sample(min(data_amount, df.shape[0]), random_state=0).reset_index(drop=True)

        embeddings = model.encode_bool(df[out_key].tolist(), batch_size=200, verbose=True).detach().cpu().numpy()
        torch.cuda.empty_cache()

        with open(cache_file, "wb") as f:
            pickle.dump((df, embeddings), f)

    return df, embeddings

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
