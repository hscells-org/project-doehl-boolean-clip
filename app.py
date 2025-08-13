from dash import Dash, callback, Output, Input, Patch
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import torch
import umap

import app_helper
from utils.boolrank import DualSiglip2Model

in_key = "nl_query"
out_key = "bool_query"
N = 1000

paths = [
    "data/training.jsonl",
    "data/TAR_data.jsonl",
    "data/sysrev_conv.jsonl",
]

model = DualSiglip2Model('BAAI/bge-small-en-v1.5')
model.load(r"models\\clip\\bge-small-en-v1.5\\b16_lr1E-05_(pubmed-que_pubmed-sea_raw-jsonl)^4\\checkpoint-11288\\model.safetensors")

# Embedding script check for model and data path
dataset = pd.concat([pd.read_json(p, lines=True) for p in paths])
dataset = dataset[dataset[in_key] != ""]
df = dataset.sample(min(N, dataset.shape[0]), random_state=0).reset_index(drop=True)

print("Calculating embeddings")
embeddings = model.encode_bool(df[out_key].tolist(), batch_size=200, verbose=True).detach().cpu().numpy()
torch.cuda.empty_cache()

print("Calculating UMAP")
um_reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=0)
trans = um_reducer.fit_transform(embeddings)
df["x"], df["y"] = trans[:, 0], trans[:, 1]

def cutoffl(cut): return lambda x: x if len(x) < cut else x[:cut] + "..."
def build_base_figure(default_opacity):
    fig = go.Figure()
    df["data_in"] = df[in_key].map(cutoffl(app_helper.DEFAULT_CHAR_AMT))
    df["data_out"] = df[out_key].map(cutoffl(app_helper.DEFAULT_CHAR_AMT))

    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(opacity=default_opacity),
        hovertemplate="<b>Data in:</b> %{customdata[0]}<br>"
                        "<b>Data out:</b> %{customdata[1]}<br>",
        customdata=np.stack((df["data_in"],df["data_out"]), axis=-1)
    ))
    fig.update_layout(width=1100, height=800, legend_title_text="Source")
    return fig

base_fig = build_base_figure(default_opacity=0.3)

app = Dash("Visualizer", title="Embedding Visualizer")
app.layout = app_helper.layout

@callback(
    Output('query-dropdown', 'options'),
    Input('query-dropdown', 'value')
)
def update_dropdown(_):
    return [{"label": q, "value": q} for q in dataset[:50][in_key]]

@callback(
    Output('embedding-graph', 'figure'),
    Input('embedding-graph', 'id'),
    prevent_initial_call='initial_duplicate'
)
def init_figure(_):
    return base_fig

last_manual_query = None
last_dropdown_query = None
last_query = None
similarities = None
@callback(
    Output('embedding-graph', 'figure', allow_duplicate=True),
    [
        Input('manual-query', 'value'),
        Input('query-dropdown', 'value'),
        Input('top-k', 'value'),
        Input('non-match-opacity', 'value'),
        Input('default-opacity', 'value'),
        Input('dropoff-strength', 'value'),
        Input('char-amt', 'value'),
    ],
    prevent_initial_call=True
)
def update_figure(manual_query, dropdown_query, topk, nonmatch_opacity, default_opacity, dropoff_strength, char_amt):
    global last_manual_query, last_dropdown_query, last_query, similarities

    query = None
    if manual_query != last_manual_query and manual_query and manual_query.strip():
        query = manual_query.strip()
        last_manual_query = query
    elif dropdown_query != last_dropdown_query:
        query = dropdown_query
        last_dropdown_query = query
    else:
        query = last_query

    if query:
        if query != last_query:
            query_emb = model.encode_text(query).detach().cpu().numpy()
            similarity = model.get_similarities(embeddings, query_emb).numpy()
            similarities = (-similarity).argsort()
        mask = np.full_like(similarities, nonmatch_opacity, dtype=float)
        top_n = similarities[:topk]
        ranks = np.arange(len(top_n))
        opacities = np.exp(-dropoff_strength * ranks / 10)
        opacities = opacities.round(2)
        mask[top_n] = np.clip(opacities, nonmatch_opacity, 1)
    else:
        mask = np.full(len(df), default_opacity)

    last_query = query

    patch = Patch()
    patch["data"][0]["marker"]["opacity"] = mask


    df["data_in"] = df[in_key].map(cutoffl(char_amt))
    df["data_out"] = df[out_key].map(cutoffl(char_amt))
    customdata = np.stack((df["data_in"], df["data_out"]), axis=-1)
    patch["data"][0]["customdata"] = customdata
    return patch

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
