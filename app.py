import pandas as pd
import torch
import umap
from dash import Dash, callback, Output, Input
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

import app_helper
from utils.boolrank import DualSiglip2Model

df = None
dataset = None

paths = [
    "data/training.jsonl",
    "data/TAR_data.jsonl",
    "data/sysrev_conv.jsonl",
]
model = DualSiglip2Model('BAAI/bge-small-en-v1.5')
model.load(r"models\\clip\\bge-small-en-v1.5\\b16_lr1E-05_(pubmed-que_pubmed-sea_raw-jsonl)^4\\checkpoint-11288\\model.safetensors")

dataset = []
for path in paths:
    df = pd.read_json(path, lines=True)
    dataset.append(df)
dataset = pd.concat(dataset)
dataset = dataset[dataset["nl_query"] != ""]

N = 1000
df = dataset.sample(min(N, dataset.shape[0])).reset_index(drop=True)

print("Calculating embeddings")
embeddings = model.encode_bool(df["bool_query"].tolist(), batch_size=200).detach().cpu().numpy()
torch.cuda.empty_cache()

print("Calculating UMAP")
um_reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=0)
trans = um_reducer.fit_transform(embeddings)
df["x"], df["y"] = trans[:, 0], trans[:, 1]

cut = 60
df["nl"] = df["nl_query"].map(lambda x: x if len(x) < cut else x[:cut] + "...")
df["bool"] = df["bool_query"].map(lambda x: x if len(x) < cut else x[:cut] + "...")

unique_sources = df['source'].unique()
color_map = {src: px.colors.qualitative.Plotly[i % 10] for i, src in enumerate(unique_sources)}

app = Dash(__name__)
app.layout = app_helper.layout

@callback(
    Output('query-dropdown', 'options'),
    Input('query-dropdown', 'value')
)
def update_dropdown(_):
    return [{"label": q, "value": q} for q in dataset[:50]["nl_query"]]

@callback(
    Output('embedding-graph', 'figure'),
    [
        Input('manual-query', 'value'),
        Input('query-dropdown', 'value'),
        Input('top-k', 'value'),
        Input('non-match-opacity', 'value')
    ]
)
def update_figure(manual_query, dropdown_query, topk, nonmatch_opacity):
    query = None
    if manual_query and manual_query.strip():
        query = manual_query.strip()
    elif dropdown_query:
        query = dropdown_query
    if query == "":
        query = None

    if not query:
        fig = go.Figure(go.Scatter(
            x=df.x,
            y=df.y,
            mode='markers',
            marker=dict(opacity=0.3),
            hovertext=df["bool"]
        ))
        fig.update_layout(width=1100, height=800)
        return fig

    query_emb = model.encode_text(query).detach().cpu().numpy()
    similarity = model.get_similarities(embeddings, query_emb).numpy()

    mask = np.full_like(similarity, nonmatch_opacity)
    top_n = (-similarity).argsort()[:topk]
    mask[top_n] = 0.9

    fig = go.Figure(go.Scatter(
        x=df.x,
        y=df.y,
        mode='markers',
        marker=dict(opacity=mask),
        hovertext=df["bool"]
    ))
    fig.update_layout(width=1100, height=800)
    return fig

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)