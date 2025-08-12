import pandas as pd
import torch
import umap
from dash import Dash, callback, Output, Input
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

import app_helper
from utils.boolrank import DualSiglip2Model

in_key = "nl_query"
out_key = "bool_query"
N = 1000

seed = 0
paths = [
    "data/training.jsonl",
    "data/TAR_data.jsonl",
    "data/sysrev_conv.jsonl",
]
model = DualSiglip2Model('BAAI/bge-small-en-v1.5')
model.load(r"models\\clip\\bge-small-en-v1.5\\b16_lr1E-05_(pubmed-que_pubmed-sea_raw-jsonl)^4\\checkpoint-11288\\model.safetensors")

dataset = []

for path in paths:
    dataset.append(pd.read_json(path, lines=True))
dataset = pd.concat(dataset)
dataset = dataset[dataset["nl_query"] != ""]

df = dataset.sample(min(N, dataset.shape[0]), random_state=0).reset_index(drop=True)

print("Calculating embeddings")
embeddings = model.encode_bool(df[out_key].tolist(), batch_size=200).detach().cpu().numpy()
torch.cuda.empty_cache()

print("Calculating UMAP")
um_reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=0)
trans = um_reducer.fit_transform(embeddings)
df["x"], df["y"] = trans[:, 0], trans[:, 1]

unique_sources = df['source'].unique()
color_map = {src: px.colors.qualitative.Plotly[i % 10] for i, src in enumerate(unique_sources)}

app = Dash("Visualizer", title="Embedding Visualizer")
app.layout = app_helper.layout

@callback(
    Output('query-dropdown', 'options'),
    Input('query-dropdown', 'value')
)
def update_dropdown(_):
    return [{"label": q, "value": q} for q in dataset[:50][in_key]]

last_manual_query = None
last_dropdown_query = None
last_query = None
@callback(
    Output('embedding-graph', 'figure'),
    [
        Input('manual-query', 'value'),
        Input('query-dropdown', 'value'),
        Input('top-k', 'value'),
        Input('non-match-opacity', 'value'),
        Input('default-opacity', 'value'),
        Input('char-amt', 'value'),
        Input('dropoff-strength', 'value')
    ]
)
def update_figure(manual_query, dropdown_query, topk, nonmatch_opacity, default_opacity, char_amt, dropoff_strength):
    global last_dropdown_query, last_manual_query, last_query
    query = None
    if manual_query != last_manual_query and manual_query.strip():
        query = manual_query.strip()
        last_manual_query = query
    elif dropdown_query != last_dropdown_query:
        query = dropdown_query
        last_dropdown_query = query
    else:
        query = last_query
    last_query = query
    if query == "":
        query = None

    if not query:
        mask = np.full(len(df), default_opacity)
    else:
        query_emb = model.encode_text(query).detach().cpu().numpy()
        similarity = model.get_similarities(embeddings, query_emb).numpy()
        mask = np.full_like(similarity, nonmatch_opacity)
        top_n = (-similarity).argsort()[:topk]

        if dropoff_strength > 0:
            ranks = np.arange(len(top_n))
            opacities = 1 * np.exp(-dropoff_strength * ranks / topk)
        else:
            opacities = np.full(len(top_n), 1)

        mask[top_n] = opacities

    df["data_in"] = df[in_key].map(lambda x: x if len(x) < char_amt else x[:char_amt] + "...")
    df["data_out"] = df[out_key].map(lambda x: x if len(x) < char_amt else x[:char_amt] + "...")

    fig = go.Figure()
    for src in df['source'].unique():
        src_mask = df['source'] == src
        fig.add_trace(go.Scatter(
            x=df.loc[src_mask, 'x'],
            y=df.loc[src_mask, 'y'],
            mode='markers',
            name=str(src),
            marker=dict(
                color=color_map[src],
                opacity=mask[src_mask]
            ),
            hovertemplate=(
                "<b>Natural Query:</b> %{customdata[0]}<br>"
                "<b>Boolean Query:</b> %{customdata[1]}<br>"
                "<b>Source:</b> %{customdata[2]}<extra></extra>"
            ),
            customdata=np.stack((
                df.loc[src_mask, "data_in"],
                df.loc[src_mask, "data_out"],
                df.loc[src_mask, "source"]
            ), axis=-1)
        ))

    fig.update_layout(
        width=1100,
        height=800,
        legend_title_text="Source"
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)