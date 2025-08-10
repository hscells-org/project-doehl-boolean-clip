import pandas as pd
import torch
import umap
from dash import Dash, html, dcc, callback, Output, Input
from boolrank import DualSiglip2Model
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

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
um_reducer = umap.UMAP(n_neighbors=15, n_components=2)
trans = um_reducer.fit_transform(embeddings)
df["x"], df["y"] = trans[:, 0], trans[:, 1]

cut = 60
df["nl"] = df["nl_query"].map(lambda x: x if len(x) < cut else x[:cut] + "...")
df["bool"] = df["bool_query"].map(lambda x: x if len(x) < cut else x[:cut] + "...")

app = Dash(__name__)
@callback(
    Output('query-dropdown', 'options'),
    Input('query-dropdown', 'value')
)
def update_dropdown(_):
    return [{"label": q, "value": q} for q in dataset[:50]["nl_query"]]

unique_sources = df['source'].unique()
color_map = {src: px.colors.qualitative.Plotly[i % 10] for i, src in enumerate(unique_sources)}

app.layout = html.Div([
    html.H1("Embedding Visualization Dashboard", style={'textAlign': 'center'}),
    dcc.Input(id='manual-query', placeholder="Type your query here...", style={'width': '60%', 'marginBottom': '10px'}),
    dcc.Dropdown(id='query-dropdown', placeholder="Or select a query..."),
    dcc.Graph(id='embedding-graph', style={"height": "800px"})
])


@app.callback(
    Output('embedding-graph', 'figure'),
    [Input('manual-query', 'value'),
     Input('query-dropdown', 'value')]
)
def update_figure(manual_query, dropdown_query):
    query = None
    if manual_query and manual_query.strip():
        query = manual_query.strip()
    elif dropdown_query:
        query = dropdown_query
    if query == "": query = None

    if not query:
        fig = go.Figure(go.Scatter(
            x=df.x,
            y=df.y,
            mode='markers',
            marker=dict(opacity=0.3),
            hovertext=df["bool"]
        ))
        fig.update_layout(width=1000, height=700)
        return fig

    query_emb = model.encode_text(query).detach().cpu().numpy()
    similarity = model.get_similarities(embeddings, query_emb).numpy()

    mask = np.full_like(similarity, 0.01)
    top_n = (-similarity).argsort()[:100]
    mask[top_n] = 0.9

    fig = go.Figure(go.Scatter(
        x=df.x,
        y=df.y,
        mode='markers',
        marker=dict(opacity=mask),
        hovertext=df["bool"]
    ))
    fig.update_layout(width=1000, height=700)
    return fig

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, dev_tools_hot_reload=False)