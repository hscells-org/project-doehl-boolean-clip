from dash import Dash, callback, Output, Input, Patch, html, MATCH, ALL, callback_context, no_update
import plotly.graph_objects as go
import numpy as np
import torch
import umap

import app_helper
from utils.boolrank import DualSiglip2Model

# -------- Adjust data and models ----------
MARKER_SIZE = 6
MARKER_HIGHLIGHT_SIZE = 12
in_key = "nl_query"
out_key = "bool_query"
N = 10000

paths = [
    "data/training.jsonl",
    "data/TAR_data.jsonl",
    "data/sysrev_conv.jsonl",
]

model_name = 'BAAI/bge-small-en-v1.5'
# model_name = 'dmis-lab/biobert-v1.1'
model_path = None
# model_path = r"models\\clip\\bge-small-en-v1.5\\b16_lr1E-05_(pubmed-que_pubmed-sea_raw-jsonl)^4\\checkpoint-11288\\model.safetensors"

model = DualSiglip2Model(model_name)
if model_path: model.load(model_path)
# -------------------------------------------

# can be run separately for caching
df, embeddings = app_helper.load_or_create_embeddings(model, paths, in_key, out_key, model_path, N)
torch.cuda.empty_cache()

print("Calculating UMAP")
um_reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=0)
trans = um_reducer.fit_transform(embeddings)
df["x"], df["y"] = trans[:, 0], trans[:, 1]

def cutoffl(cut): return lambda x: x if len(x) < cut else x[:cut] + "..."
def build_base_figure(default_opacity):
    fig = go.Figure(layout_title_text=f"Data points: {len(df)}")
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
    return [{"label": q, "value": q} for q in df[:50][in_key]]


@callback(
    Output('embedding-graph', 'figure'),
    Input('embedding-graph', 'id'),
    prevent_initial_call='initial_duplicate'
)
def init_figure(_):
    return base_fig


@callback(
    Output("embedding-graph", "figure", allow_duplicate=True),
    Input({"type": "topk-item", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def focus_point(clicks):
    ctx = callback_context
    if not ctx.triggered: return no_update
    triggered_id = ctx.triggered_id
    if not triggered_id: return no_update

    idx = triggered_id["index"]
    patch = Patch()
    patch["data"][0]["marker"]["size"] = [MARKER_HIGHLIGHT_SIZE if i == idx else MARKER_SIZE for i in range(len(df))]
    return patch


@callback(
    Output('manual-query', 'value'),
    Input('embedding-graph', 'clickData'),
    prevent_initial_call=True
)
def use_point_as_query(clickData):
    if not clickData: return no_update

    point = clickData['points'][0]
    data = df.iloc[point["pointIndex"]]
    return data["nl_query"]


last_manual_query = None
last_dropdown_query = None
last_query = None
similarities = None
@callback(
    Output('embedding-graph', 'figure', allow_duplicate=True),
    Output('topk-list', 'children'),
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

    cutoff_fun = cutoffl(char_amt)
    df["data_in"] = df[in_key].map(cutoff_fun)
    df["data_out"] = df[out_key].map(cutoff_fun)
    customdata = np.stack((df["data_in"], df["data_out"]), axis=-1)
    patch["data"][0]["customdata"] = customdata

    topk_items = []
    for rank, idx in enumerate(top_n):
        topk_items.append(html.Div(
            id={"type": "topk-item", "index": int(idx)},
            children=[
                html.B(f"Rank: {rank + 1}"), html.Br(),
                html.B("Input: "), cutoff_fun(df.iloc[idx][in_key]), html.Br(),
                html.B("Output: "), cutoff_fun(df.iloc[idx][out_key]), html.Hr()
            ],
            style={"padding": "6px", "cursor": "pointer"}
        ))
    return patch, topk_items

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
