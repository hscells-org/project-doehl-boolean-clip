import multiprocessing as mp
import pandas as pd
import torch
import umap
from dash import Dash, html, dcc, callback, Output, Input
from boolrank import DualSiglip2Model

df = None
dataset = None
data_ready = mp.Value('b', False)  # shared flag

paths = [
    "data/training.jsonl",
    "data/TAR_data.jsonl",
    "data/sysrev_conv.jsonl",
]
model = DualSiglip2Model('BAAI/bge-small-en-v1.5')
model.load(r"models\\clip\\bge-small-en-v1.5\\b16_lr1E-05_(pubmed-que_pubmed-sea_raw-jsonl)^4\\checkpoint-11288\\model.safetensors")

def compute_embeddings_and_umap(return_dict):
    dataset = []
    for path in paths:
        df = pd.read_json(path, lines=True)
        dataset.append(df)
    dataset = pd.concat(dataset)

    N = 10000
    df = dataset.sample(min(N, dataset.shape[0])).reset_index(drop=True)

    embeddings = model.encode_bool(df["bool_query"].tolist(), batch_size=200).detach().cpu().numpy()
    torch.cuda.empty_cache()

    um_reducer = umap.UMAP(n_neighbors=15, n_components=2)
    trans = um_reducer.fit_transform(embeddings)
    df["x"], df["y"] = trans[:, 0], trans[:, 1]

    cut = 60
    df["nl"] = df["nl_query"].map(lambda x: x if len(x) < cut else x[:cut] + "...")
    df["bool"] = df["bool_query"].map(lambda x: x if len(x) < cut else x[:cut] + "...")

    return_dict["df"] = df
    return_dict["dataset"] = dataset
    data_ready.value = True


app = Dash(__name__)
app.layout = html.Div([
    html.H1("Embedding Visualization Dashboard", style={'textAlign': 'center'}),
    dcc.Dropdown(id='query-dropdown', placeholder="Loading queries..."),
    dcc.Graph(id='embedding-graph', style={"height": "800px"})
])

@callback(
    Output('query-dropdown', 'options'),
    Input('query-dropdown', 'value')
)
def update_dropdown(_):
    if not data_ready.value:
        return []
    return [{"label": q, "value": q} for q in dataset["test"]["pubmed-searchrefiner"]["nl_query"][:50]]

if __name__ == "__main__":
    manager = mp.Manager()
    return_dict = manager.dict()

    p = mp.Process(target=compute_embeddings_and_umap, args=(return_dict,))
    p.start()

    app.run(debug=True)

    p.join()
    df = return_dict["df"]
    dataset = return_dict["dataset"]
