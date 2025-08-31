import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import math
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict
import pandas as pd
from IPython.display import display
from tqdm import tqdm

def _prepare_intermediates(logits):
    conf = (logits + 1) / 2
    N, M = conf.shape

    top_idxs = conf.argsort(axis=1)[:, ::-1]
    true_idxs = np.tile(np.arange(M), math.ceil(N / M))[:N][:, None]
    ranks_pos = np.argmax(top_idxs == true_idxs, axis=1)

    mask = np.tile(np.eye(M, dtype=bool), math.ceil(N / M)).T[:N]
    conf_pos = conf[mask]
    conf_neg = conf[~mask]
    conf_neg = conf_neg[conf_neg >= 0]

    return {
        "conf": conf,
        "ranks_pos": ranks_pos,
        "conf_pos": conf_pos,
        "conf_neg": conf_neg,
        "N": N,
        "M": M
    }


def compute_metrics(eval_pred):
    logits, _ = eval_pred
    inter = _prepare_intermediates(logits)
    _, ranks_pos, conf_pos, conf_neg, N, M = (
        inter["conf"], inter["ranks_pos"], inter["conf_pos"], inter["conf_neg"], inter["N"], inter["M"]
    )
    metrics = {}

    ks = [min(2**i, 30) for i in range(6)]
    metrics.update({f"recall@{k}": np.mean(ranks_pos < k) for k in ks})
    metrics['mean_rank'] = ranks_pos.mean()
    metrics['median_rank'] = np.median(ranks_pos)
    metrics['mean_rank_norm'] = ranks_pos.mean() / M
    metrics['median_rank_norm'] = np.median(ranks_pos) / M
    metrics['min_rank'] = ranks_pos.max()
    metrics['min_rank_norm'] = ranks_pos.max() / M

    for p in [1, 2, 5, 10, 25, 50]:
        metrics[f"recall@{p}%"] = np.mean(ranks_pos < max(min(N, M) * p / 100, 1))

    y_true = np.concatenate([np.ones_like(conf_pos), np.zeros_like(conf_neg)])
    y_scores = np.concatenate([conf_pos, conf_neg])

    total_pos = y_true.sum()
    total_neg = y_true.size - total_pos
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]
    cum_pos = np.cumsum(y_true_sorted)
    cum_neg = np.cumsum(1 - y_true_sorted)

    tpr = cum_pos / total_pos
    tnr = (total_neg - cum_neg) / total_neg
    score = tpr + tnr
    best_idx = np.argmax(score)
    best_thresh = float(y_scores[order][best_idx])
    best_score = float(score[best_idx])
    metrics['best_threshold'] = best_thresh
    metrics['best_score'] = best_score

    y_pred_best = (y_scores >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_best).ravel()
    metrics.update({'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)})

    return metrics


def evaluate(model, in_bool, in_text, plot=False, density=True, title=None, batch_size=30):
    model.eval()
    shape = (9, 4)

    logits_list = []
    for b in range(0, len(in_bool), batch_size):
        data_a = in_bool[b:b+batch_size]
        data_b = in_text[b:b+batch_size]

        if len(data_a) < batch_size:
            pad_size = batch_size - len(data_a)
            data_a = np.concatenate([data_a, in_bool[:pad_size]])
            data_b = np.concatenate([data_b, in_text[:pad_size]])

        with torch.no_grad():
            outputs = model(data_a, data_b, return_loss=False)
        logits = outputs['logits'].cpu().numpy()
        logits_list.append(logits)

    logits_all = np.concatenate(logits_list, axis=0)
    metrics = compute_metrics((logits_all, None))
    inter = _prepare_intermediates(logits_all)
    ranks_pos, conf_pos, conf_neg = inter["ranks_pos"], inter["conf_pos"], inter["conf_neg"]
    print(ranks_pos)
    print("Worst (idx, rank): ", ranks_pos.argmax(), ranks_pos.max())
    print("Best (idx, rank): ", ranks_pos.argmin(), ranks_pos.min())
    metrics['logits'] = logits_all
    thresh = metrics['best_threshold']

    if plot:
        bins = 50
        d_pos, d_neg = conf_pos.flatten(), conf_neg.flatten()

        def get_hist_height(data):
            return np.histogram(data, bins=bins, range=(0,1))[0].max() / data.size
        pos_height = get_hist_height(d_pos)
        neg_height = get_hist_height(d_neg)
        ymax = max(pos_height, neg_height) * bins * 1.05

        fig, axs = plt.subplots(1, 3, width_ratios=[0.45, 0.45, 0.1], figsize=shape)
        if title is not None: fig.suptitle(title)

        ax = axs[0]
        ax.grid(True, ls="--", alpha=0.5)
        ax.hist(conf_pos, bins=bins, range=(0,1), density=density)
        ax.axvline(d_pos.mean(), color='red', linestyle='dashed', linewidth=2,
                   label=f"Mean: {d_pos.mean():.2f}")
        ax.set_title('Positive Similarity Distribution')
        ax.set_ylim((0, ymax))
        ax.legend()

        ax = axs[1]
        ax.grid(True, ls="--", alpha=0.5)
        ax.hist(conf_neg, bins=bins, range=(0,1), density=density)
        ax.axvline(d_neg.mean(), color='red', linestyle='dashed', linewidth=2,
                   label=f"Mean: {d_neg.mean():.2f}")
        ax.set_title('Negative Similarity Distribution')
        ax.set_ylim((0, ymax))
        ax.legend()

        ax = axs[2]
        ax.boxplot(ranks_pos, vert=True, patch_artist=True, widths=0.6)
        ax.set_ylim((0, batch_size))
        ax.set_ylabel("Rank")
        ax.invert_yaxis()
        ax.set_title('True Rank Boxplot')

        metrics['plot1'] = fig
        plt.tight_layout()
        fig, axs = plt.subplots(1, 2, figsize=shape)

        ax = axs[0]
        recall_keys = [k for k in metrics if k.startswith('recall@') and '%' not in k]
        ks = sorted(int(k.split('@')[1]) for k in recall_keys)
        ax.plot(ks, [metrics[f'recall@{k}'] for k in ks], marker='o')
        rec1 = metrics['recall@1']
        ax.set_xlabel("Top-K")
        ax.set_ylabel("Recall@K")
        ax.set_title(f"Recall@K Curve ({rec1:.2f}@K=1)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, ls="--", alpha=0.5)

        ax = axs[1]
        y_true = np.concatenate([np.ones_like(conf_pos), np.zeros_like(conf_neg)])
        y_pred = (np.concatenate([conf_pos, conf_neg]) >= thresh)
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(cm, display_labels=['Neg','Pos'])
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title(f'Confusion Matrix (thresh={thresh:.2f})')

        metrics['plot2'] = fig
        plt.tight_layout()

    return metrics

def evaluate_on_generated(model, group_keys: list[str] = ["id", "model"]):
    main_name = group_keys[-1]
    res = defaultdict(lambda: defaultdict(list))
    df = pd.read_json("data/combined_outputs.jsonl", lines=True)
    byid = df.sort_values("f3").groupby(group_keys)
    prompt_data = list(map(lambda tpl: tpl[1], byid))

    # metric = RBMetric(phi=0.95)
    for group in tqdm(prompt_data, desc="Evaluating groups"):
        group = group[~group.duplicated(["id", "f3"])]
        if len(group) <= 3: continue

        name = group[main_name].iloc[0]
        res[name]["prompt_amt"] = len(prompt_data)
        queries = list(group["generated_query"].values)
        if len(queries) < 2: continue
        topic = group["topic"].iloc[0]

        tensors = [
            model(query, topic, False)["logits"]
                .cpu().detach()
            for query in queries
        ]
        # print(tensors)
        # concatenate them along the batch dimension
        tensor = torch.cat(tensors).squeeze()

        # Original ranks (0-based indices)
        original_ranks = torch.arange(len(tensor))

        # Sorted values → get the sorted indices → assign new ranks
        sorted_indices = torch.argsort(tensor)
        new_ranks = torch.arange(len(tensor))[sorted_indices]

        spearman_corr, _ = spearmanr(original_ranks.numpy(), new_ranks.numpy())
        pearson_corr, _ = pearsonr(group["f3"].values, tensor.detach().numpy())
        offset_sum = (original_ranks-new_ranks.float()).abs().sum()
        n = original_ranks.size()[0]
        norm_offset = ((offset_sum)*2/(n**2-n%2)).numpy()

        res[name]["spearman"].append(spearman_corr)
        res[name]["pearson"].append(pearson_corr)
        res[name]["norm_offset"].append(norm_offset)
        res[name]["query_amt"].append(n)
        res[name]["f3_variance"].append(np.sqrt(np.var(group['f3'].values)))
        res[name]["best_rank"].append(new_ranks[-1] / len(tensor))

    df = pd.DataFrame({
        main_name: list(res.keys()),
        "spearman": [np.mean(m["spearman"]) for m in res.values()],
        "pearson": [np.mean(m["pearson"]) for m in res.values()],
        # "norm_offset_sum": [np.mean(m["norm_offset"]) for m in res.values()],
        # "med_queries_per_prompt": [np.median(m["query_amt"]) for m in res.values()],
        "f3_variance": [np.mean(m["f3_variance"]) for m in res.values()],
        "best_rank": [np.mean(m["best_rank"]) for m in res.values()],
        # "avg_queries_per_prompt": [np.mean(m["query_amt"]) for m in res.values()],
    })

    avg_row = df.mean(numeric_only=True)
    avg_row[main_name] = "Average"
    df = pd.concat([df, avg_row.to_frame().T], ignore_index=True)

    df = df.astype({
        "spearman": float,
        "pearson": float,
        # "avg_queries_per_prompt": float,
        "f3_variance": float,
        "best_rank": float,
    })

    # def highlight_last_row(row):
    #     return ['font-weight: bold;' if row.name == len(df) - 1 else '' for _ in row]
    # fmt = {
    #     "spearman":            "{:.3f}",
    #     # "norm_offset_sum":     "{:.3f}",
    #     "avg_queries_per_prompt": "{:.3f}",
    #     # "med_queries_per_prompt": "{}",
    #     "f3_variance": "{:.1e}",
    # }
    # styled = df.style.apply(highlight_last_row, axis=1).format(fmt)
    # display(HTML(styled.to_html()))
    df = df.round(4)
    display(df)

