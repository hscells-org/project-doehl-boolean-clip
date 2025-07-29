import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import math
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict
import pandas as pd
from IPython.display import HTML, display

def compute_metrics(eval_pred):
    logits, _ = eval_pred
    # Convert logits to similarity scores in [0,1]
    conf: np.ndarray = (logits + 1) / 2
    N, M = conf.shape
    metrics = {}

    # Compute true ranks for recall@k
    top_idxs = conf.argsort(axis=1)[:, ::-1]
    true_idxs = np.tile(np.arange(M), math.ceil(N / M))[:N][:, None]
    ranks_pos: np.ndarray = np.argmax(top_idxs == true_idxs, axis=1)

    # Determine ks (up to 2^6 if N large)
    max_pow = int(np.log2(N))
    ks = [2**i for i in range(min(max_pow + 1, 7))]
    metrics = {f"recall@{k}": np.mean(ranks_pos < k) for k in ks}
    metrics['mean_rank'] = ranks_pos.mean()
    metrics['median_rank'] = np.median(ranks_pos)
    metrics['mean_rank_norm'] = ranks_pos.mean() / M
    metrics['median_rank_norm'] = np.median(ranks_pos) / M
    metrics['min_rank'] = ranks_pos.max()
    metrics['min_rank_norm'] = ranks_pos.max() / M

    for p in [1, 2, 5, 10, 25, 50]:
        metrics[f"recall@{p}%"] = np.mean(ranks_pos < max(min(N,M) * p / 100, 1))

    # Flatten positive (diagonal) and negative (off-diagonal) scores
    mask = np.tile(np.eye(M, dtype=bool), math.ceil(N / M)).T[:N]
    conf_pos = conf[mask]
    conf_neg = conf[~mask]
    conf_neg = conf_neg[conf_neg >= 0]
    y_true = np.concatenate([np.ones_like(conf_pos), np.zeros_like(conf_neg)])
    y_scores = np.concatenate([conf_pos, conf_neg])

    # Compute recall at decile thresholds
    total_pos = y_true.sum()
    total_neg = y_true.size - total_pos
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]
    cum_pos = np.cumsum(y_true_sorted)
    cum_neg = np.cumsum(1 - y_true_sorted)

    # Best threshold maximizing TPR + TNR
    tpr = cum_pos / total_pos
    tnr = (total_neg - cum_neg) / total_neg
    score = tpr + tnr
    best_idx = np.argmax(score)
    best_thresh = float(y_scores[order][best_idx])
    best_score = float(score[best_idx])
    metrics['best_threshold'] = best_thresh
    metrics['best_score'] = best_score

    # Confusion at best threshold
    y_pred_best = (y_scores >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_best).ravel()
    metrics.update({'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)})
    return metrics


def evaluate(model, in_bool, in_text, plot=False, density=True, title=None):
    model.eval()
    with torch.no_grad():
        outputs = model(in_bool, in_text, return_loss=False)
    logits = outputs['logits'].cpu().numpy()
    metrics = compute_metrics((logits, None))

    # Optional diagnostic plotting
    if plot:
        bins = 50

        N = logits.shape[0]
        conf = (logits + 1) / 2
        top_idxs = np.argsort(-conf, axis=1)
        true_idxs = np.arange(N)[:, None]
        ranks_pos = np.argmax(top_idxs == true_idxs, axis=1)
        mask = np.eye(N, dtype=bool)
        conf_pos = conf[mask]
        conf_neg = conf[~mask]
        thresh = metrics['best_threshold']
        d_pos = conf_pos.flatten()
        d_neg = conf_neg.flatten()
        def get_hist_height(data): return np.histogram(data, bins=bins, range=(0,1))[0].max() / data.size
        pos_height = get_hist_height(d_pos)
        neg_height = get_hist_height(d_neg)
        ymax = max(pos_height, neg_height) * bins * 1.05

        fig, axs = plt.subplots(3, 2, figsize=(10, 12))
        if title is not None: fig.suptitle(title)
        # positive histogram
        ax = axs[0,0]
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.hist(conf_pos, bins=50, range=(0,1), density=density)
        ax.axvline(d_pos.mean(), color='red', linestyle='dashed', linewidth=2,
                   label=f"Mean: {d_pos.mean():.2f}")
        ax.set_title('Positive Score Distribution')
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.set_ylim((0, ymax))
        ax.legend()

        # negative histogram
        ax = axs[0,1]
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.hist(conf_neg, bins=50, range=(0,1), density=density)
        ax.axvline(d_neg.mean(), color='red', linestyle='dashed', linewidth=2,
                   label=f"Mean: {d_neg.mean():.2f}")
        ax.set_title('Negative Score Distribution')
        ax.set_xlabel("Score")
        ax.set_ylim((0, ymax))
        ax.legend()

        # boxplot of ranks
        ax = axs[1,0]
        ax.boxplot(ranks_pos, vert=True, patch_artist=True)
        ax.set_ylim((0, conf_pos.size))
        ax.set_title("True Rank Boxplot (lower = better)")
        ax.set_ylabel("Rank")
        ax.invert_yaxis()
        ax.set_title('True Rank Boxplot')

        # recall curve
        # extract integer recall@k metrics
        ax = axs[1,1]
        recall_keys = [key for key in metrics if key.startswith('recall@') and key.endswith(str(int(key.split('@')[1].replace('%',''))))]
        ks = sorted([int(key.split('@')[1].replace('%','')) for key in recall_keys if '%' not in key])
        ax.plot(ks, [metrics[f'recall@{k}'] for k in ks], marker='o')
        rec1 = metrics['recall@1']
        ax.set_xlabel("Top-K")
        ax.set_ylabel("Recall@K")
        ax.set_title(f"Recall@K Curve ({rec1:.2f}@K=1)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, which="both", ls="--", alpha=0.5)

        # confusion matrix
        ax = axs[2,0]
        y_true = np.concatenate([np.ones_like(conf_pos), np.zeros_like(conf_neg)])
        y_pred = (np.concatenate([conf_pos, conf_neg]) >= thresh)
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(cm, display_labels=['Neg','Pos'])
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title(f'Confusion Matrix (thresh={thresh:.2f})')

        plt.tight_layout()
        metrics['plot'] = fig

    return metrics

def evaluate_on_generated(model, group_keys: list[str] = ["id", "model"]):
    main_name = group_keys[-1]
    res = defaultdict(lambda: defaultdict(list))
    df = pd.read_json("data/combined_outputs.jsonl", lines=True)
    byid = df.sort_values("f3").groupby(group_keys)
    prompt_data = list(map(lambda tpl: tpl[1], byid))
    for group in prompt_data:
        group = group[~group.duplicated(["id", "f3"])]
        name = group[main_name].iloc[0]
        res[name]["prompt_amt"] = len(prompt_data)
        queries = list(group["generated_query"].values)
        if len(queries) < 2: continue
        topic = group["topic"].iloc[0]
        logits = model(queries, topic, False)["logits"]

        tensor = logits.squeeze().cpu()
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
        res[name]["f3_variance"].append(np.var(group['f3'].values))

    df = pd.DataFrame({
        main_name: list(res.keys()),
        "spearman": [np.mean(m["spearman"]) for m in res.values()],
        "pearson": [np.mean(m["pearson"]) for m in res.values()],
        # "norm_offset_sum": [np.mean(m["norm_offset"]) for m in res.values()],
        # "med_queries_per_prompt": [np.median(m["query_amt"]) for m in res.values()],
        "f3_variance": [np.mean(m["f3_variance"]) for m in res.values()],
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