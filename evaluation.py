import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def compute_metrics(eval_pred):
    logits, _ = eval_pred
    # Convert logits to similarity scores in [0,1]
    conf: np.ndarray = (logits + 1) / 2
    N = conf.shape[0]

    # Compute true ranks for recall@k
    top_idxs = conf.argsort(axis=1)[:, ::-1]
    true_idxs = np.arange(N)[:, None]
    ranks_pos: np.ndarray = np.argmax(top_idxs == true_idxs, axis=1)

    # Determine ks (up to 2^6 if N large)
    max_pow = int(np.log2(N))
    ks = [2**i for i in range(min(max_pow + 1, 7))]
    metrics = {f"recall@{k}": np.mean(ranks_pos < k) for k in ks}
    metrics['mean_rank'] = ranks_pos.mean()
    metrics['median_rank'] = np.median(ranks_pos)
    metrics['mean_rank_norm'] = ranks_pos.mean() / N
    metrics['median_rank_norm'] = np.median(ranks_pos) / N
    metrics['min_rank'] = ranks_pos.max()
    metrics['min_rank_norm'] = ranks_pos.max() / N

    # Flatten positive (diagonal) and negative (off-diagonal) scores
    mask = np.eye(N, dtype=bool)
    probs_pos = conf[mask]
    probs_neg = conf[~mask]
    y_true = np.concatenate([np.ones_like(probs_pos), np.zeros_like(probs_neg)])
    y_scores = np.concatenate([probs_pos, probs_neg])

    # Compute recall at decile thresholds
    total_pos = y_true.sum()
    total_neg = y_true.size - total_pos
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]
    cum_pos = np.cumsum(y_true_sorted)
    cum_neg = np.cumsum(1 - y_true_sorted)
    # n = y_true_sorted.size
    for p in [1, 2, 5, 10, 25, 50]:
        # idx = max(int(n * p / 100) - 1, 0)
        # tp = cum_pos[idx]
        # fn = total_pos - tp
        # metrics[f"recall@{p}%"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics[f"recall@{p}%"] = np.mean(ranks_pos < max(probs_pos.size * p / 100, 1))

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
        probs_pos = conf[mask]
        probs_neg = conf[~mask]
        thresh = metrics['best_threshold']
        d_pos = probs_pos.flatten()
        d_neg = probs_neg.flatten()
        pos_height = np.histogram(d_pos, bins=bins, range=(0,1))[0].max() / d_pos.size
        neg_height = np.histogram(d_neg, bins=bins, range=(0,1))[0].max() / d_neg.size
        ymax = max(pos_height, neg_height) * bins * 1.05

        fig, axs = plt.subplots(3, 2, figsize=(10, 12))
        if title is not None: fig.suptitle(title)
        # positive histogram
        ax = axs[0,0]
        ax.hist(probs_pos, bins=50, range=(0,1), density=density)
        ax.axvline(d_pos.mean(), color='red', linestyle='dashed', linewidth=2,
                   label=f"Mean: {d_pos.mean():.2f}")
        ax.set_title('Positive Score Distribution')
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.set_ylim((0, ymax))
        ax.legend()

        # negative histogram
        ax = axs[0,1]
        ax.hist(probs_neg, bins=50, range=(0,1), density=density)
        ax.axvline(d_neg.mean(), color='red', linestyle='dashed', linewidth=2,
                   label=f"Mean: {d_neg.mean():.2f}")
        ax.set_title('Negative Score Distribution')
        ax.set_xlabel("Score")
        ax.set_ylim((0, ymax))
        ax.legend()

        # boxplot of ranks
        ax = axs[1,0]
        ax.boxplot(ranks_pos, vert=True, patch_artist=True)
        ax.set_ylim((0, probs_pos.size))
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
        y_true = np.concatenate([np.ones_like(probs_pos), np.zeros_like(probs_neg)])
        y_pred = (np.concatenate([probs_pos, probs_neg]) >= thresh)
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(cm, display_labels=['Neg','Pos'])
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title(f'Confusion Matrix (thresh={thresh:.2f})')

        plt.tight_layout()
        metrics['plot'] = fig

    return metrics
