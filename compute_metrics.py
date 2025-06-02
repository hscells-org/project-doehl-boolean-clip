import numpy as np
from sklearn.metrics import confusion_matrix

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
    metrics = {f"recall@{k}": float(np.mean(ranks_pos < k)) for k in ks}
    metrics['mean_rank'] = ranks_pos.mean()
    metrics['median_rank'] = np.median(ranks_pos)
    metrics['mean_rank_relative'] = ranks_pos.mean()/ranks_pos.size
    metrics['median_rank_relative'] = np.median(ranks_pos)/ranks_pos.size

    # Flatten positive (diagonal) and negative (off-diagonal) scores
    mask = np.eye(N, dtype=bool)
    probs_pos = conf[mask]
    probs_neg = conf[~mask]
    y_true = np.concatenate([np.ones_like(probs_pos), np.zeros_like(probs_neg)])
    y_scores = np.concatenate([probs_pos, probs_neg])

    # Sort by score descending for fast thresholding
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]
    total_pos = y_true.sum()
    total_neg = y_true.size - total_pos

    # Cumulative sums of positives and negatives
    cum_pos = np.cumsum(y_true_sorted)
    cum_neg = np.cumsum(1 - y_true_sorted)
    n = y_true_sorted.size

    # Percentile thresholds at deciles
    for p in range(10, 100, 10):
        idx = max(int(n * p / 100) - 1, 0)
        tp = cum_pos[idx]
        fp = cum_neg[idx]
        fn = total_pos - tp
        tn = total_neg - fp
        # metrics[f"accuracy@{p}%"] = (tp + tn) / n
        # metrics[f"precision@{p}%"] = tp / (tp + fp) if tp + fp > 0 else 0.0
        metrics[f"recall@{p}%"] = tp / (tp + fn) if tp + fn > 0 else 0.0

    # Best threshold maximizing TPR + TNR using vectorized pass
    # Compute TPR and TNR at each unique score boundary
    tpr = cum_pos / total_pos
    tnr = (total_neg - cum_neg) / total_neg
    score = tpr + tnr
    best_idx = np.argmax(score)
    best_thresh = y_scores[order][best_idx]
    best_score = float(score[best_idx])

    metrics['best_threshold'] = float(best_thresh)
    metrics['best_score'] = best_score
    # Confusion at best threshold
    y_pred_best = (y_scores >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_best).ravel()
    metrics.update({'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)})

    return metrics
