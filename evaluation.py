import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def _get_probs(self, in_bool, in_text):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(in_bool, in_text, loss=False)
            return (outputs["logits"] + 1) / 2  # NxN similarity scores

    def _compute_true_ranks(self, probs):
        # For each row, find rank of the true (diagonal) score
        top_idxs = probs.argsort(descending=True)
        true_idxs = torch.arange(probs.size(0), device=probs.device).unsqueeze(1)
        true_rank = (top_idxs == true_idxs).argsort(descending=True)[:, 0]
        return true_rank

    def _print_recall_at_k(self, true_rank, N):
        print(f"{'Top-K':>10} | {'Top-K(perc)':>10} | {'Recall@K':>10}")
        for i in range(int(np.log2(N)) + 1):
            k = 2**i
            recall = (true_rank < k).float().mean().item()
            print(f"{k:>10} | {k/N:>10.2f} | {recall:10.2f}")

    # maximize the score of true positives and true negatives
    def _search_threshold(self, y_true, y_scores, n_thresholds=100):
        best_thresh, best_score = 0.0, -1.0
        y_true = y_true.astype(bool)
        true_count = y_true.sum()
        neg_count = (~y_true).sum()
        for t in np.linspace(0, 1, n_thresholds):
            preds = (y_scores >= t)
            pos_score = (y_true & preds).astype(float).sum()
            neg_score = (~y_true & ~preds).astype(float).sum()
            score = pos_score / true_count + neg_score / neg_count
            if score > best_score:
                best_score, best_thresh = score, t
        return best_thresh

    def _plot(self, probs_pos, probs_neg, true_rank, threshold, density):
        bins = 50

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        d_pos = probs_pos.cpu().numpy().flatten()
        d_neg = probs_neg.cpu().numpy().flatten()
        pos_height = np.histogram(d_pos, bins=bins, range=(0,1))[0].max() / d_pos.size
        neg_height = np.histogram(d_neg, bins=bins, range=(0,1))[0].max() / d_neg.size
        ymax = max(pos_height, neg_height) * bins * 1.05

        # Positive histogram
        ax: axes = axs[0, 0]
        ax.hist(d_pos, bins=bins, range=(0,1), alpha=0.7, density=density)
        ax.axvline(d_pos.mean(), color='red', linestyle='dashed', linewidth=2,
                   label=f"Mean: {d_pos.mean():.2f}")
        ax.set_title("Positive Score Distribution (higher = better)")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.set_ylim((0, ymax))
        ax.legend()

        # Negative histogram
        ax = axs[0, 1]
        ax.hist(d_neg, bins=bins, range=(0,1), alpha=0.7, density=density)
        ax.axvline(d_neg.mean(), color='red', linestyle='dashed', linewidth=2,
                   label=f"Mean: {d_neg.mean():.2f}")
        ax.set_title("Negative Score Distribution (lower = better)")
        ax.set_xlabel("Score")
        ax.set_ylim((0, ymax))
        ax.legend()

        # Boxplot true ranks
        ax = axs[1, 0]
        ranks = true_rank.float().cpu().numpy()
        ax.boxplot(ranks, vert=True, patch_artist=True)
        ax.set_ylim((0, probs_pos.size(0)))
        ax.set_title("True Rank Boxplot (lower = better)")
        ax.set_ylabel("Rank")
        ax.invert_yaxis()

        # Confusion matrix
        y_true = np.concatenate([np.ones_like(d_pos), np.zeros_like(d_neg)])
        y_pred = np.concatenate([d_pos, d_neg]) > threshold
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
        disp.plot(ax=axs[1,1], cmap='Blues')
        axs[1,1].set_title(f"Confusion Matrix (threshold={threshold:.2f})")

        plt.tight_layout()
        plt.show()

    def evaluate(self, in_bool, in_text, plot=False, threshold=None, density=True, n_thresholds=100, threshold_search=True):
        probs = self._get_probs(in_bool, in_text)
        N = probs.size(0)

        # ranks
        true_rank = self._compute_true_ranks(probs)
        self._print_recall_at_k(true_rank, N)

        # flatten for thresholding
        mask = torch.eye(N, device=probs.device).bool()
        probs_pos = probs[mask]
        probs_neg = probs[~mask]
        y_true = np.concatenate([np.ones_like(probs_pos.cpu().numpy()), np.zeros_like(probs_neg.cpu().numpy())])
        y_scores = np.concatenate([probs_pos.cpu().numpy(), probs_neg.cpu().numpy()])

        # threshold
        if threshold_search:
            best_thresh = self._search_threshold(y_true, y_scores, n_thresholds)
        else:
            best_thresh = threshold if threshold is not None else 0.5

        # plot diagnostics
        if plot:
            self._plot(probs_pos, probs_neg, true_rank, best_thresh, density)

        return {
            'probs': probs.cpu().numpy(),
            'best_threshold': best_thresh,
        }
