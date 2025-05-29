import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

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

    def _search_threshold(self, y_true, y_scores, n_thresholds=100):
        best_thresh, best_f1 = 0.0, -1.0
        for t in np.linspace(0, 1, n_thresholds):
            preds = (y_scores >= t).astype(int)
            score = f1_score(y_true, preds)
            if score > best_f1:
                best_f1, best_thresh = score, t
        return best_thresh, best_f1

    def _plot(self, probs_pos, probs_neg, true_rank, threshold, density):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Positive histogram
        d_pos = probs_pos.cpu().numpy().flatten()
        ax = axes[0, 0]
        ax.hist(d_pos, bins=50, range=(0,1), alpha=0.7, density=density)
        ax.axvline(d_pos.mean(), color='red', linestyle='dashed', linewidth=2,
                   label=f"Mean: {d_pos.mean():.2f}")
        ax.set_title("Positive Score Distribution (higher = better)")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.legend()

        # Negative histogram
        d_neg = probs_neg.cpu().numpy().flatten()
        ax = axes[0, 1]
        ax.hist(d_neg, bins=50, range=(0,1), alpha=0.7, density=density)
        ax.axvline(d_neg.mean(), color='red', linestyle='dashed', linewidth=2,
                   label=f"Mean: {d_neg.mean():.2f}")
        ax.set_title("Negative Score Distribution (lower = better)")
        ax.set_xlabel("Score")
        ax.legend()

        # Boxplot true ranks
        ax = axes[1, 0]
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
        disp.plot(ax=axes[1,1], cmap='Blues')
        axes[1,1].set_title(f"Confusion Matrix (threshold={threshold:.2f})")

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
            best_thresh, best_f1 = self._search_threshold(y_true, y_scores, n_thresholds)
            print(f"Best threshold by F1: {best_thresh:.3f} (F1={best_f1:.3f})")
        else:
            best_thresh = threshold if threshold is not None else 0.5

        # plot diagnostics
        if plot:
            self._plot(probs_pos, probs_neg, true_rank, best_thresh, density)

        return {
            'probs': probs.cpu().numpy(),
            'best_threshold': best_thresh,
        }
