import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from transformers import AutoTokenizer, Siglip2TextModel
from safetensors.torch import load_file
import random
from datasets import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from tabulate import tabulate

device = "cuda" if torch.cuda.is_available() else "cpu"


class DualSiglip2Model(nn.Module):
    def __init__(self, model_name="google/siglip2-base-patch16-224"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder_bool = Siglip2TextModel.from_pretrained(model_name)
        self.encoder_text = deepcopy(self.encoder_bool)
        self.bias = nn.Parameter(torch.zeros(1))
        self.to(device)

    def tokenize(self, texts):
        return self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt", max_length=64).to(device)

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip2/modeling_siglip2.py#L952
    def forward(self, in_bool, in_text, loss=True):
        tok_bool = self.tokenize(in_bool)
        tok_text = self.tokenize(in_text)
        out_bool = self.encoder_bool(**tok_bool).pooler_output
        out_text = self.encoder_text(**tok_text).pooler_output
        out_bool = out_bool / out_bool.norm(p=2, dim=-1, keepdim=True)
        out_text = out_text / out_text.norm(p=2, dim=-1, keepdim=True)
        loss = self.loss(out_bool, out_text) if loss else None
        logits = out_bool @ out_text.t()  # + self.bias
        return {"loss": loss, "logits": logits}

    def loss(self, emb_a, emb_b):
        sim = emb_a @ emb_b.t() + self.bias
        eye = torch.eye(sim.size(0), device=sim.device)
        y = -torch.ones_like(sim) + 2 * eye
        loglik = F.logsigmoid(y * sim)
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()
        return loss

    def load(self, path):
        state_dict = load_file(path, device)
        self.load_state_dict(state_dict, strict=False)
        return self

    def preprocess(self, in_bool, in_text):
        pass  # TODO

    def evaluate(self, in_bool, in_text, plot=False, threshold=0.5, density=True):
        self.eval()
        with torch.no_grad():
            outputs = self(in_bool, in_text, loss=False)
            probs: torch.Tensor = (outputs["logits"] + 1) / 2

        # Compute ranks
        top_idxs = probs.argsort(descending=True)
        true_idxs = torch.arange(0, probs.size(0), device=probs.device).unsqueeze(1)
        true_rank = (top_idxs == true_idxs).argsort(descending=True)[:, 0]

        # Print recall metrics
        print(f"{'Top-K':>10} | {'Top-K(perc)':>10} | {'Recall@K':>10}")
        for i in range(int(np.log2(probs.size(0)) + 1)):
            k = 2**i
            recall = (true_rank < k).float().mean().item()
            print(f"{k:>10} | {k/probs.size(0):>10.2f} | {recall:10.2f}")

        # Separate positive and negative probabilities
        mask = torch.eye(probs.size(0), device=probs.device).bool()
        probs_pos = probs[mask]
        probs_neg = probs[~mask]

        if plot:
            # Prepare subplots: 2 rows x 2 cols
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Positive distribution histogram
            ax = axes[0, 0]
            d_pos = probs_pos.cpu().numpy().flatten()
            ax.hist(d_pos, bins=50, range=(0, 1), alpha=0.7, density=density)
            ax.axvline(d_pos.mean(), color='red', linestyle='dashed', linewidth=2, label=f"Mean: {d_pos.mean():.2f}")
            ax.set_title("Positive Score Distribution (higher = better)")
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            ax.legend()

            # Negative distribution histogram
            ax = axes[0, 1]
            d_neg = probs_neg.cpu().numpy().flatten()
            ax.hist(d_neg, bins=50, range=(0, 1), alpha=0.7, density=density)
            ax.axvline(d_neg.mean(), color='red', linestyle='dashed', linewidth=2, label=f"Mean: {d_neg.mean():.2f}")
            ax.set_title("Negative Score Distribution (lower = better)")
            ax.set_xlabel("Score")
            ax.legend()

            # Boxplot of true ranks
            ax = axes[1, 0]
            ranks = true_rank.float().cpu().numpy()
            ax.boxplot(ranks, vert=True, patch_artist=True)
            ax.set_title("True Rank Boxplot (lower = better)")
            ax.set_ylabel("Rank")
            ax.invert_yaxis()

            # Confusion matrix
            y_true = np.concatenate([np.ones_like(d_pos), np.zeros_like(d_neg)])
            y_pred = np.concatenate([d_pos, d_neg]) > threshold
            cm = confusion_matrix(y_true, y_pred, normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
            disp.plot(ax=axes[1, 1], cmap='Blues')
            axes[1, 1].set_title("Confusion Matrix")

            plt.tight_layout()
            plt.show()

        return probs.cpu().numpy()


class RandomAccessMismatchedPairs:
    def __init__(self, dataset, key_a='q', key_b='d'):
        self.dataset = dataset
        self.key_a = key_a
        self.key_b = key_b
        self.size = len(dataset)
        self.total_pairs = self.size * (self.size - 1)

    def __len__(self):
        return self.total_pairs

    def _index_to_pair(self, index):
        i = index // (self.size - 1)
        j = index % (self.size - 1)
        if j >= i: j += 1
        return i, j

    def __getitem__(self, index):
        if index < 0 or index >= self.total_pairs:
            raise IndexError("Index out of bounds")
        i, j = self._index_to_pair(index)
        return {
            self.key_a: self.dataset[i][self.key_a],
            self.key_b: self.dataset[j][self.key_b],
        }

    def random_sample(self, k=1, seed=42):
        rng = random.Random(seed)
        indices = rng.sample(range(self.total_pairs), k)
        return [self[i] for i in indices]

    def get_n_pairs(self, n, random_order=False, seed=42):
        if n > self.total_pairs:
            raise ValueError("Requested more pairs than available.")
        if random_order:
            return self.random_sample(n, seed)
        else:
            return [self[i] for i in range(n)]

    def to_dataset(self, n=None, random_order=False, seed=42):
        if n is None:
            n = self.total_pairs
        data = self.get_n_pairs(n, random_order, seed)
        return Dataset.from_list(data)
