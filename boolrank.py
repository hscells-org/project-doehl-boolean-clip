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

device = "cuda" if torch.cuda.is_available() else "cpu"

class DualSiglip2Model(nn.Module):
    def __init__(self, model_name="google/siglip2-base-patch16-224"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder_bool = Siglip2TextModel.from_pretrained(model_name)
        self.encoder_text = deepcopy(self.encoder_bool)
        self.bias = nn.Parameter(torch.zeros(1))
        # for stronger extremes
        self.logit_scale = nn.Parameter(torch.ones(1))
        self.to(device)

    def tokenize(self, texts):
        return self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt", max_length=64).to(device)

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip2/modeling_siglip2.py#L952
    def forward(self, in_bool, in_text):
        tok_bool = self.tokenize(in_bool)
        tok_text = self.tokenize(in_text)
        out_bool = self.encoder_bool(**tok_bool).pooler_output
        out_text = self.encoder_text(**tok_text).pooler_output
        out_bool = out_bool / out_bool.norm(p=2, dim=-1, keepdim=True)
        out_text = out_text / out_text.norm(p=2, dim=-1, keepdim=True)
        loss = self.loss(out_bool, out_text)
        logits = out_bool @ out_text.t() * self.logit_scale.clamp(1, 100).exp() + self.bias
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

    def evaluate(self, in_bool, in_text, plot=False, threshold=0.5):
        self.eval()
        with torch.no_grad():
            outputs = self(in_bool, in_text)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits)

        mask = torch.eye(probs.size(0), device=probs.device).bool()
        probs_pos = probs[mask]
        probs_neg = probs[~mask]

        if plot:
            for name, p in zip(["Positive", "Negative"], [probs_pos, probs_neg]):
                mean = p.mean().item()
                std = p.std().item()
                plt.hist(p.cpu().numpy().flatten(), bins=50, range=(0, 1), alpha=0.7, color='skyblue')
                plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {mean:.2f}")
                plt.title(f"{name} Score Distribution")
                plt.xlabel("Score")
                plt.ylabel("Frequency")
                plt.legend()
                plt.show()
                print(f"{name} mean: {mean:.4f} ± {std:.4f}")
            y_true = torch.cat([
                torch.ones_like(probs_pos),
                torch.zeros_like(probs_neg)
            ]).cpu().numpy()

            y_pred = torch.cat([
                probs_pos,
                probs_neg
            ]).cpu().numpy() > threshold

            cm = confusion_matrix(y_true, y_pred, normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
            disp.plot(cmap='Blues')
            plt.title("Confusion Matrix")
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