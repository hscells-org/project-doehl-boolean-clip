import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from transformers import AutoTokenizer, Siglip2TextModel, AutoModelForSequenceClassification
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"


class DualSiglip2Model(nn.Module):
    def __init__(self, model_name="google/siglip2-base-patch16-224"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder_bool = Siglip2TextModel.from_pretrained(model_name)
        # if "siglip" in model_name:
        #     self.encoder_bool = Siglip2TextModel.from_pretrained(model_name)
        # else:
        #     self.encoder_bool = AutoModelForSequenceClassification.from_pretrained(model_name)
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