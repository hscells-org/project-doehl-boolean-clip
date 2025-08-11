import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from transformers import AutoTokenizer, AutoModel, Siglip2TextModel
from safetensors.torch import load_file
from transformers.models.clip.modeling_clip import clip_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

class DualSiglip2Model(nn.Module):
    def __init__(self, model_name="google/siglip2-base-patch16-224", loss_type="siglip"):
        super().__init__()
        self.model_name = model_name
        self.loss_type = loss_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if "siglip" in model_name:
            self.encoder_bool = Siglip2TextModel.from_pretrained(model_name, trust_remote_code=True)
        else:
            self.encoder_bool = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.encoder_text = deepcopy(self.encoder_bool)
        self.bias = nn.Parameter(torch.zeros(1))
        self.to(device)

    def tokenize(self, texts):
        if "siglip" in self.model_name:
            return self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt", max_length=64).to(device)
        else:
            # try:
            #     return self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt").to(device)
            # except:
            return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip2/modeling_siglip2.py#L952
    def forward(self, in_bool, in_text, return_loss=True):
        out_bool = self.encode_bool(in_bool, eval_mode=not return_loss)
        out_text = self.encode_text(in_text, eval_mode=not return_loss)
        logits = self.get_similarities(out_bool, out_text)  # + self.bias
        if return_loss:
            match self.loss_type:
                case "siglip": loss = self.siglip_loss(logits)
                case "clip": loss = clip_loss(logits)
        else: loss = None
        return {"loss": loss, "logits": logits}

    def get_similarities(self, a, b): return torch.tensor(a) @ torch.tensor(b).t()

    def encode_text(self, in_text, batch_size=1, eval_mode=True):
        single = False
        if isinstance(in_text, str):
            in_text = [in_text]
            single = True

        if eval_mode: self.encoder_text.eval()
        context = torch.no_grad() if eval_mode else torch.enable_grad()

        all_emb = []
        with context:
            for i in range(0, len(in_text), batch_size):
                batch = in_text[i : i + batch_size]
                tok = self.tokenize(batch)
                out = self.encoder_text(**tok).pooler_output
                all_emb.append(out)

        emb_cat = torch.cat(all_emb, dim=0)
        emb_cat = emb_cat / emb_cat.norm(p=2, dim=-1, keepdim=True)
        return emb_cat[0] if single else emb_cat

    def encode_bool(self, in_bool, eval_mode=True, batch_size = 1):
        single = False
        if isinstance(in_bool, str):
            in_bool = [in_bool]
            single = True
            single = True

        if eval_mode: self.encoder_text.eval()
        context = torch.no_grad() if eval_mode else torch.enable_grad()

        all_emb = []
        with context:
            for i in range(0, len(in_bool), batch_size):
                batch = in_bool[i : i + batch_size]
                tok = self.tokenize(batch)
                out = self.encoder_bool(**tok).pooler_output
                all_emb.append(out)

        emb_cat = torch.cat(all_emb, dim=0)
        emb_cat = emb_cat / emb_cat.norm(p=2, dim=-1, keepdim=True)
        return emb_cat[0] if single else emb_cat

    def siglip_loss(self, logits):
        sim = logits + self.bias
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