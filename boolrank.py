import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from transformers import AutoTokenizer, AutoModel, Siglip2TextModel
from transformers import BartForConditionalGeneration, BartConfig
from transformers.modeling_outputs import BaseModelOutput
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

        self.decoder = CrossAttnDecoder(self.encoder_text.config.hidden_size)
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
        tok_bool = self.tokenize(in_bool)
        tok_text = self.tokenize(in_text)
        out_bool = self.encoder_bool(**tok_bool).pooler_output
        out_text = self.encoder_text(**tok_text).pooler_output
        out_bool = out_bool / out_bool.norm(p=2, dim=-1, keepdim=True)
        out_text = out_text / out_text.norm(p=2, dim=-1, keepdim=True)
        logits = out_bool @ out_text.t()  # + self.bias

        decoded = self.decoder(
            z_emb               = out_text,
            decoder_input_ids   = tok_bool.input_ids,
            decoder_attention_mask = tok_bool.attention_mask,
            labels              = tok_bool.input_ids
        )

        if return_loss:
            match self.loss_type:
                case "siglip": loss = self.siglip_loss(logits)
                case "clip": loss = clip_loss(logits)
            loss += decoded.loss
        else: loss = None
        return {"loss": loss, "logits": logits, "decoded": decoded}

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


class CrossAttnDecoder(nn.Module):
    def __init__(self, emb_dim, decoder=None):
        super().__init__()
        if decoder is None:
            config = BartConfig.from_pretrained("facebook/bart-base")
            config.is_decoder = True
            config.add_cross_attention = True
            decoder = BartForConditionalGeneration(config)

        self.decoder = decoder
        self.enc_proj = nn.Linear(emb_dim, decoder.config.d_model)
        self.to(device)

    def forward(
        self,
        z_emb: torch.FloatTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        decoder_attention_mask: torch.LongTensor = None,
        labels: torch.LongTensor = None
    ):
        B = z_emb.size(0)
        enc_states = self.enc_proj(z_emb).unsqueeze(1)
        enc_mask   = torch.ones(B, 1, device=z_emb.device)

        return self.decoder(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=(enc_states,),
            attention_mask=enc_mask,
            labels=labels,
            return_dict=True
        )

    def generate(self, emb):
        B, D = emb.size()
        enc_states = self.enc_proj(emb).unsqueeze(1)

        enc_mask = torch.ones(B, 1, device=emb.device)

        return self.decoder.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_states),
            attention_mask=enc_mask,
            max_length=32,
            num_beams=4,
            early_stopping=True
        )
