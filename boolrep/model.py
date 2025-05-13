from torch import nn
from transformers import AutoModel, AutoConfig


class TextEncoder(nn.Module):
    def __init__(self, model_name, requires_grad=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.embedding_dims = self.config.hidden_size

        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:,0,:]