import itertools
import lightning as L
from torch import nn
import torch

from boolrep.model import TextEncoder

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float):
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)

        x += projected

        return self.layer_norm(x)


class BooleanQueryEncoderModel(L.LightningModule):
    def __init__(self, encoder_name):
        super().__init__()
        self.d_encoder = TextEncoder(encoder_name, requires_grad=False)
        self.d_encoder.requires_grad_(False)

        self.q_encoder = TextEncoder(encoder_name)

        self.d_projection = ProjectionHead(
            embedding_dim=self.d_encoder.embedding_dims,
            projection_dim=128,
            dropout=0.0
        )
        self.q_projection = ProjectionHead(
            embedding_dim=self.q_encoder.embedding_dims,
            projection_dim=128,
            dropout=0.0
        )

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.temperature = 1.0

        self.save_hyperparameters()

    def forward(self, inputs):
        d_features = self.d_encoder(
            input_ids=inputs["input_ids_d"], 
            attention_mask=inputs["attention_mask_d"]
        )
        q_features = self.q_encoder(
            input_ids=inputs["input_ids_q"], 
            attention_mask=inputs["attention_mask_q"]
        )

        d_embeddings = self.d_projection(d_features)
        q_embeddings = self.q_projection(q_features)

        return d_embeddings, q_embeddings
    
    def _compute_losses(self, d_embeddings, q_embeddings):
        logits = (d_embeddings @ q_embeddings.T) / self.temperature
        
        labels = torch.arange(len(d_embeddings)).to(self.device)
        d_loss = torch.nn.functional.cross_entropy(logits.T,labels)
        q_loss = torch.nn.functional.cross_entropy(logits,labels)

        return (q_loss+d_loss) / 2.0

    def training_step(self, batch):
        d_embeddings, q_embeddings = self.forward(batch)
        loss = self._compute_losses(d_embeddings, q_embeddings).mean()
        train_loss = self.all_gather(loss)
        self.log("train/loss", train_loss.mean())
        return loss

    def validation_step(self, batch):
        d_embeddings, q_embeddings = self.forward(batch)
        loss = self._compute_losses(d_embeddings, q_embeddings).mean()
        train_loss = self.all_gather(loss)
        self.log("val/loss", train_loss.mean())
        return loss

    def configure_optimizers(self):
        parameters = [
            {"params": self.d_encoder.parameters(), "lr": 1e-6},
            {"params": self.q_encoder.parameters(), "lr": 1e-6},
            {
                "params": itertools.chain(
                    self.d_projection.parameters(),
                    self.q_projection.parameters(),
                ),
                "lr": 1e-4,
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.Adam(parameters, weight_decay=0.0)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=1.0,
            factor=0.8,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }