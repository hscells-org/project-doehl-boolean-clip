import lightning as L

from boolrep.model import TextEncoder


class BooleanQueryPairwiseEncoderModel(L.LightningModule):
    def __init__(self, encoder_name):
        super().__init__()

        self.d_encoder = TextEncoder(encoder_name, requires_grad=False)
        self.d_encoder.requires_grad_(False)
        self.q_encoder = TextEncoder(encoder_name)

        self.save_hyperparameters()

    def forward(self, inputs):
        d_embeddings = self.d_encoder(
            input_ids=inputs["input_ids_d"], 
            attention_mask=inputs["attention_mask_d"]
        )
        q_embeddings = self.q_encoder(
            input_ids=inputs["input_ids_q"], 
            attention_mask=inputs["attention_mask_q"]
        )
        return d_embeddings, q_embeddings
    
    def _compute_losses(self, d_embeddings, q_embeddings):
        similarity = d_embeddings @ q_embeddings