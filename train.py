from multiprocessing import freeze_support
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning import Trainer

from boolrep.data import BooleanQueryDataModule
from boolrep.model_clip import BooleanQueryEncoderModel



if __name__ == "__main__":
    freeze_support()
    model_name = "tavakolih/all-MiniLM-L6-v2-pubmed-full"
    model = BooleanQueryEncoderModel(model_name)
    data_module = BooleanQueryDataModule(
        model_name=model_name
    )

    logger = WandbLogger(project="BooleanCLIP", log_model="all")

    model_checkpoint = ModelCheckpoint()
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        accelerator="mps", 
        logger=logger,
        max_epochs=20,
        log_every_n_steps=1,
        callbacks=[model_checkpoint, lr_monitor]
    )
    trainer.fit(model, data_module)