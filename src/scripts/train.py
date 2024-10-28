import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from rdkit import rdBase

from molclip.config import DataConfig, MolClipConfig, MolConfig, TextConfig, TrainConfig
from molclip.data import MolClipDataModule
from molclip.model import MolClip
from molclip.utils import fix_seeds


def main(notes: str | None = None, version: str | None = None):
    rdBase.DisableLog("rdApp.info")
    rdBase.DisableLog("rdApp.warning")

    fix_seeds()
    config = MolClipConfig(
        data=DataConfig(),
        text=TextConfig(),
        mol=MolConfig(),
        train=TrainConfig(),
    )
    model = MolClip(config)
    tokenizer = model.get_tokenizer()

    data_module = MolClipDataModule(config.data, tokenizer)
    data_module.setup()

    logger = WandbLogger(project="molclip", config=config.model_dump(), notes=notes, version=version)
    trainer = pl.Trainer(
        accelerator=config.train.accelerator,
        devices=config.train.devices,
        gradient_clip_val=config.train.gradient_clip_val,
        max_epochs=config.train.max_epochs,
        logger=logger,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    # for experiment tracking
    notes = "Adjusted parameters for training"
    version = "0.1.1"

    main(notes, version)
