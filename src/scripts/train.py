import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from rdkit import rdBase

from molclip.config import DataConfig, MolClipConfig, MolConfig, TextConfig, TrainConfig
from molclip.data import MolClipDataModule
from molclip.model import MolClip


def main():
    rdBase.DisableLog("rdApp.info")
    rdBase.DisableLog("rdApp.warning")

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

    logger = WandbLogger(project="molclip")
    trainer = pl.Trainer(
        accelerator=config.train.accelerator,
        devices=config.train.devices,
        max_epochs=config.train.max_epochs,
        logger=logger,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
