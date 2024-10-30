from pydantic import BaseModel


class DataConfig(BaseModel):
    batch_size: int = 512
    num_workers: int = 4
    max_length: int = 512


class TextConfig(BaseModel):
    model_name: str = "dunzhang/stella_en_400M_v5"
    hidden_channels: int = 1024
    out_channels: int = 1024


class MolConfig(BaseModel):
    num_features: int = 78
    hidden_channels: int = 512
    out_channels: int = 1024
    num_heads: int = 8
    dropout: float = 0.1


class TrainConfig(BaseModel):
    accelerator: str = "gpu"
    strategy: str = "ddp"
    devices: int | str = "auto"
    max_epochs: int = 30
    learning_rate: float = 1e-5
    gradient_clip_val: float = 1.0


class MolClipConfig(BaseModel):
    data: DataConfig
    text: TextConfig
    mol: MolConfig
    train: TrainConfig
