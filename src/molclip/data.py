import polars
import pytorch_lightning as pl
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from molclip.config import DataConfig
from molclip.feature import get_mol_graph


class MolClipDataset(Dataset):
    def __init__(
        self,
        data_url: str,
        max_length: int,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ):
        self.data = polars.read_csv(data_url, separator="\t")
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Data:
        text = self.data["description"][idx]
        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"]

        smiles = self.data["SMILES"][idx]
        mol_graph = get_mol_graph(smiles, input_ids)  # type: ignore

        return mol_graph


class MolClipDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: DataConfig,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.batch_size = self.config.batch_size
        self.num_workers = self.config.num_workers
        self.train_dataset_url = (
            "https://raw.githubusercontent.com/blender-nlp/MolT5/refs/heads/main/ChEBI-20_data/train.txt"
        )
        self.validation_dataset_url = (
            "https://raw.githubusercontent.com/blender-nlp/MolT5/refs/heads/main/ChEBI-20_data/validation.txt"
        )
        self.test_dataset_url = (
            "https://raw.githubusercontent.com/blender-nlp/MolT5/refs/heads/main/ChEBI-20_data/test.txt"
        )

    def setup(self, stage: str | None = None):
        self.train_dataset = MolClipDataset(self.train_dataset_url, self.config.max_length, self.tokenizer)
        self.val_dataset = MolClipDataset(self.validation_dataset_url, self.config.max_length, self.tokenizer)
        self.test_dataset = MolClipDataset(self.test_dataset_url, self.config.max_length, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
