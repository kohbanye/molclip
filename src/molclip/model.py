import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn
from torch_geometric import nn as gnn
from torch_geometric.nn import global_mean_pool
from torchmetrics import AUROC, Accuracy
from transformers import AutoModel, AutoTokenizer

from molclip.config import MolClipConfig, MolConfig, TextConfig
from molclip.type import AnnotatedTensor


class TextEncoder(nn.Module):
    def __init__(self, config: TextConfig):
        super(TextEncoder, self).__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(self.config.model_name, trust_remote_code=True).cuda()

        self.bn = nn.BatchNorm1d(self.config.hidden_channels)
        self.linear = nn.Linear(self.config.hidden_channels, self.config.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones_like(x)
        last_hidden_state = self.model(x, attention_mask)[0]
        last_hidden_state = last_hidden_state.mean(dim=1)

        normalized = self.bn(last_hidden_state)
        embedding = self.linear(normalized)

        return embedding


class MolEncoder(nn.Module):
    def __init__(self, config: MolConfig):
        super(MolEncoder, self).__init__()
        self.config = config
        self.gat1 = gnn.GATConv(
            self.config.num_features,
            self.config.hidden_channels // self.config.num_heads,
            heads=self.config.num_heads,
            dropout=self.config.dropout,
        )
        self.gat2 = gnn.GATConv(
            self.config.hidden_channels,
            self.config.hidden_channels // self.config.num_heads,
            heads=self.config.num_heads,
            dropout=self.config.dropout,
        )

        self.linear = nn.Linear(self.config.hidden_channels, self.config.out_channels)

        self.bn1 = nn.BatchNorm1d(self.config.hidden_channels)
        self.bn2 = nn.BatchNorm1d(self.config.hidden_channels)

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)

        x = global_mean_pool(x, data.batch)
        embedding = self.linear(x)

        return embedding


class Log(BaseModel):
    train_loss: AnnotatedTensor | None = None
    val_loss: AnnotatedTensor | None = None
    accuracy_text: AnnotatedTensor
    accuracy_mol: AnnotatedTensor
    auroc_text: AnnotatedTensor
    auroc_mol: AnnotatedTensor


class MolClip(pl.LightningModule):
    def __init__(self, config: MolClipConfig):
        super(MolClip, self).__init__()
        self.config = config
        self.text_encoder = TextEncoder(config.text)
        self.mol_encoder = MolEncoder(config.mol)
        self.loss_fn = nn.CrossEntropyLoss()
        self.tokenizer = AutoTokenizer.from_pretrained(config.text.model_name, trust_remote_code=True)
        self.training_step_outputs: list[Log] = []

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, mol_graph_batch) -> tuple[torch.Tensor, torch.Tensor]:
        mol_graph_batch = mol_graph_batch.cuda()

        encoded_mol_graph = self.mol_encoder(mol_graph_batch)

        input_ids = mol_graph_batch.y
        encoded_text = self.text_encoder(input_ids)

        return encoded_text, encoded_mol_graph

    def loss(self, logits_text: torch.Tensor, logits_mol: torch.Tensor) -> torch.Tensor:
        loss_text = self.loss_fn(logits_text, torch.arange(len(logits_text), device=logits_text.device))
        loss_mol = self.loss_fn(logits_mol, torch.arange(len(logits_mol), device=logits_mol.device))

        return (loss_text + loss_mol) / 2

    def calc_metrics(self, logits_text: torch.Tensor, logits_mol: torch.Tensor) -> Log:
        accuracy = Accuracy(task="multiclass", num_classes=logits_text.shape[0]).to(logits_text.device)
        auroc = AUROC(task="multiclass", num_classes=logits_text.shape[0]).to(logits_text.device)

        target_text = torch.arange(len(logits_text), device=logits_text.device)
        target_mol = torch.arange(len(logits_mol), device=logits_mol.device)

        return Log(
            accuracy_text=accuracy(logits_text, target_text),
            accuracy_mol=accuracy(logits_mol, target_mol),
            auroc_text=auroc(logits_text, target_text),
            auroc_mol=auroc(logits_mol, target_mol),
        )

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        mol_graph = batch
        encoded_text, encoded_mol = self.forward(mol_graph)

        logits_text = encoded_mol @ encoded_text.T
        logits_mol = encoded_text @ encoded_mol.T

        loss = self.loss(logits_text, logits_mol)

        log = self.calc_metrics(logits_text, logits_mol)
        log.train_loss = loss
        self.training_step_outputs.append(log)

        self.log_dict(log.model_dump(exclude_none=True))

        return loss

    def on_train_epoch_end(self) -> None:
        avg_loss = torch.stack([x.train_loss for x in self.training_step_outputs if x.train_loss is not None]).mean()
        self.log("avg_loss", avg_loss)

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        mol_graph = batch
        encoded_text, encoded_mol = self.forward(mol_graph)

        logits_text = encoded_mol @ encoded_text.T
        logits_mol = encoded_text @ encoded_mol.T

        loss = self.loss(logits_text, logits_mol)

        log = self.calc_metrics(logits_text, logits_mol)
        log.val_loss = loss

        self.log_dict(log.model_dump(exclude_none=True))

        return loss

    def test_step(self, batch, batch_idx):
        mol_graph = batch
        encoded_text, encoded_mol = self.forward(mol_graph)

        logits_text = encoded_mol @ encoded_text.T
        logits_mol = encoded_text @ encoded_mol.T

        log = self.calc_metrics(logits_text, logits_mol)

        self.log_dict(log.model_dump(exclude_none=True))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.train.learning_rate)  # type: ignore
        return optimizer
