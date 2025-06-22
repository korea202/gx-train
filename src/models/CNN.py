import torch
import torch.nn as nn
import torchmetrics
import timm

from pytorch_lightning import LightningModule, Trainer
from hydra.utils import instantiate

class SimpleCNN(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.learning_rate = cfg.optimizer.lr
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.model.model.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.scheduler

        self.num_classes = cfg.model.model.num_classes
        self.dropout_ratio = cfg.model.model.dropout_ratio

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(self.dropout_ratio),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(self.dropout_ratio),
        )
        self.fc_layer = nn.Linear(64 * 2 * 2, self.num_classes)

    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        return self.fc_layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits.softmax(dim=-1), y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, self.parameters())
        scheduler = instantiate(self.scheduler_cfg, optimizer)
        return [optimizer], [scheduler]


class ResNet(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.learning_rate = cfg.optimizer.lr
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.model.model.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.scheduler

        self.model = instantiate(cfg.model.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits.softmax(dim=-1), y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, self.parameters())
        scheduler = instantiate(self.scheduler_cfg, optimizer)
        return [optimizer], [scheduler]