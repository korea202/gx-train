import torch
import torch.nn as nn
import torchmetrics
import timm

from pytorch_lightning import LightningModule, Trainer
from hydra.utils import instantiate

class ClassificationDNN(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.learning_rate = cfg.optimizer.lr
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.criterion = nn.CrossEntropyLoss()

        self.hidden_dims = cfg.model.hidden_dims
        self.apply_batchnorm = cfg.model.apply_batchnorm
        self.apply_activation = cfg.model.apply_activation
        self.apply_dropout = cfg.model.apply_dropout
        self.dropout_ratio = cfg.model.dropout_ratio
        self.num_classes = cfg.model.num_classes


        self.layers = nn.ModuleList()
        
        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            
            if self.apply_batchnorm:
                self.layers.append(nn.BatchNorm1d(self.hidden_dims[i+1]))
            
            if self.apply_activation:
                self.layers.append(nn.ReLU())
            
            if self.apply_dropout:
                self.layers.append(nn.Dropout(self.dropout_ratio))

        self.fc_layer = nn.Linear(self.hidden_dims[-1], self.num_classes)


    def forward(self, x):
        x = x.view(x.shape[0], -1)  # [batch_size, 784]

        for layer in self.layers:
            x = layer(x)
        
        return self.fc_layer(x)    
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        acc = self.train_accuracy(torch.argmax(logits, dim=1), y)
        # 로깅
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 정확도 계산
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        
        # 로깅
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True)
        
        return loss      
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 정확도 계산
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, y)
        
        # 로깅
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]

