import torch
import torch.nn as nn
import torchmetrics
import timm

from pytorch_lightning import LightningModule, Trainer
from hydra.utils import instantiate
import torchmetrics.regression

class RegressionDNN(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.learning_rate = cfg.optimizer.lr
        self.weight_decay = cfg.optimizer.weight_decay
        self.train_rmse  =  torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse  = torchmetrics.MeanSquaredError(squared=False)
        self.test_rmse  = torchmetrics.MeanSquaredError(squared=False)
        self.criterion = nn.MSELoss()

        self.input_dim = cfg.model.input_dim
        self.hidden_dims = cfg.model.hidden_dims
        self.apply_batchnorm = cfg.model.apply_batchnorm
        self.apply_activation = cfg.model.apply_activation
        self.apply_dropout = cfg.model.apply_dropout
        self.dropout_ratio = cfg.model.dropout_ratio
        self.output_dim = cfg.model.output_dim


        self.layers = nn.ModuleList()
        
        # 첫 번째 레이어
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        
        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            
            if self.apply_batchnorm:
                self.layers.append(nn.BatchNorm1d(self.hidden_dims[i+1]))
            
            if self.apply_activation:
                self.layers.append(nn.ReLU())
            
            if self.apply_dropout:
                self.layers.append(nn.Dropout(self.dropout_ratio))

        self.fc_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)

        # 가중치 초기화 적용
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(0.01)


    def forward(self, x):
        # 디버깅용 - 나중에 제거
        #print(f"Input shape: {x.shape}")
        
        for layer in self.layers:
            x = layer(x)
        
        return self.fc_layer(x)    
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)

        # Shape 확인
        #print(f"Preds shape: {preds.shape}")
        #print(f"Y shape: {y.shape}")
        #print(f"x:{x[0]}")
        #print(f"preds:{preds[0]}")
        #print(f"y:{y}")

        loss = self.criterion(preds, y)

        # 훈련용 RMSE 누적
        self.train_rmse.update(preds, y)
        
        # 로깅
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_rmse', self.train_rmse, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        
        # 검증용 RMSE 누적
        self.val_rmse.update(preds, y)
        
        # 로깅
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_rmse', self.val_rmse, on_step=False, on_epoch=True)
        
        return loss      
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        
        # 테스트용 RMSE 누적
        self.test_rmse.update(preds, y)
        
        # 로깅
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_rmse', self.test_rmse, on_step=False, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return [optimizer]

