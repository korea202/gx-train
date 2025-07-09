import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import timm
from torchmetrics import Accuracy, F1Score
import numpy as np

        

class TuneModel(LightningModule):
    def __init__(self, model_name: str, config, pretrained: bool = True):
        super().__init__()

        # config 값들을 네이티브 타입으로 변환
        self.config = {
            key: float(value) if isinstance(value, (np.floating, np.integer)) else value
            for key, value in config.items()
        }
        self.save_hyperparameters()
        
        # timm 모델 생성
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes= config["num_classes"]
        )

        # 분류기 교체
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(config["dropout_rate"]),
            torch.nn.Linear(self.model.classifier.in_features, 17)
        )

        # 메트릭 정의
        self.train_accuracy = Accuracy(task="multiclass", num_classes=config["num_classes"])
        self.train_f1 = F1Score(task="multiclass", num_classes=config["num_classes"], average='macro')
        
        self.val_accuracy = Accuracy(task="multiclass", num_classes=config["num_classes"])
        self.val_f1 = F1Score(task="multiclass", num_classes=config["num_classes"],  average='macro')
        
        # 손실 함수
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 메트릭 계산
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.train_f1(preds, y)
                
        # 로깅
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 메트릭 계산
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.val_f1(preds, y)

        # 로깅
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])


        if self.config["scheduler_type"] == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=self.config["factor"], 
                patience=self.config["patience"]
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        
        elif self.config["scheduler_type"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.config["T_max"],
                eta_min=self.config.get("eta_min", 0)
            )
            return [optimizer], [scheduler]

 

