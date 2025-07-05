import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import timm
from torchmetrics import Accuracy, F1Score
import wandb
import numpy as np

class CustomModel(LightningModule):
    def __init__(
        self, 
        model_name: str,
        num_classes: int = 17,
        learning_rate: float = 1e-3,
        drop_rate:float = 0.0,
        pretrained: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # timm 모델 생성
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate = drop_rate
        )
        
        # 메트릭 정의
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes,  average='macro')
        
        # 손실 함수
        self.criterion = nn.CrossEntropyLoss()

        # 테스트 결과 수집용 리스트 초기화
        self.test_step_outputs = []
        
        # 최종 결과 저장용 변수 초기화
        self.test_predictions = None
        
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

        # 오답 샘플 수집
        if self.current_epoch == self.trainer.max_epochs - 1: 
            misclassified = []
            for i in range(len(preds)):
                if preds[i] != y[i]:
                    img = prepare_image_for_wandb(x[i]) 
                    wandb_image = wandb.Image(img, caption=f"Pred: {preds[i]}, Label: {y[i]}")
                    misclassified.append(wandb_image)    
            # WandB Table에 기록
            wandb.log({"misclassified_images": misclassified})

        # 로깅
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        
        # 테스트 결과 저장
        
        result =  {'preds': preds, 'targets': y}
        self.test_step_outputs.append(result)
        
        return result

    def on_test_epoch_end(self):
        # 모든 배치 결과 수집
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs])
        
        # 인스턴스 변수에 저장
        self.test_predictions = all_preds.detach().cpu().numpy()
        
        # 메모리 정리
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        # 스케줄러 추가 (선택사항)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

def prepare_image_for_wandb(img):
    """WandB 업로드용 이미지 전처리"""
    # GPU 텐서를 CPU로 이동
    img = img.detach().cpu()
    
    # 정규화 역변환 (ImageNet 기준)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = img * std.view(3, 1, 1) + mean.view(3, 1, 1)
    
    # 값 범위 클램핑 (0~1)
    img = torch.clamp(img, 0, 1)
    
    # CHW -> HWC 변환
    img = img.permute(1, 2, 0)
    
    # NumPy 배열로 변환
    img = img.numpy()
    
    # 0~255 범위로 변환 (선택사항)
    img = (img * 255).astype(np.uint8)
    
    return img