import os
import sys

import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from dotenv import load_dotenv, dotenv_values

sys.argv = ['']
# 환경변수 읽기
if (python_path := dotenv_values().get('PYTHONPATH')) and python_path not in sys.path: sys.path.append(python_path)

from src.models.RegressionDNN import RegressionDNN
from src.dataset.SimpleRegressionDataset import get_datasets
from src.utils import config

# 데이터 준비 함수
def prepare_data(batch_size=64, num_workers=4):
    
   # 데이터셋 생성
    train_dataset, val_dataset, test_dataset = get_datasets()

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

@hydra.main(config_path=config.CONFIGS_DIR, config_name="config", version_base=None)
def main(cfg):
    print(cfg)

    pl.seed_everything(42)
    
    # 데이터 로더 준비
    train_loader, val_loader, test_loader = prepare_data(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )

    model = RegressionDNN(cfg)

     # 콜백을 직접 생성
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min',
            min_delta=0.001,
            verbose=True
        ),
        LearningRateMonitor(
            logging_interval='epoch',
            log_momentum=False
        )
    ]

    trainer = Trainer(default_root_dir=config.OUTPUTS_DIR, max_epochs=cfg.trainer.max_epochs, accelerator=cfg.trainer.accelerator, callbacks=callbacks)
    
    # 훈련
    trainer.fit(model, train_loader, val_loader)
    
    # 테스트
    test_results = trainer.test(model, test_loader)

    # 회귀 모델의 테스트 결과 출력
    if test_results and len(test_results) > 0:
        result = test_results[0]
        
        # 회귀 메트릭들 출력
        test_loss = result.get('test_loss', 0.0)
        test_rmse = result.get('test_rmse', 0.0)
        
        print(f"최종 테스트 손실 (MSE): {test_loss:.4f}")
        print(f"최종 테스트 RMSE: {test_rmse:.4f}")
        
        # R² Score가 있다면
        test_r2 = result.get('test_r2', None)
        if test_r2 is not None:
            print(f"최종 테스트 R² Score: {test_r2:.4f}")
    else:
        print("테스트 결과를 가져올 수 없습니다.")

if __name__ == "__main__":
    main()