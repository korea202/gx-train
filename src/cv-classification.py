import os
import sys
import torch

import pandas as pd
import numpy as np
import torch.nn as nn
import wandb

from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from PIL import Image
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from dotenv import load_dotenv, dotenv_values

import hydra
from omegaconf import DictConfig

# 하이드라와 주피터 노트북은 아규먼트 관련 충돌이 발생하므로 초기화 해줌
sys.argv = ['']
# 환경변수 읽기

load_dotenv()
if (python_path := dotenv_values().get('PYTHONPATH')) and python_path not in sys.path: sys.path.append(python_path)

from src.dataset.CvImageDatasetFastEx import get_datasets
#from src.dataset.CvImageDataset import get_datasets
from src.models.CustomModelEx import CustomModelEx
from src.utils import config, utils

# 시드 고정
def random_seed(seed_num=42):

    """ SEED = seed_num
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True """
    
    # seed_everything 은 위의 내용 제어 + 밑에내용
    pl.seed_everything(seed_num)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

# 데이터 준비 함수
def prepare_data(default_cfg, batch_size=32, num_workers=4):
    
   # 데이터셋 생성
    train_dataset, val_dataset, test_dataset = get_datasets(default_cfg)

    # DataLoader 정의
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,  # 별도의 검증 데이터셋
        batch_size=batch_size,
        shuffle=False,  # 검증 시에는 셔플하지 않음
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader

def test(trainer, model, test_loader):

    # 테스트
    trainer.test(model, test_loader)

    print("테스트 갯수=",len(model.test_predictions))
    
    if len(model.test_predictions) > 0:
        # 모든 예측값과 실제값 합치기
        all_preds = model.test_predictions
        
        pred_df = pd.DataFrame(test_loader.dataset.df, columns=['ID', 'target'])
        pred_df['target'] = all_preds

        sample_submission_df = pd.read_csv(config.CV_CLS_TEST_CSV)
        assert (sample_submission_df['ID'] == pred_df['ID']).all()
        pred_df.to_csv(config.OUTPUTS_DIR + "/pred.csv", index=False)

    else:
        print("테스트 결과를 가져올 수 없습니다.")

@hydra.main(config_path=config.CONFIGS_DIR, config_name="config", version_base=None)
def main(cfg):

    # 모델 초기화 전에 설정
    torch.set_float32_matmul_precision('medium')
    
    # WandB Logger 초기화
    wandb_logger = WandbLogger(
        project="document-image-classification",                                                                            # 프로젝트 이름
        name=utils.generate_experiment_name(cfg.model.model_name, cfg.optimizer.learning_rate, cfg.data.batch_size),        # 실험 이름 (선택사항)
        job_type="train",                                                                                                   # 작업 타입 (선택사항)
        save_dir=config.OUTPUTS_DIR,
        log_model=True

    )
   
    random_seed(cfg.custom.seed_num)

    model = CustomModelEx(
        model_name= cfg.model.model_name,
        num_classes= cfg.model.num_classes,
        learning_rate= cfg.optimizer.learning_rate,
        dropout_rate= cfg.model.dropout_rate,
        pretrained= cfg.model.pretrained )

    # 데이터 로더 준비
    train_loader, val_loader, test_loader = prepare_data(default_cfg=model.model.default_cfg, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    

    # 콜백을 직접 생성
    early_stopping = hydra.utils.instantiate(cfg.callbacks.early_stopping)
    lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)
    model_checkpoint = hydra.utils.instantiate(cfg.callbacks.model_checkpoint)

    callbacks = [early_stopping, lr_monitor, model_checkpoint]

    trainer = Trainer(default_root_dir=config.OUTPUTS_DIR, max_epochs=cfg.trainer.max_epochs, accelerator=cfg.trainer.accelerator, 
                      callbacks=callbacks, logger=wandb_logger, num_sanity_val_steps=cfg.trainer.num_sanity_val_steps)
    
    # 훈련
    if cfg.custom.do_checkpoint == True and os.path.exists(cfg.custom.ckpt_path):
        trainer.fit(model, train_loader, val_loader, ckpt_path=cfg.custom.ckpt_path)
    else:
        trainer.fit(model, train_loader, val_loader)
    
    # 테스트
    if(cfg.custom.do_test == True): 
        test(trainer, model, test_loader)

    # WandB 종료
    wandb.finish()
if __name__ == "__main__":
    main()