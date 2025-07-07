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
def prepare_data(model, batch_size=32, num_workers=4):
    
   # 데이터셋 생성
    train_dataset, val_dataset, test_dataset = get_datasets(model)

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

def main():

    # model config
    model_name = 'tf_efficientnet_b4' # 'resnet50' 'efficientnet_b4', ...

    # training config
    EPOCHS = 100
    BATCH_SIZE = 16
    num_workers = 0
    num_classes = 17
    learning_rate = 1e-4
    drop_out = 0.4
    do_test = True

    # 모델 초기화 전에 설정
    torch.set_float32_matmul_precision('medium')
    
    # WandB Logger 초기화
    wandb_logger = WandbLogger(
        project="cv-classification",                                                            # 프로젝트 이름
        name=utils.generate_experiment_name(model_name, learning_rate, BATCH_SIZE),             # 실험 이름 (선택사항)
        job_type="train",                                                                        # 작업 타입 (선택사항)
        save_dir=config.OUTPUTS_DIR
    )
   
    random_seed(42)

    model = CustomModelEx(
        model_name= model_name,
        num_classes=num_classes,
        learning_rate=learning_rate,
        drop_rate = drop_out
    )

    # 데이터 로더 준비
    train_loader, val_loader, test_loader = prepare_data(model=model, batch_size=BATCH_SIZE, num_workers=num_workers)
    

    # 콜백을 직접 생성
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
            min_delta=0.001,
            verbose=True
        ),
        ModelCheckpoint(
            save_top_k=1,
            monitor='val_loss',
            mode='min',
            filename='best-{epoch:02d}-{val_loss:.3f}'
        ), 
        LearningRateMonitor(
            logging_interval='epoch',
            log_momentum=False
        )
    ]

    trainer = Trainer(default_root_dir=config.OUTPUTS_DIR, max_epochs=EPOCHS, accelerator='auto', callbacks=callbacks, logger=wandb_logger)
    
    # 훈련
    trainer.fit(model, train_loader, val_loader)
    #trainer.fit(model, train_loader, val_loader, ckpt_path=config.OUTPUTS_DIR + "/cv-classification/69cazm6k/checkpoints/best-epoch=37-val_loss=0.650.ckpt")
    
    # 테스트
    if(do_test == True): 
        test(trainer, model, test_loader)

    # WandB 종료
    wandb.finish()
if __name__ == "__main__":
    main()