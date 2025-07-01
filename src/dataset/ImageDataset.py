import os

from torch.utils.data import Dataset, random_split
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
import pandas as pd
import numpy as np

from src.utils import config 

class ImageDataset(Dataset):
    def __init__(self, csv, path, transform=None):
        self.df = pd.read_csv(csv).values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target

img_size = 32

# augmentation을 위한 transform 코드
trn_transform = A.Compose([
    # 이미지 크기 조정
    A.Resize(height=img_size, width=img_size),
    # images normalization
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # numpy 이미지나 PIL 이미지를 PyTorch 텐서로 변환
    ToTensorV2(),
])

# test image 변환을 위한 transform 코드
tst_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])  


def get_datasets():
    
    # Dataset 정의
    train_dataset = ImageDataset(
        config.CV_CLS_TRAIN_CSV,
        config.CV_CLS_TRAIN_DIR,
        transform=trn_transform
    )

    # 전체 데이터셋을 8:2로 분할
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size]
        #generator=torch.Generator().manual_seed(42)  # 재현 가능성을 위한 시드
    )

    test_dataset = ImageDataset(
        config.CV_CLS_TEST_CSV,
        config.CV_CLS_TEST_DIR,
        transform=tst_transform
    )

    return train_dataset, val_dataset, test_dataset   