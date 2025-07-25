import os

from torch.utils.data import Dataset, ConcatDataset, random_split
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2

from src.dataset.MemoryImageSource import MemoryImageSource
from src.dataset.FullAugraphyPipelineQueue import FullAugraphyPipelineQueue
from src.utils import config 

class CvImageDatasetFast(Dataset):
    def __init__(self, source, transform=None, img_size=config.IMAGE_SIZE ):
        self.df = source.df.values
        self.img_source = source
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.img_source.df)

    def __getitem__(self, idx):
        name, target = self.img_source.df.iloc[idx]
        img = self.img_source[name]
        if self.transform:
            img = self.transform(img)
        return img, target


# 파이프라인
augmentation_pipeline = FullAugraphyPipelineQueue(max_effects=2)

train_source = MemoryImageSource(config.CV_CLS_TRAIN_CSV, config.CV_CLS_TRAIN_DIR)
test_source = MemoryImageSource(config.CV_CLS_TEST_CSV,config.CV_CLS_TEST_DIR)

# 커스텀 변환 클래스 정의
class To_BGR(object):
    """PIL RGB 이미지를 numpy BGR 이미지로 변환"""
    def __call__(self, image):
        image_numpy = np.array(image)
        if len(image_numpy.shape) < 3:
            return cv2.cvtColor(image_numpy, cv2.COLOR_GRAY2BGR)
        else:
            return cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)

# 수정된 변환 파이프라인
dirty_transforms = transforms.Compose([
    To_BGR(),
    augmentation_pipeline,  # Augraphy 적용
    transforms.ToTensor(),  # numpy -> tensor
    # 정규화
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

clean_transforms = transforms.Compose([
    transforms.ToTensor(),
    # 정규화
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_datasets(model=None):
    
    # 데이터셋 생성
    d1 = CvImageDatasetFast(train_source, transform=dirty_transforms)
    d2 = CvImageDatasetFast(train_source, transform=clean_transforms)

    train_dataset = ConcatDataset([d1, d2])

    # 전체 데이터셋을 8:2로 분할
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size]
        #generator=torch.Generator().manual_seed(42)  # 재현 가능성을 위한 시드
    )

    test_dataset = CvImageDatasetFast(test_source, transform=clean_transforms
    )

    return train_dataset, val_dataset, test_dataset   