import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.utils import config

class HousePricingDataset(Dataset):
    def __init__(self, df, scaler=None, label_scaler=None, label_encoders=None):
        self.df = df
        self.features = None
        self.labels = None
        self.scaler = scaler
        self.label_scaler = label_scaler
        self.label_encoders = label_encoders
        self._preprocessing()

    def _preprocessing(self):
        target_nm = 'target'
        # 타겟 및 피처 정의
        labels = self.df[target_nm].to_numpy()
        features = self.df.drop(columns=[target_nm], axis=1).to_numpy()

        # 라벨 스케일링
        if self.label_scaler:
            self.labels = self.label_scaler.transform(labels.reshape(-1, 1))
        else:
            self.label_scaler = StandardScaler()
            self.labels = self.label_scaler.fit_transform(labels.reshape(-1, 1))

        # 피처 스케일링
        if self.scaler:
            self.features = self.scaler.transform(features)
        else:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features)

    @property
    def features_dim(self):
        return self.features.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # PyTorch 텐서로 변환하여 반환
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # 회귀 문제라면 float32, 분류라면 long

        # 1차원으로 반환 (스칼라를 1차원으로)
        if label.dim() == 0:
            label = label.unsqueeze(0)  # [] → [1]

        return features, label


def read_dataset():
    return pd.read_csv(config.HOUSE_PRICING_DATA)


# 사용 컬럼 셋팅 
def set_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        #'계약년', 
        #'계약월', 
        #'계약일', 
        '중개사소재지',
        #시군구랑 시는 삭제합니다.
        '시군구', 
        '시', 
        '번지',
        #'k-전체세대수',
        'k-전체동수', 
        'k-연면적', 
        'k-주거전용면적', 
        'k-관리비부과면적',
        '경비비관리형태', 
        '청소비관리형태',
        #'k-사용검사일-사용승인일',
        '단지승인일', 
        '사용허가여부',
        'k-단지분류(아파트,주상복합등등)', 
        'k-세대타입(분양형태)',
        'k-관리방식', 
        'k-복도유형', 
        #'k-난방방식',
        '세대전기계약방법',
        '기타/의무/임대/임의=1/2/3/4', 
        'k-팩스번호', 
        '관리비 업로드',
        '단지신청일', 
        '등기신청일자', 
        '거래유형',
        'k-사용검사일-사용승인일', 
        'k-수정일자', 
        '고용보험관리번호', 
        'k-시행사', 
        '신축여부',
        'k-전용면적별세대현황(60㎡이하)', 
        'k-전용면적별세대현황(60㎡~85㎡이하)'
    ]

    # 존재하지 않는 컬럼이 있을 경우 오류를 방지하기 위해 errors='ignore' 옵션 사용
    df.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')

    numerical_list = []  # 숫자형 변수를 담아두기 위한 빈 List 생성
    categorical_list = []  # 문자형 변수를 담아두기 위한 빈 List 생성

    target_nm = 'target'

    for i in df.drop(columns=[target_nm], axis=1).columns:
        if df[i].dtypes == 'O':  # Object 타입 (문자형)
            categorical_list.append(i)
        else:
            numerical_list.append(i)
    
    print("숫자형 변수:", numerical_list)
    print("문자형 변수:", categorical_list)

    label_encoders = {}
    
    for col in categorical_list:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str)) 
        label_encoders[col] = encoder

    return df, label_encoders 


def split_dataset(df):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
    return train_df, val_df, test_df


def get_datasets(scaler=None, label_scaler=None, label_encoders=None):
    # 1. 데이터 읽기
    df = read_dataset()

    # 2. 컬럼 셋팅
    df, label_encoders = set_columns(df)

    # 3. 데이터 분할
    train_df, val_df, test_df = split_dataset(df)

    # PyTorch Dataset 객체 생성
    train_dataset = HousePricingDataset(train_df, scaler, label_scaler, label_encoders)
    val_dataset = HousePricingDataset(val_df, scaler=train_dataset.scaler, label_scaler=train_dataset.label_scaler, label_encoders=train_dataset.label_encoders)
    test_dataset = HousePricingDataset(test_df, scaler=train_dataset.scaler, label_scaler=train_dataset.label_scaler, label_encoders=train_dataset.label_encoders)
    
    return train_dataset, val_dataset, test_dataset

""" 
# 사용 예제
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # 데이터셋 생성
    train_dataset, val_dataset, test_dataset = get_datasets()
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 데이터 확인
    for batch_idx, (features, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Features shape = {features.shape}, Labels shape = {labels.shape}")
        if batch_idx == 0:  # 첫 번째 배치만 확인
            break """
