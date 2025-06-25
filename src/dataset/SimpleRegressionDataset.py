import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_california_housing

class SimpleRegressionDataset(Dataset):
    def __init__(self, X, y, scaler=None, label_scaler=None, label_encoders=None):

        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(X)

        if label_scaler is None:
            self.label_scaler = StandardScaler()
            self.labels = self.label_scaler.fit_transform(y.reshape(-1, 1)).flatten()        
        else:
            self.label_scaler = label_scaler
            self.labels = self.label_scaler.transform(y.reshape(-1, 1)).flatten()   
       
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if label.dim() == 0:
            label = label.unsqueeze(0)
        
        return features, label
    
def split_dataset(X, y):
    
    X_tmp, X_val, y_tmp, y_val = train_test_split(X ,y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_datasets(scaler=None, label_scaler=None, label_encoders=None):
    
     # 1. 데이터 로드
    data = fetch_california_housing()
    X, y = data.data, data.target

    # 3. 데이터 분할
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    # PyTorch Dataset 객체 생성
    train_dataset = SimpleRegressionDataset(X_train, y_train, scaler, label_scaler)
    val_dataset = SimpleRegressionDataset(X_val, y_val, scaler=train_dataset.scaler, label_scaler=train_dataset.label_scaler)
    test_dataset = SimpleRegressionDataset(X_test, y_test, scaler=train_dataset.scaler, label_scaler=train_dataset.label_scaler)
    
    return train_dataset, val_dataset, test_dataset

