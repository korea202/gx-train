import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

class SimpleRegressionDataset(Dataset):
    def __init__(self, X, y, scaler=None):
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(X)
        
        self.target_scaler = StandardScaler()
        self.labels = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if label.dim() == 0:
            label = label.unsqueeze(0)
        
        return features, label
