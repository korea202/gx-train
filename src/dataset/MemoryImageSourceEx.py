import os

from PIL import Image
import pandas as pd

from src.utils import config 

class MemoryImageSourceEx:
    def __init__(self, csv, path, target_cls=None):
     
        if target_cls is None: target_cls = []

        df = pd.read_csv(csv)
        if len(target_cls) == 0:
            filtered_df = df
        else:
            filtered_df = df[df['target'].isin(target_cls)]

        self.df = filtered_df 
        self.image_dict = {}

        for index, row in self.df.iterrows():
            if len(target_cls) == 0 or row['target'] in target_cls:
                # 파일명 추출
                self.image_dict[row['ID']] = Image.open(os.path.join(path, row['ID'])).convert('RGB')
        
    def __getitem__(self, filename):
        # 파일명으로 이미지 반환
        return self.image_dict[filename]

    def __contains__(self, filename):
        return filename in self.image_dict

    def __len__(self):
        return len(self.image_dict)

    def keys(self):
        return self.image_dict.keys()        

   