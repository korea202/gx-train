import os

from PIL import Image
import pandas as pd

from src.utils import config 

class MemoryImageSource:
    def __init__(self, csv, path, img_size=config.IMAGE_SIZE ):
     
        self.df = pd.read_csv(csv)
        self.image_dict = {}
        for index, row in self.df.iterrows():
            # 파일명 추출
            self.image_dict[row['ID']] = Image.open(os.path.join(path, row['ID'])).convert('RGB').resize((img_size, img_size), Image.LANCZOS)
        
    def __getitem__(self, filename):
        # 파일명으로 이미지 반환
        return self.image_dict[filename]

    def __contains__(self, filename):
        return filename in self.image_dict

    def __len__(self):
        return len(self.image_dict)

    def keys(self):
        return self.image_dict.keys()        

   