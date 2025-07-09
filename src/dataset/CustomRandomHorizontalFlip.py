import torch
import random
from PIL import Image
import numpy as np

class CustomRandomHorizontalFlip:
    """
    RandomHorizontalFlip(p=0.3) 전용 커스텀 클래스
    """
    
    def __init__(self, p=0.2):
        """
        Args:
            p (float): 좌우 뒤집기를 적용할 확률 (기본값: 0.4)
        """
        self.p = p
    
    def __call__(self, img):
        """
        이미지에 랜덤 좌우 뒤집기를 적용
        
        Args:
            img (PIL Image): 입력 이미지
            
        Returns:
            PIL Image: 변환된 이미지
        """
        target = img.target

        if random.random() < self.p:
            image =  img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            image  = img

        setattr(image, 'target', target)
        return image
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"
