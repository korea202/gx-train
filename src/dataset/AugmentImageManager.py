import pandas as pd
import os
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
import base64
from io import BytesIO

class AugmentImageManager:
    """
    이미지 데이터를 pandas DataFrame으로 관리하는 클래스
    기능: 읽기, 쓰기, 카운트, 자동 파일 생성
    """
    
    def __init__(self, csv_path: str = "image_data.csv"):
        """
        초기화
        Args:
            csv_path: CSV 파일 경로
        """
        self.csv_path = csv_path
        self.df = None
        self._load_or_create_dataframe()
    
    def _load_or_create_dataframe(self):
        """DataFrame 로드 또는 생성"""
        if os.path.exists(self.csv_path):
            try:
                self.df = pd.read_csv(self.csv_path)
                #print(f"기존 데이터 로드 완료: {self.csv_path}")
            except Exception as e:
                print(f"파일 로드 실패: {e}")
                self._create_new_dataframe()
        else:
            self._create_new_dataframe()
    
    def _create_new_dataframe(self):
        """새 DataFrame 생성"""
        self.df = pd.DataFrame(columns=["ID","target"])
        print(f"새 데이터프레임 생성 완료")
    
    def add_image_data(self, image_path: str, image_name: str = None, target: str = None) -> bool:

        try:
            if not os.path.exists(image_path):
                print(f"이미지 파일이 존재하지 않습니다: {image_path}")
                return False
            
            # 새 행 데이터
            new_row = {
                'ID': image_name,
                'target': target
            }
            
            # DataFrame에 추가
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            
            # 자동 저장
            self._save_dataframe()
            
            #print(f"이미지 데이터 추가 완료: {image_name}")
            return True
            
        except Exception as e:
            print(f"이미지 데이터 추가 실패: {e}")
            return False
    
    def get_count(self) -> int:
        """
        총 이미지 개수 반환
        Returns:
            int: 이미지 개수
        """
        if self.df is None:
            return 0
        return len(self.df)
    
    def _save_dataframe(self):
        """DataFrame을 CSV 파일로 저장"""
        try:
            self.df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"파일 저장 실패: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        데이터 요약 정보 반환
        Returns:
            Dict: 요약 정보
        """
        if self.df is None or self.df.empty:
            return {"총 이미지 수": 0}
        
        summary = {
            "총 이미지 수": len(self.df),
            "총 파일 크기 (MB)": round(self.df['image_size'].sum() / (1024*1024), 2),
            "평균 파일 크기 (KB)": round(self.df['image_size'].mean() / 1024, 2),
            "포맷별 개수": self.get_count_by_format(),
            "최근 추가된 이미지": self.df.iloc[-1]['image_name'] if len(self.df) > 0 else "없음"
        }
        
        return summary
    

