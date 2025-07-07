from augraphy import *
import random
import numpy as np
import cv2
from PIL import Image
from datetime import datetime

from src.dataset.AugmentImageManager import AugmentImageManager 
from src.utils import config

class FullAugraphyPipelineQueueEx:
    def __init__(self, num_pipelines=20):
        self.pipelines = []
        self.augument_image_manager = AugmentImageManager(csv_path=config.CV_CLS_AUGMENT_CSV)

        """     
        # 실제 존재하는 효과들로 수정
        self.ink_effects = [
            Dithering(),
            InkBleed(),
            BleedThrough(),
            Letterpress(),
            LowInkRandomLines(),
            LowInkPeriodicLines(),
            InkShifter(),
        ]
        
        self.paper_effects = [
            ColorPaper(),
            #WaterMark(watermark_word="random"),
            PaperFactory(),
            NoiseTexturize(),
            BrightnessTexturize(),
            Brightness(),
            VoronoiTessellation(),
            DelaunayTessellation(),
            SubtleNoise(),
        ]
        
        self.post_effects = [
            # 물리적 변형
            Folding(),
            #BindingsAndFasteners(),
            
            # 인쇄/스캔 효과
            BadPhotoCopy(),
            DirtyDrum(),
            DirtyRollers(),
            Faxify(),
            Jpeg(),
            
            # 마킹 및 주석
            Markup(),
            Scribbles(),
            
            # 노이즈 및 왜곡 (Noisify 대신 실제 존재하는 것들)
            GlitchEffect(),
            LightingGradient(),
            ShadowCast(),
            Gamma(),
            
            # 추가 효과들
            #BookBinding(),
            PageBorder(),
            Hollow(),
            
            # 색상 효과
            ColorShift(),
            
            # 기타 효과 (NoisyLines 추가)
            NoisyLines(),
        ] """

        self.list_watermark = []

        for i in range(7):  # 7개의 서로 다른 워터마크
            watermark = WaterMark(
                watermark_word="random",
                watermark_font_size=(1, 2),  
                watermark_font_thickness=(1, 2),
                watermark_rotation=(i*30, i*30 + 60),  # 각기 다른 회전
                watermark_location="random",
                watermark_color=(200, 200, 200),
                watermark_method="darken",
                p=0.3
            )
            self.list_watermark.append(watermark)

        # 각 파이프라인마다 다른 설정으로 생성
        for _ in range(num_pipelines):
            # 매번 새로운 랜덤 값으로 효과 생성
            ink_effects = []
            
            paper_effects = [
                SubtleNoise(
                    subtle_range=random.randint(60, 90),  # 매번 다른 값
                    p=0.3
                ),
                ColorPaper(
                    hue_range=(0, 360),
                    saturation_range=(0, 5),
                    p=0.3
                )
            ]
            paper_effects.extend(self.list_watermark)
            
            post_effects = [
                Geometric(
                    scale=(1, 1.1),
                    translation=probabilistic_translation(),  # 매번 다른 값
                    fliplr=0.2,
                    flipud=0.2,
                    p=0.5
                ),
            ]

            pipeline = AugraphyPipeline(
                ink_phase= ink_effects,
                paper_phase= paper_effects,
                post_phase= post_effects
            )
            self.pipelines.append(pipeline)

       
    def __call__(self, image):
   
        target = image[0, 0, 0] & 0x1F
        
        rotated_image  = rotate_first_then_augraphy(image)

        pipeline = random.choice(self.pipelines)
        augmented = pipeline(rotated_image)

        # 채널 수에 따른 처리
        if len(augmented.shape) == 2:  # 그레이스케일 (H, W)
            augmented = np.stack([augmented, augmented, augmented], axis=-1)
        elif len(augmented.shape) == 3:
            if augmented.shape[2] == 1:  # (H, W, 1) 형태의 그레이스케일
                augmented = np.repeat(augmented, 3, axis=2)
            elif augmented.shape[2] == 3:  # BGR 컬러
                augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
            elif augmented.shape[2] == 4:  # BGRA
                augmented = cv2.cvtColor(augmented, cv2.COLOR_BGRA2RGB)
        
        # PIL Image로 변환
        pil_result = Image.fromarray(augmented.astype(np.uint8))

        if self.augument_image_manager.get_count() < 10000:
            self.save_augment_file(pil_result, target)

        return pil_result
    
    def save_augment_file(self, pil_image, target, prefix="augraphy", output_dir=config.CV_CLS_AUGMENT_DIR):
    
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        id = f"{self.augument_image_manager.get_count() + 1:07d}"
        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")  # 20250706_204900
                
        # 파일명 생성
        filename = f"{prefix}_{time_str}_{target}_{id}.png"
        filepath = os.path.join(output_dir, filename)
        
        # 저장
        pil_image.save(filepath)
        #print(f"저장 완료: {filepath}")
        self.augument_image_manager.add_image_data(image_path=filepath, image_name=filename, target=target)

        return filepath  



def probabilistic_translation(probability=0.2, max_movement=100):
    
    if random.random() < probability:
        # 20% 확률로 랜덤한 이동 적용
        x_move = random.randint(-max_movement, max_movement)
        y_move = random.randint(-max_movement, max_movement)
        return (x_move, y_move)
    else:
        # 80% 확률로 이동 없음
        return (0, 0)

def rotate_first_then_augraphy(image):
     
    # 1. 원본 크기 유지 회전
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    M = cv2.getRotationMatrix2D(center, random.randint(0, 300), 1.0)
    rotated = cv2.warpAffine(
        image, M, (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    return rotated   

     