from augraphy import *
import random
import numpy as np
import cv2
from PIL import Image

from src.utils import config

class FullAugraphyPipelineQueue:
    def __init__(self, max_effects=5, num_pipelines=10):
        self.pipelines = []
        self.max_effects = max_effects
        
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
            BindingsAndFasteners(),
            
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
        ]

        for _ in range(num_pipelines):

            # 모든 효과에서 랜덤 선택
            all_effects = self.ink_effects + self.paper_effects + self.post_effects
            num_effects = random.randint(1, min(self.max_effects, len(all_effects)))
            #selected_effects = random.sample(all_effects, num_effects)
            selected_effects = random.sample(all_effects, max_effects) 
            selected_effects.append(
                Geometric(
                    fliplr=0.5,
                    flipud=0.5,
                    rotate_range=(1, 355),
                    scale=(0.8, 1.3),
                    translation=(0.05, 0.05),
                    p=0.8
                ) )
            
            # 효과들을 단계별로 분류
            ink_selected = []
            paper_selected = []
            post_selected = []
            
            for effect in selected_effects:
                if effect in self.ink_effects:
                    ink_selected.append(effect)
                elif effect in self.paper_effects:
                    paper_selected.append(effect)
                else:
                    post_selected.append(effect)
            
            # 파이프라인 생성 및 적용
            pipeline = AugraphyPipeline(
                ink_phase=ink_selected if ink_selected else [],
                paper_phase=paper_selected if paper_selected else [],
                post_phase=post_selected if post_selected else []
            )
            self.pipelines.append(pipeline)
    
    def __call__(self, image):
   
        pipeline = random.choice(self.pipelines)
        augmented = pipeline(image)

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
        
        # 최종 크기 조정
        pil_result = pil_result.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
        
        return pil_result