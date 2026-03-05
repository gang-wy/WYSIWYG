# src/core/sam_wrapper.py
import numpy as np
from PIL import Image
import torch
import gc # 가비지 컬렉터 추가

class SAMWrapper:
    def __init__(self):
        self.model = None
        self.processor = None
        self.text_model = None
        self.text_processor = None
        
        # M1/M2 GPU 지원 추가
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.is_loaded = False
        self.is_text_loaded = False  # [NEW] 텍스트 모델 로드 상태

    def load_model(self):
        if self.is_loaded:
            return

        print(f"Loading SAM3 Tracker Model (device: {self.device})...", flush=True)

        try:
            import os
            from transformers import Sam3TrackerProcessor, Sam3TrackerModel
            
            model_name = "facebook/sam3"
            
            # ✅ 1단계: 로컬 캐시 확인
            try:
                self.model = Sam3TrackerModel.from_pretrained(
                    model_name,
                    local_files_only=True
                ).to(self.device)
                self.processor = Sam3TrackerProcessor.from_pretrained(
                    model_name,
                    local_files_only=True
                )
                print("✅ SAM3 Model loaded from cache!", flush=True)
                
            except Exception as cache_error:
                # ✅ 2단계: 캐시 없으면 다운로드
                print("📦 Cache not found. Downloading model (this may take a few minutes)...")
                print(f"   Cache error: {cache_error}")
                
                # 타임아웃 및 재시도 설정
                os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
                
                self.model = Sam3TrackerModel.from_pretrained(
                    model_name,
                    resume_download=True
                ).to(self.device)
                self.processor = Sam3TrackerProcessor.from_pretrained(
                    model_name,
                    resume_download=True
                )
                print("✅ SAM3 Model downloaded and loaded successfully!", flush=True)
            
            self.is_loaded = True
            
        except Exception as e:
            print(f"❌ Failed to load SAM3 Tracker model: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def load_text_model(self):
        """[NEW] 텍스트 전용 SAM3 모델 로드"""
        if self.is_text_loaded:
            return

        print(f"Loading SAM3 Text Model (device: {self.device})...", flush=True)

        try:
            import os
            from transformers import Sam3Processor, Sam3Model
            
            model_name = "facebook/sam3"
            
            # ✅ 1단계: 로컬 캐시 확인
            try:
                print("🔍 Checking local cache for text model...")
                self.text_model = Sam3Model.from_pretrained(
                    model_name,
                    local_files_only=True
                ).to(self.device)
                self.text_processor = Sam3Processor.from_pretrained(
                    model_name,
                    local_files_only=True
                )
                print("✅ SAM3 Text Model loaded from cache!", flush=True)
                
            except Exception as cache_error:
                # ✅ 2단계: 캐시 없으면 다운로드
                print("📦 Cache not found. Downloading text model...")
                print(f"   Cache error: {cache_error}")
                
                # 타임아웃 설정
                os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
                
                self.text_model = Sam3Model.from_pretrained(
                    model_name,
                    resume_download=True
                ).to(self.device)
                self.text_processor = Sam3Processor.from_pretrained(
                    model_name,
                    resume_download=True
                )
                print("✅ SAM3 Text Model downloaded and loaded successfully!", flush=True)
            
            self.is_text_loaded = True
            
        except Exception as e:
            print(f"❌ Failed to load SAM3 Text model: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def predict(self, image_path, points, labels):
        """
        Args:
            image_path (str): 이미지 경로
            points (list): [(x, y), ...] 형태의 2D 좌표 리스트
            labels (list): [1, 0, ...] 형태의 라벨 리스트
        Returns:
            mask: (H, W) boolean numpy array
            logits_2d: (H, W) float numpy array — raw logits resized to original resolution
        """
        if not self.is_loaded:
            self.load_model()

        best_mask = None
        logits_2d = None
        inputs = None
        outputs = None

        try:
            with Image.open(image_path) as f:
                raw_image = f.convert("RGB")

            if not isinstance(points, list):
                points = [list(points)]

            clean_points = [ [float(p[0]), float(p[1])] for p in points ]
            clean_labels = [ int(l) for l in labels ]

            input_points = [[clean_points]]
            input_labels = [[clean_labels]]

            inputs = self.processor(
                images=[raw_image],
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Raw logits (before sigmoid)
            raw_logits = outputs.pred_masks.cpu()  # (B, F, num_masks, H_logit, W_logit)

            # Post-process mask
            masks = self.processor.post_process_masks(
                raw_logits,
                inputs["original_sizes"].cpu()
            )

            if len(masks) > 0:
                best_mask = masks[0][0, 0, :, :].numpy()

                # Resize logits to original resolution (same as mask)
                original_h, original_w = inputs["original_sizes"][0].tolist()
                logit_lowres = raw_logits[0, 0, 0:1, :, :].unsqueeze(0).float()
                logit_highres = torch.nn.functional.interpolate(
                    logit_lowres,
                    size=(int(original_h), int(original_w)),
                    mode='bilinear',
                    align_corners=False
                )
                logits_2d = logit_highres.squeeze().numpy()  # (H, W) float

        except Exception as e:
            print(f"Prediction Error: {e}")
            import traceback
            traceback.print_exc()
            raise e

        finally:
            inputs = None
            outputs = None

            if self.device == "cuda":
                torch.cuda.empty_cache()

            gc.collect()

        return best_mask, logits_2d
    
    def predict_text(self, image_path, text_prompt):
        """
        [NEW] 텍스트 프롬프트 기반 예측 (Sam3Processor + Sam3Model 사용)
        Args:
            image_path (str): 이미지 경로
            text_prompt (str): "kidney", "tumor" 등의 텍스트 프롬프트
        """
        if not self.is_text_loaded:
            self.load_text_model()

        final_mask = None
        inputs = None
        outputs = None
        
        try:
            with Image.open(image_path) as f:
                raw_image = f.convert("RGB")

            print(f"Running Text Inference: '{text_prompt}'")
            
            # [핵심] Sam3Processor는 text 파라미터를 직접 지원
            inputs = self.text_processor(
                images=raw_image,  # 단일 이미지 (배치 차원 자동 추가됨)
                text=text_prompt,  # 단일 텍스트 프롬프트
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.text_model(**inputs)

            # 후처리 (Post-process)
            results = self.text_processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]  # 첫 번째 배치의 결과

            # 결과 마스크 병합 (여러 객체가 찾아질 수 있음)
            # results['masks'] shape: (N, H, W) -> Boolean
            found_masks = results['masks'].cpu().numpy()
            
            if len(found_masks) > 0:
                # 찾아진 모든 객체를 하나의 마스크로 합침 (Logical OR)
                final_mask = np.any(found_masks, axis=0) 
                print(f"Text SAM found {len(found_masks)} objects.")
            else:
                print("Text SAM found no objects.")
                final_mask = None

        except Exception as e:
            print(f"Text Prediction Error: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        finally:
            # 메모리 정리
            inputs = None
            outputs = None
            self._cleanup_memory()

        return final_mask

    def _cleanup_memory(self):
        """메모리 정리 헬퍼 메서드"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
        gc.collect()