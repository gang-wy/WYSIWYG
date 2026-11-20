"""MobileSAM(TinyViT encoder) 포인트 프롬프트 segmentation. SAMWrapper와 같은 predict 인터페이스."""
import warnings

import numpy as np
import torch
from PIL import Image

DEFAULT_CHECKPOINT = "resources/models/mobile_sam.pt"


class MobileSAMWrapper:
    """MobileSAM(vit_t)을 로드하고 이미지 + 포인트로 2D mask를 예측한다. 장치: cuda > cpu."""

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device=None):
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = None
        self.is_loaded = False

    def load_model(self):
        if self.is_loaded:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # timm deprecation 경고
            from mobile_sam import sam_model_registry, SamPredictor
        model = sam_model_registry["vit_t"](checkpoint=self.checkpoint).to(self.device).eval()
        self.predictor = SamPredictor(model)
        self.is_loaded = True
        print(f"✅ MobileSAM loaded (device: {self.device})", flush=True)

    def predict(self, image_path, points, labels):
        """
        Args:
            image_path: 입력 이미지 경로
            points: [(x, y), ...] 이미지 좌표(좌상단 원점)
            labels: [1(positive) | 0(negative), ...]
        Returns: (mask (H, W) bool, logits (H, W) float32 원본 해상도)

        포인트가 1개면 후보 3개 중 예측 IoU가 가장 높은 mask를, 2개 이상이면 단일 mask 출력을 쓴다.
        """
        if not self.is_loaded:
            self.load_model()

        with Image.open(image_path) as f:
            image = np.array(f.convert("RGB"))
        coords = np.array([[float(p[0]), float(p[1])] for p in points], dtype=np.float32)
        point_labels = np.array([int(l) for l in labels], dtype=np.int32)

        multimask = len(coords) == 1
        with torch.no_grad():
            self.predictor.set_image(image)
            masks, scores, low_res_logits = self.predictor.predict(
                point_coords=coords, point_labels=point_labels,
                multimask_output=multimask, return_logits=True)
        best = int(np.argmax(scores)) if multimask else 0
        logits = masks[best].astype(np.float32)   # return_logits=True → 원본 해상도 logits
        return logits > self.predictor.model.mask_threshold, logits
