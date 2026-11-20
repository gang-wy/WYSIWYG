"""MobileSAM을 on-device(Unity)용 ONNX 두 개로 export하고, PyTorch 결과와 같은지 검증한다.

  encoder.onnx  image (1,3,1024,1024) 정규화·패딩된 이미지 → image_embeddings (1,256,64,64)
  decoder.onnx  image_embeddings, point_coords (1,N,2), point_labels (1,N), mask_input (1,1,256,256),
                has_mask_input (1,) → low_res_masks (1,1,256,256) logits, iou_predictions (1,1)

decoder는 원본 크기로 되돌리는 보간을 포함하지 않는다(동적 크기 연산을 피하기 위함). 앞뒤 처리는
이 파일의 preprocess / transform_points / postprocess와 똑같이 C#에서 구현하면 된다.
  전처리   긴 변을 1024로 bilinear 리사이즈 → (픽셀 - mean) / std → 오른쪽·아래를 0으로 채워 1024×1024
  포인트   좌표 × (리사이즈 크기 / 원본 크기), 끝에 패딩 포인트 (0,0) 라벨 -1 추가
  후처리   256×256 logits → 1024×1024 bilinear → 리사이즈 영역만 자르기 → 원본 크기 bilinear → logits > 0
  mask 선택 포인트 1개면 후보 3개 중 IoU 예측 최고, 2개 이상이면 단일 mask (decoder 안에서 처리)

실행 예시 (프로젝트 루트에서):
    .venv\\Scripts\\python.exe scripts\\export_mobile_sam_onnx.py export
    .venv\\Scripts\\python.exe scripts\\export_mobile_sam_onnx.py verify
"""
import argparse
import glob
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from src.core.sam.mobile_sam_wrapper import MobileSAMWrapper, DEFAULT_CHECKPOINT

OUT_DIR = "resources/models"
ENCODER_PATH = os.path.join(OUT_DIR, "mobile_sam_encoder.onnx")
DECODER_PATH = os.path.join(OUT_DIR, "mobile_sam_decoder.onnx")
IMG_SIZE = 1024
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
OPSET = 15


def load_sam():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from mobile_sam import sam_model_registry
    return sam_model_registry["vit_t"](checkpoint=DEFAULT_CHECKPOINT).eval()


def build_decoder_module(sam):
    """SamOnnxModel에서 원본 크기 보간을 뺀 decoder. Returns: nn.Module"""
    from mobile_sam.utils.onnx import SamOnnxModel

    class LowResDecoder(SamOnnxModel):
        @torch.no_grad()
        def forward(self, image_embeddings, point_coords, point_labels, mask_input, has_mask_input):
            sparse = self._embed_points(point_coords, point_labels)
            dense = self._embed_masks(mask_input, has_mask_input)
            masks, scores = self.model.mask_decoder.predict_masks(
                image_embeddings=image_embeddings,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense)
            return self.select_masks(masks, scores, point_coords.shape[1])

    return LowResDecoder(sam, return_single_mask=True).eval()


def export():
    """encoder / decoder ONNX를 OUT_DIR에 저장한다."""
    sam = load_sam()
    os.makedirs(OUT_DIR, exist_ok=True)

    torch.onnx.export(sam.image_encoder, torch.randn(1, 3, IMG_SIZE, IMG_SIZE), ENCODER_PATH,
                      input_names=["image"], output_names=["image_embeddings"],
                      opset_version=OPSET, do_constant_folding=True, dynamo=False)
    print(f"encoder: {ENCODER_PATH} ({os.path.getsize(ENCODER_PATH) / 2**20:.1f}MB)")

    decoder = build_decoder_module(sam)
    dummy = (torch.randn(1, 256, 64, 64), torch.randint(0, 1024, (1, 3, 2), dtype=torch.float),
             torch.tensor([[1, 1, -1]], dtype=torch.float), torch.zeros(1, 1, 256, 256),
             torch.zeros(1, dtype=torch.float))
    torch.onnx.export(decoder, dummy, DECODER_PATH,
                      input_names=["image_embeddings", "point_coords", "point_labels",
                                   "mask_input", "has_mask_input"],
                      output_names=["low_res_masks", "iou_predictions"],
                      dynamic_axes={"point_coords": {1: "num_points"}, "point_labels": {1: "num_points"}},
                      opset_version=OPSET, do_constant_folding=True, dynamo=False)
    print(f"decoder: {DECODER_PATH} ({os.path.getsize(DECODER_PATH) / 2**20:.1f}MB)")


# ---------------------------------------------------------------------------
# C#으로 옮길 앞뒤 처리 (numpy 기준 구현)
# ---------------------------------------------------------------------------
def resized_size(h, w):
    """Returns: 긴 변을 1024로 맞춘 (new_h, new_w)"""
    scale = IMG_SIZE / max(h, w)
    return int(h * scale + 0.5), int(w * scale + 0.5)


def preprocess(image):
    """(H,W,3) uint8 → (1,3,1024,1024) float32: 리사이즈, 정규화, 0 패딩."""
    h, w = image.shape[:2]
    new_h, new_w = resized_size(h, w)
    resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR), dtype=np.float32)
    normalized = (resized - PIXEL_MEAN) / PIXEL_STD
    padded = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    padded[:new_h, :new_w] = normalized
    return padded.transpose(2, 0, 1)[None]


def transform_points(points, labels, h, w):
    """원본 이미지 좌표 → 1024 기준 좌표 + 패딩 포인트. Returns: (coords (1,N+1,2), labels (1,N+1))"""
    new_h, new_w = resized_size(h, w)
    coords = np.asarray(points, dtype=np.float32) * np.array([new_w / w, new_h / h], dtype=np.float32)
    coords = np.concatenate([coords, np.zeros((1, 2), np.float32)])[None]
    lab = np.concatenate([np.asarray(labels, np.float32), np.array([-1], np.float32)])[None]
    return coords, lab


def bilinear_resize(arr, out_h, out_w):
    """align_corners=False bilinear (F.interpolate와 동일). arr: (H,W) float32"""
    t = torch.from_numpy(arr)[None, None]
    return torch.nn.functional.interpolate(t, size=(out_h, out_w), mode="bilinear",
                                           align_corners=False)[0, 0].numpy()


def postprocess(low_res, h, w):
    """(256,256) logits → (H,W) logits."""
    new_h, new_w = resized_size(h, w)
    up = bilinear_resize(low_res, IMG_SIZE, IMG_SIZE)[:new_h, :new_w]
    return bilinear_resize(np.ascontiguousarray(up), h, w)


class OnnxMobileSAM:
    """ONNX Runtime으로 encoder → decoder를 실행한다(C# 구현의 기준)."""

    def __init__(self):
        import onnxruntime as ort
        self.encoder = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(DECODER_PATH, providers=["CPUExecutionProvider"])

    def predict(self, image, points, labels):
        """Returns: (mask (H,W) bool, logits (H,W) float32, encoder 초, decoder 초)"""
        h, w = image.shape[:2]
        t0 = time.time()
        emb = self.encoder.run(None, {"image": preprocess(image)})[0]
        t1 = time.time()
        coords, lab = transform_points(points, labels, h, w)
        low_res, _ = self.decoder.run(None, {
            "image_embeddings": emb, "point_coords": coords, "point_labels": lab,
            "mask_input": np.zeros((1, 1, 256, 256), np.float32),
            "has_mask_input": np.zeros(1, np.float32)})
        t2 = time.time()
        logits = postprocess(low_res[0, 0], h, w)
        return logits > 0.0, logits, t1 - t0, t2 - t1


def verify():
    """기준 데이터(*_uint8_mobilesam.npz)의 캡처 이미지·포인트로 PyTorch와 ONNX 결과를 비교한다."""
    onnx_sam = OnnxMobileSAM()
    torch_sam = MobileSAMWrapper(device="cpu")
    for path in sorted(glob.glob("resources/reference_data/*_uint8_mobilesam.npz")):
        d = np.load(path, allow_pickle=True)
        image, points, labels = d["sam_image"], d["sam_points_img"], d["sam_labels"]

        tmp = os.path.join(OUT_DIR, "_verify_input.png")
        Image.fromarray(image).save(tmp)
        mask_t, logits_t = torch_sam.predict(tmp, points, labels)
        mask_o, logits_o, t_enc, t_dec = onnx_sam.predict(image, points, labels)
        os.remove(tmp)

        iou = (mask_t & mask_o).sum() / max((mask_t | mask_o).sum(), 1)
        ref_iou = (mask_o & d["sam_mask"]).sum() / max((mask_o | d["sam_mask"]).sum(), 1)
        print(f"{os.path.basename(path):<34} PyTorch vs ONNX IoU {iou:.4f}, logits 최대 오차 "
              f"{np.abs(logits_t - logits_o).max():.4f} | 기준 mask IoU {ref_iou:.4f} | "
              f"CPU encoder {t_enc * 1000:.0f}ms decoder {t_dec * 1000:.0f}ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["export", "verify"])
    args = parser.parse_args()
    export() if args.command == "export" else verify()


if __name__ == "__main__":
    main()
