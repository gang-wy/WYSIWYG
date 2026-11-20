"""BRAINIX는 GT가 없어서, 종양/두피 voxel mask를 규칙으로 만든다(analyze_group_separation.py의 --a-mask/--b-mask 입력용).

사용:
    .venv\\Scripts\\python.exe experiments\\make_brainix_tumor_scalp_masks.py [출력 폴더]
    → <출력 폴더>/brainix_tumor.nii, brainix_scalp.nii (기본 resources/Experiments/tumor_scalp)

규칙 (intensity는 볼륨 min-max 0~1 기준, TF = 프리셋 brainix_initial_black_cuboid.json):
  머리 mask   intensity > 0.06 → closing 3회 → 최대 연결성분 → z 슬라이스별 구멍 채움
  깊이        머리 mask 안에서 표면까지 거리(mm, voxel spacing 반영)
  종양        TF로 보이고(opacity > 0) intensity ≥ 0.305이며 깊이 ≥ 20mm인 voxel의 최대 연결성분,
              보간 샘플이 경계에 걸리는 것을 허용하려고 2 voxel 팽창
  두피        머리 안, 깊이 ≤ 10mm, 종양 아님, intensity가 [0.305, 0.523)
  0.305 / 0.523 = TF 두 번째 텐트 시작(0.3055)과 세 번째 텐트 끝(0.5231) — 종양과 두피가 같은 색으로 보이는 구간
사람이 그린 GT가 아니라 규칙 기반 라벨이다.
"""
import json
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOLUME = os.path.join(ROOT, "resources/Volume_Data/brainix.nii")
TF = os.path.join(ROOT, "resources/TFs/BRAINIX/brainix_initial_black_cuboid.json")

HEAD_THRESHOLD = 0.06
BAND_LOW, BAND_HIGH = 0.305, 0.523
TUMOR_MIN_DEPTH_MM = 20
TUMOR_DILATE = 2
SCALP_MAX_DEPTH_MM = 10


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "resources/Experiments/tumor_scalp")
    os.makedirs(out_dir, exist_ok=True)
    img = nib.load(VOLUME)
    v = np.asarray(img.dataobj).astype(np.float32)
    v = (v - v.min()) / (v.max() - v.min())
    spacing = img.header.get_zooms()[:3]

    head = ndimage.binary_fill_holes(ndimage.binary_closing(v > HEAD_THRESHOLD, iterations=3))
    lab, _ = ndimage.label(head)
    head = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    for z in range(head.shape[2]):
        head[:, :, z] = ndimage.binary_fill_holes(head[:, :, z])
    depth_mm = ndimage.distance_transform_edt(head, sampling=spacing)

    tf = json.load(open(TF))["global"]
    visible = np.interp(v, [n[0] for n in tf], [n[4] for n in tf], left=0, right=0) > 0
    lab, _ = ndimage.label(visible & (v >= BAND_LOW) & (depth_mm >= TUMOR_MIN_DEPTH_MM))
    tumor = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    tumor = ndimage.binary_dilation(tumor, iterations=TUMOR_DILATE)
    scalp = head & (depth_mm <= SCALP_MAX_DEPTH_MM) & ~tumor & (v >= BAND_LOW) & (v < BAND_HIGH)

    for name, m in (("brainix_tumor", tumor), ("brainix_scalp", scalp)):
        path = os.path.join(out_dir, f"{name}.nii")
        nib.save(nib.Nifti1Image(m.astype(np.uint8), img.affine), path)
        print(f"{path}: {int(m.sum()):,} voxel")
    print(f"spacing {tuple(float(s) for s in spacing)}")


if __name__ == "__main__":
    main()
