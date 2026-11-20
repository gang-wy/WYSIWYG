"""데이터셋 프리셋을 Galaxy XR(Unity)용 NIfTI 세트로 변환하고, 원본과 같은 값인지 검증한다.

출력: resources/gxr_data/<preset>/
  volume.nii   볼륨. 원본이 .nii면 그대로 복사, raw면 원래 값·dtype·spacing으로 .nii 저장
               (정규화는 하지 않음 — Unity가 앱과 같은 min-max [0,1] 정규화를 한다)
  inr8.nii     INR latent embedding 8채널, uint8 (값 × 255 반올림), shape (X,Y,Z,8)
               좌표 3채널은 저장하지 않는다 — 원본에서 정확히 index / (size - 1)이므로 위치로 계산
  tf.tfi       프리셋 TF(json "global")를 Voreen XML로 변환 (HoloLens NodeManager.Load2TFI 형식)
  camera.json  프리셋 VTK 카메라(위치, 초점, view up, 화각) 원본 복사 — Unity 초기 시점에 사용
  meta.json    크기, spacing, 파일별 설명

실행 예시 (프로젝트 루트에서):
    .venv\\Scripts\\python.exe scripts\\export_gxr_nifti.py              (모든 프리셋)
    .venv\\Scripts\\python.exe scripts\\export_gxr_nifti.py --dataset tooth
"""
import argparse
import json
import os
import shutil
import sys
from xml.sax.saxutils import quoteattr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nibabel as nib
import numpy as np

from src.core.dataset_presets import DATASET_PRESETS
from src.gui.data.volume_loader import VolumeLoader, load_raw

OUT_ROOT = "resources/gxr_data"
NUM_COORD_CHANNELS = 3


def tf_to_tfi(nodes):
    """TF node list [[x, r, g, b, a], ...] (0~1) → Voreen .tfi XML 문자열."""
    keys = []
    for x, r, g, b, a in sorted(nodes, key=lambda n: n[0]):
        rgba = [int(round(min(max(v, 0.0), 1.0) * 255)) for v in (r, g, b, a)]
        keys.append(
            '      <key type="TransFuncMappingKey">\n'
            f'        <intensity value={quoteattr(repr(float(x)))} />\n'
            '        <split value="false" />\n'
            f'        <colorL r="{rgba[0]}" g="{rgba[1]}" b="{rgba[2]}" a="{rgba[3]}" />\n'
            '      </key>')
    return ('<?xml version="1.0" encoding="utf-8"?>\n'
            '<VoreenData version="1">\n'
            '  <TransFuncIntensity type="TransFuncIntensity">\n'
            '    <alphaMode value="1" />\n'
            '    <texDimensions x="256" y="1" z="1" />\n'
            '    <filter value="0" />\n'
            '    <dataType value="5121" />\n'
            '    <gammaValue value="1" />\n'
            '    <domain x="0" y="255" />\n'
            '    <threshold x="0" y="1" />\n'
            '    <Keys>\n' + "\n".join(keys) + '\n    </Keys>\n'
            '  </TransFuncIntensity>\n'
            '</VoreenData>\n')


def write_volume(preset, path):
    """volume.nii를 만든다. Returns: (dims, spacing)"""
    src = preset["volume"]
    if src.lower().endswith(".nii"):
        shutil.copyfile(src, path)
        img = nib.load(path)
        return img.shape[:3], tuple(float(s) for s in img.header.get_zooms()[:3])

    raw = preset["raw"]
    data, spacing = load_raw(src, raw["shape"], raw["dtype_str"], raw["endian"], raw["voxel_spacing"])
    data = data.astype(np.dtype(raw["dtype_str"]))   # load_raw는 float32로 바꾸므로 원래 dtype으로 되돌림
    img = nib.Nifti1Image(data, np.diag(list(spacing) + [1.0]))
    img.header.set_zooms(spacing)
    nib.save(img, path)
    return data.shape, tuple(float(s) for s in spacing)


def write_inr8(preset, path, spacing):
    """inr8.nii를 만든다. Returns: (원본 채널 수, 기대값 uint8 배열)"""
    src = nib.load(preset["inr"])
    inr = np.asarray(src.dataobj, dtype=np.float32)
    C = inr.shape[-1]
    u8 = np.round(np.clip(inr[..., :C - NUM_COORD_CHANNELS], 0.0, 1.0) * 255.0).astype(np.uint8)
    del inr
    img = nib.Nifti1Image(u8, src.affine)
    img.header.set_zooms(tuple(spacing) + (1.0,))
    nib.save(img, path)
    return C, u8


def export_preset(name):
    """프리셋 하나를 변환하고 다시 읽어 검증한다."""
    preset = DATASET_PRESETS[name]
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    vol_path = os.path.join(out_dir, "volume.nii")
    inr_path = os.path.join(out_dir, "inr8.nii")
    dims, spacing = write_volume(preset, vol_path)
    C, expected_u8 = write_inr8(preset, inr_path, spacing)

    with open(preset["tf"], "r") as f:
        tf_nodes = json.load(f)["global"]
    with open(os.path.join(out_dir, "tf.tfi"), "w", encoding="utf-8") as f:
        f.write(tf_to_tfi(tf_nodes))
    shutil.copyfile(preset["camera"], os.path.join(out_dir, "camera.json"))

    meta = {
        "dataset": name,
        "dims": [int(d) for d in dims],
        "spacing": list(spacing),
        "volume": {"file": "volume.nii", "normalize": "min-max to [0,1] over the whole volume"},
        "inr": {"file": "inr8.nii", "dtype": "uint8", "scale": "value / 255",
                "channels": "latent embedding 0..7 (원본 INR 채널 0..7)",
                "coords": "저장하지 않음: 원본 채널 8..10 = voxel index / (size - 1)",
                "original_channels": C},
        "tf": {"file": "tf.tfi", "num_nodes": len(tf_nodes)},
        "camera": {"file": "camera.json",
                   "coords": "VTK world(mm), voxel (i,j,k) 중심 = (i,j,k) * spacing, 볼륨 actor는 identity"},
        "source": {k: preset.get(k) for k in ("volume", "inr", "tf", "camera")},
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 검증: 다시 읽은 값이 원본 경로로 읽은 값과 같은지
    vol_same = np.array_equal(VolumeLoader().load(vol_path)[0],
                              VolumeLoader().load(preset["volume"], preset.get("raw"))[0])
    inr_back = np.asarray(nib.load(inr_path).dataobj)
    inr_same = inr_back.dtype == np.uint8 and np.array_equal(inr_back, expected_u8)
    size_mb = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / 2**20
    inr_mb = os.path.getsize(inr_path) / 2**20
    print(f"{name:<11} {dims[0]}x{dims[1]}x{dims[2]}  볼륨 일치 {vol_same}  INR 일치 {inr_same} "
          f"(inr8 {inr_mb:.0f}MB)  TF 노드 {len(tf_nodes)}개  폴더 합계 {size_mb:.0f}MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASET_PRESETS))
    args = parser.parse_args()
    for name in ([args.dataset] if args.dataset else DATASET_PRESETS):
        export_preset(name)


if __name__ == "__main__":
    main()
