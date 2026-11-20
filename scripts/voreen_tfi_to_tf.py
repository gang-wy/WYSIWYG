"""Voreen(5.x)에서 편집해 저장한 .tfi → main.py TF JSON + Unity .tfi.

Voreen key의 intensity는 TF domain [lo, hi](볼륨 실제 값 단위)에 대한 0~1 비율이고,
우리 파이프라인의 x는 볼륨 min-max 정규화값이다 → x = (lo + k*(hi - lo) - vmin) / (vmax - vmin).
그래서 반드시 Voreen에서 렌더링한 것과 같은 볼륨 파일을 --volume으로 넘겨야 한다.

실행 예시 (프로젝트 루트에서):
    .venv\\Scripts\\python.exe scripts\\voreen_tfi_to_tf.py my_tf.tfi ^
        --volume resources\\Volume_Data\\BRAINIX\\sT2WFLAIR401_enlarged.nii
출력: <tfi와 같은 폴더>/<이름>_wysiwyg.json (main.py TF 패널 Load), <이름>_wysiwyg.tfi (Unity tf.tfi)
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.export_gxr_nifti import tf_to_tfi
from src.gui.data.volume_loader import load_nifti


def read_voreen_tfi(path):
    """Returns: (domain (lo, hi), threshold (lo, hi), keys [(k, r, g, b, a)] 0~1)"""
    tf = ET.parse(path).getroot().find(".//TransFuncIntensity")
    if tf is None:
        raise ValueError("TransFuncIntensity(1D TF)가 없습니다 — 2D/Gaussian TF는 변환하지 않습니다.")
    d, t = tf.find("domain"), tf.find("threshold")
    domain = (float(d.get("x")), float(d.get("y"))) if d is not None else (0.0, 1.0)
    threshold = (float(t.get("x")), float(t.get("y"))) if t is not None else (0.0, 1.0)
    keys = []
    for k in tf.find("Keys"):
        it = k.find("intensity")
        kv = float(it.get("value")) if it is not None else float(k.get("intensity"))
        split = (k.find("split").get("value") if k.find("split") is not None else k.get("split")) == "true"
        cl = k.find("colorL")
        keys.append((kv,) + tuple(int(cl.get(c)) / 255.0 for c in "rgba"))
        if split:   # split key: 왼쪽/오른쪽 색이 다름 → 같은 위치 근처에 노드 두 개
            cr = k.find("colorR")
            keys.append((kv + 1e-4,) + tuple(int(cr.get(c)) / 255.0 for c in "rgba"))
    return domain, threshold, sorted(keys)


def convert(tfi_path, volume_path):
    vol, _ = load_nifti(volume_path)
    vmin, vmax = float(vol.min()), float(vol.max())
    (lo, hi), (tlo, thi), keys = read_voreen_tfi(tfi_path)
    to_x = lambda k: (lo + k * (hi - lo) - vmin) / (vmax - vmin)

    nodes = [[to_x(k), r, g, b, a] for k, r, g, b, a in keys]
    # threshold 밖은 투명: 노드가 threshold 밖으로 나가 있으면 경계에 alpha 0 노드를 넣고 밖 노드는 버린다
    if (tlo, thi) != (0.0, 1.0):
        x_lo, x_hi = to_x(tlo), to_x(thi)
        inside = [n for n in nodes if x_lo <= n[0] <= x_hi]
        if len(inside) != len(nodes):
            first, last = inside[0], inside[-1]
            nodes = [[x_lo, *first[1:4], 0.0]] + inside + [[x_hi, *last[1:4], 0.0]]
    out_of_range = [n[0] for n in nodes if not 0.0 <= n[0] <= 1.0]
    nodes = [[min(max(n[0], 0.0), 1.0)] + n[1:] for n in nodes]

    stem = os.path.splitext(tfi_path)[0] + "_wysiwyg"
    data = {"global": nodes,
            "shading": {"enabled": False, "ambient": 0.4, "diffuse": 0.8, "specular": 0.6},
            "metadata": {"version": "3.0", "source": os.path.basename(tfi_path),
                         "volume": volume_path, "voreen_domain": [lo, hi], "volume_minmax": [vmin, vmax]}}
    with open(stem + ".json", "w") as f:
        json.dump(data, f, indent=2)
    with open(stem + ".tfi", "w", encoding="utf-8") as f:
        f.write(tf_to_tfi(nodes))

    print(f"Voreen domain {lo:g}~{hi:g}, 볼륨 min-max {vmin:g}~{vmax:g} → x = (값 - {vmin:g}) / {vmax - vmin:g}")
    print(f"노드 {len(nodes)}개, x {nodes[0][0]:.3f} ~ {nodes[-1][0]:.3f}")
    if out_of_range:
        print(f"⚠️ 볼륨 범위 밖 노드 {len(out_of_range)}개를 0~1로 잘랐습니다: {out_of_range}")
    print(f"저장: {stem}.json, {stem}.tfi")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tfi", help="Voreen에서 저장한 .tfi")
    p.add_argument("--volume", required=True, help="Voreen에서 렌더링한 것과 같은 볼륨(.nii)")
    a = p.parse_args()
    convert(a.tfi, a.volume)


if __name__ == "__main__":
    main()
