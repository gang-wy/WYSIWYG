"""데이터셋 프리셋이 가리키는 파일(볼륨·INR·TF·카메라·SAM 포인트·ROI별 파일·GT)을 프리셋마다 zip 하나로 묶는다.

zip 안 경로 = repo 루트 기준 상대 경로(resources/...)라 받는 쪽은 repo 루트에서 풀면 제자리에 들어간다.
옛 INR("inr_old")은 넣지 않는다.

사용:
    .venv\\Scripts\\python.exe scripts\\export_dataset_bundle.py calix tooth engine [--out <폴더>]
    출력: <폴더>/<프리셋>.zip (기본 resources/dataset_bundles)
"""
import argparse
import glob
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.core.dataset_presets import DATASET_PRESETS  # noqa: E402


def preset_files(key):
    """프리셋이 쓰는 파일의 repo 기준 상대 경로 목록(있는 것만)과 없는 경로 목록."""
    p = DATASET_PRESETS[key]
    files = [p.get(k) for k in ("volume", "inr", "tf", "camera", "points")]
    for roi in (p.get("rois") or {}).values():
        files += [roi.get("points"), roi.get("another")]
    # GT: run_gt_evaluation.find_gt_path와 같은 규칙(<볼륨 이름>_GT.*, <프리셋>_GT.*)
    stem = os.path.basename(p["volume"]).split(".nii")[0].rsplit(".", 1)[0]
    for pattern in (f"{stem}_GT.*", f"{key}_GT.*", f"{key.capitalize()}_GT.*"):
        hits = glob.glob(os.path.join(ROOT, os.path.dirname(p["volume"]), pattern))
        if hits:
            files.append(os.path.relpath(hits[0], ROOT).replace(os.sep, "/"))
            break
    files = sorted({f for f in files if f})
    return ([f for f in files if os.path.isfile(os.path.join(ROOT, f))],
            [f for f in files if not os.path.isfile(os.path.join(ROOT, f))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("presets", nargs="+", choices=sorted(DATASET_PRESETS))
    ap.add_argument("--out", default=os.path.join(ROOT, "resources", "dataset_bundles"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for key in args.presets:
        files, missing = preset_files(key)
        path = os.path.join(args.out, f"{key}.zip")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in files:
                z.write(os.path.join(ROOT, f), f)
        raw = sum(os.path.getsize(os.path.join(ROOT, f)) for f in files)
        print(f"{key}: {len(files)}개 파일 {raw / 2**20:,.0f} MB → {path} ({os.path.getsize(path) / 2**20:,.0f} MB)")
        for f in files:
            print(f"    {f}")
        if missing:
            print(f"    ⚠️ 없는 파일(빠짐): {missing}")


if __name__ == "__main__":
    main()
