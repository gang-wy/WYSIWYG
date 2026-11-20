"""결과 공유 패키지: ROI마다 Stage A·B 덤프 한 쌍을 볼륨과 함께 한 폴더로 묶는다.
폴더를 압축해 보내면 받는 쪽은 numpy·matplotlib만 있으면 open_viewer.bat(또는 python viewer.py)으로 연다.
뷰어 = scripts/result_viewer.py(저장소 코드를 import하지 않는 단독 파일)를 viewer.py로 복사한 것.

사용:
    .venv\\Scripts\\python.exe scripts\\export_result_package.py <stage_a_*.npz ...> [--names 이름 ...]
    .venv\\Scripts\\python.exe scripts\\export_result_package.py --since 20261120_150000
    옵션: --out <폴더>   기본 resources/Experiments/result_package/<오늘 날짜>
출력: <out>.zip(보낼 파일)과 <out>/viewer.py, open_viewer.bat, data/<데이터셋>/volume.npz(8bit 볼륨), data/<데이터셋>/<ROI>.npz,
      data/<데이터셋>/<ROI>_render.png·_mask.png(ROI를 고른 화면·SAM mask)
ROI 이름: --names(덤프 순서대로), 없으면 데이터셋 안에서 시간순 roi1, roi2, ...
"""
import argparse
import glob
import os
import shutil
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_stage_a_slices as cmp  # noqa: E402  (ROOT를 sys.path에 넣음)
from src.gui.widget.renderer_widget import cluster_colors  # noqa: E402

DUMP_DIR = os.path.join(cmp.ROOT, "resources/Experiments/stage_a_dump")
VIEWER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_viewer.py")
COORD_SCALE = 8   # 샘플 voxel 좌표를 1/8 voxel 단위 int16으로 저장(화면 표시용 정밀도면 충분, 용량 절반)


def channel_names(layout):
    """layout ["inr_embedding:0:8", "coords:8:11", ...] → 열 이름과 원래 값 배열 키(열마다)."""
    names, raw_keys = [], []
    for s in layout:
        group, a, b = s.split(":")
        n = int(b) - int(a)
        if group == "coords":
            names += list("xyz"[:n]); raw_keys += [("vox", i) for i in range(n)]
        elif group == "visibility":
            names.append("V"); raw_keys.append(("visibility", 0))
        elif n == 1:
            names.append(group); raw_keys.append((group, 0))
        else:
            names += [f"{group.split('_')[0]}{i}" for i in range(n)]; raw_keys += [(None, 0)] * n
    return names, raw_keys


def cluster_table(d, run):
    """cluster × 채널 평균: K-means 입력(정규화) 값과, 해석 가능한 채널은 원래 값(INR은 잠재값이라 NaN)."""
    norm, labels, k = d["normalized"], run["labels"], run["k"]
    names, raw_keys = channel_names([str(s) for s in d["layout"]])
    assert len(names) == norm.shape[1], (names, norm.shape)
    raw_src = {"vox": run["vox"], "visibility": d["visibility"], "depth": d["depths"],
               "opacity": d["opacity"], "intensity": d["intensities"]}
    z = np.full((k, len(names)), np.nan, np.float32)
    raw = np.full((k, len(names)), np.nan, np.float32)
    counts = np.bincount(labels, minlength=k)
    inten = np.full(k, np.nan, np.float32)
    for c in range(k):
        sel = labels == c
        if not sel.any():
            continue
        z[c] = norm[sel].mean(0)
        inten[c] = d["intensities"][sel].mean()
        for j, (key, i) in enumerate(raw_keys):
            if key in raw_src:
                arr = np.asarray(raw_src[key])
                raw[c, j] = (arr[sel, i] if arr.ndim == 2 else arr[sel]).mean()
    return names, z, raw, counts, inten


def to_uint8(vol):
    """표시용 8bit: 0.5~99.9 백분위 창(최소·최대 창은 CT에서 연조직이 다 검게 뭉개짐)."""
    sub = np.asarray(vol[::4, ::4, ::4], dtype=np.float32)
    lo, hi = np.percentile(sub, [0.5, 99.9])
    hi = hi if hi > lo else lo + 1
    return np.clip((np.asarray(vol, dtype=np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def copy_png(src, dst):
    src = src if os.path.isabs(src) else os.path.join(cmp.ROOT, src)
    if src and os.path.isfile(src):
        shutil.copyfile(src, dst)
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="*", help="stage_a_*.npz (생략하면 --since 이후 덤프 전체)")
    ap.add_argument("--since", help="YYYYMMDD_HHMMSS 이후 덤프만")
    ap.add_argument("--names", nargs="+", help="ROI 이름(덤프 순서대로)")
    ap.add_argument("--out", help="출력 폴더")
    args = ap.parse_args()

    paths = list(args.dumps)
    if args.since:
        paths += [p for p in sorted(glob.glob(os.path.join(DUMP_DIR, "stage_a_*.npz")))
                  if os.path.basename(p)[len("stage_a_"):-len(".npz")] >= args.since]
    if not paths:
        sys.exit("덤프를 지정하세요(경로 또는 --since).")
    if args.names and len(args.names) != len(paths):
        sys.exit(f"--names {len(args.names)}개 ≠ 덤프 {len(paths)}개")
    out = args.out or os.path.join(cmp.ROOT, "resources/Experiments/result_package",
                                   datetime.now().strftime("%Y%m%d"))
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    shutil.copyfile(VIEWER, os.path.join(out, "viewer.py"))
    with open(os.path.join(out, "open_viewer.bat"), "w", encoding="ascii") as f:   # cmd 코드페이지 문제로 영문만
        f.write('@echo off\r\ncd /d "%~dp0"\r\npython viewer.py || pause\r\n')

    volumes, counter = {}, {}
    for i, path in enumerate(paths):
        d = np.load(path)
        vpath, raw_params = cmp.dump_volume_source(d)
        if vpath is None:
            print(f"⚠️ 건너뜀(볼륨 경로 없는 옛 덤프): {path}")
            continue
        dataset = cmp.preset_for_volume(vpath) or os.path.splitext(os.path.basename(vpath))[0]
        folder = os.path.join(out, "data", dataset)
        os.makedirs(folder, exist_ok=True)
        if dataset not in volumes:
            vol, spacing = cmp.load_volume(volume=vpath, raw=raw_params)
            volumes[dataset] = (vol.shape[:3], spacing)
            np.savez_compressed(os.path.join(folder, "volume.npz"), volume=to_uint8(vol),
                                spacing=np.asarray(spacing, np.float32))
        shape, spacing = volumes[dataset]

        counter[dataset] = counter.get(dataset, 0) + 1
        name = args.names[i] if args.names else f"roi{counter[dataset]}"
        run = cmp.load_run(path, spacing)
        names, z, raw, counts, inten = cluster_table(d, run)

        b_path = os.path.join(os.path.dirname(path), os.path.basename(path).replace("stage_a_", "stage_b_"))
        extra = {}
        if os.path.exists(b_path):
            mask = np.load(b_path)["mask"].astype(bool)
            if mask.shape == tuple(shape):
                extra["stage_b_mask"] = mask
            else:
                print(f"⚠️ 최종 ROI(Stage B mask) shape {mask.shape} ≠ 볼륨 {tuple(shape)} — 뺌: {b_path}")
        has_render = copy_png(str(d["sam_image_path"]), os.path.join(folder, f"{name}_render.png")) \
            if "sam_image_path" in d.files else False
        has_mask = copy_png(str(d["sam_mask_path"]), os.path.join(folder, f"{name}_mask.png")) \
            if "sam_mask_path" in d.files else False

        np.savez_compressed(
            os.path.join(folder, f"{name}.npz"),
            vox=np.round(run["vox"] * COORD_SCALE).astype(np.int16), coord_scale=COORD_SCALE,
            labels=run["labels"].astype(np.int8), best=run["best"],
            best_mask=run["best_mask"], detached=run["detached"],
            colors=cluster_colors(np.arange(run["k"]), run["best"]),
            channels=np.array(names), heat_z=z, heat_raw=raw, counts=counts, mean_intensity=inten,
            k=run["k"], seed=run["seed"], n_init=int(d["kmeans_n_init"]) if "kmeans_n_init" in d.files else 1,
            dump=os.path.basename(path), dataset=dataset, name=name,
            sam_points=d["sam_points_image"] if "sam_points_image" in d.files else np.zeros((0, 2), np.int32),
            sam_types=d["sam_point_types"] if "sam_point_types" in d.files else np.zeros(0, "U8"),
            has_render=has_render, has_mask=has_mask, **extra)
        print(f"  {dataset}/{name}  ← {os.path.basename(path)} | 샘플 {len(run['labels']):,} k {run['k']} "
              f"best {run['best']} ({run['best_n']:,}개) | 최종 ROI "
              f"{int(extra['stage_b_mask'].sum()):,} voxel" if "stage_b_mask" in extra else
              f"  {dataset}/{name}  ← {os.path.basename(path)} | 최종 ROI 없음(Stage B 덤프 없음)")

    zip_path = shutil.make_archive(out, "zip", os.path.dirname(out), os.path.basename(out))   # 풀면 폴더 하나
    print(f"\n패키지: {out}\n압축본: {zip_path}  ({os.path.getsize(zip_path) / 2**20:.1f} MB) — 이 zip을 보내면 됩니다.")


if __name__ == "__main__":
    main()
