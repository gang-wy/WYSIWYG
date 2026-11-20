"""Stage A 덤프 폴더를 훑어 'INR 실행'과 'intensity 실행'의 짝을 자동으로 찾고 비교 창을 연다.

짝 조건(전부 같아야 함):
  - 샘플: world_points가 완전히 같다(같은 SAM mask·카메라·TF로 뽑은 같은 샘플) — 해시로 비교
  - k, K-means seed (덤프에 기록된 값 = 그 실행에서 UI seed로 K-means를 돌린 값)
  - INR/intensity 외의 feature 구성(coords, visibility, depth, opacity 등)
  - 한쪽은 INR 블록, 다른 쪽은 intensity 블록을 쓴다
같은 조건의 실행이 여러 개면 시간이 가장 가까운 것끼리 짝짓는다.

사용:
    .venv\\Scripts\\python.exe experiments\\pair_stage_a_dumps.py                       # 덤프 폴더 전체
    .venv\\Scripts\\python.exe experiments\\pair_stage_a_dumps.py --since 20261112_140000  # 이 시각 이후 덤프만
    옵션: --dataset brainix   볼륨 경로가 없는 옛 덤프용
          --list              표만 출력(짝 파일 안 만듦)
          --out-dir <폴더>    기본 resources/Experiments/pair_viewers
          --test test_1       이 test 폴더에 넣음(생략하면 test_<다음 번호>를 새로 만듦)
출력: 짝 표 + 짝마다 <out-dir>/<test 폴더>/<데이터셋>/<데이터셋>_k<k>_seed<seed>.bat
      — 탐색기에서 더블클릭하면 비교 창(왼쪽 INR, 오른쪽 intensity)이 열린다.
      같은 이름의 다른 실험 파일이 있으면 _2, _3을 붙이고, 같은 덤프 짝이면 덮어쓴다.
창 조작: 슬라이더 / ←→ 1칸 / ↑↓ 10칸 / b best만↔전체
"""
import argparse
import glob
import hashlib
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import compare_stage_a_slices as cmp  # noqa: E402

DUMP_DIR = os.path.join(cmp.ROOT, "resources/Experiments/stage_a_dump")
OUT_DIR = os.path.join(cmp.ROOT, "resources/Experiments/pair_viewers")


def read_meta(path):
    d = np.load(path)
    kind, rest = cmp.feature_kind(d["layout"])
    wp = np.round(d["world_points"].astype(np.float64), 3).astype(np.float32)
    vpath, raw = cmp.dump_volume_source(d)
    return {
        "path": path, "time": os.path.basename(path)[len("stage_a_"):-len(".npz")],
        "kind": kind, "rest": rest, "k": len(d["centroids"]), "seed": int(d["seed"]),
        "n": len(wp), "fp": hashlib.sha1(wp.tobytes()).hexdigest()[:10],
        "labels": np.asarray(d["labels"]), "best": int(d["best"]), "vpath": vpath, "raw": raw,
        # best 공간 분리를 켰으면 분리 후 남은 best(없으면 best 라벨 전체)
        "best_mask": (d["best_cluster_mask"].astype(bool) if "best_cluster_mask" in d.files
                      else np.asarray(d["labels"]) == int(d["best"])),
        "detached_n": int(d["detached_best_mask"].sum()) if "detached_best_mask" in d.files else 0,
        # K-means·정규화·후처리 설정 (기록이 없는 옛 덤프는 그때 기본값: 재시작 1, z-score, 분리 없음)
        "cfg": (int(d["kmeans_n_init"]) if "kmeans_n_init" in d.files else 1,
                bool(d["sparse_minmax"]) if "sparse_minmax" in d.files else False,
                bool(d["best_spatial_split"]) if "best_spatial_split" in d.files else False,
                "opacity_unit_distance" in d.files),   # step 보정 전/후 덤프는 짝짓지 않는다
        "inr_file": os.path.basename(str(d["inr_path"])) if "inr_path" in d.files else "",
        "roi": str(d["experiment_roi"]).replace(" ", "") if "experiment_roi" in d.files else "",
        # "depth_slabs" = Run 깊이 구간 덤프 (K-means 짝 비교 대상 아님)
        "method": str(d["stage_a_method"]) if "stage_a_method" in d.files else "kmeans",
    }


def dataset_name(m, fallback=None):
    """프리셋 볼륨과 경로가 같으면 프리셋 이름, 아니면 볼륨 파일 이름(확장자 제외)."""
    if m["vpath"]:
        norm = os.path.normcase(os.path.abspath(m["vpath"]))
        for key, preset in cmp.DATASET_PRESETS.items():
            if os.path.normcase(os.path.abspath(os.path.join(cmp.ROOT, preset["volume"]))) == norm:
                return key
        return os.path.basename(m["vpath"]).split(".")[0]
    return fallback or "unknown"


def tsec(t):
    return datetime.strptime(t, "%Y%m%d_%H%M%S").timestamp()


def find_pairs(metas):
    """Returns: (pairs [(inr, int)], 짝 없는 meta 목록)"""
    groups = {}
    for m in metas:
        if m["kind"] in ("INR", "intensity") and m["method"] == "kmeans":
            groups.setdefault((m["fp"], m["k"], m["seed"], m["rest"], m["cfg"]),
                              {"INR": [], "intensity": []})[m["kind"]].append(m)
    pairs, used = [], set()
    for g in groups.values():
        cands = sorted(((abs(tsec(a["time"]) - tsec(b["time"])), i, j)
                        for i, a in enumerate(g["INR"]) for j, b in enumerate(g["intensity"])))
        ui, uj = set(), set()
        for _, i, j in cands:
            if i not in ui and j not in uj:
                ui.add(i); uj.add(j)
                pairs.append((g["INR"][i], g["intensity"][j]))
                used.update((g["INR"][i]["path"], g["intensity"][j]["path"]))
    pairs.sort(key=lambda p: max(p[0]["time"], p[1]["time"]))
    return pairs, [m for m in metas if m["path"] not in used]


def why_unpaired(m, metas):
    """같은 샘플의 반대 종류 실행이 있는데 조건이 다르면 무엇이 다른지 알려준다."""
    want = "intensity" if m["kind"] == "INR" else "INR"
    if m["kind"] not in ("INR", "intensity"):
        return f"INR/intensity 블록이 없음({m['kind']})"
    same = [o for o in metas if o["kind"] == want and o["fp"] == m["fp"]]
    if not same:
        return f"같은 샘플의 {want} 실행 없음"
    diffs = []
    for o in same:
        dd = [f"{name} {m[key]}≠{o[key]}" for key, name in (("k", "k"), ("seed", "seed"), ("rest", "나머지 feature"),
                                                            ("cfg", "(재시작, V/O min-max, 공간 분리, step 보정)"))
              if m[key] != o[key]]
        diffs.append(f"{o['time']}({', '.join(dd) or '이미 다른 실행과 짝'})")
    return f"같은 샘플의 {want} 실행은 있으나 조건이 다름: " + "; ".join(diffs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="*", help="생략하면 덤프 폴더 전체")
    ap.add_argument("--since", help="YYYYMMDD_HHMMSS 이후 덤프만")
    ap.add_argument("--dataset", choices=sorted(cmp.DATASET_PRESETS), help="볼륨 경로가 없는 옛 덤프용")
    ap.add_argument("--list", action="store_true", help="표만 출력(짝 파일 안 만듦)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--test", help="넣을 test 폴더(예: test_1, out-dir 기준). 생략하면 test_<다음 번호>를 새로 만든다")
    args = ap.parse_args()

    paths = args.dumps or sorted(glob.glob(os.path.join(DUMP_DIR, "stage_a_*.npz")))
    if args.since:
        paths = [p for p in paths if os.path.basename(p)[len("stage_a_"):-len(".npz")] >= args.since]
    metas = [read_meta(p) for p in paths]
    pairs, alone = find_pairs(metas)

    # 짝 이름: 데이터셋_k_seed (같은 조건 짝이 여럿이면 _2, _3)
    names, used = [], {}
    for a, _ in pairs:
        roi = f"_{a['roi']}" if a["roi"] else ""     # Experiment 그룹으로 돌린 덤프는 ROI 이름이 있다
        stem = f"{dataset_name(a, args.dataset)}{roi}_k{a['k']}_seed{a['seed']}"
        used[stem] = used.get(stem, 0) + 1
        names.append(stem if used[stem] == 1 else f"{stem}_{used[stem]}")

    print(f"덤프 {len(metas)}개, 짝 {len(pairs)}개")
    print(f"{'#':>2}  {'이름':<22} {'INR 덤프':<16} {'intensity 덤프':<16} {'샘플':>8}  {'best INR':>9} {'best int':>9}  "
          f"겹침(Jaccard)  INR best 중 int best  떼어냄(INR/int)  재시작  INR 파일")
    for i, ((a, b), name) in enumerate(zip(pairs, names), 1):
        ba, bb = a["best_mask"], b["best_mask"]
        inter = int(np.sum(ba & bb))
        print(f"{i:>2}  {name:<22} {a['time']:<16} {b['time']:<16} {a['n']:>8,}  {int(ba.sum()):>9,} {int(bb.sum()):>9,}  "
              f"{inter / max(int((ba | bb).sum()), 1):>12.3f}  {inter / max(int(ba.sum()), 1):>18.1%}  "
              f"{a['detached_n']:,} / {b['detached_n']:,}  {a['cfg'][0]:>6}  {a['inr_file']}")
    if alone:
        print("\n짝 없는 덤프:")
        for m in alone:
            print(f"    {m['time']} [{m['kind']}, k {m['k']}, seed {m['seed']}, 샘플 {m['n']:,}] — {why_unpaired(m, metas)}")
    if not pairs or args.list:
        return

    # 짝마다 탐색기에서 더블클릭해 여는 .bat 저장(.py는 이 PC에서 더블클릭하면 편집기로 열려서 .bat을 쓴다)
    # 위치: <pair_viewers>/<test 폴더>/<데이터셋>/ — test 폴더는 --test로 지정, 없으면 test_<다음 번호> 새로 만듦
    if args.test:
        test_dir = args.test if os.path.isabs(args.test) else os.path.join(args.out_dir, args.test)
    else:
        nums = [int(d[5:]) for d in os.listdir(args.out_dir) if d.startswith("test_") and d[5:].isdigit()] \
            if os.path.isdir(args.out_dir) else []
        test_dir = os.path.join(args.out_dir, f"test_{max(nums, default=0) + 1}")
    venv_py = os.path.join(cmp.ROOT, ".venv", "Scripts", "python.exe")
    script = os.path.join(cmp.ROOT, "scripts", "compare_stage_a_slices.py")
    print(f"\n짝 파일 ({test_dir}):")
    for (a, b), name in zip(pairs, names):
        dumps = [os.path.abspath(a["path"]), os.path.abspath(b["path"])]
        folder = os.path.join(test_dir, dataset_name(a, args.dataset))
        os.makedirs(folder, exist_ok=True)
        title, n = name, 1
        while True:                                  # 같은 이름의 다른 실험 파일은 덮어쓰지 않는다(_2, _3)
            path = os.path.join(folder, f"{title}.bat")
            if not os.path.exists(path) or all(d in open(path, encoding="ascii", errors="ignore").read() for d in dumps):
                break
            n += 1
            title = f"{name}_{n}"
        dataset = args.dataset if not a["vpath"] else None
        with open(path, "w", encoding="ascii") as f:                 # cmd 코드페이지 문제로 영문만 쓴다
            f.write(f"@echo off\r\nrem {title}: INR vs intensity Stage A viewer (generated by experiments\\pair_stage_a_dumps.py)\r\n"
                    f"cd /d \"{cmp.ROOT}\"\r\n"
                    f"\"{venv_py}\" \"{script}\" \"{dumps[0]}\" \"{dumps[1]}\" --title {title}"
                    + (f" --dataset {dataset}" if dataset else "") + " || pause\r\n")
        print(f"    {os.path.relpath(path, test_dir)}  ← INR {a['time']} / intensity {b['time']}")


if __name__ == "__main__":
    main()
